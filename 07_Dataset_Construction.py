#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
07_Dataset_Construction.py

Build a year-long training dataset by pairing Sentinel-1 features with
Sentinel-2 NDVI/Cloud/Water masks using a precomputed S1->S2 date mapping.

Inputs (expected filenames in the folders; customizable via CLI patterns):
  - S1 rasters:          S1_<YYYYMMDD>.tif  (bands: VV, VH)
  - S2 NDVI+Cloud:       NDVI_CloudMask_<YYYYMMDD>_WGS84.tif  (bands: NDVI, CloudMask)
  - S2 Water mask:       WaterMask_<YYYYMMDD}_WGS84.tif       (1 band: 1=water, 0=non-water)
  - Mapping JSON:        {"<S1_YYYYMMDD>": "<S2_YYYYMMDD>", ...}

Outputs (in --out-dir):
  - fullyear_S1.npy   : shape (T, 5, H, W) with normalized S1 feature channels
  - fullyear_S2.npy   : shape (T, H, W, 3) with [NDVI_norm, CloudMask, WaterMask]
  - normalization_params.pkl : dict with min/max per S1 channel and NDVI
  - geo_info.pkl      : dict with {"transform": tuple, "crs": "EPSG:XXXX" or WKT string}
  - good_pixel.npy    : (H, W) boolean mask for pixels having >= MIN_CLEAR_FRAMES clear observations
  - preview/good_pixel_map.png : quicklook of good pixels

S1 feature channels (computed in linear domain):
  [0] vh,
  [1] rvi       = 4 * VH / (VV + VH + EPS)
  [2] vv_div    = VV / (VH + EPS)
  [3] vv_diff   = VV - VH
  [4] log_ratio = log1p( clip(VV / (VH + EPS), min=-0.999) )

Clear/valid pixel per frame:
  CloudMask == 0  AND  WaterMask == 0

Temporal smoothing for NDVI:
  - Robust HANTS-like harmonic fit (cos/sin up to --hants-harmonics with outlier rejection)
  - Whittaker smoothing (lambda = --whit-lambda)

CLI example:
  python 07_Dataset_Construction.py \
    --mapping ./s1_to_s2_mapping_filtered.json \
    --s1-dir ./S12024_clipped_HS \
    --s2-dir ./S22024_clipped_HS \
    --out-dir ./fullyear_2024_HS \
    --min-clear 12 --threads 8 --hants-harmonics 3 --whit-lambda 1e5
"""

import os
import re
import json
import pickle
import argparse
import datetime as dt
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import cpu_count

import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
from tqdm import tqdm
from scipy import sparse
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt


# ------------------------------ CLI ------------------------------ #

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Construct S1+S2 NDVI dataset with QA and smoothing.")
    p.add_argument("--mapping", required=True, help="JSON file with S1->S2 date mapping.")
    p.add_argument("--s1-dir", required=True, help="Directory containing S1 GeoTIFFs (VV,VH).")
    p.add_argument("--s2-dir", required=True, help="Directory containing S2 NDVI/Cloud/Water products.")
    p.add_argument("--out-dir", required=True, help="Output directory.")
    p.add_argument("--s1-pattern", default=r"^S1_(\d{8})\.tif$", help="Regex for S1 filenames.")
    p.add_argument("--s2-nc-pattern", default="NDVI_CloudMask_{date}_WGS84.tif",
                   help="S2 NDVI+Cloud filename template with {date}=YYYYMMDD.")
    p.add_argument("--s2-wm-pattern", default="WaterMask_{date}_WGS84.tif",
                   help="S2 Water mask filename template with {date}=YYYYMMDD.")
    p.add_argument("--min-clear", type=int, default=12,
                   help="Minimum number of clear frames per pixel to be considered good (default: 12).")
    p.add_argument("--threads", type=int, default=max(1, cpu_count() - 1),
                   help="Max workers for I/O reading (default: CPU-1).")
    p.add_argument("--hants-harmonics", type=int, default=3,
                   help="Number of harmonics in robust harmonic fit (default: 3).")
    p.add_argument("--hants-outlier", type=float, default=0.10,
                   help="Outlier proportion for robust fit (default: 0.10).")
    p.add_argument("--hants-iter", type=int, default=3,
                   help="Max iterations for robust fit (default: 3).")
    p.add_argument("--whit-lambda", type=float, default=1e5,
                   help="Whittaker smoothing lambda (default: 1e5).")
    p.add_argument("--preview", action="store_true",
                   help="Save preview good_pixel_map.png (default: on).")
    return p.parse_args()


# --------------------------- Parameters -------------------------- #

EPS = 1e-6
CH_NAMES = ["vh", "rvi", "vv_div", "vv_diff", "log_ratio"]  # 5 channels in linear domain


# ------------------------ Feature Construction ------------------- #

def feats_s1(vv: np.ndarray, vh: np.ndarray) -> np.ndarray:
    """Compute S1 feature stack in linear power domain with NaN/Inf guards."""
    vv = np.nan_to_num(vv, nan=0.0, posinf=1.0, neginf=-1.0)
    vh = np.nan_to_num(vh, nan=0.0, posinf=1.0, neginf=-1.0)

    rvi = 4.0 * vh / (vv + vh + EPS)
    vv_div = vv / (vh + EPS)
    vv_diff = vv - vh
    log_ratio = np.log1p(np.clip(vv / (vh + EPS), a_min=-0.999, a_max=None))

    out = np.stack([vh, rvi, vv_div, vv_diff, log_ratio], axis=0).astype(np.float32)
    return np.nan_to_num(out, nan=0.0, posinf=1.0, neginf=-1.0)


def reproj(src_arr, src_tr, src_crs, dst_tr, dst_crs, H, W):
    dst = np.zeros((H, W), dtype=src_arr.dtype)
    reproject(
        src_arr, dst,
        src_transform=src_tr, src_crs=src_crs,
        dst_transform=dst_tr, dst_crs=dst_crs,
        resampling=Resampling.bilinear
    )
    return dst


# --------------------------- Smoothing --------------------------- #

def robust_harmonic_fit(ts: np.ndarray,
                        dates: list[dt.datetime],
                        n_harm: int,
                        outlier_pct: float,
                        max_iter: int) -> np.ndarray:
    """HANTS-like robust harmonic regression with iterative outlier rejection."""
    T = len(ts)
    t = np.array([d.timetuple().tm_yday for d in dates], dtype=np.float64)

    X = [np.ones(T, dtype=np.float64)]
    for k in range(1, n_harm + 1):
        X.append(np.cos(2 * np.pi * k * t / 365.0))
        X.append(np.sin(2 * np.pi * k * t / 365.0))
    X = np.stack(X, axis=1)

    mask = ~np.isnan(ts)
    fit = np.full(T, np.nan, dtype=np.float64)
    if mask.sum() < X.shape[1]:
        # Not enough valid points to solve; return zeros (will be handled downstream)
        return np.nan_to_num(fit)

    for _ in range(max_iter):
        coef, *_ = np.linalg.lstsq(X[mask], ts[mask], rcond=None)
        fit = X @ coef
        err = np.abs(ts - fit)
        thr = np.percentile(err[mask], 100.0 * (1.0 - outlier_pct))
        new_mask = mask & (err <= thr)
        if new_mask.sum() == mask.sum():
            break
        mask = new_mask
        if mask.sum() < X.shape[1]:
            break

    return fit.astype(np.float32)


def whittaker_smooth(y: np.ndarray, w: np.ndarray, lam: float) -> np.ndarray:
    """Classic Whittaker smoother: (W + λ DᵀD) z = Wy."""
    m = len(y)
    if m < 3:
        return y.astype(np.float32)
    D = sparse.diags([1, -2, 1], [0, 1, 2], shape=(m - 2, m))
    W = sparse.diags(w.astype(float), 0)
    A = W + lam * (D.T @ D)
    z = W @ y
    return spsolve(A, z).astype(np.float32)


# ------------------------------ I/O ------------------------------ #

def load_mapping(mapping_fp: str) -> list[tuple[str, str]]:
    with open(mapping_fp, "r", encoding="utf-8") as f:
        mapping = json.load(f)
    # mapping: {S1_YYYYMMDD: S2_YYYYMMDD}
    pairs = list(mapping.items())
    # sort by S2 date to get chronological target sequence
    pairs.sort(key=lambda kv: kv[1])
    return pairs


def find_reference_grid(s2_dir: str, s2_nc_pat: str):
    # Pick first NDVI_CloudMask file as reference grid
    for fn in sorted(os.listdir(s2_dir)):
        if "ndvi_cloudmask_" in fn.lower() and fn.lower().endswith(".tif"):
            ref_fp = os.path.join(s2_dir, fn)
            with rasterio.open(ref_fp) as ref:
                return ref.transform, ref.crs, ref.height, ref.width
    raise FileNotFoundError("No NDVI_CloudMask_* GeoTIFF found to derive reference grid.")


def read_pair(item, s1_dir, s2_dir, dst_tr, dst_crs, H, W,
              s1_regex: re.Pattern, s2_nc_tmpl: str, s2_wm_tmpl: str):
    idx, (s1_d, s2_d) = item
    try:
        # S2 products
        s2_nc = os.path.join(s2_dir, s2_nc_tmpl.format(date=s2_d))
        if not os.path.exists(s2_nc):
            return None

        with rasterio.open(s2_nc) as src:
            ndvi = src.read(1).astype(np.float32)
            cmask = src.read(2).astype(np.uint8)

        if np.isnan(ndvi).all() or (cmask > 0).all():
            return None  # degenerate frame

        s2_wm = os.path.join(s2_dir, s2_wm_tmpl.format(date=s2_d))
        if not os.path.exists(s2_wm):
            return None
        with rasterio.open(s2_wm) as src:
            w_arr = src.read(1).astype(np.uint8)
            wmask = reproj(w_arr, src.transform, src.crs, dst_tr, dst_crs, H, W).astype(np.uint8)

        # S1 product
        s1_name = f"S1_{s1_d}.tif"
        if not s1_regex.search(s1_name):
            return None
        s1_fp = os.path.join(s1_dir, s1_name)
        if not os.path.exists(s1_fp):
            return None

        with rasterio.open(s1_fp) as src:
            vv = reproj(src.read(1).astype(np.float32), src.transform, src.crs, dst_tr, dst_crs, H, W)
            vh = reproj(src.read(2).astype(np.float32), src.transform, src.crs, dst_tr, dst_crs, H, W)

        feat = feats_s1(vv, vh)

        # Valid mask: cloud==0 & water==0 (clear land/vegetation)
        vmask = (cmask == 0) & (wmask == 0)
        date = dt.datetime.strptime(s2_d, "%Y%m%d")
        return idx, feat, ndvi, cmask, wmask, vmask, date

    except Exception as e:
        print(f"[WARN] Pair {s1_d}->{s2_d} skipped: {e}")
        return None


# ------------------------------ Main ----------------------------- #

def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, "preview"), exist_ok=True)

    # Load mapping and reference grid
    pairs = load_mapping(args.mapping)
    if not pairs:
        raise RuntimeError("Empty mapping.")
    print(f"[INFO] Loaded {len(pairs)} S1->S2 pairs.")

    dst_tr, dst_crs, H, W = find_reference_grid(args.s2_dir, args.s2_nc_pattern)
    print(f"[INFO] Reference grid: {W}x{H}, CRS={dst_crs}")

    s1_regex = re.compile(args.s1_pattern)

    # Read all pairs in parallel
    reader = (read_pair(item,
                        args.s1_dir, args.s2_dir,
                        dst_tr, dst_crs, H, W,
                        s1_regex, args.s2_nc_pattern, args.s2_wm_pattern)
              for item in enumerate(pairs))

    results = list(tqdm(ThreadPoolExecutor(max_workers=args.threads).map(lambda x: x, reader),
                        total=len(pairs), desc="Reading pairs"))
    results = [r for r in results if r is not None]
    if not results:
        raise RuntimeError("No valid S1/S2 pairs after filtering.")

    # Sort by original index to preserve chronological order
    results.sort(key=lambda x: x[0])
    _, feats_list, ndvi_list, cmask_list, wmask_list, mask_list, dates = zip(*results)
    T = len(dates)

    # Stack arrays
    raw_feats = np.stack(feats_list).astype(np.float32)   # (T, 5, H, W)
    raw_ndvi  = np.stack(ndvi_list).astype(np.float32)    # (T, H, W)
    raw_cmask = np.stack(cmask_list).astype(np.uint8)     # (T, H, W)
    raw_wmask = np.stack(wmask_list).astype(np.uint8)     # (T, H, W)
    raw_mask  = np.stack(mask_list).astype(bool)          # (T, H, W)

    # Build "good pixel" mask
    flatN = H * W
    mask_flat = raw_mask.reshape(T, flatN)
    clean_cnt = mask_flat.sum(axis=0)
    good_pixel = clean_cnt >= int(args.min_clear)
    good_idxs = np.where(good_pixel)[0]
    print(f"[INFO] Good pixels: {good_pixel.sum()} (>= {args.min_clear} clear frames)")

    # --- NDVI smoothing (HANTS-like + Whittaker) ---
    ndvi_flat = raw_ndvi.reshape(T, flatN)
    sm_flat = np.full((T, flatN), np.nan, dtype=np.float32)
    for idx in tqdm(good_idxs, desc="Smoothing NDVI (Harmonic+Whittaker)"):
        ts = ndvi_flat[:, idx]
        w = mask_flat[:, idx].astype(float)
        fit = robust_harmonic_fit(ts, list(dates),
                                  n_harm=args.hants_harmonics,
                                  outlier_pct=args.hants_outlier,
                                  max_iter=args.hants_iter)
        sm_flat[:, idx] = whittaker_smooth(fit, w, lam=float(args.whit_lambda))

    # Remove spikes (simple gradient-based rule)
    doys = np.array([d.timetuple().tm_yday for d in dates], dtype=int)
    for idx in tqdm(good_idxs, desc="Spike removal"):
        ts = sm_flat[:, idx]
        for t in range(1, T - 1):
            Ca = (ts[t] - ts[t - 1]) / max(doys[t] - doys[t - 1], 1)
            Cb = (ts[t + 1] - ts[t]) / max(doys[t + 1] - doys[t], 1)
            Cg = (ts[t + 1] - ts[t - 1]) / max(doys[t + 1] - doys[t - 1], 1)
            if (Ca <= -0.15) and (Cb >= 0.15) and (Cg >= 0.05):
                mask_flat[t, idx] = False

    # Final valid mask and NDVI normalization range
    mask_flat[:, ~good_pixel] = False
    sm_flat[:, ~good_pixel] = np.nan
    valid_flat = mask_flat & (~np.isnan(sm_flat))

    if not np.any(valid_flat):
        raise RuntimeError("No valid NDVI samples after smoothing and masking.")

    vals = sm_flat[valid_flat]
    loN, hiN = float(np.min(vals)), float(np.max(vals))
    print(f"[INFO] NDVI range after smoothing: min={loN:.4f}, max={hiN:.4f}")

    # --- Normalize S1 channels based on valid pixels only ---
    feats_flat = raw_feats.reshape(T, 5, flatN)
    S1_norm_flat = np.zeros_like(feats_flat, dtype=np.float32)
    S1_lohi: dict[str, dict[str, float]] = {}

    print("\n[INFO] Sentinel-1 channel QA:")
    for ci, ch in enumerate(CH_NAMES):
        ch_vals = feats_flat[:, ci]
        ch_valid = ch_vals[valid_flat]
        lo = float(np.min(ch_valid))
        hi = float(np.max(ch_valid))
        S1_lohi[ch] = {"min": lo, "max": hi}
        S1_norm_flat[:, ci] = (ch_vals - lo) / (hi - lo + EPS)

        has_nan = bool(np.isnan(ch_vals).any())
        has_inf = bool(np.isinf(ch_vals).any())
        all_zero = bool(np.allclose(ch_vals, 0))
        print(f"  - {ch:9s} | NaN={has_nan}  Inf={has_inf}  AllZero={all_zero}  min={lo:.4f}  max={hi:.4f}")

    # --- Pack S2 tensor: [NDVI_norm, CloudMask, WaterMask] ---
    S2_norm = np.zeros((T, H, W, 3), dtype=np.float32)
    S2_norm[..., 0] = (sm_flat.reshape(T, H, W) - loN) / (hiN - loN + EPS)
    S2_norm[..., 1] = raw_cmask.astype(np.float32)
    S2_norm[..., 2] = raw_wmask.astype(np.float32)

    # --- Save outputs ---
    np.save(os.path.join(args.out_dir, "fullyear_S1.npy"),
            S1_norm_flat.reshape(T, 5, H, W))
    np.save(os.path.join(args.out_dir, "fullyear_S2.npy"), S2_norm)

    with open(os.path.join(args.out_dir, "normalization_params.pkl"), "wb") as f:
        pickle.dump({
            "dates": [d.strftime("%Y%m%d") for d in dates],
            "S1": S1_lohi,
            "S2": {"ndvi": {"min": loN, "max": hiN}}
        }, f)

    # Serialize geo info in a portable way
    crs_str = None
    try:
        crs_str = dst_crs.to_string()
    except Exception:
        crs_str = str(dst_crs)

    with open(os.path.join(args.out_dir, "geo_info.pkl"), "wb") as f:
        pickle.dump({"transform": tuple(dst_tr), "crs": crs_str}, f)

    np.save(os.path.join(args.out_dir, "good_pixel.npy"), good_pixel.reshape(H, W))

    if args.preview:
        plt.figure(figsize=(6, 6))
        plt.axis("off")
        plt.imshow(good_pixel.reshape(H, W).astype(np.uint8), cmap="gray")
        plt.tight_layout()
        plt.savefig(os.path.join(args.out_dir, "preview", "good_pixel_map.png"), dpi=200)
        plt.close()

    print(f"\n[INFO] Dataset built. Good pixels: {int(good_pixel.sum())} (>= {args.min_clear} clear frames)")
    print(f"[INFO] Output directory: {args.out_dir}")


if __name__ == "__main__":
    main()
