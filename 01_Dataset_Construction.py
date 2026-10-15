# -*- coding: utf-8 -*-
"""
Build an S1-centric dataset for STReF-NDVI.

The temporal axis follows the Sentinel-1 dates in the matching table.
Missing Sentinel-2 labels do not remove Sentinel-1 dates.

Inputs:
    S1/S1_YYYYMMDD.tif
        Two bands: VV and VH, in dB by default.

    S2_NDVI_CloudMask/NDVI_CloudMask_YYYYMMDD_WGS84.tif
        Two bands: NDVI and cloud mask.

    WaterMask/WaterMask_YYYYMMDD_WGS84.tif
        One band: water mask.

    Matching/S1_S2_Label_Mapping.csv
        Sentinel-1 to Sentinel-2 label mapping.

Outputs:
    S1.npy              (T, 5, H, W)
    S2.npy              (T, H, W, 3): NDVI, cloud, water
    target_mask.npy     (T, H, W)
    S1_valid.npy        (T, H, W)
    good_pixel.npy      (H, W)
    dates.csv
    metadata.json
    normalization.pkl

Input rasters must already be spatially aligned.
Cloud and water masks use zero for clear and non-water pixels, respectively.
"""

import os
import json
import pickle

import numpy as np
import pandas as pd
import rasterio

from tqdm import tqdm


# ==========================================================
# PATHS
# ==========================================================

BASE_DIR = os.path.join("data")

S1_DIR = os.path.join(BASE_DIR, "S1")
S2_DIR = os.path.join(BASE_DIR, "S2_NDVI_CloudMask")
WATER_DIR = os.path.join(BASE_DIR, "WaterMask")

MAPPING_FILE = os.path.join(BASE_DIR, "Matching", "S1_S2_Label_Mapping.csv")

OUT_DIR = os.path.join(BASE_DIR, "Dataset")
os.makedirs(OUT_DIR, exist_ok=True)


# ==========================================================
# CONSTANTS
# ==========================================================

EPS = 1e-6

# S1 raw is in dB; convert to linear power before feature computation.
S1_IN_DB = True

S1_CHANNELS = ["vh", "rvi", "vv_div", "vv_diff", "log_ratio"]

# percentile clip for S1 normalization
S1_PCT_LOW = 1.0
S1_PCT_HIGH = 99.0

# NDVI supervision bounds (tightened from [-1, 1])
NDVI_MIN = -0.2
NDVI_MAX = 1.0

# good_pixel: minimum number of supervised frames per pixel
MIN_CLEAR_FRAMES = 8


# ==========================================================
# Sentinel-1 feature
# ==========================================================

def db_to_linear(x):
    """dB power -> linear power."""
    return np.power(10.0, x / 10.0)


def build_s1_features(vv, vh):
    """
    Compute 5 S1 channels.

    vv, vh: raw raster values.
      If S1_IN_DB is True they are dB, otherwise already linear.

    Returns (5, H, W) float32.
    """
    if S1_IN_DB:
        vv_lin = db_to_linear(vv)
        vh_lin = db_to_linear(vh)
    else:
        vv_lin = vv
        vh_lin = vh

    vv_lin = np.nan_to_num(vv_lin, nan=0.0, posinf=0.0, neginf=0.0)
    vh_lin = np.nan_to_num(vh_lin, nan=0.0, posinf=0.0, neginf=0.0)

    rvi = 4.0 * vh_lin / (vv_lin + vh_lin + EPS)
    vv_div = vv_lin / (vh_lin + EPS)
    vv_diff = vv_lin - vh_lin
    log_ratio = np.log1p(np.clip(vv_lin / (vh_lin + EPS), a_min=-0.999, a_max=None))

    out = np.stack(
        [vh_lin, rvi, vv_div, vv_diff, log_ratio],
        axis=0
    ).astype(np.float32)

    out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    return out


# ==========================================================
# Reading
# ==========================================================

def read_s1(fp):
    """
    Returns (features (5,H,W) float32, valid_mask (H,W) bool).
    """
    with rasterio.open(fp) as src:
        vv = src.read(1).astype(np.float32)
        vh = src.read(2).astype(np.float32)

    valid_mask = np.isfinite(vv) & np.isfinite(vh)
    feats = build_s1_features(vv, vh)
    return feats, valid_mask


def read_s2(fp):
    """Returns (ndvi (H,W) float32, cloud (H,W) float32)."""
    with rasterio.open(fp) as src:
        ndvi = src.read(1).astype(np.float32)
        cloud = src.read(2).astype(np.float32)
    return ndvi, cloud


def read_water(fp):
    """Returns water (H,W) float32."""
    with rasterio.open(fp) as src:
        water = src.read(1).astype(np.float32)
    return water


# ==========================================================
# Load mapping
# ==========================================================

print("\nLoading S1-S2 mapping")

mapping = pd.read_csv(MAPPING_FILE, dtype=str)
print(mapping.head())

for col in ["S1_date", "S2_label_date", "S1_file", "S2_label_file"]:
    mapping[col] = mapping[col].fillna("").astype(str).str.strip()

mapping["matched_bool"] = (
    mapping["matched"].astype(str).str.lower().isin(["true", "1", "yes"])
)

mapping_dict = {}
for _, row in mapping.iterrows():
    mapping_dict[row["S1_date"]] = {
        "S1_file": row["S1_file"],
        "S2_label_file": row["S2_label_file"],
        "S2_label_date": row["S2_label_date"],
        "matched": bool(row["matched_bool"]),
    }


# ==========================================================
# Get all S1 dates
# ==========================================================

print("\nCollecting S1 dates")

s1_dates = sorted(mapping["S1_date"].tolist())
print(f"Total S1 dates: {len(s1_dates)}")


# ==========================================================
# Build dataset
# ==========================================================

S1_list = []
S2_list = []
target_list = []
valid_list = []
records = []

print("\nBuilding dataset")

for d in tqdm(s1_dates):

    info = mapping_dict.get(d, {})
    s1_file = info.get("S1_file", f"S1_{d}.tif")
    s2_file = info.get("S2_label_file", "")
    s2_date = info.get("S2_label_date", "")
    is_matched = info.get("matched", False)

    # ---------------- S1 ----------------
    s1_path = os.path.join(S1_DIR, s1_file)
    s1, s1_valid = read_s1(s1_path)

    H, W = s1.shape[1:]
    S1_list.append(s1)
    valid_list.append(s1_valid)

    # ---------------- S2 label ----------------
    s2_loaded = False

    if is_matched and s2_file and s2_date:

        ndvi_path = os.path.join(S2_DIR, s2_file)
        wm_name = f"WaterMask_{s2_date}_WGS84.tif"
        wm_path = os.path.join(WATER_DIR, wm_name)

        if os.path.exists(ndvi_path) and os.path.exists(wm_path):

            ndvi, cloud = read_s2(ndvi_path)
            water = read_water(wm_path)

            # tightened NDVI bound (was -1.0)
            target = (
                (cloud == 0)
                & (water == 0)
                & np.isfinite(ndvi)
                & (ndvi >= NDVI_MIN)
                & (ndvi <= NDVI_MAX)
            )

            s2 = np.stack([ndvi, cloud, water], axis=-1).astype(np.float32)
            s2_loaded = True

        else:
            print(f"  [WARN] {d}: matched but file missing "
                  f"({s2_file} / {wm_name})")

    # fallback: no S2 supervision for this frame
    if not s2_loaded:
        s2 = np.zeros((H, W, 3), dtype=np.float32)
        s2[:, :, 0] = np.nan
        s2[:, :, 1] = 1.0
        s2[:, :, 2] = 1.0
        target = np.zeros((H, W), dtype=bool)

    S2_list.append(s2)
    target_list.append(target)

    records.append({
        "S1_date": d,
        "S2_label_date": s2_date if s2_loaded else "",
        "supervised": bool(target.any()),
    })


# ==========================================================
# Stack
# ==========================================================

S1 = np.stack(S1_list)                # (T, 5, H, W)
S2 = np.stack(S2_list)                # (T, H, W, 3)
target_mask = np.stack(target_list)   # (T, H, W)
S1_valid = np.stack(valid_list)       # (T, H, W)

T = S1.shape[0]

print("\nDataset shape")
print("S1          :", S1.shape)
print("S2          :", S2.shape)
print("Target mask :", target_mask.shape)
print("S1 valid    :", S1_valid.shape)


# ==========================================================
# Normalization  (percentile clip, NOT min/max)
# ==========================================================

norm = {}
S1_norm = np.zeros_like(S1, dtype=np.float32)

print("\nNormalizing S1 channels (percentile clip):")
print(f"{'channel':10s} {'p_lo':>12s} {'p_hi':>12s} {'min':>10s} {'max':>10s} {'mean':>10s}")

for i, ch in enumerate(S1_CHANNELS):

    vals = S1[:, i]

    p_lo = float(np.percentile(vals, S1_PCT_LOW))
    p_hi = float(np.percentile(vals, S1_PCT_HIGH))

    norm[ch] = {
        "p_low": p_lo,
        "p_high": p_hi,
        "pct_low": S1_PCT_LOW,
        "pct_high": S1_PCT_HIGH,
    }

    S1_norm[:, i] = np.clip(
        (vals - p_lo) / (p_hi - p_lo + EPS),
        0.0, 1.0
    )

    print(f"{ch:10s} {p_lo:12.6f} {p_hi:12.6f} "
          f"{float(S1_norm[:, i].min()):10.4f} "
          f"{float(S1_norm[:, i].max()):10.4f} "
          f"{float(S1_norm[:, i].mean()):10.4f}")

S1 = S1_norm


# ==========================================================
# Good pixel
# ==========================================================

good_pixel = (target_mask.sum(axis=0) >= MIN_CLEAR_FRAMES)
print(f"\nGood pixels: {int(good_pixel.sum())} (>= {MIN_CLEAR_FRAMES} supervised frames)")


# ==========================================================
# Save
# ==========================================================

np.save(os.path.join(OUT_DIR, "S1.npy"), S1)
np.save(os.path.join(OUT_DIR, "S2.npy"), S2)
np.save(os.path.join(OUT_DIR, "target_mask.npy"), target_mask)
np.save(os.path.join(OUT_DIR, "S1_valid.npy"), S1_valid)
np.save(os.path.join(OUT_DIR, "good_pixel.npy"), good_pixel)

pd.DataFrame(records).to_csv(
    os.path.join(OUT_DIR, "dates.csv"),
    index=False,
    encoding="utf-8-sig",
)

metadata = {
    "time_steps": int(T),
    "height": int(H),
    "width": int(W),
    "dates_s1": s1_dates,
    "n_supervised": int(target_mask.any(axis=(1, 2)).sum()),
    "channels": S1_CHANNELS,
    "s1_in_db": bool(S1_IN_DB),
    "min_clear_frames": int(MIN_CLEAR_FRAMES),
    "normalization": "percentile_clip",
    "s1_pct_low": S1_PCT_LOW,
    "s1_pct_high": S1_PCT_HIGH,
    "ndvi_min": NDVI_MIN,
    "ndvi_max": NDVI_MAX,
}

with open(os.path.join(OUT_DIR, "metadata.json"), "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=2, ensure_ascii=False)

with open(os.path.join(OUT_DIR, "normalization.pkl"), "wb") as f:
    pickle.dump(norm, f)

print("\n==========================")
print("Dataset construction DONE")
print("==========================")
print(f"T = {T}")
print(f"Supervised frames = {metadata['n_supervised']}")
print(f"Output: {OUT_DIR}")
if __name__=='__main__':
    main()
