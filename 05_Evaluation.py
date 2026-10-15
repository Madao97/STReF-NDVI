# -*- coding: utf-8 -*-
r"""
Evaluate reconstructed NDVI at supplied evaluation coordinates.

Inputs:
    ndvi_pred.npy       (T, H, W): reconstructed NDVI
    ndvi_true.npy       (T, H, W): reference NDVI, when available
    val_coords.npy      (N, 3): coordinates ordered as (t, row, col)
    metadata.json       Sentinel-1 dates in dates_s1

If ndvi_true.npy is unavailable, reference NDVI is read from S2.npy
and masked using target_mask.npy.

Evaluation:
    Samples with nonfinite reference or predicted NDVI are excluded.
    R-squared, RMSE, MAE, and a linear regression fit are calculated.
    Per-date results are generated only when the sample count reaches
    --min-samples-per-frame.
    Pooled results use all retained samples, including samples from
    dates below the per-date reporting threshold.

The supplied coordinates define the evaluation population.
This script does not establish independence from training or
model-selection data and does not recheck target_mask when
ndvi_true.npy is supplied.

Outputs:
    per_date_metrics.csv
    yearly_metrics.txt
    plots/scatter_<DATE>.png
    plots/scatter_year_all.png

The existing yearly output filenames are retained for compatibility.
Pooled evaluation covers all supplied dates; it does not group by year.

Plots:
    Times New Roman by default.
    White-to-Jet density coloring with Gaussian smoothing.
    Default axis limits: [-0.5, 1.0].
    Equal aspect ratio, no title, and no grid.
    Metrics and regression use all retained samples, including values
    outside the displayed axis limits.

"""

import os
import csv
import json
import argparse

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from scipy.ndimage import gaussian_filter
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from tqdm import tqdm

# =============================================================
# DEFAULT PATHS
# Override these paths using command-line arguments.
# =============================================================
DATASET_DIR = os.path.join("data", "Dataset")
PRED_DIR = os.path.join("outputs", "Reconstruction")
OUT_DIR = os.path.join("outputs", "Evaluation")
VAL_NPY = os.path.join("outputs", "Evaluation_set", "val_coords.npy")


# ------------------------------ CLI -------------------------------- #

def parse_args():
p = argparse.ArgumentParser(
    description="Evaluate reconstructed NDVI at supplied evaluation coordinates."
)
    p.add_argument("--dataset-dir", default=DATASET_DIR,
                   help="Directory with S2.npy, target_mask.npy, metadata.json")
    p.add_argument("--pred-dir", default=PRED_DIR,
                   help="Directory with ndvi_pred.npy (and optionally ndvi_true.npy)")
    p.add_argument("--out-dir", default=OUT_DIR)
    p.add_argument("--val-npy", default=VAL_NPY,
                   help="Path to val_coords.npy (N,3) with columns (t,row,col)")
    p.add_argument("--min-samples-per-frame", type=int, default=20000)
    p.add_argument("--bins2d", type=int, default=200)
    p.add_argument("--sigma", type=float, default=1.0)
    p.add_argument("--font-family", default="Times New Roman")
    p.add_argument("--font-size", type=int, default=16)
    p.add_argument("--dpi", type=int, default=500)
    p.add_argument("--xlim", nargs=2, type=float, default=[-0.5, 1.0])
    p.add_argument("--ylim", nargs=2, type=float, default=[-0.5, 1.0])
    return p.parse_args()


# --------------------------- Plot Styling -------------------------- #

def setup_matplotlib(font_family, font_size):
    # White → Jet colormap: zero-density bin rendered as pure white
    base_jet = plt.get_cmap("jet", 256)
    colors = base_jet(np.linspace(0, 1, 256))
    colors[0] = [1, 1, 1, 1]
    cmap_white_jet = LinearSegmentedColormap.from_list("white_jet", colors)

    plt.rcParams.update({
        "font.family": font_family,
        "font.size": font_size,
        "axes.linewidth": 1.0,
        "xtick.direction": "out", "ytick.direction": "out",
        "xtick.major.width": 1.0, "ytick.major.width": 1.0,
        "xtick.major.size": 4,    "ytick.major.size": 4,
    })
    return cmap_white_jet


# --------------------------- Plot Function ------------------------- #

def density_scatter(y_true, y_pred, out_png, cmap,
                    bins=200, sigma=1.0, dpi=500,
                    xlim=(-0.5, 1.0), ylim=(-0.5, 1.0)):
    """White→Jet density scatter with linear fit and metrics box."""
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_t = y_true[mask]
    y_p = y_pred[mask]
    if y_t.size == 0:
        raise ValueError("No valid points to plot.")

    vmin, vmax = xlim[0], xlim[1]

    H2d, _, _ = np.histogram2d(
        y_t, y_p, bins=bins,
        range=[[vmin, vmax], [vmin, vmax]]
    )
    H2d = gaussian_filter(H2d, sigma=sigma)

    slope, intercept = np.polyfit(y_t, y_p, 1)
    R2   = float(r2_score(y_t, y_p))
    RMSE = float(np.sqrt(mean_squared_error(y_t, y_p)))
    MAE  = float(mean_absolute_error(y_t, y_p))

    fig, ax = plt.subplots(figsize=(6, 6), facecolor="white")
    normc = Normalize(vmin=H2d.min(), vmax=H2d.max(), clip=False)
    im = ax.imshow(
        H2d.T, origin="lower",
        extent=[vmin, vmax, vmin, vmax],
        cmap=cmap, norm=normc, aspect="auto"
    )
    ax.plot([vmin, vmax], [vmin, vmax], "k--", lw=1, zorder=3)
    ax.plot([vmin, vmax],
            [slope * vmin + intercept, slope * vmax + intercept],
            "r-", lw=2, zorder=3)

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal", adjustable="box")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xticks(np.linspace(xlim[0], xlim[1], 4))
    ax.set_yticks(np.linspace(ylim[0], ylim[1], 4))
    ax.set_xlabel("True NDVI")
    ax.set_ylabel("Predicted NDVI")

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Counts in bin")

    txt = (f"y = {slope:.3f}x + {intercept:.3f}\n"
           f"R² = {R2:.3f}\n"
           f"RMSE = {RMSE:.3f}\n"
           f"MAE = {MAE:.3f}")
    ax.text(0.02, 0.98, txt, transform=ax.transAxes,
            va="top", ha="left",
            bbox=dict(facecolor="white", alpha=0.8, pad=4))

    plt.tight_layout()
    fig.savefig(out_png, dpi=dpi)
    plt.close(fig)
    return R2, RMSE, MAE, slope, intercept


# ------------------------------- Main ------------------------------ #

def main():
    args = parse_args()
    cmap = setup_matplotlib(args.font_family, args.font_size)

    os.makedirs(args.out_dir, exist_ok=True)
    plots_dir = os.path.join(args.out_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    per_date_csv = os.path.join(args.out_dir, "per_date_metrics.csv")
    yearly_txt   = os.path.join(args.out_dir, "yearly_metrics.txt")

    # ---- 1) Load predictions + truth ----
    pred_fp = os.path.join(args.pred_dir, "ndvi_pred.npy")
    true_fp = os.path.join(args.pred_dir, "ndvi_true.npy")

    if not os.path.exists(pred_fp):
        raise FileNotFoundError(pred_fp)
    NDVI_pred = np.load(pred_fp).astype(np.float32)                # (T,H,W) raw NDVI

    if os.path.exists(true_fp):
        NDVI_true = np.load(true_fp).astype(np.float32)            # (T,H,W) raw, NaN where no label
    else:
        # fall back: read from S2.npy (raw NDVI in band 0), mask with target_mask
        S2 = np.load(os.path.join(args.dataset_dir, "S2.npy")).astype(np.float32)
        tm = np.load(os.path.join(args.dataset_dir, "target_mask.npy")).astype(bool)
        NDVI_true = np.where(tm, S2[..., 0], np.nan).astype(np.float32)

    T, H, W = NDVI_pred.shape
    if NDVI_true.shape != (T, H, W):
        raise ValueError(f"shape mismatch: pred {NDVI_pred.shape} vs true {NDVI_true.shape}")
    print(f"[INFO] pred/true shape: {NDVI_pred.shape}")

    # ---- 2) Load dates ----
    with open(os.path.join(args.dataset_dir, "metadata.json"), "r", encoding="utf-8") as f:
        meta = json.load(f)
    dates = meta["dates_s1"]

    # ---- 3) Load validation coordinates ----
    if not os.path.exists(args.val_npy):
        raise FileNotFoundError(
            f"val_coords.npy not found at {args.val_npy}\n"
            f"Run 04_Build_Evaluation_Set.py first."
        )
    coords = np.load(args.val_npy).astype(np.int64)
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError("val_coords.npy must be (N,3) with columns (t,row,col)")
    print(f"[INFO] validation coords: {coords.shape[0]:,}")

    # ---- 4) Gather samples (only target_mask==True points are in val_coords) ----
    tt = coords[:, 0]
    rr = coords[:, 1]
    cc = coords[:, 2]

    yt = NDVI_true[tt, rr, cc]
    yp = NDVI_pred[tt, rr, cc]

    ok = np.isfinite(yt) & np.isfinite(yp)
    n_drop = int((~ok).sum())
    if n_drop > 0:
        print(f"[INFO] dropping {n_drop:,} points with NaN true/pred")
        tt = tt[ok]; rr = rr[ok]; cc = cc[ok]
        yt = yt[ok]; yp = yp[ok]

    if yt.size == 0:
        raise RuntimeError("No valid points to evaluate.")

    # ---- 5) Per-date evaluation ----
    print("\n[INFO] Per-date evaluation ...")
    per_date_rows = []
    n_skipped = 0

    for i, d in enumerate(tqdm(dates, desc="Per-date")):
        sel = (tt == i)
        n = int(sel.sum())
        if n < args.min_samples_per_frame:
            n_skipped += 1
            continue

        yti = yt[sel]
        ypi = yp[sel]

        png = os.path.join(plots_dir, f"scatter_{d}.png")
        R2, RMSE, MAE, k, b = density_scatter(
            yti, ypi, png, cmap,
            bins=args.bins2d, sigma=args.sigma, dpi=args.dpi,
            xlim=tuple(args.xlim), ylim=tuple(args.ylim),
        )
        per_date_rows.append([d, n, f"{R2:.3f}", f"{RMSE:.3f}",
                              f"{MAE:.3f}", f"{k:.3f}", f"{b:.3f}"])
        print(f"  {d}: N={n:>8d}  R2={R2:.3f}  RMSE={RMSE:.3f}  MAE={MAE:.3f}")

    with open(per_date_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["date", "N", "R2", "RMSE", "MAE", "slope", "intercept"])
        w.writerows(per_date_rows)
    print(f"[INFO] skipped {n_skipped} frames with N < {args.min_samples_per_frame}")

    # ---- 6) Global (yearly) evaluation ----
    print("\n[INFO] Global evaluation ...")
    year_png = os.path.join(plots_dir, "scatter_year_all.png")
    R2, RMSE, MAE, k, b = density_scatter(
        yt, yp, year_png, cmap,
        bins=args.bins2d, sigma=args.sigma, dpi=args.dpi,
        xlim=tuple(args.xlim), ylim=tuple(args.ylim),
    )
    with open(yearly_txt, "w", encoding="utf-8") as f:
        f.write(f"y = {k:.3f}x + {b:.3f}\n")
        f.write(f"R² = {R2:.3f}\n")
        f.write(f"RMSE = {RMSE:.3f}\n")
        f.write(f"MAE = {MAE:.3f}\n")
        f.write(f"N = {yt.size}\n")

    print("\n" + "=" * 60)
    print("[INFO] Global metrics")
    print("=" * 60)
    print(f"  y    = {k:.3f}x + {b:.3f}")
    print(f"  R²   = {R2:.4f}")
    print(f"  RMSE = {RMSE:.4f}")
    print(f"  MAE  = {MAE:.4f}")
    print(f"  N    = {yt.size:,}")
    print(f"\n[INFO] Outputs: {args.out_dir}")


if __name__ == "__main__":
    main()
