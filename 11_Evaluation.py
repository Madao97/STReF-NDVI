#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
11_Evaluation.py

Evaluate reconstructed NDVI using a validation set of clear pixels.
Outputs:
  • per-date metrics CSV + density scatter plots
  • yearly metrics TXT + density scatter plot

Plot style:
  - Custom white→Jet colormap (zero-density = pure white)
  - No density truncation (full distribution)
  - Times New Roman fonts, thicker axes, no grid
  - Fixed axes limits [-0.5, 1] on both axes, equal aspect, same ticks
  - No titles or sample-size text on the figure
"""

import os
import re
import glob
import csv
import argparse
import pickle
import warnings

import numpy as np
import rasterio
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from scipy.ndimage import gaussian_filter
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from tqdm import tqdm


# ------------------------------ CLI -------------------------------- #

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate NDVI predictions on a clear-pixel validation set.")
    p.add_argument("--data-dir", required=True,
                   help="Directory containing fullyear_S2.npy and normalization_params.pkl.")
    p.add_argument("--pred-dir", required=True,
                   help="Directory containing NDVI_Predict_<DATE>.tif (or NDVI_pred.npy cache).")
    p.add_argument("--out-dir", required=True,
                   help="Directory to save evaluation outputs (CSV/TXT/plots).")
    p.add_argument("--val-npy", default=None,
                   help="Path to val_coords.npy. If not provided, will search common locations.")
    p.add_argument("--min-samples-per-frame", type=int, default=20000,
                   help="Minimum valid points required to evaluate a date (default: 20000).")
    p.add_argument("--bins2d", type=int, default=200,
                   help="2D histogram bins per axis for density scatter (default: 200).")
    p.add_argument("--sigma", type=float, default=1.0,
                   help="Gaussian filter sigma for density smooth (default: 1.0).")
    p.add_argument("--font-family", default="Times New Roman",
                   help="Matplotlib font family (default: Times New Roman).")
    p.add_argument("--font-size", type=int, default=16,
                   help="Matplotlib base font size (default: 16).")
    p.add_argument("--dpi", type=int, default=500,
                   help="DPI for saved PNGs (default: 500).")
    return p.parse_args()


# --------------------------- Plot Styling --------------------------- #

def setup_matplotlib(font_family: str, font_size: int):
    # Custom white→Jet colormap where zero-density = pure white
    base_jet = plt.get_cmap("jet", 256)
    colors = base_jet(np.linspace(0, 1, 256))
    colors[0] = [1, 1, 1, 1]  # pure white at the lowest bin
    cmap_white_jet = LinearSegmentedColormap.from_list("white_jet", colors)

    plt.rcParams.update({
        "font.family": font_family,
        "font.size": font_size,
        "axes.linewidth": 1.0,
        "xtick.direction": "out", "ytick.direction": "out",
        "xtick.major.width": 1.0, "ytick.major.width": 1.0,
        "xtick.major.size": 4, "ytick.major.size": 4,
    })
    return cmap_white_jet


# --------------------------- I/O Utilities -------------------------- #

def find_val_coords_path(data_dir: str, out_dir: str) -> str | None:
    # Priority: explicit out_dir/Validation_set → data_dir/Validation_set → any val_coords*.npy under pred/out
    candidates = []
    for d in [os.path.join(data_dir, "Validation_set"),
              os.path.join(out_dir, "Validation_set"),
              data_dir, out_dir]:
        if os.path.isdir(d):
            candidates += glob.glob(os.path.join(d, "val_coords*.npy"))
    return candidates[0] if candidates else None


def load_true_ndvi_and_dates(data_dir: str) -> tuple[np.ndarray, list[str], float, float]:
    with open(os.path.join(data_dir, "normalization_params.pkl"), "rb") as f:
        norm = pickle.load(f)
    dates = norm["dates"]
    nd_min = float(norm["S2"]["ndvi"]["min"])
    nd_max = float(norm["S2"]["ndvi"]["max"])

    S2 = np.load(os.path.join(data_dir, "fullyear_S2.npy"))  # (T,H,W,3)
    NDVI_true = S2[..., 0] * (nd_max - nd_min) + nd_min     # denormalized
    return NDVI_true, dates, nd_min, nd_max


def load_pred_ndvi_stack(pred_dir: str, dates: list[str], cache_name: str = "NDVI_pred.npy") -> np.ndarray:
    cache_fp = os.path.join(pred_dir, cache_name)
    if os.path.exists(cache_fp):
        return np.load(cache_fp)

    # Build from GeoTIFFs
    tifs = sorted(glob.glob(os.path.join(pred_dir, "NDVI_Predict_*.tif")))
    if not tifs:
        raise FileNotFoundError(f"No prediction GeoTIFFs found under: {pred_dir}")
    name2fp = {}
    for fp in tifs:
        # accept both "NDVI_Predict_YYYYMMDD.tif" and any suffix
        base = os.path.basename(fp)
        m = re.search(r"NDVI_Predict_(\d{8})", base)
        if m:
            name2fp[m.group(1)] = fp

    T = len(dates)
    sample = rasterio.open(name2fp[dates[0]])
    H, W = sample.height, sample.width
    sample.close()

    NDVI_pred = np.empty((T, H, W), dtype=np.float32)
    for i, d in enumerate(tqdm(dates, desc="Loading predictions")):
        if d not in name2fp:
            raise FileNotFoundError(f"Missing prediction for date {d} in {pred_dir}")
        with rasterio.open(name2fp[d]) as src:
            NDVI_pred[i] = src.read(1)
    np.save(cache_fp, NDVI_pred)
    return NDVI_pred


# --------------------------- Plot Function ------------------------- #

def density_scatter(y_true: np.ndarray,
                    y_pred: np.ndarray,
                    out_png: str,
                    cmap,
                    bins: int = 200,
                    sigma: float = 1.0,
                    dpi: int = 500) -> tuple[float, float, float, float, float]:
    """Save a density scatter and return (R2, RMSE, MAE, slope, intercept)."""
    mask = (~np.isnan(y_true)) & (~np.isnan(y_pred))
    y_t = y_true[mask]; y_p = y_pred[mask]
    if y_t.size == 0:
        raise ValueError("No valid points to plot.")

    vmin, vmax = y_t.min(), y_t.max()

    H2d, _, _ = np.histogram2d(y_t, y_p, bins=bins, range=[[vmin, vmax], [vmin, vmax]])
    H2d = gaussian_filter(H2d, sigma=sigma)

    # Metrics
    slope, intercept = np.polyfit(y_t, y_p, 1)
    R2 = float(r2_score(y_t, y_p))
    RMSE = float(np.sqrt(mean_squared_error(y_t, y_p)))
    MAE = float(mean_absolute_error(y_t, y_p))

    # Figure (no title; RSE-like styling)
    fig, ax = plt.subplots(figsize=(6, 6), facecolor="white")
    normc = Normalize(vmin=H2d.min(), vmax=H2d.max(), clip=False)
    im = ax.imshow(H2d.T, origin="lower",
                   extent=[vmin, vmax, vmin, vmax],
                   cmap=cmap, norm=normc)
    ax.plot([vmin, vmax], [vmin, vmax], "k--", lw=1)
    ax.plot([vmin, vmax], [slope * vmin + intercept, slope * vmax + intercept], "r-", lw=2)

    ax.set_xlim(-0.5, 1.0)
    ax.set_ylim(-0.5, 1.0)
    ax.set_aspect("equal", adjustable="box")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xticks(np.linspace(-0.5, 1.0, 4))
    ax.set_yticks(np.linspace(-0.5, 1.0, 4))
    ax.set_xlabel("True NDVI")
    ax.set_ylabel("Predicted NDVI")

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Counts in bin")

    txt = (f"y = {slope:.3f}x + {intercept:.3f}\n"
           f"R² = {R2:.3f}\n"
           f"RMSE = {RMSE:.3f}\n"
           f"MAE = {MAE:.3f}")
    ax.text(0.02, 0.98, txt, transform=ax.transAxes, va="top", ha="left",
            bbox=dict(facecolor="white", alpha=0.8, pad=4))

    plt.tight_layout()
    fig.savefig(out_png, dpi=dpi)
    plt.close(fig)
    return R2, RMSE, MAE, slope, intercept


# ------------------------------- Main ----------------------------- #

def main():
    args = parse_args()
    cmap = setup_matplotlib(args.font_family, args.font_size)

    os.makedirs(args.out_dir, exist_ok=True)
    plots_dir = os.path.join(args.out_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    per_date_csv = os.path.join(args.out_dir, "per_date_metrics.csv")
    yearly_txt = os.path.join(args.out_dir, "yearly_metrics.txt")

    # 1) Load truth and dates; denormalize NDVI
    NDVI_true, dates, nd_min, nd_max = load_true_ndvi_and_dates(args.data_dir)
    T, H, W = NDVI_true.shape

    # 2) Load validation coordinates
    val_path = args.val_npy or find_val_coords_path(args.data_dir, args.out_dir)
    if not val_path:
        raise FileNotFoundError("Validation set (val_coords*.npy) not found. "
                                "Provide --val-npy or place it under <data-dir>/Validation_set.")
    coords = np.load(val_path).astype(int)
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError("val_coords.npy must have shape (N, 3) with columns (t, row, col).")

    # 3) Load predictions
    NDVI_pred = load_pred_ndvi_stack(args.pred_dir, dates)

    # 4) Gather samples (true vs pred)
    yt = NDVI_true[coords[:, 0], coords[:, 1], coords[:, 2]]
    yp = NDVI_pred[coords[:, 0], coords[:, 1], coords[:, 2]]

    # 5) Per-date evaluation
    with open(per_date_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["date", "N", "R2", "RMSE", "MAE", "slope", "intercept"])
        for i, d in enumerate(tqdm(dates, desc="Per-date")):
            idxs = np.where(coords[:, 0] == i)[0]
            if idxs.size == 0:
                continue
            yti, ypi = yt[idxs], yp[idxs]
            valid_n = int((~np.isnan(yti) & ~np.isnan(ypi)).sum())
            if valid_n < args.min_samples_per_frame:
                continue
            png = os.path.join(plots_dir, f"scatter_{d}.png")
            R2, RMSE, MAE, k, b = density_scatter(
                yti, ypi, png, cmap, bins=args.bins2d, sigma=args.sigma, dpi=args.dpi
            )
            w.writerow([d, valid_n, f"{R2:.3f}", f"{RMSE:.3f}", f"{MAE:.3f}", f"{k:.3f}", f"{b:.3f}"])

    # 6) Yearly/global evaluation
    year_png = os.path.join(plots_dir, "scatter_year_all.png")
    R2, RMSE, MAE, k, b = density_scatter(
        yt, yp, year_png, cmap, bins=args.bins2d, sigma=args.sigma, dpi=args.dpi
    )
    with open(yearly_txt, "w") as f:
        f.write(f"y = {k:.3f}x + {b:.3f}\n")
        f.write(f"R² = {R2:.3f}\n")
        f.write(f"RMSE = {RMSE:.3f}\n")
        f.write(f"MAE = {MAE:.3f}\n")

    print("[INFO] Done:")
    print("       Per-date CSV:", per_date_csv)
    print("       Yearly  TXT:", yearly_txt)
    print("       Plots saved under:", plots_dir)


if __name__ == "__main__":
    main()
