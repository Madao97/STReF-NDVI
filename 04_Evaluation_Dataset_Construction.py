#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Build a validation set of "smoothed & clear" NDVI pixels by:
  • Using S2_norm[..., 0] (smoothed, normalized NDVI in [0,1])
  • Sampling only where cloud == 0 & water == 0 & good_pixel == True
  • Balanced reservoir sampling across NDVI bins (default: 10 equal-width bins)
  • Optionally excluding training coordinates (coords_train.npy)

Outputs (in --out-dir):
  - val_coords.npy  : int32 array of shape (N, 3) with (t, row, col)
  - val_coords.csv  : CSV with columns t,row,col
  - Console summary of per-bin counts
"""

import os
import csv
import argparse
import warnings
import random
import numpy as np
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Construct evaluation coordinates from smoothed NDVI.")
    p.add_argument("--data-dir", required=True,
                   help="Directory containing fullyear_S2.npy and good_pixel.npy.")
    p.add_argument("--out-dir", default=None,
                   help="Output directory (default: <data-dir>/Validation_set).")
    p.add_argument("--samples-per-bin", type=int, default=250_000,
                   help="Max samples to keep per NDVI bin (reservoir).")
    p.add_argument("--bins", type=int, default=10,
                   help="Number of equal-width NDVI bins in [0,1] (default: 10).")
    p.add_argument("--seed", type=int, default=42, help="Random seed.")
    p.add_argument("--train-coords", default=None,
                   help="Path to coords_train.npy to exclude (default: <data-dir>/coords_train.npy if exists).")
    return p.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    data_dir = args.data_dir
    out_dir = args.out_dir or os.path.join(data_dir, "Validation_set")
    os.makedirs(out_dir, exist_ok=True)

    # ---- Load inputs ----
    s2_path = os.path.join(data_dir, "fullyear_S2.npy")
    gp_path = os.path.join(data_dir, "good_pixel.npy")
    if not os.path.exists(s2_path) or not os.path.exists(gp_path):
        raise FileNotFoundError("Required files not found: fullyear_S2.npy and/or good_pixel.npy")

    S2_norm = np.load(s2_path)                 # (T, H, W, 3) = [ndvi_norm_smoothed, cloud, water]
    NDVI = S2_norm[..., 0]                     # (T, H, W) normalized NDVI ∈ [0,1]
    cloud = S2_norm[..., 1].astype(bool)       # (T, H, W)
    water = S2_norm[..., 2].astype(bool)       # (T, H, W)
    good_pixel = np.load(gp_path).astype(bool) # (H, W)

    T, H, W = NDVI.shape

    # ---- Optional: exclude training coords ----
    train_coords_path = args.train_coords or os.path.join(data_dir, "coords_train.npy")
    train_set = set()
    if os.path.exists(train_coords_path):
        coords_train = np.load(train_coords_path)
        # Expect shape (N,3) with (t,row,col) or (N,2) with (row,col)
        if coords_train.ndim == 2 and coords_train.shape[1] in (2, 3):
            if coords_train.shape[1] == 3:
                train_set = {tuple(map(int, c)) for c in coords_train}
            else:
                # (row,col) → exclude across all times
                rc = {tuple(map(int, c)) for c in coords_train}
                train_set = {(t, r, c) for t in range(T) for (r, c) in rc}
            print(f"[INFO] Excluding training pixels: {len(train_set):,}")
        else:
            warnings.warn("coords_train.npy has unexpected shape; ignoring exclusion.")
    else:
        warnings.warn("coords_train.npy not found; no training pixels excluded.")

    # ---- Reservoir sampling across NDVI bins ----
    nb = max(1, int(args.bins))
    bin_edges = np.linspace(0.0, 1.0, nb + 1)
    bins = {i: [] for i in range(nb)}
    counts = {i: 0 for i in range(nb)}

    for t in tqdm(range(T), desc="Scanning frames"):
        mask = (~cloud[t]) & (~water[t]) & good_pixel
        rows, cols = np.where(mask)
        if rows.size == 0:
            continue

        vals = NDVI[t, rows, cols]
        seg = np.digitize(vals, bin_edges) - 1
        seg = np.clip(seg, 0, nb - 1)  # guard for value==1.0 edge

        for k, b in enumerate(seg):
            coord = (t, int(rows[k]), int(cols[k]))
            if train_set and coord in train_set:
                continue
            counts[int(b)] += 1
            if len(bins[int(b)]) < args.samples_per_bin:
                bins[int(b)].append(coord)
            else:
                # Classic reservoir replacement
                j = random.randint(0, counts[int(b)] - 1)
                if j < args.samples_per_bin:
                    bins[int(b)][j] = coord

    # ---- Merge, shuffle, and save ----
    val_coords = []
    for b in range(nb):
        val_coords.extend(bins[b])
    random.shuffle(val_coords)

    np.save(os.path.join(out_dir, "val_coords.npy"),
            np.array(val_coords, dtype=np.int32))

    with open(os.path.join(out_dir, "val_coords.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["t", "row", "col"])
        writer.writerows(val_coords)

    # ---- Report ----
    per_bin = {f"{i}": len(bins[i]) for i in range(nb)}
    print("\n[INFO] Validation set construction done.")
    print("[INFO] Per-bin sample counts:", per_bin)
    print(f"[INFO] Total samples: {len(val_coords):,}")
    print(f"[INFO] Files saved to: {out_dir}")


if __name__ == "__main__":
    main()
