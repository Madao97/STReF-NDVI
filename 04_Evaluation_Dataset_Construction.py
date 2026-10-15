# -*- coding: utf-8 -*-
r"""
Build evaluation coordinates for STReF-NDVI reconstruction assessment.

Eligibility:
    target_mask is True.
    good_pixel is True.
    Observed NDVI is finite.

Optical label validity is inherited from target_mask, which is produced
by the dataset construction script. This script does not recompute
cloud, water, or NDVI-range screening.

Sampling:
    Reservoir sampling is performed separately within equal-width
    NDVI bins. Each bin retains at most --samples-per-bin coordinates.
    Underpopulated bins retain all eligible coordinates, so the final
    sample counts are not necessarily equal across bins.

Inputs:
    S2.npy              (T, H, W, 3)
    target_mask.npy     (T, H, W)
    good_pixel.npy      (H, W)
    metadata.json

Outputs:
    val_coords.npy      (N, 3), int32 columns: t, row, col
    val_coords.csv
    val_build_summary.json

The existing val_* output filenames are retained for compatibility.

Evaluation scope:
    Coordinates are sampled from the supplied dataset.
    This script does not exclude training or model-selection pixels.
    An independent test set requires a separately defined held-out
    population before running this script.

Examples:
    python 04_Build_Evaluation_Set.py
    python 04_Build_Evaluation_Set.py --data-dir ./data/Dataset --out-dir ./outputs/Evaluation_set
    python 04_Build_Evaluation_Set.py --samples-per-bin 250000 --bins 10 --seed 42
"""

import os
import csv
import json
import argparse
import random

import numpy as np
from tqdm import tqdm


# =============================================================
# DEFAULT PATHS
# Override these paths using command-line arguments.
# =============================================================
DATA_DIR = os.path.join("data", "Dataset")
OUT_DIR = os.path.join("outputs", "Evaluation_set")


# ------------------------------ CLI -------------------------------- #

def parse_args():
    p = argparse.ArgumentParser(
        description="Sample evaluation coordinates from valid optical labels."
    )
    p.add_argument(
        "--data-dir",
        default=DATA_DIR,
        help="Directory containing S2.npy, target_mask.npy, "
             "good_pixel.npy, and metadata.json.",
    )
    p.add_argument("--out-dir", default=OUT_DIR)
    p.add_argument(
        "--samples-per-bin",
        type=int,
        default=250_000,
        help="Maximum number of coordinates retained per NDVI bin.",
    )
    p.add_argument(
        "--bins",
        type=int,
        default=10,
        help="Number of equal-width NDVI bins defined by the metadata bounds.",
    )
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()
# ------------------------------- Main ------------------------------ #

def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    # ---- Load ----
    s2_fp  = os.path.join(args.data_dir, "S2.npy")
    tm_fp  = os.path.join(args.data_dir, "target_mask.npy")
    gp_fp  = os.path.join(args.data_dir, "good_pixel.npy")
    md_fp  = os.path.join(args.data_dir, "metadata.json")
    for fp in (s2_fp, tm_fp, gp_fp, md_fp):
        if not os.path.exists(fp):
            raise FileNotFoundError(fp)

    print("[INFO] Loading arrays ...")
    S2 = np.load(s2_fp)                                # (T,H,W,3)
    target_mask = np.load(tm_fp).astype(bool)          # (T,H,W)
    good_pixel  = np.load(gp_fp).astype(bool)          # (H,W)

    with open(md_fp, "r", encoding="utf-8") as f:
        meta = json.load(f)
    ndvi_min = float(meta.get("ndvi_min", -0.2))
    ndvi_max = float(meta.get("ndvi_max", 1.0))

    T, H, W = target_mask.shape
    print(f"[INFO] T={T}, H={H}, W={W}")
    print(f"[INFO] NDVI range: [{ndvi_min}, {ndvi_max}]")
    print(f"[INFO] target_mask pixels: {int(target_mask.sum()):,}")
    print(f"[INFO] good_pixel        : {int(good_pixel.sum()):,} / {H*W:,}")

    # NDVI in raw scale (S2[...,0] is raw NDVI, NaN where no S2 label)
    NDVI = S2[..., 0].astype(np.float32)

    # eligible mask: has S2 label + target_mask + good_pixel + finite
    eligible = target_mask & good_pixel[None, :, :] & np.isfinite(NDVI)
    print(f"[INFO] eligible pixels   : {int(eligible.sum()):,}")
    if eligible.sum() == 0:
        raise RuntimeError("No eligible pixels.")

    # ---- Reservoir sampling per NDVI bin ----
    nb = max(1, int(args.bins))
    bin_edges = np.linspace(ndvi_min, ndvi_max, nb + 1)

    bins   = {i: [] for i in range(nb)}
    counts = {i: 0  for i in range(nb)}

    print("[INFO] Sampling (bin-balanced reservoir) ...")
    for t in tqdm(range(T), desc="Scanning frames"):
        m = eligible[t]
        if not m.any():
            continue
        rr, cc = np.where(m)
        vals = NDVI[t, rr, cc]
        seg = np.clip(np.digitize(vals, bin_edges) - 1, 0, nb - 1)

        # vectorized replacement is non-trivial; keep per-sample loop but only
        # for pixels that need it (this is fast enough for ~1M samples)
        for k, b in enumerate(seg):
            b = int(b)
            coord = (t, int(rr[k]), int(cc[k]))
            counts[b] += 1
            if len(bins[b]) < args.samples_per_bin:
                bins[b].append(coord)
            else:
                j = random.randint(0, counts[b] - 1)
                if j < args.samples_per_bin:
                    bins[b][j] = coord

    # ---- Merge / shuffle / save ----
    val_coords = []
    for b in range(nb):
        val_coords.extend(bins[b])
    random.shuffle(val_coords)

    arr = np.array(val_coords, dtype=np.int32)          # (N,3)
    np.save(os.path.join(args.out_dir, "val_coords.npy"), arr)

    with open(os.path.join(args.out_dir, "val_coords.csv"),
              "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["t", "row", "col"])
        writer.writerows(val_coords)

    per_bin = {str(i): len(bins[i]) for i in range(nb)}

    summary = {
        "data_dir": args.data_dir,
        "out_dir":  args.out_dir,
        "T": int(T), "H": int(H), "W": int(W),
        "ndvi_min": ndvi_min,
        "ndvi_max": ndvi_max,
        "n_bins": int(nb),
        "samples_per_bin": int(args.samples_per_bin),
        "per_bin_counts": per_bin,
        "total_samples": int(len(val_coords)),
        "eligible_pixels": int(eligible.sum()),
    }
    with open(os.path.join(args.out_dir, "val_build_summary.json"),
              "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n[INFO] Per-bin counts:", per_bin)
    print(f"[INFO] Total validation samples: {len(val_coords):,}")
    print(f"[INFO] Saved to: {args.out_dir}")


if __name__ == "__main__":
    main()
