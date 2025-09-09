#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
03_S1_Resampling.py

Resample all Sentinel-1 GeoTIFF images to match the resolution, extent,
and CRS of a reference Sentinel-2 product.

Inputs:
    --s1-dir   Directory containing Sentinel-1 GeoTIFFs
    --s2-ref   Path to a Sentinel-2 reference GeoTIFF
    --out-dir  Directory to save resampled Sentinel-1 GeoTIFFs

Output:
    Each input S1 file is resampled and saved under the same filename in --out-dir
"""

import os
import argparse
import rasterio
from rasterio.enums import Resampling
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser(description="Resample Sentinel-1 images to Sentinel-2 reference grid.")
    parser.add_argument("--s1-dir", required=True, help="Input directory with Sentinel-1 GeoTIFFs.")
    parser.add_argument("--s2-ref", required=True, help="Reference Sentinel-2 GeoTIFF for alignment.")
    parser.add_argument("--out-dir", required=True, help="Output directory for resampled Sentinel-1 images.")
    parser.add_argument("--resampling", default="bilinear",
                        choices=["nearest", "bilinear", "cubic"],
                        help="Resampling method (default: bilinear).")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Load reference raster grid
    with rasterio.open(args.s2_ref) as ref:
        ref_transform = ref.transform
        ref_width = ref.width
        ref_height = ref.height
        ref_crs = ref.crs

    print(f"[INFO] Reference grid: {ref_width} x {ref_height}, CRS={ref_crs}")

    # Collect Sentinel-1 files
    s1_files = [f for f in os.listdir(args.s1_dir) if f.lower().endswith(".tif")]
    print(f"[INFO] Found {len(s1_files)} Sentinel-1 images to process")

    # Map resampling method
    resampling_map = {
        "nearest": Resampling.nearest,
        "bilinear": Resampling.bilinear,
        "cubic": Resampling.cubic,
    }
    resample_mode = resampling_map[args.resampling]

    # Process each file
    for fname in tqdm(s1_files, desc="Resampling"):
        in_path = os.path.join(args.s1_dir, fname)
        out_path = os.path.join(args.out_dir, fname)

        try:
            with rasterio.open(in_path) as src:
                data = src.read(
                    out_shape=(src.count, ref_height, ref_width),
                    resampling=resample_mode,
                )
                out_meta = src.meta.copy()
                out_meta.update({
                    "height": ref_height,
                    "width": ref_width,
                    "transform": ref_transform,
                    "crs": ref_crs,
                })

            with rasterio.open(out_path, "w", **out_meta) as dst:
                dst.write(data)

        except Exception as e:
            print(f"[ERROR] Failed to process {fname}: {e}")

    print(f"[INFO] All Sentinel-1 images resampled and saved to: {args.out_dir}")


if __name__ == "__main__":
    main()
