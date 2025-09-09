#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
01_S2_Generate_water_mask_from_scl.py

Convert raw SCL-based water mask files (WaterMask_<DATE>_WGS84.tif)
into binary masks:

    - Input value == 6 → Output 1 (water)
    - Other values     → Output 0 (non-water)

Notes:
    - Input and output directories are separated to avoid overwriting original files.
    - The script processes all matching GeoTIFFs in the input directory.
"""

import os
import glob
import numpy as np
import rasterio

# Define your input and output directories here
INPUT_DIR = r"/path/to/input"
OUTPUT_DIR = r"/path/to/output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Search for all candidate files
tifs = sorted(glob.glob(os.path.join(INPUT_DIR, "WaterMask_*_WGS84.tif")))
print(f"Detected {len(tifs)} input WaterMask files")

for fp in tifs:
    fname = os.path.basename(fp)
    out_fp = os.path.join(OUTPUT_DIR, fname)

    # Skip if already exists
    if os.path.exists(out_fp):
        print(f"Skipped (already exists): {fname}")
        continue

    # Read input file
    with rasterio.open(fp) as src:
        raw = src.read(1)
        meta = src.meta

    # Convert to binary mask
    fixed = (raw == 6).astype(np.uint8)

    # Update metadata and save
    meta.update(dtype="uint8", count=1, nodata=0)
    with rasterio.open(out_fp, "w", **meta) as dst:
        dst.write(fixed, 1)

    print(f"Output written: {out_fp}")

print("\nAll WaterMask files have been converted to binary 0/1 masks.")
