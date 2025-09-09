#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
04_Clip.py

Batch clip GeoTIFF rasters by a vector boundary (shapefile/GeoPackage/etc.).
For each input .tif, the script reprojects the boundary to the raster CRS,
clips with crop=True, and writes the result to the output directory.

Usage example:
  python 04_Clip.py \
    --input-dir ./S1_resampled \
    --shp ./AOI/Kuala_Selangor.shp \
    --output-dir ./S1_clipped_KS \
    --pattern "_WGS84.tif" \
    --compress LZW

Dependencies: rasterio, geopandas, tqdm
"""

import os
import sys
import glob
import argparse
import rasterio
import geopandas as gpd
from rasterio.mask import mask
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Clip rasters by a vector boundary (crop to AOI).")
    p.add_argument("--input-dir", required=True, help="Directory containing input GeoTIFFs.")
    p.add_argument("--shp", required=True, help="Vector boundary file (SHP/GeoPackage/GeoJSON, etc.).")
    p.add_argument("--output-dir", required=True, help="Directory to save clipped rasters.")
    p.add_argument("--pattern", default=".tif",
                   help="Filename filter (substring or glob, default: .tif).")
    p.add_argument("--recursive", action="store_true",
                   help="Scan input directory recursively.")
    p.add_argument("--overwrite", action="store_true",
                   help="Overwrite existing outputs.")
    p.add_argument("--compress", default="LZW",
                   help="GTiff compression (e.g., LZW, DEFLATE, ZSTD; default: LZW).")
    return p.parse_args()


def list_tifs(input_dir: str, pattern: str, recursive: bool) -> list[str]:
    # Accept substring or glob pattern
    if any(ch in pattern for ch in ["*", "?", "["]):
        pat = pattern
    else:
        pat = f"*{pattern}"
    if recursive:
        files = glob.glob(os.path.join(input_dir, "**", pat), recursive=True)
    else:
        files = glob.glob(os.path.join(input_dir, pat))
    # Filter to common raster extensions
    return [f for f in files if os.path.splitext(f)[1].lower() in (".tif", ".tiff")]


def load_aoi(shp_path: str):
    gdf = gpd.read_file(shp_path)
    if gdf.empty:
        raise ValueError("AOI file contains no features.")
    # dissolve to a single (multi)polygon to avoid per-feature repetition
    aoi = gdf.unary_union
    if aoi.is_empty:
        raise ValueError("AOI geometry is empty after dissolve.")
    return gdf, aoi


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    try:
        gdf, aoi = load_aoi(args.shp)
    except Exception as e:
        print(f"[ERROR] Failed to load AOI: {e}")
        sys.exit(1)

    files = list_tifs(args.input_dir, args.pattern, args.recursive)
    print(f"[INFO] Found {len(files)} raster(s) to clip.")

    for in_path in tqdm(files, desc="Clipping"):
        # Preserve relative structure if recursive
        rel = os.path.relpath(in_path, args.input_dir)
        out_path = os.path.join(args.output_dir, rel)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        if (not args.overwrite) and os.path.exists(out_path):
            # Skip existing
            continue

        try:
            with rasterio.open(in_path) as src:
                # Reproject AOI to raster CRS if needed
                if gdf.crs is None:
                    raise ValueError("AOI has no CRS. Please define/set a CRS for the vector file.")
                if src.crs is None:
                    raise ValueError("Raster has no CRS. Cannot reproject AOI to match.")
                if gdf.crs != src.crs:
                    gdf_proj = gdf.to_crs(src.crs)
                    aoi_proj = gdf_proj.unary_union
                else:
                    aoi_proj = aoi

                # Prepare GeoJSON-like sequence
                try:
                    geoms = [aoi_proj.__geo_interface__]
                except Exception:
                    # fallback for some geometry types
                    geoms = [g.__geo_interface__ for g in gdf_proj.geometry]

                out_image, out_transform = mask(src, geoms, crop=True)
                meta = src.meta.copy()

            # Update metadata
            meta.update({
                "height": out_image.shape[1],
                "width": out_image.shape[2],
                "transform": out_transform,
                "driver": "GTiff",
                "compress": args.compress,
                "BIGTIFF": "YES",
            })

            # Write output
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            with rasterio.open(out_path, "w", **meta) as dst:
                dst.write(out_image)

        except Exception as e:
            print(f"[ERROR] Clip failed: {in_path} -> {e}")

    print(f"[INFO] Finished. Clipped rasters saved under: {args.output_dir}")


if __name__ == "__main__":
    main()
