#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
01_S2_DataPreparation_2022+.py

Batch preprocessing for Sentinel-2 L2A (ESA 2022 radiometric offset + SCL + Otsu-based
shadow enhancement). For each date (two tiles required), the script outputs:

  • NoCloud_<DATE>_RGB_NDVI_CloudMask_WGS84.tif   (5 bands: B04, B03, B02, NDVI, CloudMask)
  • NDVI_CloudMask_<DATE>_WGS84.tif               (2 bands: NDVI, CloudMask)
  • WaterMask_<DATE>_WGS84.tif                    (1 band: 1 = water, 0 = non-water)

Key steps:
  1) Pick the latest (highest Nxxxx) SAFE ZIP per (date, tile).
  2) Read radiometric offsets from MTD_TL.xml and subtract per-band.
  3) Mosaic JP2 per band to 10 m GeoTIFF (B02, B03, B04, B08, SCL; extra bands for shadow).
  4) Compute NDVI; build cloud mask from SCL (2, 3, 8–10) + Otsu(B02) + shadow expansion.
  5) Reproject to EPSG:4326 and write final products.
  6) Save a simple JSON log (cloud percentage and Otsu threshold).

Dependencies:
  rasterio, GDAL, numpy, scikit-image, scipy

Example:
  python 01_S2_DataPreparation_2022+.py \
      --input /path/to/S2_ZIP_folder \
      --temp /path/to/temp_dir \
      --output /path/to/output_dir \
      --tiles T47NQD T47NQE
"""

import os
import re
import glob
import json
import shutil
import zipfile
import argparse
import xml.etree.ElementTree as ET
from collections import defaultdict

import numpy as np
import rasterio
from skimage.filters import threshold_otsu
from scipy.ndimage import binary_dilation
from osgeo import gdal


# ----------------------------- CLI ----------------------------- #

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Sentinel-2 L2A preprocessing (ESA 2022 offset + SCL + Otsu shadow)."
    )
    p.add_argument("--input", required=True, help="Folder containing S2 L2A SAFE ZIPs.")
    p.add_argument("--temp", required=True, help="Working directory for intermediate mosaics.")
    p.add_argument("--output", required=True, help="Output directory for final products.")
    p.add_argument("--tiles", nargs="+", required=True, help="Tile IDs, e.g., T47NQD T47NQE (two tiles).")
    p.add_argument("--nir-shadow-thr", type=float, default=0.08,
                   help="NIR-like threshold for shadow test on B05/B06/B11 (default: 0.08).")
    p.add_argument("--epsg", type=int, default=4326, help="Target EPSG (default: 4326).")
    p.add_argument("--compress", default="LZW", help="GTiff compression (default: LZW).")
    return p.parse_args()


# ----------------------- Utilities & IO ------------------------ #

BAND_ID_MAP = {
    1: "B01", 2: "B02", 3: "B03", 4: "B04", 5: "B05", 6: "B06",
    7: "B07", 8: "B08", 9: "B8A", 10: "B09", 11: "B10", 12: "B11", 13: "B12"
}

CORE_BANDS = ["B02", "B03", "B04", "B08", "SCL"]
EXTRA_BANDS = ["B05", "B06", "B11"]  # for shadow detection


def read_offsets(xml_path: str) -> dict:
    """Parse radiometric offsets from MTD_TL.xml (ESA 2022+)."""
    offsets = {}
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        for tag in root.findall(".//Band_Radiometric_Offset"):
            band_id = int(tag.attrib.get("bandId"))
            val = float(tag.text)
            band = BAND_ID_MAP.get(band_id)
            if band:
                offsets[band] = val
    except Exception as e:
        print(f"[WARN] Failed to parse radiometric offsets: {e}")
    return offsets


def latest_zip_pairs(folder: str, tiles: tuple[str, str]) -> dict:
    """
    For each date, pick the highest-processing (Nxxxx) SAFE ZIP for each tile.
    Return only dates where *both* tiles are present.
    """
    pat = re.compile(r"_N(\d{4})_")  # processing baseline Nxxxx
    groups: dict[str, dict[str, tuple[int, str]]] = defaultdict(dict)

    for fn in os.listdir(folder):
        if not fn.lower().endswith(".zip"):
            continue
        parts = fn.split("_")
        if len(parts) < 6:
            continue
        date = parts[2][:8]
        tile = parts[5]
        if tile not in tiles:
            continue
        ver = int(pat.search(fn).group(1)) if pat.search(fn) else 0
        cur = groups[date].get(tile, (-1, ""))
        if ver > cur[0]:
            groups[date][tile] = (ver, os.path.join(folder, fn))

    out: dict[str, dict[str, str]] = {}
    for d, mapping in groups.items():
        if len(mapping) == 2:
            out[d] = {t: p for t, (_, p) in mapping.items()}
    return out


def find_band_files(band: str, folders: list[str]) -> list[str]:
    files = []
    for d in folders:
        files.extend(glob.glob(os.path.join(d, "**", f"*_{band}_*.jp2"), recursive=True))
    return files


def read_band(fp: str, offset: float = 0.0) -> np.ndarray:
    with rasterio.open(fp) as src:
        arr = src.read(1).astype(np.float32)
    return arr - offset


def mosaic_to_tif(jp2_list: list[str], out_fp: str, xres: float = 10.0, yres: float = 10.0,
                  compress: str = "LZW") -> str:
    gdal.Warp(
        out_fp,
        jp2_list,
        format="GTiff",
        xRes=xres,
        yRes=yres,
        resampleAlg="bilinear",
        options=[f"COMPRESS={compress}", "BIGTIFF=YES", "TARGETALIGNEDPIXELS=YES"]
    )
    return out_fp


def compute_ndvi(b08: np.ndarray, b04: np.ndarray) -> np.ndarray:
    return ((b08 - b04) / (b08 + b04 + 1e-6)).astype(np.float32)


def otsu_binary(arr: np.ndarray) -> tuple[np.ndarray, float]:
    thr = float(threshold_otsu(arr))
    mask = (arr > thr).astype(np.uint8)
    return mask, thr


# ----------------------------- Main ---------------------------- #

def main():
    args = parse_args()
    tiles = tuple(args.tiles)
    if len(tiles) != 2:
        raise ValueError("This script expects exactly two tiles (e.g., --tiles T47NQD T47NQE).")

    os.makedirs(args.temp, exist_ok=True)
    os.makedirs(args.output, exist_ok=True)

    dates = latest_zip_pairs(args.input, tiles)
    print(f"[INFO] Found {len(dates)} valid dates with both tiles: {tiles}")

    log = {}

    for date, zip_map in sorted(dates.items()):
        print(f"\n[INFO] Processing date {date}")
        extracted_dirs = []
        try:
            # Unzip both tiles
            for tile, zip_fp in zip_map.items():
                out_dir = os.path.join(args.input, f"tmp_{date}_{tile}")
                try:
                    zipfile.ZipFile(zip_fp).extractall(out_dir)
                    extracted_dirs.append(out_dir)
                except Exception as e:
                    print(f"[ERROR] Failed to unzip {zip_fp}: {e}")
                    raise

            if len(extracted_dirs) != 2:
                print("[WARN] Missing extracted folders; skip this date.")
                continue

            # Read radiometric offsets (from the first tile's TL metadata)
            xmls = glob.glob(os.path.join(extracted_dirs[0], "**", "MTD_TL.xml"), recursive=True)
            offsets = read_offsets(xmls[0]) if xmls else {}

            # Mosaic core bands
            mosaic_paths = {}
            for b in CORE_BANDS:
                fps = find_band_files(b, extracted_dirs)
                if not fps:
                    print(f"[WARN] Missing band {b}; skip this date.")
                    raise RuntimeError(f"Missing band {b}")
                out_fp = os.path.join(args.temp, f"{date}_{b}.tif")
                mosaic_to_tif(fps, out_fp, compress=args.compress)
                mosaic_paths[b] = out_fp

            # Load bands with offsets
            b04 = read_band(mosaic_paths["B04"], offsets.get("B04", 0.0))
            b08 = read_band(mosaic_paths["B08"], offsets.get("B08", 0.0))
            b03 = read_band(mosaic_paths["B03"], offsets.get("B03", 0.0))
            b02 = read_band(mosaic_paths["B02"], offsets.get("B02", 0.0))
            scl = read_band(mosaic_paths["SCL"])  # SCL has no radiometric offset

            ndvi = compute_ndvi(b08, b04)

            # Base cloud & water from SCL
            cloud = np.isin(scl, [2, 3, 8, 9, 10]).astype(np.uint8)
            water = (scl == 6).astype(np.uint8)

            # Otsu on B02 (blue) + shadow enhancement (B05/B06/B11 + NDVI < 0)
            otsu_mask, thr = otsu_binary(b02)
            cloud = np.clip(cloud + otsu_mask, 0, 1).astype(np.uint8)

            def ensure_extra(band_code: str) -> np.ndarray:
                jp2s = find_band_files(band_code, extracted_dirs)
                if not jp2s:
                    return np.ones_like(b02)  # fallback to non-shadow
                tiff = os.path.join(args.temp, f"{date}_{band_code}.tif")
                mosaic_to_tif(jp2s, tiff, compress=args.compress)
                return read_band(tiff, offsets.get(band_code, 0.0))

            b05 = ensure_extra("B05")
            b06 = ensure_extra("B06")
            b11 = ensure_extra("B11")

            shadow_core = (ndvi < 0) & (b05 < args.nir_shadow_thr) & \
                          (b06 < args.nir_shadow_thr) & (b11 < args.nir_shadow_thr)
            shadow = binary_dilation(cloud.astype(bool), iterations=3) & shadow_core
            shadow[water > 0] = False

            # In final cloud band: 1 = cloud, 2 = shadow (optional encoding)
            cloud_final = cloud.copy()
            cloud_final[shadow] = 2

            # Write 5-band product (reference from B04 mosaic)
            mb = os.path.join(args.temp, f"NoCloud_{date}_RGB_NDVI_CM.tif")
            with rasterio.open(mosaic_paths["B04"]) as ref:
                meta = ref.profile
            meta.update(
                driver="GTiff",
                count=5,
                dtype="float32",
                compress=args.compress,
                BIGTIFF="YES",
                interleave="band",
                nodata=0
            )
            # Remove tiling hints if present (avoid mismatched writer options)
            for k in ("tiled", "blockxsize", "blockysize"):
                meta.pop(k, None)

            with rasterio.open(mb, "w", **meta) as dst:
                dst.write(b04, 1)
                dst.write(b03, 2)
                dst.write(b02, 3)
                dst.write(ndvi, 4)
                dst.write(cloud_final.astype(np.float32), 5)

            out_full = os.path.join(args.output, f"NoCloud_{date}_RGB_NDVI_CloudMask_WGS84.tif")
            out_min  = os.path.join(args.output, f"NDVI_CloudMask_{date}_WGS84.tif")
            out_wat  = os.path.join(args.output, f"WaterMask_{date}_WGS84.tif")

            # Reproject to target CRS
            gdal.Warp(out_full, mb, dstSRS=f"EPSG:{args.epsg}",
                      options=[f"COMPRESS={args.compress}", "BIGTIFF=YES"])
            # Extract NDVI + CloudMask
            gdal.Translate(out_min, out_full, format="GTiff", bandList=[4, 5],
                           creationOptions=[f"COMPRESS={args.compress}", "BIGTIFF=YES"])
            # Water mask from SCL (keep original classes, downstream scripts can binarize to (SCL==6))
            gdal.Translate(out_wat, mosaic_paths["SCL"], format="GTiff",
                           creationOptions=[f"COMPRESS={args.compress}", "BIGTIFF=YES"])

            cloud_pct = round(float((cloud_final > 0).sum()) / cloud_final.size * 100, 2)
            log[date] = {"cloud_pct": cloud_pct, "otsu_thr": round(thr, 4)}
            print(f"[INFO] Done {date}: cloud={cloud_pct}%  otsu={thr:.4f}")

        except Exception as e:
            print(f"[ERROR] Date {date} failed: {e}")

        finally:
            # Cleanup mosaics & extracted SAFE dirs
            try:
                for p in list(glob.glob(os.path.join(args.temp, f"{date}_*.tif"))):
                    os.remove(p)
            except Exception:
                pass
            for d in extracted_dirs:
                shutil.rmtree(d, ignore_errors=True)

    # Write processing log
    log_fp = os.path.join(args.output, "process_log.json")
    with open(log_fp, "w", encoding="utf-8") as jf:
        json.dump(log, jf, indent=2)
    print(f"\n[INFO] All done. Outputs: {args.output}")
    print(f"[INFO] Log saved to: {log_fp}")


if __name__ == "__main__":
    main()
