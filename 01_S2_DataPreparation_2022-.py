#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
01_S2_DataPreparation_2022-.py

Batch preprocessing for Sentinel-2 L2A (pre-2022 radiometric scheme; no band
offset subtraction). For each valid date (two tiles present) this script outputs:

  • NoCloud_<DATE>_RGB_NDVI_CloudMask_WGS84.tif   (5 bands: B04, B03, B02, NDVI, CloudMask)
  • NDVI_CloudMask_<DATE>_WGS84.tif               (2 bands: NDVI, CloudMask)
  • WaterMask_<DATE>_WGS84.tif                    (1 band: 1 = water, 0 = non-water)

Steps:
  1) Pick the latest (highest Nxxxx) SAFE ZIP per (date, tile).
  2) Mosaic JP2 per band to 10 m GeoTIFF (B02, B03, B04, B08, plus SCL; B05/B06/B11 for shadow).
  3) Compute NDVI.
  4) Build cloud mask: SCL clouds (2, 3, 8–10) + Otsu(B02) + shadow expansion using NDVI and NIR-like bands.
  5) Reproject to EPSG:4326 and write final products.
  6) Save a JSON log per date.

Dependencies:
  rasterio, GDAL, numpy, scikit-image, scipy
"""

import os
import re
import glob
import json
import shutil
import zipfile
import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import rasterio
from skimage.filters import threshold_otsu
from scipy.ndimage import binary_dilation
from osgeo import gdal


# ----------------------------- CLI ----------------------------- #

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Sentinel-2 L2A preprocessing (pre-2022; SCL+Otsu shadow, water mask)."
    )
    p.add_argument("--input", required=True, help="Folder containing S2 L2A SAFE ZIPs.")
    p.add_argument("--temp", required=True, help="Working directory for intermediate mosaics.")
    p.add_argument("--output", required=True, help="Output directory for final products.")
    p.add_argument("--tiles", nargs="+", required=True,
                   help="Two tile IDs, e.g., T47NQD T47NQE.")
    p.add_argument("--epsg", type=int, default=4326, help="Target EPSG (default: 4326).")
    p.add_argument("--compress", default="LZW", help="GTiff compression (default: LZW).")
    p.add_argument("--nir-shadow-thr", type=float, default=0.08,
                   help="Threshold for B05/B06/B11 in shadow test (default: 0.08).")
    p.add_argument("--dilate-iter", type=int, default=3,
                   help="Binary dilation iterations for shadow expansion (default: 3).")
    return p.parse_args()


# ----------------------- Utilities & IO ------------------------ #

BANDS_10M = ["B02", "B03", "B04", "B08"]
BANDS_20M = ["B05", "B06", "B11"]  # used for shadow detection
SCL_RES = ["10m", "20m", "60m"]    # SCL may exist at different native resolutions


def latest_zip_pairs(folder: str, tiles: tuple[str, str]) -> dict:
    """
    For each date, pick the highest-processing (Nxxxx) SAFE ZIP for each tile.
    Return only dates where *both* tiles are present.
    """
    pat = re.compile(r"_N(\d{4})_")
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


def pick_band_files(band: str, res_opts: list[str], folders: list[str]) -> list[str]:
    """Search band JP2s across extracted SAFE dirs, preferring higher native resolutions."""
    for res in res_opts:
        pat = f"*_{band}_{res}.jp2"
        files = []
        for d in folders:
            files.extend(glob.glob(os.path.join(d, "**", pat), recursive=True))
        if files:
            return files
    return []


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


def rarr(path: str, dtype=np.float32) -> np.ndarray:
    with rasterio.open(path) as ds:
        return ds.read(1).astype(dtype)


# ----------------------------- Main ---------------------------- #

def main():
    args = parse_args()
    tiles = tuple(args.tiles)
    if len(tiles) != 2:
        raise ValueError("Provide exactly two tiles, e.g., --tiles T47NQD T47NQE")

    os.makedirs(args.temp, exist_ok=True)
    os.makedirs(args.output, exist_ok=True)

    dates = latest_zip_pairs(args.input, tiles)
    print(f"[INFO] Found {len(dates)} valid dates with both tiles: {tiles}")

    log = {}

    for date, zips in sorted(dates.items()):
        print(f"\n[INFO] Processing date {date}")
        extracted_dirs = []

        # 1) Unzip both tiles
        try:
            for tile, zip_fp in zips.items():
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

            # 2) Mosaic + Warp target bands to 10 m
            tasks = []
            mosaic_paths = {}

            for b in BANDS_10M:
                tasks.append((
                    pick_band_files(b, ["10m"], extracted_dirs),
                    os.path.join(args.temp, f"{date}_{b}_10m.tif"),
                    b
                ))
            for b in BANDS_20M:
                tasks.append((
                    pick_band_files(b, ["20m"], extracted_dirs),
                    os.path.join(args.temp, f"{date}_{b}_10m.tif"),
                    b
                ))

            scl_files = pick_band_files("SCL", SCL_RES, extracted_dirs)
            if not all(t[0] for t in tasks) or not scl_files:
                print("[WARN] Missing required bands; skip this date.")
                continue
            tasks.append((scl_files, os.path.join(args.temp, f"{date}_SCL_10m.tif"), "SCL"))

            with ThreadPoolExecutor() as ex:
                futures = {ex.submit(mosaic_to_tif, jp2s, out_fp, 10.0, 10.0, args.compress): key
                           for jp2s, out_fp, key in tasks}
                for fut in as_completed(futures):
                    key = futures[fut]
                    mosaic_paths[key] = fut.result()

            # 3) NDVI
            b08 = rarr(mosaic_paths["B08"])
            b04 = rarr(mosaic_paths["B04"])
            ndvi = ((b08 - b04) / (b08 + b04 + 1e-6)).astype(np.float32)

            # 4) Cloud + water + shadow
            b02 = rarr(mosaic_paths["B02"])
            thr = float(threshold_otsu(b02))
            cloud_otsu = (b02 > thr).astype(np.uint8)

            scl = rarr(mosaic_paths["SCL"])
            cloud_scl = np.isin(scl, [2, 3, 8, 9, 10]).astype(np.uint8)
            water_mask = (scl == 6)

            cloud = np.clip(cloud_otsu + cloud_scl, 0, 1).astype(np.uint8)

            # Shadow expansion using NDVI and NIR-like bands
            b05 = rarr(mosaic_paths["B05"]) if "B05" in mosaic_paths else np.ones_like(b02)
            b06 = rarr(mosaic_paths["B06"]) if "B06" in mosaic_paths else np.ones_like(b02)
            b11 = rarr(mosaic_paths["B11"]) if "B11" in mosaic_paths else np.ones_like(b02)

            shadow_core = (ndvi < 0) & (b05 < args.nir_shadow_thr) & \
                          (b06 < args.nir_shadow_thr) & (b11 < args.nir_shadow_thr)
            shadow = binary_dilation(cloud.astype(bool), iterations=args.dilate_iter) & shadow_core
            shadow[water_mask > 0] = False

            cloud_final = cloud.copy()
            cloud_final[shadow] = 2  # encode shadow as 2 (optional)

            print("[INFO] NDVI and cloud/water/shadow masks computed")

            # 5) Write 5-band GeoTIFF (native CRS), then reproject to target EPSG
            mb = os.path.join(args.temp, f"NoCloud_{date}_RGB_NDVI_CM.tif")
            if os.path.exists(mb):
                os.remove(mb)

            with rasterio.open(mosaic_paths["B04"]) as ref:
                meta = ref.profile
            meta.update(driver="GTiff", count=5, dtype="float32",
                        compress=args.compress, BIGTIFF="YES",
                        interleave="band", nodata=0)
            for key in ("tiled", "blockxsize", "blockysize"):
                meta.pop(key, None)

            with rasterio.open(mb, "w", **meta) as dst:
                dst.write(b04, 1)
                dst.write(rarr(mosaic_paths["B03"]), 2)
                dst.write(b02, 3)
                dst.write(ndvi, 4)
                dst.write(cloud_final.astype(np.float32), 5)

            out_full = os.path.join(args.output, f"NoCloud_{date}_RGB_NDVI_CloudMask_WGS84.tif")
            out_min  = os.path.join(args.output, f"NDVI_CloudMask_{date}_WGS84.tif")
            gdal.Warp(out_full, mb, dstSRS=f"EPSG:{args.epsg}",
                      options=[f"COMPRESS={args.compress}", "BIGTIFF=YES"])
            gdal.Translate(out_min, out_full, format="GTiff", bandList=[4, 5],
                           creationOptions=[f"COMPRESS={args.compress}", "BIGTIFF=YES"])
            print("[INFO] 5-band and 2-band products written")

            # 6) WaterMask (1 = water, 0 = non-water), reprojected to target EPSG
            wm_temp = os.path.join(args.temp, f"WaterMask_{date}.tif")
            if os.path.exists(wm_temp):
                os.remove(wm_temp)

            with rasterio.open(mosaic_paths["B04"]) as ref:
                wm_meta = ref.profile
            wm_meta.update(driver="GTiff", count=1, dtype="uint8",
                           compress=args.compress, BIGTIFF="YES",
                           interleave="band", nodata=0)
            for key in ("tiled", "blockxsize", "blockysize"):
                wm_meta.pop(key, None)

            with rasterio.open(wm_temp, "w", **wm_meta) as dst:
                dst.write(water_mask.astype(np.uint8), 1)

            wm_fp = os.path.join(args.output, f"WaterMask_{date}_WGS84.tif")
            gdal.Warp(wm_fp, wm_temp, dstSRS=f"EPSG:{args.epsg}",
                      options=[f"COMPRESS={args.compress}", "BIGTIFF=YES"])
            print("[INFO] Water mask written")

            # 7) Log and cleanup
            cloud_pct = round(float((cloud_final > 0).sum()) / cloud_final.size * 100, 2)
            log[date] = {"cloud_pct": cloud_pct, "otsu_thr": round(thr, 4)}
            print(f"[INFO] Done {date}: cloud={cloud_pct}%  otsu={thr:.4f}")

        except Exception as e:
            print(f"[ERROR] Date {date} failed: {e}")

        finally:
            # Remove mosaics & temps; remove extracted SAFE dirs
            try:
                for p in list(glob.glob(os.path.join(args.temp, f"{date}_*.tif"))):
                    try: os.remove(p)
                    except: pass
                for p in (os.path.join(args.temp, f"NoCloud_{date}_RGB_NDVI_CM.tif"),
                          os.path.join(args.temp, f"WaterMask_{date}.tif")):
                    if os.path.exists(p):
                        try: os.remove(p)
                        except: pass
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
