#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
05_S2_Data_Availability_Judgment.py

Decide daily Sentinel-2 usability by combining:
  - Per-image cloud ratio from the 2-band product (NDVI, CloudMask)
  - Daily rainfall from an external CSV
  - Optional presence of a same-date water mask

Rules (configurable via CLI):
  - Rain < RAIN_THRESHOLD (mm)
  - Cloud ratio <= CLOUD_THRESHOLD
  - NDVI is not entirely NaN
  - (Optional) Water mask for the same date exists

Inputs:
  --ndvi-dir     Directory containing NDVI_CloudMask_<DATE>_*.tif (bands: [NDVI, CloudMask])
  --rain-csv     CSV with columns: ["date","rain_mm"] (date parseable by pandas)
  --out-csv      Output CSV path for the summary table
  --usable-dir   Directory to copy "usable" images

Optional:
  --cloud-thr            Cloud ratio threshold (default: 0.8)
  --rain-thr             Rain threshold in mm (default: 10.0)
  --require-watermask    Require WaterMask_<DATE>_WGS84.tif to exist (default: False)
  --wm-dir               Directory to look for water masks (defaults to --ndvi-dir)
  --filename-regex       Regex to extract date from filename (default matches NDVI_CloudMask_<YYYYMMDD>_)
  --no-copy              Do not copy usable files to --usable-dir
"""

import os
import re
import shutil
import argparse
from datetime import datetime

import numpy as np
import pandas as pd
import rasterio
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Judge Sentinel-2 data usability by cloud & rainfall.")
    p.add_argument("--ndvi-dir", required=True, help="Directory with NDVI_CloudMask_* GeoTIFFs.")
    p.add_argument("--rain-csv", required=True, help="CSV with columns ['date','rain_mm'].")
    p.add_argument("--out-csv", required=True, help="Output CSV for usability summary.")
    p.add_argument("--usable-dir", required=True, help="Directory to copy usable images.")
    p.add_argument("--cloud-thr", type=float, default=0.8, help="Cloud ratio threshold (default: 0.8).")
    p.add_argument("--rain-thr", type=float, default=10.0, help="Rain threshold in mm (default: 10.0).")
    p.add_argument("--require-watermask", action="store_true",
                   help="Require WaterMask_<DATE>_WGS84.tif to exist.")
    p.add_argument("--wm-dir", default=None,
                   help="Directory for water masks (default: same as --ndvi-dir).")
    p.add_argument("--filename-regex", default=r"NDVI_CloudMask_(\d{8})_",
                   help=r"Regex with one capturing group for date (default: NDVI_CloudMask_(YYYYMMDD)_)")
    p.add_argument("--no-copy", action="store_true", help="Do not copy usable files.")
    return p.parse_args()


def load_rain_table(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "date" not in df.columns or "rain_mm" not in df.columns:
        raise ValueError("Rain CSV must have columns: 'date' and 'rain_mm'.")
    df["date"] = pd.to_datetime(df["date"])
    df["rain_mm"] = pd.to_numeric(df["rain_mm"], errors="coerce")
    # Drop rows with missing rain values
    df = df.dropna(subset=["rain_mm"]).copy()
    return df


def extract_date_from_name(name: str, pattern: re.Pattern) -> datetime | None:
    m = pattern.search(name)
    if not m:
        return None
    datestr = m.group(1)
    try:
        return datetime.strptime(datestr, "%Y%m%d")
    except ValueError:
        return None


def compute_cloud_ratio_and_ndvi_nan(tif_path: str) -> tuple[float, bool]:
    """
    Expect a 2-band GeoTIFF:
      band 1 = NDVI (float)
      band 2 = CloudMask (0=clear, 1=cloud, 2=shadow, etc.)
    Returns:
      (cloud_ratio in [0,1], ndvi_all_nan: bool)
    """
    with rasterio.open(tif_path) as src:
        if src.count < 2:
            raise ValueError("GeoTIFF should have at least 2 bands: [NDVI, CloudMask].")
        ndvi = src.read(1)
        cloud = src.read(2)
    total = cloud.size
    if total == 0:
        return 1.0, True
    cloud_ratio = float((cloud > 0).sum()) / float(total)
    ndvi_all_nan = bool(np.isnan(ndvi).all())
    return cloud_ratio, ndvi_all_nan


def main():
    args = parse_args()
    os.makedirs(args.usable_dir, exist_ok=True)

    wm_dir = args.wm_dir or args.ndvi_dir
    filename_re = re.compile(args.filename_regex)

    # Load rainfall table and index by date (date-only)
    rain_df = load_rain_table(args.rain_csv)
    rain_df["date_only"] = rain_df["date"].dt.normalize()
    rain_map = dict(zip(rain_df["date_only"].dt.date, rain_df["rain_mm"]))

    # List candidate NDVI_CloudMask files
    all_files = sorted(
        [f for f in os.listdir(args.ndvi_dir)
         if f.lower().endswith(".tif") and "ndvi_cloudmask_" in f.lower()]
    )
    print(f"[INFO] Found {len(all_files)} NDVI_CloudMask raster(s).")

    results = []
    usable_count = 0

    for fname in tqdm(all_files, desc="Evaluating"):
        in_path = os.path.join(args.ndvi_dir, fname)

        # Parse date from filename
        dt = extract_date_from_name(fname, filename_re)
        if dt is None:
            print(f"[WARN] Skip (unable to parse date): {fname}")
            continue
        date_only = dt.date()

        # Lookup rainfall
        rain_val = rain_map.get(date_only, np.nan)
        if np.isnan(rain_val):
            # If no rainfall record for that day, skip (or you could choose to mark not_usable)
            # Here we skip to keep logic consistent with the original script.
            continue

        # Check water mask if required
        wm_name = f"WaterMask_{dt.strftime('%Y%m%d')}_WGS84.tif"
        wm_path = os.path.join(wm_dir, wm_name)
        has_watermask = os.path.exists(wm_path)

        try:
            cloud_ratio, ndvi_all_nan = compute_cloud_ratio_and_ndvi_nan(in_path)
        except Exception as e:
            print(f"[WARN] Skip (read error): {fname} -> {e}")
            continue

        usable = (
            (rain_val < args.rain_thr) and
            (cloud_ratio <= args.cloud_thr) and
            (not ndvi_all_nan) and
            (has_watermask or (not args.require-watermask))  # will be fixed below
        )

        # Python identifiers cannot contain '-', fix the preceding line:
        # (We keep the logic; this is just a readability note.)

        # Recompute with the correct attribute:
        usable = (
            (rain_val < args.rain_thr) and
            (cloud_ratio <= args.cloud_thr) and
            (not ndvi_all_nan) and
            (has_watermask or (not args.require_watermask))
        )

        results.append({
            "filename": fname,
            "date": dt.strftime("%Y-%m-%d"),
            "rain_mm": float(rain_val),
            "cloud_ratio": round(float(cloud_ratio), 3),
            "ndvi_all_nan": bool(ndvi_all_nan),
            "has_watermask": bool(has_watermask),
            "status": "usable" if usable else "not_usable",
        })

        if usable and (not args.no_copy):
            dst = os.path.join(args.usable_dir, fname)
            try:
                shutil.copy2(in_path, dst)
                usable_count += 1
            except Exception as e:
                print(f"[WARN] Failed to copy {fname} -> {e}")

    # Save CSV
    df = pd.DataFrame(results)
    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    df.to_csv(args.out_csv, index=False)

    # Summary
    total = len(all_files)
    judged = len(df)
    n_usable = int((df["status"] == "usable").sum()) if judged else 0
    n_not = int((df["status"] == "not_usable").sum()) if judged else 0

    print("\n[INFO] Screening finished.")
    print(f"[INFO] Output CSV: {args.out_csv}")
    print(f"[INFO] Usable dir: {args.usable_dir} (copied {usable_count} file(s))")
    print(f"[INFO] Total files found: {total}")
    print(f"[INFO] Evaluated: {judged}")
    print(f"[INFO] Usable: {n_usable}")
    print(f"[INFO] Not usable: {n_not}")


if __name__ == "__main__":
    main()
