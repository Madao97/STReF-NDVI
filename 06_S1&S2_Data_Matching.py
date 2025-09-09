#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
06_S1&S2_Data_Matching.py

Match Sentinel-1 acquisition dates to "usable" Sentinel-2 dates within a time window.

Inputs:
  --s1-dir    Directory containing S1 rasters (default filename pattern: S1_YYYYMMDD*.tif)
  --s2-csv    CSV produced by 05_S2_Data_Availability_Judgment.py (must include columns: date, status)
  --out-json  Output JSON path for the S1->S2 date mapping

Options:
  --max-delta DAYS         Maximum absolute difference in days (default: 7)
  --s1-regex PATTERN       Regex to extract S1 date (one capturing group), default: r"S1_(\\d{8})"
  --status-filter VALUE    Only use S2 rows with this status (default: usable)
  --strategy STR           Matching strategy: nearest|past|future (default: nearest)
  --allow-reuse            Allow a single S2 date to be matched to multiple S1 dates (default: False)

Output JSON format:
  {
    "20230103": "20230102",
    "20230115": "20230116",
    ...
  }
"""

import os
import re
import json
import argparse
from datetime import datetime, timedelta

import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Match S1 dates to usable S2 dates within a window.")
    p.add_argument("--s1-dir", required=True, help="Directory containing S1 GeoTIFFs.")
    p.add_argument("--s2-csv", required=True, help="CSV with columns [date, status].")
    p.add_argument("--out-json", required=True, help="Output JSON path (S1->S2 mapping).")
    p.add_argument("--max-delta", type=int, default=7, help="Max allowed |date diff| in days (default: 7).")
    p.add_argument("--s1-regex", default=r"S1_(\d{8})",
                   help=r"Regex with one capturing group for S1 date (default: S1_(YYYYMMDD)).")
    p.add_argument("--status-filter", default="usable",
                   help="Use only S2 rows with this status (default: usable).")
    p.add_argument("--strategy", choices=["nearest", "past", "future"], default="nearest",
                   help="Matching strategy when multiple S2 candidates exist (default: nearest).")
    p.add_argument("--allow-reuse", action="store_true",
                   help="Allow the same S2 date to be used multiple times.")
    return p.parse_args()


def list_s1_dates(folder: str, rx: re.Pattern) -> list[datetime]:
    dates = []
    for fname in os.listdir(folder):
        if not fname.lower().endswith(".tif"):
            continue
        m = rx.search(fname)
        if not m:
            continue
        try:
            dt = datetime.strptime(m.group(1), "%Y%m%d")
            dates.append(dt)
        except ValueError:
            continue
    dates = sorted(set(dates))
    return dates


def list_s2_usable_dates(csv_path: str, status_filter: str) -> list[datetime]:
    df = pd.read_csv(csv_path)
    if "date" not in df.columns or "status" not in df.columns:
        raise ValueError("S2 CSV must contain 'date' and 'status' columns.")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    if status_filter:
        df = df[df["status"].astype(str).str.lower() == status_filter.lower()]
    dates = sorted(set(dt.to_pydatetime() for dt in df["date"]))
    return dates


def choose_candidate(s1_dt: datetime, cands: list[datetime], strategy: str) -> datetime | None:
    if not cands:
        return None
    if strategy == "nearest":
        # tie-breaker: prefer the one with smaller absolute delta, then earlier date
        return sorted(cands, key=lambda d: (abs((d - s1_dt).days), d))[0]
    elif strategy == "past":
        past = [d for d in cands if d <= s1_dt]
        if past:
            return max(past)  # most recent in the past
        # fallback to nearest if no past
        return sorted(cands, key=lambda d: (abs((d - s1_dt).days), d))[0]
    elif strategy == "future":
        future = [d for d in cands if d >= s1_dt]
        if future:
            return min(future)  # soonest in the future
        # fallback to nearest if no future
        return sorted(cands, key=lambda d: (abs((d - s1_dt).days), d))[0]
    return None


def match_dates(
    s1_dates: list[datetime],
    s2_dates: list[datetime],
    max_delta: int,
    strategy: str = "nearest",
    allow_reuse: bool = False,
) -> dict[str, str]:
    s2_available = set(s2_dates)
    mapping: dict[str, str] = {}

    for s1_dt in s1_dates:
        window = [d for d in s2_available if abs((d - s1_dt).days) <= max_delta]
        best = choose_candidate(s1_dt, window, strategy)
        if best is None:
            continue
        mapping[s1_dt.strftime("%Y%m%d")] = best.strftime("%Y%m%d")
        if not allow_reuse:
            # consume the chosen S2 date
            s2_available.discard(best)
    return mapping


def main():
    args = parse_args()
    rx = re.compile(args.s1_regex)

    print("[INFO] Scanning S1 dates ...")
    s1_dates = list_s1_dates(args.s1_dir, rx)
    print(f"[INFO] Found {len(s1_dates)} unique S1 date(s).")

    print("[INFO] Reading usable S2 dates ...")
    s2_dates = list_s2_usable_dates(args.s2_csv, args.status_filter)
    print(f"[INFO] Found {len(s2_dates)} S2 date(s) with status = {args.status_filter!r}.")

    print("[INFO] Matching S1 ⇔ S2 ...")
    mapping = match_dates(
        s1_dates=s1_dates,
        s2_dates=s2_dates,
        max_delta=args.max_delta,
        strategy=args.strategy,
        allow_reuse=args.allow_reuse,
    )

    print(f"[INFO] Matched {len(mapping)} S1 date(s). Example preview (up to 10):")
    for i, (k, v) in enumerate(mapping.items()):
        if i >= 10:
            break
        print(f"  S1: {k}  ←  S2: {v}")

    os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(mapping, f, indent=2)

    print(f"[INFO] Mapping saved to: {args.out_json}")


if __name__ == "__main__":
    main()
