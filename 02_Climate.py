#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
02_Climate.py

Process daily CHIRPS rainfall data into a Sentinel usability indicator.
Rule:
    - Rainfall < 10 mm → "usable"
    - Rainfall >= 10 mm → "disturbed"

Inputs:
    CSV with columns including "date" and "rain_mm"
Outputs:
    CSV with an extra column "sentinel_support"
"""

import pandas as pd
import argparse


def main():
    parser = argparse.ArgumentParser(description="Convert CHIRPS rainfall to Sentinel usability flag.")
    parser.add_argument("--input", required=True, help="Input CSV (daily CHIRPS rainfall).")
    parser.add_argument("--output", required=True, help="Output CSV with usability flag.")
    parser.add_argument("--threshold", type=float, default=10.0,
                        help="Rainfall threshold in mm (default: 10.0).")
    args = parser.parse_args()

    # Read data
    df = pd.read_csv(args.input)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")

    # Apply rule
    df["sentinel_support"] = df["rain_mm"].apply(
        lambda x: "usable" if x < args.threshold else "disturbed"
    )

    # Save
    df.to_csv(args.output, index=False)

    print("[INFO] Daily Sentinel usability classification done.")
    print(f"[INFO] Saved to: {args.output}")


if __name__ == "__main__":
    main()
