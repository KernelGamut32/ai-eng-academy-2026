"""Profile the raw orders CSV - always the first step, before any cleaning.

Prints shape, dtypes, per-column null ratios, and value distributions for the
categorical columns, matching the deck's "Profile First" step. Run this and read
it before you touch a single transformation.

Usage:
    python scripts/profile_raw.py
    python scripts/profile_raw.py --raw data/raw/cordwell_orders_raw.csv
"""
from __future__ import annotations

import argparse

import pandas as pd

from cordwell.data.cleaning import profile_dataframe


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile the raw orders CSV.")
    parser.add_argument("--raw", default="data/raw/cordwell_orders_raw.csv", help="raw input CSV path")
    parser.add_argument("--top", type=int, default=10, help="value_counts rows to show per column")
    args = parser.parse_args()

    df = pd.read_csv(args.raw)
    report = profile_dataframe(df)

    print(f"shape: {report['n_rows']:,} rows x {report['n_cols']} columns\n")
    print("dtypes:")
    for col, dt in report["dtypes"].items():
        print(f"  {col:18s} {dt}")

    print("\nnull ratio per column (nonzero only):")
    for col, ratio in sorted(report["null_ratio"].items(), key=lambda kv: -kv[1]):
        if ratio > 0:
            print(f"  {col:18s} {ratio:6.2%}")

    print("\ncategorical distributions:")
    for col in ("store_region", "channel", "product_category"):
        print(f"\n--- {col} ---")
        print(df[col].value_counts(dropna=False).head(args.top).to_string())


if __name__ == "__main__":
    main()
