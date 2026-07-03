"""Run the Cordwell order-cleaning pipeline from the command line.

Usage:
    python scripts/run_pipeline.py
    python scripts/run_pipeline.py --raw data/raw/cordwell_orders_raw.csv \
                                   --out data/processed/orders_clean.parquet

This is the production entry point: the same `run_cleaning_pipeline` function the
tests exercise, wrapped in argument parsing and logging. No cleaning logic lives
here - it all lives in the tested module.
"""
from __future__ import annotations

import argparse
import logging

from cordwell.data.cleaning import profile_dataframe, run_cleaning_pipeline
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean raw Cordwell orders into a validated Parquet file.")
    parser.add_argument("--raw", default="data/raw/cordwell_orders_raw.csv", help="raw input CSV path")
    parser.add_argument("--out", default="data/processed/orders_clean.parquet", help="clean output Parquet path")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")

    raw_profile = profile_dataframe(pd.read_csv(args.raw))
    logging.info("raw rows=%d cols=%d", raw_profile["n_rows"], raw_profile["n_cols"])

    clean = run_cleaning_pipeline(args.raw, args.out)
    logging.info("DONE: %d clean rows written to %s", len(clean), args.out)


if __name__ == "__main__":
    main()
