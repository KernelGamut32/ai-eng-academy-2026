"""Shared pytest fixtures for the cleaning tests.

``conftest.py`` fixtures are available to every test in this directory and its
subdirectories without importing them.
"""
from __future__ import annotations

import pandas as pd
import pytest


@pytest.fixture
def messy_orders_df() -> pd.DataFrame:
    """A tiny, messy raw orders DataFrame matching the pipeline's input schema.

    Deliberately exercises every cleaning step, with known expected outcomes:

    * ``order_id``   - one null row (dropped) + a duplicate key ``ORD-5`` (deduped).
    * ``store_region`` - alias spellings ("se", "MW", "W") + one null (-> "Unknown").
    * ``channel``    - alias spellings ("web", "in store") + one null (-> "Unknown").
    * ``product_category`` / ``product_name`` - inconsistent case and whitespace.
    * ``quantity``   - one null (median-imputed) and one outlier 9999 (capped to 500).
    * ``unit_price`` - one null (median-imputed) and one negative -5 (capped to 0).
    * ``discount_pct`` - one null (-> 0) and one out-of-range 150 (capped to 100).
    * ``updated_at`` - the duplicate ``ORD-5`` rows differ so dedup keeps the latest.

    After cleaning this yields exactly **5** unique orders (ORD-1..ORD-5).
    """
    return pd.DataFrame(
        {
            "order_id": ["ORD-1", "ORD-2", "ORD-3", "ORD-4", "ORD-5", "ORD-5", None],
            "order_date": ["2024-01-05", "2024-02-11", "2024-03-01", "2024-04-01",
                           "2024-05-01", "2024-05-01", "2024-06-01"],
            "store_id": [101, 102, 103, 104, 105, 105, 106],
            "store_region": ["se", " Southeast ", "MW", None, "west", "W", "northeast"],
            "product_sku": ["SKU-00001", "SKU-00002", "SKU-00003", "SKU-00004",
                            "SKU-00005", "SKU-00005", "SKU-00006"],
            "product_category": [" Power Tools ", "LUMBER", "plumbing", "Paint",
                                 "hand tools", "hand tools", "Garden"],
            "product_name": [" 20V Cordless Drill ", "2x4x8 PRIME STUD", "1/2 in. PVC Pipe",
                             "Interior Latex Paint", "16 OZ. CLAW HAMMER",
                             "16 oz. claw hammer", "Potting Soil"],
            "quantity": [2, None, 9999, 5, 3, 3, 1],
            "unit_price": [99.00, 4.28, -5.00, None, 14.98, 14.98, 12.98],
            "discount_pct": [10, None, 150, 0, 5, 5, 0],
            "channel": ["in_store", "web", None, "online", "in store", "in store", "pro desk"],
            "updated_at": ["2024-06-01", "2024-06-02", "2024-06-03", "2024-06-04",
                           "2024-06-05", "2024-06-10", "2024-06-07"],
        }
    )


@pytest.fixture
def clean_result(tmp_path, messy_orders_df) -> pd.DataFrame:
    """Run the full pipeline on the messy fixture and return the clean DataFrame.

    Writes the fixture to a raw CSV (so the pipeline reads it exactly as it would
    a real file), runs the pipeline, and returns the validated output.
    """
    from cordwell.data.cleaning import run_cleaning_pipeline

    raw_csv = tmp_path / "raw.csv"
    messy_orders_df.to_csv(raw_csv, index=False)
    out_parquet = tmp_path / "clean.parquet"
    return run_cleaning_pipeline(raw_csv, out_parquet)
