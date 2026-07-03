"""Unit and integration tests for the cleaning pipeline.

Organized to mirror ``src/cordwell/data/cleaning.py``. Tests follow
Arrange-Act-Assert and assert on the four dimensions that matter for data
transforms: schema, row counts, null handling, and known edge cases.
"""
from __future__ import annotations

import pandas as pd
import pandera.pandas as pa
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from cordwell.data.cleaning import (
    CATEGORY_CANON,
    CHANNEL_CANON,
    REGION_CANON,
    SCHEMA_COLUMNS,
    coerce_dtypes,
    drop_duplicates,
    handle_missing,
    handle_outliers,
    normalize_label,
    normalize_strings,
    profile_dataframe,
    validate,
)


# --------------------------------------------------------------------------- #
# normalize_label
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "raw, expected",
    [
        ("  Power Tools  ", "power tools"),
        ("LUMBER", "lumber"),
        ("south   east", "south east"),   # internal whitespace collapsed
        ("Plumbing", "plumbing"),
        ("   ", ""),
        ("", ""),
    ],
)
def test_normalize_label_cases(raw, expected):
    # Arrange / Act
    result = normalize_label(raw)
    # Assert
    assert result == expected


def test_normalize_label_handles_none_and_nan():
    assert normalize_label(None) == ""
    assert normalize_label(float("nan")) == ""
    assert normalize_label(pd.NA) == ""


@given(st.text())
def test_normalize_label_never_raises_and_returns_str(text):
    """normalize_label must be total: a str out for any str in."""
    result = normalize_label(text)
    assert isinstance(result, str)


@given(st.text())
@settings(max_examples=300)
def test_normalize_label_is_idempotent(text):
    """Normalizing twice equals normalizing once."""
    once = normalize_label(text)
    twice = normalize_label(once)
    assert once == twice


# --------------------------------------------------------------------------- #
# handle_missing
# --------------------------------------------------------------------------- #
def test_handle_missing_drops_null_keys(messy_orders_df):
    # Act
    result = handle_missing(messy_orders_df)
    # Assert - the one null-order_id row is gone (7 -> 6)
    assert len(result) == 6
    assert result["order_id"].notna().all()


def test_handle_missing_imputes_and_sentinels(messy_orders_df):
    result = handle_missing(messy_orders_df)
    # quantity/unit_price nulls filled; discount null -> 0; region/channel -> Unknown
    assert result["quantity"].notna().all()
    assert result["unit_price"].notna().all()
    assert (result["discount_pct"].notna()).all()
    assert (result["store_region"] == "Unknown").sum() == 1
    assert (result["channel"] == "Unknown").sum() == 1


# --------------------------------------------------------------------------- #
# drop_duplicates
# --------------------------------------------------------------------------- #
def test_drop_duplicates_keeps_latest():
    # Arrange - same key, different updated_at and value
    df = pd.DataFrame({
        "order_id": ["A", "A", "B"],
        "updated_at": ["2024-01-01", "2024-02-01", "2024-01-15"],
        "quantity": [1, 2, 3],
    })
    # Act
    result = drop_duplicates(df, key="order_id")
    # Assert - one row per key; the LATER "A" (quantity 2) survived
    assert len(result) == 2
    assert result.loc[result["order_id"] == "A", "quantity"].iloc[0] == 2


# --------------------------------------------------------------------------- #
# normalize_strings
# --------------------------------------------------------------------------- #
def test_normalize_strings_canonicalizes():
    # Arrange
    df = pd.DataFrame({
        "product_name": [" 20V Cordless Drill ", "16 OZ. CLAW HAMMER"],
        "product_category": [" Power Tools ", "hand tools"],
        "store_region": ["se", "W"],
        "channel": ["web", "in store"],
    })
    # Act
    result = normalize_strings(df)
    # Assert
    assert result["product_name"].tolist() == ["20v cordless drill", "16 oz. claw hammer"]
    assert result["product_category"].tolist() == ["Power Tools", "Hand Tools"]
    assert result["store_region"].tolist() == ["Southeast", "West"]
    assert result["channel"].tolist() == ["Online", "In-Store"]


# --------------------------------------------------------------------------- #
# handle_outliers
# --------------------------------------------------------------------------- #
def test_handle_outliers_caps_and_flags():
    # Arrange - one bad quantity, one bad (negative) price, one bad discount
    df = pd.DataFrame({
        "quantity": pd.array([2, 9999, 5], dtype="int64"),
        "unit_price": [99.0, -5.0, 20.0],
        "discount_pct": [10.0, 150.0, 0.0],
    })
    # Act
    result = handle_outliers(df)
    # Assert - values clipped into range
    assert result["quantity"].between(1, 500).all()
    assert result["unit_price"].between(0, 10_000).all()
    assert result["discount_pct"].between(0, 100).all()
    # Flags: exactly the offending rows
    assert result["quantity_capped"].tolist() == [False, True, False]
    assert result["unit_price_capped"].tolist() == [False, True, False]
    assert result["discount_pct"].iloc[1] == 100.0


# --------------------------------------------------------------------------- #
# validate
# --------------------------------------------------------------------------- #
def _minimal_valid_frame() -> pd.DataFrame:
    """A one-row frame that satisfies CLEAN_SCHEMA, for negative tests."""
    return pd.DataFrame({
        "order_id": pd.array(["ORD-1"], dtype="string"),
        "order_date": pd.to_datetime(["2024-01-01"], utc=True),
        "store_id": pd.array([101], dtype="int64"),
        "store_region": pd.array(["Southeast"], dtype="string"),
        "product_sku": pd.array(["SKU-1"], dtype="string"),
        "product_category": pd.array(["Power Tools"], dtype="string"),
        "product_name": pd.array(["drill"], dtype="string"),
        "quantity": pd.array([3], dtype="int64"),
        "quantity_capped": [False],
        "unit_price": [9.99],
        "unit_price_capped": [False],
        "discount_pct": [10.0],
        "channel": pd.array(["Online"], dtype="string"),
    })


def test_validate_accepts_valid_frame():
    validate(_minimal_valid_frame())  # should not raise


def test_validate_rejects_out_of_range_quantity():
    bad = _minimal_valid_frame()
    bad.loc[0, "quantity"] = 9999
    with pytest.raises(pa.errors.SchemaError):
        validate(bad)


def test_validate_rejects_unknown_region():
    bad = _minimal_valid_frame()
    bad["store_region"] = pd.array(["Atlantis"], dtype="string")
    with pytest.raises(pa.errors.SchemaError):
        validate(bad)


# --------------------------------------------------------------------------- #
# profile_dataframe
# --------------------------------------------------------------------------- #
def test_profile_dataframe(messy_orders_df):
    report = profile_dataframe(messy_orders_df)
    assert report["n_rows"] == 7
    assert report["n_cols"] == 12
    assert 0 < report["null_ratio"]["store_region"] < 1


# --------------------------------------------------------------------------- #
# Full pipeline (integration) - the deck's four dimensions
# --------------------------------------------------------------------------- #
def test_pipeline_schema_and_rowcount(clean_result):
    # Schema: exactly the declared clean columns
    assert set(clean_result.columns) == set(SCHEMA_COLUMNS)
    # Row count: null-id row dropped, ORD-5 duplicate collapsed -> 5 unique orders
    assert len(clean_result) == 5


def test_pipeline_keys_unique_and_present(clean_result):
    assert clean_result["order_id"].notna().all()
    assert clean_result["order_id"].is_unique


def test_pipeline_no_remaining_nulls(clean_result):
    assert int(clean_result.isnull().sum().sum()) == 0


def test_pipeline_canonical_categoricals(clean_result):
    assert set(clean_result["store_region"]).issubset(set(REGION_CANON))
    assert set(clean_result["channel"]).issubset(set(CHANNEL_CANON))
    assert set(clean_result["product_category"]).issubset(set(CATEGORY_CANON))


def test_pipeline_outlier_handling(clean_result):
    # ORD-3 carried the outliers: capped and flagged exactly once each
    assert clean_result["quantity"].between(1, 500).all()
    assert clean_result["unit_price"].between(0, 10_000).all()
    assert clean_result["discount_pct"].between(0, 100).all()
    assert clean_result["quantity_capped"].dtype == "bool"
    assert clean_result["quantity_capped"].sum() == 1
    assert clean_result["unit_price_capped"].sum() == 1


def test_pipeline_dtypes(clean_result):
    assert str(clean_result["order_id"].dtype) == "string"
    assert clean_result["store_id"].dtype == "int64"
    assert clean_result["quantity"].dtype == "int64"
    assert clean_result["unit_price"].dtype == "float64"
    assert str(clean_result["order_date"].dtype).startswith("datetime64")


def test_pipeline_writes_parquet(tmp_path, messy_orders_df):
    from cordwell.data.cleaning import run_cleaning_pipeline

    raw_csv = tmp_path / "raw.csv"
    messy_orders_df.to_csv(raw_csv, index=False)
    out = tmp_path / "out.parquet"
    result = run_cleaning_pipeline(raw_csv, out)
    assert out.exists()
    reloaded = pd.read_parquet(out)
    assert len(reloaded) == len(result)
    assert list(reloaded.columns) == list(result.columns)
