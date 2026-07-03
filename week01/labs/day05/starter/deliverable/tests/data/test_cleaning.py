"""Unit and integration tests for the cleaning pipeline.  [STUDENT]

Most tests are PROVIDED - they are your acceptance criteria. As you implement
``cleaning.py``, run ``pytest -q`` and watch them turn from errors to passes.

THREE tests are marked ``TODO`` (they currently ``pytest.skip``). Implement them
to practice writing tests yourself - a parametrized case table, a Hypothesis
property, and an edge case. Remove the ``pytest.skip`` line once written.
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
def test_normalize_label_cases():
    """TODO (write this test).

    Use @pytest.mark.parametrize to check normalize_label on at least these
    (raw -> expected) pairs, then assert equality:
        "  Power Tools  " -> "power tools"
        "LUMBER"          -> "lumber"
        "south   east"    -> "south east"   (internal whitespace collapsed)
        "   "             -> ""
        ""                -> ""
    """
    pytest.skip("TODO: implement a parametrized test for normalize_label")


def test_normalize_label_handles_none_and_nan():
    assert normalize_label(None) == ""
    assert normalize_label(float("nan")) == ""
    assert normalize_label(pd.NA) == ""


@given(st.text())
def test_normalize_label_never_raises_and_returns_str(text):
    """normalize_label must be total: a str out for any str in."""
    result = normalize_label(text)
    assert isinstance(result, str)


def test_normalize_label_is_idempotent():
    """TODO (write this test).

    Use Hypothesis (@given(st.text())) to assert that normalizing twice equals
    normalizing once:  normalize_label(normalize_label(x)) == normalize_label(x).
    """
    pytest.skip("TODO: implement a Hypothesis idempotency test for normalize_label")


# --------------------------------------------------------------------------- #
# handle_missing
# --------------------------------------------------------------------------- #
def test_handle_missing_drops_null_keys(messy_orders_df):
    result = handle_missing(messy_orders_df)
    assert len(result) == 6
    assert result["order_id"].notna().all()


def test_handle_missing_imputes_and_sentinels(messy_orders_df):
    result = handle_missing(messy_orders_df)
    assert result["quantity"].notna().all()
    assert result["unit_price"].notna().all()
    assert (result["discount_pct"].notna()).all()
    assert (result["store_region"] == "Unknown").sum() == 1
    assert (result["channel"] == "Unknown").sum() == 1


# --------------------------------------------------------------------------- #
# drop_duplicates
# --------------------------------------------------------------------------- #
def test_drop_duplicates_keeps_latest():
    """TODO (write this test).

    Arrange a small DataFrame with a repeated key "A" (two rows) and a key "B",
    where the two "A" rows have different ``updated_at`` values and a different
    ``quantity``. Act: call drop_duplicates(df, key="order_id"). Assert the result
    has one row per key AND that the row kept for "A" is the one with the LATER
    ``updated_at``.
    """
    pytest.skip("TODO: implement the keeps-latest edge-case test")


# --------------------------------------------------------------------------- #
# normalize_strings
# --------------------------------------------------------------------------- #
def test_normalize_strings_canonicalizes():
    df = pd.DataFrame({
        "product_name": [" 20V Cordless Drill ", "16 OZ. CLAW HAMMER"],
        "product_category": [" Power Tools ", "hand tools"],
        "store_region": ["se", "W"],
        "channel": ["web", "in store"],
    })
    result = normalize_strings(df)
    assert result["product_name"].tolist() == ["20v cordless drill", "16 oz. claw hammer"]
    assert result["product_category"].tolist() == ["Power Tools", "Hand Tools"]
    assert result["store_region"].tolist() == ["Southeast", "West"]
    assert result["channel"].tolist() == ["Online", "In-Store"]


# --------------------------------------------------------------------------- #
# handle_outliers
# --------------------------------------------------------------------------- #
def test_handle_outliers_caps_and_flags():
    df = pd.DataFrame({
        "quantity": pd.array([2, 9999, 5], dtype="int64"),
        "unit_price": [99.0, -5.0, 20.0],
        "discount_pct": [10.0, 150.0, 0.0],
    })
    result = handle_outliers(df)
    assert result["quantity"].between(1, 500).all()
    assert result["unit_price"].between(0, 10_000).all()
    assert result["discount_pct"].between(0, 100).all()
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
    assert set(clean_result.columns) == set(SCHEMA_COLUMNS)
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
