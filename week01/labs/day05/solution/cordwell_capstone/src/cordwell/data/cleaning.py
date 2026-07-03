"""Dataset cleaning pipeline for Cordwell Home & Hardware order data.

This module turns a messy raw orders CSV into a clean, validated DataFrame and
Parquet artifact. Every transformation is a small, single-purpose, tested
function; :func:`run_cleaning_pipeline` composes them in a fixed order so the same
logic runs in tests and in the production script.

The data is synthetic and clearly fictional. Docstrings follow Google style.
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

# NOTE (currency): the Day-5 deck imports pandera the old way
#   `import pandera as pa` / `from pandera import Column, DataFrameSchema, Check`
# which still works but now emits a FutureWarning in pandera >= 0.20. The modern,
# warning-free idiom (used here) routes through the pandas submodule:
import pandera.pandas as pa
from pandera.pandas import Check, Column, DataFrameSchema

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Canonical vocabularies and alias maps
# --------------------------------------------------------------------------- #
# Keys are the *normalized* (stripped, lower-cased, whitespace-collapsed) form of
# every messy variant we expect; values are the single canonical label.
REGION_ALIASES: dict[str, str] = {
    "southeast": "Southeast", "se": "Southeast", "south east": "Southeast",
    "northeast": "Northeast", "ne": "Northeast", "north east": "Northeast",
    "midwest": "Midwest", "mw": "Midwest", "mid west": "Midwest",
    "west": "West", "w": "West",
    "southwest": "Southwest", "sw": "Southwest", "south west": "Southwest",
    "unknown": "Unknown",
}
CHANNEL_ALIASES: dict[str, str] = {
    "in-store": "In-Store", "in_store": "In-Store", "in store": "In-Store", "instore": "In-Store",
    "online": "Online", "web": "Online",
    "pro-desk": "Pro-Desk", "pro_desk": "Pro-Desk", "pro desk": "Pro-Desk", "prodesk": "Pro-Desk",
    "unknown": "Unknown",
}
# Product categories canonicalize to Title case from their lower-cased form.
CATEGORY_CANON: list[str] = [
    "Power Tools", "Hand Tools", "Lumber", "Plumbing", "Electrical",
    "Paint", "Flooring", "Hardware", "Garden", "Appliances",
]
CATEGORY_ALIASES: dict[str, str] = {c.lower(): c for c in CATEGORY_CANON}

REGION_CANON = ["Southeast", "Northeast", "Midwest", "West", "Southwest", "Unknown"]
CHANNEL_CANON = ["In-Store", "Online", "Pro-Desk", "Unknown"]

# Outlier bounds (business rules).
QUANTITY_MIN, QUANTITY_MAX = 1, 500
PRICE_MIN, PRICE_MAX = 0.0, 10_000.0
DISCOUNT_MIN, DISCOUNT_MAX = 0.0, 100.0


# --------------------------------------------------------------------------- #
# Small pure helper
# --------------------------------------------------------------------------- #
def normalize_label(value: object) -> str:
    """Return a canonical, comparable form of a text label.

    Strips leading/trailing whitespace, collapses internal runs of whitespace to
    a single space, and lower-cases the result. Non-string / missing values
    become the empty string. The function is total (never raises) and idempotent
    (``normalize_label(normalize_label(x)) == normalize_label(x)``).

    Args:
        value: Any value; typically a raw string from an ingested column.

    Returns:
        The normalized lower-case label, or ``""`` for null/blank input.
    """
    if value is None or (isinstance(value, float) and pd.isna(value)) or value is pd.NA:
        return ""
    return " ".join(str(value).split()).lower()


# --------------------------------------------------------------------------- #
# Profiling
# --------------------------------------------------------------------------- #
def profile_dataframe(df: pd.DataFrame) -> dict:
    """Summarize a DataFrame's shape, dtypes, and null ratios.

    Profiling is the mandatory first step: understand the data before changing
    it. This returns a machine-readable summary (rather than printing) so it can
    be asserted on in tests and logged in scripts.

    Args:
        df: The DataFrame to profile.

    Returns:
        A dict with keys ``n_rows``, ``n_cols``, ``dtypes`` (column -> dtype str),
        and ``null_ratio`` (column -> fraction of nulls).
    """
    return {
        "n_rows": int(len(df)),
        "n_cols": int(df.shape[1]),
        "dtypes": {c: str(t) for c, t in df.dtypes.items()},
        "null_ratio": {c: float(r) for c, r in df.isnull().mean().items()},
    }


# --------------------------------------------------------------------------- #
# Cleaning steps
# --------------------------------------------------------------------------- #
def handle_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the per-column missing-data strategy.

    Strategy (one decision per column, documented here and in the data dictionary):
      * ``order_id``: drop the row - it is the required business key.
      * ``quantity``: median-impute - numeric, missing-at-random.
      * ``unit_price``: median-impute - numeric, missing-at-random.
      * ``discount_pct``: fill ``0.0`` - a missing discount means no discount.
      * ``store_region``: sentinel ``"Unknown"`` - missingness is itself a category.
      * ``channel``: sentinel ``"Unknown"`` - same rationale.

    Args:
        df: DataFrame straight from ``read_csv``.

    Returns:
        A copy with nulls resolved per the strategy above.
    """
    df = df.dropna(subset=["order_id"]).copy()
    n_dropped = 0  # recomputed below for logging
    df["quantity"] = df["quantity"].fillna(df["quantity"].median())
    df["unit_price"] = df["unit_price"].fillna(df["unit_price"].median())
    df["discount_pct"] = df["discount_pct"].fillna(0.0)
    df["store_region"] = df["store_region"].fillna("Unknown")
    df["channel"] = df["channel"].fillna("Unknown")
    logger.info("handle_missing: %d rows remain after dropping null keys", len(df))
    return df


def drop_duplicates(df: pd.DataFrame, key: str = "order_id") -> pd.DataFrame:
    """Remove duplicate business keys, keeping the most recently updated row.

    Exact and key-based duplicates both inflate metrics. Here we sort by
    ``updated_at`` descending and keep the first row per ``key`` - i.e. the
    latest record wins.

    Args:
        df: Input DataFrame with a ``updated_at`` column.
        key: Business-key column to deduplicate on.

    Returns:
        A DataFrame with one row per ``key`` value.
    """
    n_before = len(df)
    df = (
        df.sort_values("updated_at", ascending=False)
        .drop_duplicates(subset=[key], keep="first")
        .reset_index(drop=True)
    )
    logger.info("drop_duplicates: dropped %d duplicate rows on %s", n_before - len(df), key)
    return df


def coerce_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce columns to their intended dtypes.

    Pandas infers types on read and is often wrong (nullable int columns become
    float, dates stay strings). Explicit coercion prevents downstream errors.

    Args:
        df: DataFrame after missing-data handling and deduplication.

    Returns:
        A copy with corrected dtypes: an explicit UTC datetime, integer key/ids,
        and PyArrow-friendly string columns.
    """
    df = df.copy()
    df["order_date"] = pd.to_datetime(df["order_date"], format="%Y-%m-%d", utc=True)
    df["store_id"] = df["store_id"].astype("int64")
    df["quantity"] = df["quantity"].astype("int64")
    df["unit_price"] = df["unit_price"].astype("float64")
    df["discount_pct"] = df["discount_pct"].astype("float64")
    for col in ("order_id", "product_sku", "product_category", "product_name",
                "store_region", "channel"):
        df[col] = df[col].astype("string")
    return df


def normalize_strings(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize string columns to a canonical form.

    Inconsistent case/whitespace creates phantom categories that break grouping
    and joins. ``product_name`` is lower-cased and whitespace-normalized;
    ``product_category``, ``store_region``, and ``channel`` are additionally
    mapped through their alias tables to a single canonical label.

    Args:
        df: DataFrame after dtype coercion.

    Returns:
        A copy with canonicalized string columns (dtype ``string``).
    """
    df = df.copy()
    df["product_name"] = df["product_name"].map(normalize_label)
    df["product_category"] = df["product_category"].map(
        lambda v: CATEGORY_ALIASES.get(normalize_label(v), normalize_label(v).title())
    )
    df["store_region"] = df["store_region"].map(
        lambda v: REGION_ALIASES.get(normalize_label(v), str(v))
    )
    df["channel"] = df["channel"].map(
        lambda v: CHANNEL_ALIASES.get(normalize_label(v), str(v))
    )
    for col in ("product_name", "product_category", "store_region", "channel"):
        df[col] = df[col].astype("string")
    return df


def handle_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Cap out-of-range numeric values and flag the rows that were capped.

    ``quantity`` and ``unit_price`` are clipped to plausible business ranges,
    each with a boolean indicator column so downstream models can learn from the
    flag. ``discount_pct`` is clipped to ``[0, 100]`` (no flag - a bounded
    percentage).

    Args:
        df: DataFrame after string normalization.

    Returns:
        A copy with capped values and the ``quantity_capped`` /
        ``unit_price_capped`` indicator columns.
    """
    df = df.copy()
    df["quantity_capped"] = (df["quantity"] < QUANTITY_MIN) | (df["quantity"] > QUANTITY_MAX)
    df["quantity"] = df["quantity"].clip(lower=QUANTITY_MIN, upper=QUANTITY_MAX)

    df["unit_price_capped"] = (df["unit_price"] < PRICE_MIN) | (df["unit_price"] > PRICE_MAX)
    df["unit_price"] = df["unit_price"].clip(lower=PRICE_MIN, upper=PRICE_MAX)

    df["discount_pct"] = df["discount_pct"].clip(lower=DISCOUNT_MIN, upper=DISCOUNT_MAX)

    logger.info(
        "handle_outliers: capped %d quantities, %d prices",
        int(df["quantity_capped"].sum()), int(df["unit_price_capped"].sum()),
    )
    return df


# --------------------------------------------------------------------------- #
# Schema validation
# --------------------------------------------------------------------------- #
CLEAN_SCHEMA = DataFrameSchema(
    columns={
        "order_id": Column(pa.String, nullable=False, unique=True),
        "order_date": Column(pd.DatetimeTZDtype(tz="UTC"), coerce=True, nullable=False),
        "store_id": Column(pa.Int64, nullable=False),
        "store_region": Column(pa.String, Check.isin(REGION_CANON), nullable=False),
        "product_sku": Column(pa.String, nullable=False),
        "product_category": Column(pa.String, Check.isin(CATEGORY_CANON), nullable=False),
        "product_name": Column(pa.String, nullable=False),
        "quantity": Column(pa.Int64, Check.in_range(QUANTITY_MIN, QUANTITY_MAX), nullable=False),
        "quantity_capped": Column(pa.Bool, nullable=False),
        "unit_price": Column(pa.Float64, Check.in_range(PRICE_MIN, PRICE_MAX), nullable=False),
        "unit_price_capped": Column(pa.Bool, nullable=False),
        "discount_pct": Column(pa.Float64, Check.in_range(DISCOUNT_MIN, DISCOUNT_MAX), nullable=False),
        "channel": Column(pa.String, Check.isin(CHANNEL_CANON), nullable=False),
    },
    strict=True,  # no unexpected columns allowed
    ordered=False,
)

# The exact clean-output columns, in order. Selecting these before validation
# drops raw-only helper columns (e.g. updated_at).
SCHEMA_COLUMNS: list[str] = list(CLEAN_SCHEMA.columns)


def validate(df: pd.DataFrame) -> pd.DataFrame:
    """Validate ``df`` against :data:`CLEAN_SCHEMA`.

    Turns implicit assumptions into an executable contract. A violation raises
    ``pandera.errors.SchemaError`` immediately - before bad data is written.

    Args:
        df: The cleaned DataFrame, containing exactly the schema columns.

    Returns:
        The validated DataFrame (unchanged if valid).

    Raises:
        pandera.errors.SchemaError: If any column, dtype, or value check fails.
    """
    return CLEAN_SCHEMA.validate(df)


# --------------------------------------------------------------------------- #
# Output
# --------------------------------------------------------------------------- #
def write_parquet(df: pd.DataFrame, path: str | Path, compression: str = "snappy") -> Path:
    """Write a validated DataFrame to Parquet.

    Parquet stores schema and dtypes, so there is no re-inference on load - the
    preferred format for inter-stage data transfer.

    Args:
        df: Clean, validated DataFrame.
        path: Destination file path.
        compression: Parquet compression codec (default ``"snappy"``).

    Returns:
        The resolved output path.
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, compression=compression, index=False)
    logger.info("write_parquet: wrote %d rows to %s", len(df), out)
    return out


# --------------------------------------------------------------------------- #
# The reproducible pipeline
# --------------------------------------------------------------------------- #
def run_cleaning_pipeline(raw_path: str | Path, output_path: str | Path) -> pd.DataFrame:
    """Run the full cleaning pipeline end-to-end.

    Composes every step in a fixed order: read -> handle missing -> deduplicate
    -> coerce dtypes -> normalize strings -> handle outliers -> select schema
    columns -> validate -> write Parquet. Pure input/output; no hidden state.

    Args:
        raw_path: Path to the raw CSV file.
        output_path: Destination Parquet path.

    Returns:
        The cleaned, validated DataFrame.
    """
    df = pd.read_csv(raw_path)
    df = handle_missing(df)
    df = drop_duplicates(df, key="order_id")
    df = coerce_dtypes(df)
    df = normalize_strings(df)
    df = handle_outliers(df)
    df = df[SCHEMA_COLUMNS]
    df = validate(df)
    write_parquet(df, output_path)
    logger.info("run_cleaning_pipeline: produced %d clean rows", len(df))
    return df
