"""
generate_raw_data.py
====================
Generate a realistic, messy raw orders CSV for the Week 1 capstone cleaning lab.

The data is synthetic and clearly fictional: point-of-sale order lines for
**Cordwell Home & Hardware**, a made-up home-improvement retailer. No real
company, store, customer, or product data is used.

The generator first builds a clean, internally-consistent dataset, then
deliberately injects the messiness the lab teaches you to clean:

  * order_id      - a few blank rows (dropped) and duplicate ids (deduped)
  * store_region  - alias spellings ("se", "South East") and some nulls
  * channel       - alias spellings ("in store", "web") and some nulls
  * product_category / product_name - inconsistent case and whitespace
  * quantity      - some nulls (median-imputed) and absurd outliers (capped)
  * unit_price    - some nulls (median-imputed) and bad values (negatives/huge)
  * discount_pct  - some nulls (filled 0) and out-of-range values (>100, capped)

Run:
    python scripts/generate_raw_data.py                       # 45,000 rows -> data/raw/cordwell_orders_raw.csv
    python scripts/generate_raw_data.py --rows 60000          # a bigger file
    python scripts/generate_raw_data.py --clean-reference     # also write the pre-messy clean CSV
    python scripts/generate_raw_data.py --seed 7 --out /tmp/x.csv

Everything is seeded, so a given (--rows, --seed) always produces the same file.

Target environment: Python 3.13, pandas 3.x, numpy 2.x.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Fictional Cordwell Home & Hardware catalog
# --------------------------------------------------------------------------- #
# category -> list of (product_name, typical_unit_price)
CATALOG: dict[str, list[tuple[str, float]]] = {
    "Power Tools": [
        ("20V Cordless Drill/Driver", 99.00), ("7-1/4 in. Circular Saw", 129.00),
        ("Impact Driver Kit", 149.00), ("4-1/2 in. Angle Grinder", 59.00),
        ("Random Orbital Sander", 69.00), ("Compact Reciprocating Saw", 119.00),
    ],
    "Hand Tools": [
        ("16 oz. Claw Hammer", 14.98), ("25 ft. Tape Measure", 19.97),
        ("Screwdriver Set (10-pc)", 24.99), ("Adjustable Wrench 10 in.", 12.49),
        ("Utility Knife", 8.97), ("Locking Pliers 7 in.", 11.98),
    ],
    "Lumber": [
        ("2x4x8 Prime Stud", 4.28), ("1/2 in. 4x8 Plywood Sheet", 38.50),
        ("7/16 in. OSB Board", 21.75), ("1x6 Cedar Board", 9.85),
        ("Pressure-Treated 4x4x8", 12.60),
    ],
    "Plumbing": [
        ("1/2 in. PVC Pipe 10 ft.", 5.42), ("Brass Ball Valve 3/4 in.", 13.95),
        ("Single-Handle Kitchen Faucet", 89.00), ("P-Trap Kit", 7.68),
        ("Pipe Wrench 14 in.", 27.50),
    ],
    "Electrical": [
        ("12/2 Romex 50 ft.", 62.00), ("Duplex Outlet (10-pk)", 8.90),
        ("Single-Pole Breaker 20A", 6.75), ("LED A19 Bulb (4-pk)", 11.24),
        ("Wire Connectors (100-ct)", 9.48),
    ],
    "Paint": [
        ("Interior Latex Paint 1 gal.", 34.98), ("Bonding Primer 1 gal.", 22.50),
        ("9 in. Roller Kit", 15.97), ("Painter's Tape 1.88 in.", 6.47),
        ("Angled Brush Set (3-pc)", 12.98),
    ],
    "Flooring": [
        ("Laminate Plank (per case)", 42.90), ("Ceramic Tile 12x12 (per box)", 28.75),
        ("Luxury Vinyl Plank (per case)", 55.20), ("Foam Underlayment Roll", 24.60),
    ],
    "Hardware": [
        ("Deck Screws 5 lb.", 34.98), ("Wood Glue 16 oz.", 6.98),
        ("Cabinet Hinges (10-pk)", 18.40), ("Drywall Anchors (50-ct)", 7.25),
        ("Cabinet Knobs (10-pk)", 22.10),
    ],
    "Garden": [
        ("Potting Soil 2 cu. ft.", 12.98), ("Hardwood Mulch 2 cu. ft.", 4.47),
        ("50 ft. Garden Hose", 29.99), ("Bypass Pruning Shears", 16.98),
        ("All-Purpose Fertilizer 25 lb.", 21.50),
    ],
    "Appliances": [
        ("40 gal. Electric Water Heater", 449.00), ("52 in. Ceiling Fan", 119.00),
        ("30 in. Range Hood", 89.00), ("1.6 cu. ft. Microwave", 129.00),
    ],
}

CATEGORIES = list(CATALOG.keys())

# Canonical store regions and the messy alias variants seen in raw data.
REGION_CANON = ["Southeast", "Northeast", "Midwest", "West", "Southwest"]
REGION_VARIANTS: dict[str, list[str]] = {
    "Southeast": ["Southeast", "southeast", "SE", "South East", " southeast "],
    "Northeast": ["Northeast", "northeast", "NE", "North East"],
    "Midwest": ["Midwest", "midwest", "MW", "Mid West"],
    "West": ["West", "west", "W"],
    "Southwest": ["Southwest", "southwest", "SW", "South West"],
}

# Canonical channels and messy variants.
CHANNEL_CANON = ["In-Store", "Online", "Pro-Desk"]
CHANNEL_VARIANTS: dict[str, list[str]] = {
    "In-Store": ["In-Store", "in_store", "in store", "instore", "IN-STORE"],
    "Online": ["Online", "online", "web", "ONLINE"],
    "Pro-Desk": ["Pro-Desk", "pro_desk", "pro desk", "prodesk", "Pro Desk"],
}


def _messy_case_whitespace(rng: np.random.Generator, value: str) -> str:
    """Randomly perturb case and surrounding whitespace of a clean string."""
    roll = rng.random()
    if roll < 0.25:
        value = value.upper()
    elif roll < 0.45:
        value = value.lower()
    pad_left = " " * int(rng.integers(0, 3))
    pad_right = " " * int(rng.integers(0, 3))
    return f"{pad_left}{value}{pad_right}"


def build_clean(n_rows: int, seed: int) -> pd.DataFrame:
    """Build the clean, internally-consistent base dataset (pre-messiness)."""
    rng = np.random.default_rng(seed)

    # Flatten the catalog into aligned arrays for fast sampling.
    cat_names, prod_names, prod_prices = [], [], []
    for cat, items in CATALOG.items():
        for name, price in items:
            cat_names.append(cat)
            prod_names.append(name)
            prod_prices.append(price)
    cat_names = np.array(cat_names)
    prod_names = np.array(prod_names, dtype=object)
    prod_prices = np.array(prod_prices)

    pick = rng.integers(0, len(prod_names), size=n_rows)
    category = cat_names[pick]
    product_name = prod_names[pick]
    base_price = prod_prices[pick]

    # order_id: unique, zero-padded
    order_id = np.array([f"ORD-{100000 + i:08d}" for i in range(n_rows)], dtype=object)

    # order_date across 2023-2024
    start = np.datetime64("2023-01-01")
    order_date = start + rng.integers(0, 730, size=n_rows).astype("timedelta64[D]")

    # updated_at: order_date + 0-30 days (used to pick the latest on dedup)
    updated_at = order_date + rng.integers(0, 30, size=n_rows).astype("timedelta64[D]")

    store_id = rng.integers(100, 150, size=n_rows)
    region = rng.choice(REGION_CANON, size=n_rows, p=[0.30, 0.22, 0.20, 0.18, 0.10])
    channel = rng.choice(CHANNEL_CANON, size=n_rows, p=[0.55, 0.30, 0.15])

    quantity = rng.integers(1, 13, size=n_rows)
    # unit_price jitters +/-8% around the catalog price
    unit_price = (base_price * rng.normal(1.0, 0.08, size=n_rows)).round(2)
    discount_pct = np.clip(rng.choice([0, 0, 0, 5, 10, 15, 20, 25], size=n_rows)
                           + rng.integers(0, 5, size=n_rows), 0, 40).astype(float)

    df = pd.DataFrame(
        {
            "order_id": order_id,
            "order_date": pd.to_datetime(order_date).strftime("%Y-%m-%d"),
            "store_id": store_id,
            "store_region": region,
            "product_sku": [f"SKU-{p:05d}" for p in pick],
            "product_category": category,
            "product_name": product_name,
            "quantity": quantity,
            "unit_price": unit_price,
            "discount_pct": discount_pct,
            "channel": channel,
            "updated_at": pd.to_datetime(updated_at).strftime("%Y-%m-%d"),
        }
    )
    return df


def inject_messiness(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Deliberately dirty the clean dataset to exercise every cleaning step."""
    rng = np.random.default_rng(seed + 1)
    df = df.copy()
    n = len(df)

    def sample(frac: float) -> np.ndarray:
        k = max(1, int(frac * n))
        return rng.choice(n, size=k, replace=False)

    # --- string case/whitespace noise on category + product_name ---
    df["product_category"] = [
        _messy_case_whitespace(rng, v) if rng.random() < 0.55 else v
        for v in df["product_category"]
    ]
    df["product_name"] = [
        _messy_case_whitespace(rng, v) if rng.random() < 0.45 else v
        for v in df["product_name"]
    ]

    # --- region + channel: swap canonical values for messy aliases ---
    df["store_region"] = [
        rng.choice(REGION_VARIANTS[v]) for v in df["store_region"]
    ]
    df["channel"] = [rng.choice(CHANNEL_VARIANTS[v]) for v in df["channel"]]

    # --- nulls ---
    df.loc[sample(0.04), "store_region"] = np.nan     # sentinel -> "Unknown"
    df.loc[sample(0.04), "channel"] = np.nan           # sentinel -> "Unknown"
    df.loc[sample(0.03), "quantity"] = np.nan          # median impute
    df.loc[sample(0.03), "unit_price"] = np.nan        # median impute
    df.loc[sample(0.05), "discount_pct"] = np.nan      # fill 0.0

    # --- numeric outliers ---
    q_out = sample(0.004)
    df.loc[q_out, "quantity"] = rng.choice([5000, 9999, 12000], size=len(q_out))
    p_hi = sample(0.003)
    df.loc[p_hi, "unit_price"] = rng.choice([88888.0, 99999.0], size=len(p_hi))
    p_neg = sample(0.003)
    df.loc[p_neg, "unit_price"] = -rng.random(len(p_neg)).round(2) * 50 - 1.0
    d_out = sample(0.004)
    df.loc[d_out, "discount_pct"] = rng.choice([150.0, 200.0, 999.0], size=len(d_out))

    # --- duplicate business keys: copy some rows with a LATER updated_at + tweaks ---
    dup_src = sample(0.015)
    dups = df.loc[dup_src].copy()
    dups["updated_at"] = (
        pd.to_datetime(dups["updated_at"]) + pd.Timedelta(days=10)
    ).dt.strftime("%Y-%m-%d")
    # the newer duplicate has a slightly different (corrected) quantity
    dups["quantity"] = dups["quantity"].fillna(1)
    dups = dups.reset_index(drop=True)
    df = pd.concat([df, dups], ignore_index=True)

    # --- blank order_id on a few rows (required key -> dropped) ---
    df.loc[rng.choice(len(df), size=max(1, int(0.005 * len(df))), replace=False), "order_id"] = np.nan

    # shuffle so duplicates/nulls are not all at the bottom
    df = df.sample(frac=1.0, random_state=seed + 2).reset_index(drop=True)
    return df


def generate(n_rows: int, seed: int) -> pd.DataFrame:
    return inject_messiness(build_clean(n_rows, seed), seed)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rows", type=int, default=45_000, help="number of clean rows before messiness (default 45000)")
    parser.add_argument("--seed", type=int, default=2025, help="random seed (default 2025)")
    parser.add_argument("--out", type=str, default="data/raw/cordwell_orders_raw.csv",
                        help="output CSV path (default data/raw/cordwell_orders_raw.csv)")
    parser.add_argument("--clean-reference", action="store_true",
                        help="also write the pre-messy clean CSV next to the raw file")
    args = parser.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    if args.clean_reference:
        clean = build_clean(args.rows, args.seed)
        ref = out.with_name(out.stem + "_clean_reference.csv")
        clean.to_csv(ref, index=False)
        print(f"wrote clean reference: {ref}  ({len(clean):,} rows)")

    messy = generate(args.rows, args.seed)
    messy.to_csv(out, index=False)
    print(f"wrote messy raw data:  {out}  ({len(messy):,} rows, {messy.shape[1]} columns)")
    print(f"  blank order_id rows : {int(messy['order_id'].isna().sum())}")
    print(f"  duplicate order_ids : {int(messy['order_id'].dropna().duplicated().sum())}")
    print(f"  null quantity       : {int(messy['quantity'].isna().sum())}")
    print(f"  null unit_price     : {int(messy['unit_price'].isna().sum())}")


if __name__ == "__main__":
    main()
