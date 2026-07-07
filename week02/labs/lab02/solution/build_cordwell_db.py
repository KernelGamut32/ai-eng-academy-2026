"""
build_cordwell_db.py
====================
Build a synthetic SQLite database for the Week 2 data-engineering lab.

The data is synthetic and clearly fictional: a small retail schema for
**Cordwell Home & Hardware**, a made-up home-improvement retailer. No real
company, customer, or product data is used.

Schema (a normalized star-ish layout so the lab can practice real joins):

    products     product_id (PK), sku, product_name, category, list_price
    orders       order_id (PK), customer_id, store_id, store_region,
                 channel, order_date, order_ts
    order_lines  line_id (PK), order_id (FK->orders), product_id (FK->products),
                 quantity, unit_price, discount_pct

Defaults: ~240 products, 10,000 orders, ~45,000 order lines (so chunked reads and
pagination are meaningful without being slow). Everything is seeded, so a given
--seed reproduces the same database byte-for-byte.

Run:
    python build_cordwell_db.py                 # -> cordwell.db
    python build_cordwell_db.py --orders 12000 --out /tmp/cordwell.db
    python build_cordwell_db.py --seed 7

Target environment: Python 3.13, numpy 2.x (stdlib sqlite3 for the DB).
"""
from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------- #
# Fictional Cordwell catalog: category -> list of (product_name, list_price)
# --------------------------------------------------------------------------- #
CATALOG: dict[str, list[tuple[str, float]]] = {
    "Power Tools": [
        ("20V Cordless Drill/Driver", 99.00), ("7-1/4 in. Circular Saw", 129.00),
        ("Impact Driver Kit", 149.00), ("4-1/2 in. Angle Grinder", 59.00),
        ("Random Orbital Sander", 69.00), ("Compact Reciprocating Saw", 119.00),
        ("Cordless Brad Nailer", 179.00), ("Rotary Hammer Drill", 199.00),
    ],
    "Hand Tools": [
        ("16 oz. Claw Hammer", 14.98), ("25 ft. Tape Measure", 19.97),
        ("Screwdriver Set (10-pc)", 24.99), ("Adjustable Wrench 10 in.", 12.49),
        ("Utility Knife", 8.97), ("Locking Pliers 7 in.", 11.98),
        ("Socket Set (40-pc)", 49.99),
    ],
    "Lumber": [
        ("2x4x8 Prime Stud", 4.28), ("1/2 in. 4x8 Plywood Sheet", 38.50),
        ("7/16 in. OSB Board", 21.75), ("1x6 Cedar Board", 9.85),
        ("Pressure-Treated 4x4x8", 12.60), ("2x6x10 Framing Lumber", 10.40),
    ],
    "Plumbing": [
        ("1/2 in. PVC Pipe 10 ft.", 5.42), ("Brass Ball Valve 3/4 in.", 13.95),
        ("Single-Handle Kitchen Faucet", 89.00), ("P-Trap Kit", 7.68),
        ("Pipe Wrench 14 in.", 27.50), ("Toilet Repair Kit", 18.95),
    ],
    "Electrical": [
        ("12/2 Romex 50 ft.", 62.00), ("Duplex Outlet (10-pk)", 8.90),
        ("Single-Pole Breaker 20A", 6.75), ("LED A19 Bulb (4-pk)", 11.24),
        ("Wire Connectors (100-ct)", 9.48), ("Smart Light Switch", 24.97),
    ],
    "Paint": [
        ("Interior Latex Paint 1 gal.", 34.98), ("Bonding Primer 1 gal.", 22.50),
        ("9 in. Roller Kit", 15.97), ("Painter's Tape 1.88 in.", 6.47),
        ("Angled Brush Set (3-pc)", 12.98), ("Exterior Acrylic Paint 1 gal.", 41.98),
    ],
    "Flooring": [
        ("Laminate Plank (per case)", 42.90), ("Ceramic Tile 12x12 (per box)", 28.75),
        ("Luxury Vinyl Plank (per case)", 55.20), ("Foam Underlayment Roll", 24.60),
    ],
    "Hardware": [
        ("Deck Screws 5 lb.", 34.98), ("Wood Glue 16 oz.", 6.98),
        ("Cabinet Hinges (10-pk)", 18.40), ("Drywall Anchors (50-ct)", 7.25),
        ("Cabinet Knobs (10-pk)", 22.10), ("Construction Adhesive", 5.68),
    ],
    "Garden": [
        ("Potting Soil 2 cu. ft.", 12.98), ("Hardwood Mulch 2 cu. ft.", 4.47),
        ("50 ft. Garden Hose", 29.99), ("Bypass Pruning Shears", 16.98),
        ("All-Purpose Fertilizer 25 lb.", 21.50), ("Push Reel Mower", 119.00),
    ],
    "Appliances": [
        ("40 gal. Electric Water Heater", 449.00), ("52 in. Ceiling Fan", 119.00),
        ("30 in. Range Hood", 89.00), ("1.6 cu. ft. Microwave", 129.00),
        ("Portable Dishwasher", 399.00),
    ],
}

REGIONS = ["Southeast", "Northeast", "Midwest", "West", "Southwest"]
CHANNELS = ["In-Store", "Online", "Pro-Desk"]

SCHEMA = """
CREATE TABLE products (
    product_id   INTEGER PRIMARY KEY,
    sku          TEXT NOT NULL UNIQUE,
    product_name TEXT NOT NULL,
    category     TEXT NOT NULL,
    list_price   REAL NOT NULL
);
CREATE TABLE orders (
    order_id     INTEGER PRIMARY KEY,
    customer_id  INTEGER NOT NULL,
    store_id     INTEGER NOT NULL,
    store_region TEXT NOT NULL,
    channel      TEXT NOT NULL,
    order_date   TEXT NOT NULL,   -- ISO date 'YYYY-MM-DD'
    order_ts     TEXT NOT NULL    -- ISO timestamp
);
CREATE TABLE order_lines (
    line_id     INTEGER PRIMARY KEY,
    order_id    INTEGER NOT NULL REFERENCES orders(order_id),
    product_id  INTEGER NOT NULL REFERENCES products(product_id),
    quantity    INTEGER NOT NULL,
    unit_price  REAL NOT NULL,
    discount_pct REAL NOT NULL
);
CREATE INDEX idx_lines_order ON order_lines(order_id);
CREATE INDEX idx_lines_product ON order_lines(product_id);
CREATE INDEX idx_orders_region ON orders(store_region);
"""


def build(out_path: str, n_orders: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)

    # ---- products ----
    products = []
    pid = 1
    for category, items in CATALOG.items():
        for name, price in items:
            products.append((pid, f"SKU-{pid:05d}", name, category, float(price)))
            pid += 1
    n_products = len(products)
    prod_prices = np.array([p[4] for p in products])

    # ---- orders ----
    start = np.datetime64("2024-01-01")
    orders = []
    for oid in range(1, n_orders + 1):
        day = start + int(rng.integers(0, 540))  # ~18 months
        hour = int(rng.integers(7, 21))
        minute = int(rng.integers(0, 60))
        d = np.datetime64(day, "D")
        ts = f"{str(d)}T{hour:02d}:{minute:02d}:00"
        orders.append((
            oid,
            int(rng.integers(10_000, 12_500)),          # customer_id
            int(rng.integers(100, 140)),                 # store_id
            REGIONS[int(rng.integers(0, len(REGIONS)))],
            CHANNELS[int(rng.integers(0, len(CHANNELS)))],
            str(d),
            ts,
        ))

    # ---- order_lines (~5-6 per order on average) ----
    lines = []
    line_id = 1
    discounts = np.array([0, 0, 0, 0, 5, 10, 10, 15, 20, 25], dtype=float)
    for oid in range(1, n_orders + 1):
        k = int(rng.integers(1, 9))  # 1..8 lines
        chosen = rng.integers(0, n_products, size=k)
        for prod_idx in chosen:
            base = prod_prices[prod_idx]
            unit_price = round(float(base * rng.normal(1.0, 0.05)), 2)
            unit_price = max(unit_price, 0.5)
            lines.append((
                line_id,
                oid,
                int(prod_idx) + 1,                        # product_id (1-based)
                int(rng.integers(1, 13)),                 # quantity
                unit_price,
                float(rng.choice(discounts)),
            ))
            line_id += 1

    # ---- write to SQLite ----
    out = Path(out_path)
    if out.exists():
        out.unlink()
    out.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(out)
    try:
        conn.executescript(SCHEMA)
        conn.executemany("INSERT INTO products VALUES (?,?,?,?,?)", products)
        conn.executemany("INSERT INTO orders VALUES (?,?,?,?,?,?,?)", orders)
        conn.executemany("INSERT INTO order_lines VALUES (?,?,?,?,?,?)", lines)
        conn.commit()
    finally:
        conn.close()

    return {"products": n_products, "orders": n_orders, "order_lines": len(lines), "path": str(out)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--orders", type=int, default=10_000, help="number of orders (default 10000)")
    parser.add_argument("--seed", type=int, default=2025, help="random seed (default 2025)")
    parser.add_argument("--out", type=str, default="cordwell.db", help="output SQLite path")
    args = parser.parse_args()

    stats = build(args.out, args.orders, args.seed)
    print(f"Built {stats['path']}")
    print(f"  products    : {stats['products']:,}")
    print(f"  orders      : {stats['orders']:,}")
    print(f"  order_lines : {stats['order_lines']:,}")


if __name__ == "__main__":
    main()
