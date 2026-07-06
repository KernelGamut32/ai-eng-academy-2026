# Cordwell Home & Hardware — Database Data Dictionary

**Week 2 · Lab 01** · companion to `cordwell.db` (built by `build_cordwell_db.py`).
Synthetic, clearly-fictional data for a made-up home-improvement retailer. One
database, three tables in a normalized star-ish layout so the lab can practice real
joins. Built seeded (default `--orders 10000 --seed 2025`): **60 products, 10,000
orders, ~44,962 order lines**.

## `products` (60 rows) — the product dimension
| Column | Type | Notes |
|---|---|---|
| `product_id` | INTEGER PK | 1-based identifier |
| `sku` | TEXT, unique | `SKU-#####` |
| `product_name` | TEXT | e.g. "20V Cordless Drill/Driver" |
| `category` | TEXT | one of 10 departments (Power Tools, Plumbing, …) |
| `list_price` | REAL | catalog price (USD) |

## `orders` (10,000 rows) — the order dimension
| Column | Type | Notes |
|---|---|---|
| `order_id` | INTEGER PK | 1-based; the pagination cursor key |
| `customer_id` | INTEGER | 10000–12499 |
| `store_id` | INTEGER | 100–139 |
| `store_region` | TEXT | Southeast, Northeast, Midwest, West, Southwest |
| `channel` | TEXT | In-Store, Online, Pro-Desk |
| `order_date` | TEXT | ISO `YYYY-MM-DD` (2024-01-01 … ~2025-06) |
| `order_ts` | TEXT | ISO timestamp `YYYY-MM-DDThh:mm:ss` |

## `order_lines` (~44,962 rows) — the fact table
| Column | Type | Notes |
|---|---|---|
| `line_id` | INTEGER PK | 1-based |
| `order_id` | INTEGER FK → orders | 1–8 lines per order |
| `product_id` | INTEGER FK → products | |
| `quantity` | INTEGER | 1–12 |
| `unit_price` | REAL | list price ±5% jitter, floor 0.50 |
| `discount_pct` | REAL | 0–25 (percent) |

**Indexes:** `order_lines(order_id)`, `order_lines(product_id)`, `orders(store_region)`.

**Referential integrity:** every `order_lines.order_id` exists in `orders`; every
`order_lines.product_id` exists in `products` (no orphans — an inner join returns
exactly the `order_lines` row count).

**Derived (computed in the lab, not stored):**
`line_total = quantity * unit_price * (1 - discount_pct/100)`.
