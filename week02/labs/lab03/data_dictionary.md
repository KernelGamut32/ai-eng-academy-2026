# Data Dictionary — Lab 03 synthetic datasets

Both datasets are **synthetic, clearly fictional, and generated in the notebook** (seeded).
No files or internet needed.

## `customers` (1,000 rows) — Cordwell online-store customers
Generated with `np.random.default_rng(42)`.

| Column | Type | Notes |
|---|---|---|
| `customer_id` | int64 | 0…999 |
| `age` | int64 | 16–79 |
| `country` | str | **Deliberately messy:** `US`, `U.S.A.`, `USA` (three spellings of one country), plus `SG`, `DE`, `BR`, `IN`. Drives the C2 normalization lesson. |
| `sessions` | int64 | Poisson(λ=3) |
| `avg_session_sec` | float64 | ~Normal(300, 60), clipped to [30, 1500] |
| `spend_usd` | float64 | Lognormal (right-skewed spend) |

Columns **added during the lab:** `is_adult` (B1), `vip` (B2), `engagement` (D3).

## `orders` (2,000 rows) — partitioned Parquet for Part E
Generated with `np.random.default_rng(7)`, written to `artifacts/orders/store_region=<R>.parquet`
(one file per region).

| Column | Type | Notes |
|---|---|---|
| `order_id` | int64 | 0…1999 |
| `store_region` | str | Southeast, Northeast, Midwest, West, Southwest |
| `order_date` | datetime64[ns] | 2024-01-01 … ~2025-06 |
| `order_amount` | float64 | Lognormal |
