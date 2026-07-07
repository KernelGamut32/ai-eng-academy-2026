# Data Dictionary — Lab 04 Synthetic Dataset
## Cordwell Home & Hardware — online customers (fictional)

**Everything here is synthetic and clearly fictional.** *Cordwell Home & Hardware* is a
made-up retailer used only as a teaching context. No real Lowe's, customer, or product data
is present. The dataset is generated in-notebook from a fixed seed
(`np.random.default_rng(123)`, `n = 1500`), so every learner gets identical values — no files
or network required.

---

## How the raw frame is built
- **1,500 base rows**, then **60 duplicate-key rows** appended (a random sample of existing
  `customer_id`s, re-inserted with `spend = "$0.00"` and `is_marketing_opt_in = False`) →
  **1,560 raw rows**.
- Missingness is injected on purpose so each cleaning decision is real.

## Raw columns (as generated — *before* cleaning)
| Column | Raw dtype (pandas 3.0) | Mess injected | Notes |
|---|---|---|---|
| `customer_id` | `int64` | none | Business key. `np.arange(1500)`; **duplicated** by the 60 appended rows. |
| `email` | `str` | ~2% `None` | Required key — rows missing it are dropped. |
| `age` | `float64` | ~4% `NaN` | Measure; kept nullable, not force-imputed. |
| `country` | `str` | 3 spellings of USA (`US`,`U.S.A.`,`usa`) + ~4% `None` | Descriptive; normalized, nulls → `UNKNOWN`. |
| `signup_date` | `str` | 4 formats (`2025-01-05`,`01/06/2025`,`06-01-2025`,`2025/01/07`) + ~4% `None` | Sequencing field; needs `format="mixed"`. |
| `spend` | `str` | `$` + thousands-comma, European decimal-comma (`€45,00`), blanks, `None` | Money-as-text; two comma meanings. |
| `is_marketing_opt_in` | `object` | `True`/`False`/`None` mix (~5% `None`) | Stays `object` because of mixed types. |

> **pandas-3.0 note:** text columns arrive as the new **`str`** dtype (not `object`). A mixed
> `True/False/None` column stays `object`. This affects `pandera` (`Column(str)` vs `Column(object)`).

## Derived / cleaned columns (added during the lab)
| Column | Cleaned dtype | Derivation |
|---|---|---|
| `country_norm` | `str` (no nulls) | `country` mapped to canonical codes; nulls → `"UNKNOWN"`. |
| `spend_usd` | `float64` (no nulls) | `spend` parsed to number (both comma meanings handled), then per-country median impute, then `0.0` backstop. |
| `signup_dt` | `datetime64[us]` | `signup_date` parsed with `format="mixed"`, `dayfirst=False`. `us` = pandas-3.0 default resolution. |

## Cleaned type targets for the two coerced source columns
| Column | Final dtype | Policy |
|---|---|---|
| `age` | `Float64` (nullable) | Keep the ~4% nulls; missingness may be informative. |
| `is_marketing_opt_in` | `bool` (no nulls) | Nulls default to `False` (safe marketing policy) then cast. |

---

## Verified pipeline landmarks (seed-locked)
| Stage | Value |
|---|---|
| Raw rows | 1560 |
| After dropping missing required keys | 1528 (dropped 32) |
| `country_norm` counts | USA 511 · BR 311 · DE 264 · SG 244 · IN 136 · UNKNOWN 62 |
| `spend_usd` nulls before impute | 129 |
| Median `spend_usd` after impute | 99.0 |
| Dates → `NaT`: naive vs `format="mixed"` | 1163 vs 66 |
| Duplicate-key rows / distinct keys | 118 / 59 |
| Rows after de-dup | 1469 |

## Target export schema (`pandera`, Part F)
| Field | Type | Constraints |
|---|---|---|
| `customer_id` | int | non-null, **unique** |
| `email` | str | non-null |
| `country_norm` | str | non-null |
| `spend_usd` | float | non-null, `>= 0` |
| `is_marketing_opt_in` | bool | non-null |

Output artifact: `artifacts/customers_clean.parquet` — 1469 rows × 10 columns.

---

## Stretch dataset (Part E) — synthetic order lines
Separate small frame generated in-notebook (`default_rng(7)`): `order_id`, `product_id`,
`quantity`, `unit_price`, with duplicate `(order_id, product_id)` pairs injected. **450 rows →
322** after keeping the max extended-price line per composite key.
