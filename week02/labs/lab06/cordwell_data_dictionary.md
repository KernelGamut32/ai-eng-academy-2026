# Cordwell Home & Hardware — Synthetic Data Dictionary
### Week 2 · Lab 06 — Clean & Standardize · Join & Aggregate

> **All data is synthetic and clearly fictional**, generated in the notebook's Setup
> cell (`build_cordwell_raw()`, `np.random.default_rng(2025)`). "Cordwell Home &
> Hardware" is invented; nothing here reflects any real company, person, or system.
> Emails use the reserved `.example` domain.

Unlike Lab 05's already-clean tables, this lab ships **deliberately dirty** data —
the mess *is* the lesson. Cordwell sells to trade accounts across a few countries, and
the raw profile export is inconsistent in exactly the ways real exports are.

---

## 1. `customers_raw` — messy customer profiles (1,000 rows)

| Column | Raw dtype | The mess | Cleaned to |
|---|---|---|---|
| `customer_id` | `str` | clean key, `C00001`–`C01000`, unique | (key) |
| `email` | `str` | ~2% `None` | required — rows dropped in A1 |
| `age` | `float64` | 16–80 | → `is_adult` (≥18) |
| `country` | `str` | 13 spellings: `US`, `U.S.A.`, `usa`, `United States`, `SG`, `sg`, `DE`, `Germany`, `Brasil`, `BR`, `Deutschland`, `Brazil`, `N/A` | → `country_norm` via reference dimension |
| `signup_date` | `str` | **4 formats** + `None`: `2025-01-05`, `01/06/2025`, `2025/01/07`, `06-01-2025` | → `signup_dt` (`datetime64[us]`) |
| `lifetime_value` | `str` | **6 currency conventions** + blanks/nulls | → `ltv_usd` (`float64`) |

### 1.1 The currency encodings (all represent real USD amounts)

| Style | Example | Convention |
|---|---|---|
| US dollar | `$1,234.50` | comma = thousands, dot = decimal |
| Euro | `€1.234,50` | dot = thousands, comma = decimal |
| Brazilian real | `R$ 45,00` | comma = decimal |
| USD code | `USD 99.95` | prefix code |
| Plain thousands | `1,234` | comma = thousands, **cents dropped** (lossy by ≤ \$0.50) |
| Plain | `99.95` | bare number |

> **The decimal-separator trap:** `"45,00"` = 45 (comma is the decimal), but `"1,234"`
> = 1234 (comma is thousands). The parser disambiguates on the digit count after the
> comma: exactly 1–2 digits ⇒ decimal; 3 ⇒ thousands.

### 1.2 The date trap (pandas 3.0)

`signup_date` mixes ISO (`2025-01-05`), US-slash (`01/06/2025`), slash-ISO
(`2025/01/07`), and dash-US (`06-01-2025`). See §4.

---

## 2. `orders` — order facts (3,000 rows)

| Column | dtype | Notes |
|---|---|---|
| `order_id` | `int64` | `50001`–`53000`, unique |
| `customer_id` | `str` | FK into `customers_raw`. Every value exists in the *raw* table — but **cleaning filters orphan ~7.8%** of orders (their customer was dropped for a bad email/date). |
| `order_date` | `str` → `datetime64[us]` | clean ISO (`2025-01-06…09`) |
| `freight` | `float64` | lognormal shipping cost |

## 3. `country_dim` — the reference dimension (14 rows)

Maps raw spellings → canonical codes (`USA`, `BR`, `DE`, `SG`, and `N/A → UNKNOWN`).
**`Deutschland`, `Brazil`** appear in the data but **not** in the dimension — they fall
through to `UNKNOWN`, which is what makes the A2 coverage story real (146 rows land in
`UNKNOWN`).

---

## 4. ⚠️ Currency flags (verified against the cohort stack)

- **`pd.to_datetime(mixed_formats, errors="coerce")` silently mass-coerces to `NaT`.**
  In pandas 3.0 the parser infers **one** format from the first non-null value and
  coerces everything that doesn't match — **no warning**. On this data the naive call
  yields **776 `NaT`**; `format="mixed"` yields **55** (the true `None`s only). This is
  the lab's headline lesson. *(A4.)*
- **`datetime64[us]`** is the pandas 3.0 default resolution (not `[ns]`). *(A4.)*
- **`str` dtype**, not `object`, for text columns. *(Setup.)*
- **`pd.qcut` on a lumpy distribution** raises `ValueError: Bin edges must be unique`
  without `duplicates="drop"`. *(Stretch S1.)*
- **Anti-join via `indicator=True`**, not `.isna()`-hunting. *(B2.)*

---

## 5. Expected checkpoints (seed 2025 — verify against these)

| Stage | Value |
|---|---|
| `customers_raw` · `orders` | (1000, 6) · (3000, 4) |
| A1 rows after required-drop | 978 (22 dropped) |
| A2 `country_norm`: USA / DE / SG / BR / UNKNOWN | 421 / 155 / 135 / 121 / **146** |
| A3 missing `ltv_usd` after imputation | 0 |
| **A4 naive-parse `NaT` vs `format="mixed"`** | **776 vs 55** |
| A4 rows after date filter (`users2`) | **923** |
| A4 90th-pct LTV · adults · high-value | \$1343.13 · 895 · **93 (10.1%)** |
| B1 customers dim · joined | 923 · **2765** |
| B2 orphaned orders · anti-rate | 235 · **7.83%** |
| B3 per-segment rows · order total | 15 · 2765 |
| B4 per-customer rows · order total | 873 · 2765 |
| S1 decile bands | 10 |

Segment precedence: `high_value` > `adult` > `general` (via `np.select`).
