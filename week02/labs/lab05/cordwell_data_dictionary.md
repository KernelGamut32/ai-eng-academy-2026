# Cordwell Home & Hardware — Synthetic Data Dictionary
### Week 2 · Lab 05 — GroupBy & Joins

> **All data in this lab is synthetic and clearly fictional.** It is generated
> programmatically in the notebook's Setup cell (`build_cordwell()`,
> `np.random.default_rng(2025)`), not loaded from any file. "Cordwell Home &
> Hardware" is an invented retailer; nothing here reflects any real company,
> customer, or system.

The lab models a **B2B account** world: Cordwell sells to trade accounts
(contractors, property managers) who place many orders over time. That gives us the
two shapes every join lesson needs — a **fact** table (`orders`) and a **dimension**
table (`customers`) — and the data is deliberately imperfect so the join lessons are
real, not hypothetical.

---

## 1. Generation parameters

| Parameter | Value | Purpose |
|---|---|---|
| RNG | `np.random.default_rng(2025)` | Modern NumPy 2.x Generator; fixes the seed so every learner gets identical rows. |
| `customers` rows | 40 | One row per `customer_id` (the dimension). |
| `orders` rows | 200 | Many orders per customer (the fact). |
| Orphan rate | ~10% | Orders whose `customer_id` is **not** in `customers` → anti-join lesson. |
| Order-less customers | 5 | Customers who never order (only the first 35 of 40 place orders) → outer-join `right_only` lesson. |

---

## 2. `customers` (dimension — one row per `customer_id`)

| Column | dtype (pandas 3.0) | Description |
|---|---|---|
| `customer_id` | `str` | Business key, `C0001`–`C0040`. **Unique.** |
| `company_name` | `str` | Account name, `Cordwell Acct 0001` … |
| `region` | `str` | The account's **home** region: Southeast / Northeast / Midwest / West. |

## 3. `orders` (fact — many rows per `customer_id`)

| Column | dtype (pandas 3.0) | Description | Deliberate wrinkle |
|---|---|---|---|
| `order_id` | `int64` | Order key, `10001`–`10200`. Unique. | — |
| `customer_id` | `str` | FK to `customers`. | ~10% reference an **unknown** id (`C9xxx`) not in `customers`. |
| `ship_region` | `str` | Region this order **shipped to**. | Same *concept* as `customers.region` but a different value → the join collision in B4. |
| `order_total` | `float64` | Order value in USD (gamma-distributed). | — |
| `freight` | `float64` | Freight/shipping cost in USD (gamma-distributed). | — |

> **pandas 3.0 note.** String columns land as the new **`str`** dtype, not `object`.
> `select_dtypes(include=["object"])` will **not** return them — a common
> stale-tutorial trap carried over from Lab 04.

### 3.1 Why two "region" columns?

`customers.region` (home) and `orders.ship_region` (ship-to) are genuinely different
facts — a Southeast account can have an order shipped West. In **B4** we rename
`ship_region → region` to simulate an export where *both* tables call it `region`,
and watch pandas auto-suffix them to `region_x` / `region_y`. The fix is explicit
`suffixes=("_ship", "_home")`.

---

## 4. Expected checkpoints (seed 2025 — verify against these)

| Quantity | Value |
|---|---|
| `orders` shape · `customers` shape | (200, 5) · (40, 3) |
| Total freight (A1) | **5190.84** |
| Region×customer combinations (A2) | 121 |
| `groupby.size()` total vs `count()` of all-null col (A3) | 200 vs **0** |
| Join lengths (B1): inner / left / outer | **179 / 200 / 205** |
| Outer breakdown | 179 both · 21 left_only · 5 right_only |
| Orphan orders / distinct unknown ids (B2) | **21 / 12** |
| Fan-out (B3): clean vs duplicated | 179 rows / \$4685.15 → **184 rows / \$4781.46** |
| Per-customer rows incl. unknowns (C1) | 47 |
| Segment join (C2): inner / left / NaN attrs | 35 / 47 / 12 |
| Spend segments low / mid / high (C2) | **7 / 10 / 18** |
| Region rollup (C3): rows · customers · orders | 4 · 35 · 179 |
| Scaled parquet (D): partitions · rows | 4 · 5000 |

*(Part D uses a second, self-contained frame — `np.random.default_rng(7)`, 5,000
rows — written to `artifacts/orders_big/ship_region=*/`. It does not depend on any
other lab's artifacts.)*

**Segment bins** (`pd.cut`, `right=False`): `low` = [0, 600), `mid` = [600, 1100),
`high` = [1100, ∞).
