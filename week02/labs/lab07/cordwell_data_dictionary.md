# Cordwell Feature Table — Data Contract & Drift Catalog
### Week 2 · Lab 07 — Why Schema Validation in ML/LLM Pipelines

> **All data is synthetic and clearly fictional**, generated in the notebook's Setup
> cell (`build_cordwell_features()`, `np.random.default_rng(42)`, 300 rows). This is the
> *validated* shape a pipeline like Lab 06 would emit — the lab is about **keeping** it
> that shape as upstream data drifts.

---

## 1. The clean feature table (the contract)

| Column | pandas 3.0 dtype | Rule | pandera declaration |
|---|---|---|---|
| `customer_id` | `str` | non-null, **unique** | `pa.Column(str, nullable=False, unique=True)` |
| `country_norm` | `str` | in `{USA, DE, SG, BR}` | `pa.Column(str, pa.Check.isin(allowed), nullable=False)` |
| `age` | `int64` | integer in `[0, 120]` | `pa.Column(pa.Int64, pa.Check.in_range(0,120), nullable=False)` |
| `ltv_usd` | `float64` | `≥ 0` | `pa.Column(float, pa.Check.ge(0), nullable=False)` |
| `is_adult` | `bool` | non-null | `pa.Column(bool, nullable=False)` |
| `is_high_value` | `bool` | non-null | `pa.Column(bool, nullable=False)` |

Data is internally coherent: `is_adult = age >= 18`, `is_high_value = ltv_usd ≥ 85th
percentile`.

> ⚠️ **The dtype trap (pandas 3.0).** `customer_id` and `country_norm` are **`str`**
> dtype, **not `object`**. A schema that declares them `pa.Column(object, …)` — the
> spelling in every pre-3.0 tutorial — **rejects this clean table**
> (`expected type object, got str`). Declare string columns `pa.Column(str, …)`.

---

## 2. The drift catalog (Part A plants these on a `broken` copy)

| # | Drift | Kind | How it's planted | Schema check that catches it |
|---|---|---|---|---|
| 1 | `age` arrives as text | **structural** | `broken["age"] = broken["age"].astype(str)` (whole column) | `age` dtype (`Int64`) |
| 2 | out-of-policy country labels | **semantic** | rows 10–13 → `["U.S.A.","United States","usa","US"]` | `country_norm` `isin` |
| 3 | negative lifetime value | **semantic (range)** | rows 50–52 → `[-10, -5, -1]` | `ltv_usd` `ge(0)` |

> **pandas 3.0 note:** the source lab's structural drift spliced strings into *part* of
> the `int64` column (`broken.loc[:20,'age'] = ...astype(str)`). That **raises
> `TypeError`** on pandas 3.0 (no more silent upcast to `object`). The whole-column
> flip above is the reliable way to model a type drift.

---

## 3. Expected validation results (seed 42 — verify against these)

| Stage | Value |
|---|---|
| clean table shape | (300, 6) |
| `Schema.validate(clean, lazy=True)` | 300 rows, passes |
| out-of-policy country rows (A2) | 4 |
| negative ltv rows (A3) | 3 |
| **`Schema.validate(broken)` → `SchemaErrors.failure_cases`** | **9 rows** |
| failure roll-up (`groupby(['column','check'])`) | **4 groups** |
| — `country_norm` / `isin` | 4 |
| — `ltv_usd` / `ge(0)` | 3 |
| — `age` / `dtype('int64')` | 1 |
| — `age` / `in_range(0,120)` | 1 |
| columns flagged | `age, country_norm, ltv_usd` |
| pydantic clean row | validates |
| pydantic row 50 (negative ltv) | `ValidationError` |
| gate: clean batch | returns 300 rows |
| gate: broken batch | raises `RuntimeError` + writes CSV |

> **Why `age` shows two failures:** once the column's *type* is wrong, the dtype check
> fails **and** the value check (`in_range`) can't pass either — a small, honest cascade
> worth pointing out. That's why the total is 9, not 7.

---

## 4. `failure_cases` structure (pandera 0.32)

`SchemaErrors.failure_cases` is a DataFrame with columns:
`schema_context · column · check · check_number · failure_case · index`.
The `index` column points back to the offending row in the validated frame — the hook
for triage.

---

## 5. Tooling notes (verified against the cohort stack)

- **`import pandera.pandas as pa`** is the modern, warning-free import. Use
  `pa.Column`, `pa.Check`, `pa.DataFrameSchema`, `pa.Int64`, `pa.errors.SchemaErrors`.
- **`lazy=True`** collects *every* violation into one report; without it, validation
  stops at the first failure.
- **`.to_dict()` returns native Python scalars** on pandas 3.0 (not numpy), so pydantic
  models accept row dicts directly — no `.item()` juggling.
- **pydantic v2 coerces in lax mode** — `"63"` → `63`. To make a row model *catch* a
  type drift, enable strict mode (`ConfigDict(strict=True)` / `Field(strict=True)`).
