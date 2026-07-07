# Cordwell Feature Table — Schema Reference & Check Catalog
### Week 2 · Lab 08 — Authoring Schemas & CI Hooks (Pandera + Pydantic)

> **All data is synthetic and clearly fictional**, generated in the notebook's Setup
> cell (`build_cordwell()`, `np.random.default_rng(1)`, 500 rows). This is the *how*
> lab — the data is a fixed, clean target so the focus stays on **authoring** the
> contract, not cleaning.

---

## 1. The feature table (500 rows × 7 columns)

| Column | pandas 3.0 dtype | Rule | pandera | pydantic |
|---|---|---|---|---|
| `customer_id` | `str` | `^C\d{5}$`, **unique** | `Column(str, Check.str_matches(ID_RE), unique=True)` | `Field(pattern=ID_RE)` |
| `country_norm` | `str` | in `{USA,DE,SG,BR}` | `Column(str, Check.isin(ALLOWED))` | `Literal["USA","DE","SG","BR"]` |
| `age` | `int64` | int in `[0,120]` | `Column(pa.Int64, Check.in_range(0,120))` | `Field(ge=0, le=120)` |
| `ltv_usd` | `float64` | `≥ 0` | `Column(float, Check.ge(0))` | `Field(ge=0)` |
| `email` | `str` | matches `EMAIL_RE` | `Column(str, Check.str_matches(EMAIL_RE))` | `Field(pattern=EMAIL_RE)` |
| `is_adult` | `bool` | `age ≥ 18 ⇒ True` | frame-level `Check` | `@field_validator` |
| `is_high_value` | `bool` | non-null | `Column(bool)` | `bool` |

Regexes: `ID_RE = r"^C\d{5}$"` · `EMAIL_RE = r"^[^@\s]+@[^@\s]+\.[^@\s]+$"`.
Data is coherent: `is_adult = age ≥ 18` (15 minors), `is_high_value = top-decile LTV`.

---

## 2. Two levels of pandera check

**Column-level** (pinpoint the failing *row*): `str_matches`, `isin`, `in_range`, `ge`.
**Frame-level** (assert an *invariant*), passed in `checks=[...]` at construction:
- **Cross-column:** `age ≥ 18 ⇒ is_adult` — `pa.Check(lambda df: bool(((~(df["age"]>=18)) | df["is_adult"]).all()), error=...)`
- **Sanity band:** `median(ltv_usd) ≤ 100_000`.

> ⚠️ Return a **single bool** from a frame-level check. A returned **Series** makes
> pandera broadcast the failure across **every column** (one bad row → 7 failure cases),
> which clutters triage.

---

## 3. Currency fixes over the source lab (verified against the cohort stack)

| Source spelling | Problem on pandas 3.0 / pandera 0.32 | Corrected |
|---|---|---|
| `import pandera as pa` + `from pandera import Column, Check` | legacy entry point | `import pandera.pandas as pa`; use `pa.Column`, `pa.Check` |
| `Column(object, ...)` for strings | **rejects clean data** (`str ≠ object`) | `pa.Column(str, ...)` |
| `schema.update_checks({...})` / `schema.add_checks([...])` | **methods don't exist** → `AttributeError` | pass `checks=[...]` at construction |
| `Check(lambda s, df: ...)` (two-arg) | not a valid check signature | frame-level `pa.Check(lambda df: <bool>)` |
| `EmailStr` | needs `email-validator` (not installed) → ImportError | regex `Field(pattern=EMAIL_RE)`; note `pydantic[email]` |
| `raise SystemExit(...)` in the gate | kills the process, not just the step | `raise RuntimeError(...)` |
| `from lab3b_context import ...` | phantom module reference | self-contained `test_cordwell_schema.py` |

---

## 4. Expected results (seed 1 — verify against these)

| Stage | Value |
|---|---|
| table shape · minors | (500, 7) · 15 |
| A1 clean validates | 500 rows |
| A1 malformed id | rejected |
| A2 frame-level checks attached | 2 |
| A2 enriched still validates clean | 500 |
| A2 cross-column violation | caught (**1** clean failure case) |
| **A3 broken → `failure_cases`** | **8 rows** |
| A3 roll-up (worst-first) | email 3 · age 2 · country_norm 2 · customer_id 1 |
| B1 clean row | validates |
| B1 bad row (`email='nope'`, `age=200`) | rejected on **`age, email`** |
| B1 cross-field (`age=40, is_adult=False`) | `ValidationError` |
| B2 batch (2 planted bad rows) | 2 rejected |
| C1 gate: clean / broken | returns 500 / raises `RuntimeError` + writes CSV |
| C2 pytest smoke test | green (returncode 0) |

`failure_cases` columns (pandera 0.32): `schema_context · column · check · check_number
· failure_case · index`.

---

## 5. Tooling notes

- **`.to_dict()` returns native Python scalars** on pandas 3.0, so `Model(**row)` works
  without numpy unwrapping.
- **pydantic `@field_validator` sees earlier fields via `info.data`** — declare `age`
  before `is_adult` so the cross-field rule can read it.
- **pydantic is lax by default** — it coerces `"63"`→`63`. Use strict mode to catch a
  type drift per-row.
