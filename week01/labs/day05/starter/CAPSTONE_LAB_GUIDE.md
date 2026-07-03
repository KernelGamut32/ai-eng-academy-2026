# Week 1 Capstone — Lab Guide
## Writing Maintainable Code & Dataset Cleaning

> **AI Engineering Academy** · Gamut Technology Services
> **Time:** ~3–4 hours · **You will produce:** a tested, importable cleaning
> package that turns a messy raw CSV into a **clean, validated, documented**
> Parquet artifact.

This capstone synthesizes the whole week: **modules & packages** (src/ layout,
imports), **docstrings** (contracts), **pytest** (fixtures, parametrize, coverage,
Hypothesis), and the **10-step dataset-cleaning workflow** (profile → clean →
validate → Parquet + data dictionary).

**The tests are the spec.** You're done when `pytest` is green, the pipeline runs
on the full dataset, the data dictionary is written, and the three `TODO` tests
are implemented.

### What's given vs. what you build
| Given (don't rewrite) | You build |
|---|---|
| Project structure, `setup.py`, `requirements.txt` | The body of every function in `cleaning.py` |
| Alias maps, outlier bounds, `CLEAN_SCHEMA`, `SCHEMA_COLUMNS` | Three `TODO` tests in `test_cleaning.py` |
| `scripts/` (generate, profile, run) | The data dictionary (`orders_clean_dict.md`) |
| The acceptance test suite + messy fixture | |

---

## Task 0 — Set up (10 min)
```bash
# from cordwell_capstone
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .            # editable install: makes `import cordwell` work anywhere
```
**Acceptance:** `python -c "import cordwell; print('ok')"` prints `ok`.

> Why editable install? The **src/ layout** keeps importable code in `src/cordwell/`.
> `pip install -e .` puts that package on your path so `from cordwell.data.cleaning
> import ...` resolves the same in tests, scripts, and (later) production. No `sys.path`
> hacks.

---

## Task 1 — Generate and PROFILE the data (15 min)
Never clean data you haven't profiled.
```bash
# from deliverable
python scripts/generate_raw_data.py      # -> data/raw/cordwell_orders_raw.csv (~45,675 rows)
python scripts/profile_raw.py            # read this carefully before writing any code
```
Read the profile output and answer for yourself:
- Which columns have nulls, and roughly what fraction?
- Which columns hold **alias variants** that need canonicalizing? (look at the
  `store_region` and `channel` value counts)
- Which numeric columns show **outliers**?

**Acceptance:** you can name, from the profile alone, at least one column for each
cleaning strategy: *drop*, *median-impute*, *sentinel-fill*, *canonicalize*, *cap*.

---

## Task 2 — Implement `cleaning.py` (110–140 min) — the core
Open `src/cordwell/data/cleaning.py`. Implement each function's body (replace
`raise NotImplementedError`). **Work top-to-bottom** and run the tests as
you go:

```bash
pytest -q                                                     # the whole suite
```

Recommended order (each unlocks more passing tests):

1. **`normalize_label(value)`** — strip, collapse internal whitespace, lower-case;
   `None`/`NaN`/`pd.NA` → `""`. Must be *total* (never raises) and *idempotent*.
   *Hint:* `" ".join(str(value).split()).lower()`.
2. **`profile_dataframe(df)`** — return `{"n_rows","n_cols","dtypes","null_ratio"}`.
   *Hint:* `df.isnull().mean()` gives null fractions.
3. **`handle_missing(df)`** — drop null `order_id` rows; median-impute `quantity` &
   `unit_price`; `discount_pct` → `0.0`; `store_region` & `channel` → `"Unknown"`.
   Return a **copy**.
4. **`drop_duplicates(df, key)`** — sort by `updated_at` descending, keep first per
   `key`. Latest record wins.
5. **`coerce_dtypes(df)`** — `order_date` → UTC datetime with an explicit
   `format="%Y-%m-%d"`; `store_id`/`quantity` → `int64`; prices → `float64`; the six
   text columns → `"string"`.
6. **`normalize_strings(df)`** — `product_name` via `normalize_label`;
   `product_category`, `store_region`, `channel` via their alias maps
   (`ALIASES.get(normalize_label(v), v)`). Re-cast to `"string"`.
7. **`handle_outliers(df)`** — compute the `quantity_capped` / `unit_price_capped`
   boolean flags **before** clipping, then `clip(lower=, upper=)` each numeric column.
8. **`validate(df)`** — one line: `return CLEAN_SCHEMA.validate(df)`.
9. **`write_parquet(df, path)`** — make parent dirs, `to_parquet(index=False)`, return
   the `Path`.
10. **`run_cleaning_pipeline(raw_path, output_path)`** — compose in order:
    read → `handle_missing` → `drop_duplicates` → `coerce_dtypes` →
    `normalize_strings` → `handle_outliers` → select `SCHEMA_COLUMNS` → `validate` →
    `write_parquet`; return the clean frame.

**Acceptance:** `pytest -q` shows **all provided tests passing** (only the 3 `TODO`
tests remain skipped).

> **Docstrings are contracts.** Each function already has a Google-style docstring
> describing its promise. If you change behavior, update the docstring in the same
> edit — an out-of-date docstring is worse than none (deck slide 13).

---

## Task 3 — Run the pipeline on the full dataset (10 min)
```bash
python scripts/run_pipeline.py
```
**Acceptance:** it logs the row counts (≈45,675 raw → ≈44,775 clean), writes
`data/processed/orders_clean.parquet`, and raises **no** `SchemaError`. Load it back
and eyeball it:
```python
import pandas as pd; df = pd.read_parquet("data/processed/orders_clean.parquet")
print(df.shape); print(df.dtypes); print(df["store_region"].value_counts())
```

---

## Task 4 — Write the data dictionary (20 min)
Fill in `data/processed/orders_clean_dict_TEMPLATE.md` (rename it to
`orders_clean_dict.md`). For every clean column give the **type**, **nullability**,
**meaning**, and any **cleaning caveat** (imputed? sentinel? capped? canonicalized?).
Explain *meaning*, not just type — write it for "you in six weeks" (deck slide 36).

**Acceptance:** every column has a real description and every cleaning decision from
Task 2 is recorded somewhere in the table.

---

## Task 5 — Write the three `TODO` tests (25 min)
In `tests/data/test_cleaning.py`, implement the three tests currently marked
`pytest.skip(...)` (remove the skip line once written):
1. **`test_normalize_label_cases`** — a `@pytest.mark.parametrize` case table (deck slide 18).
2. **`test_normalize_label_is_idempotent`** — a Hypothesis property (`@given(st.text())`), deck slide 21.
3. **`test_drop_duplicates_keeps_latest`** — an edge-case test proving the latest row survives.

**Acceptance:** `pytest -q` shows **0 skipped** and everything passing.

---

## Task 6 — Coverage check (5 min)
```bash
pytest --cov=cordwell --cov-report=html
```
**Acceptance:** the suite passes and coverage on `cleaning.py` is at/near 100%.
Remember coverage is a *signal*, not a target (deck slide 20 & 23) — the point is
that your tests exercise every behavior, not that a number is green.

---

## Deliverables checklist (deck slide 44, adapted)
- [ ] `src/cordwell/data/cleaning.py` — all functions implemented, docstrings accurate.
- [ ] `tests/data/test_cleaning.py` — all tests passing, including your three.
- [ ] `data/processed/orders_clean.parquet` — validated output on the full dataset.
- [ ] `data/processed/orders_clean_dict.md` — the data dictionary.
- [ ] `pytest --cov=src` passes with no failures.

## Stretch goals (for fast finishers)
- **Tighten the schema:** add a `Check` that `unit_price >= 0` *without* relying on the
  cap, or enforce `product_sku` matches `^SKU-\d{5}$` with `Check.str_matches`.
- **Record-level validation:** add a Pydantic `OrderRecord` model (deck slide 34) and a
  test that validates a few `df.to_dict("records")` rows through it.
- **A `line_total` feature:** add `quantity * unit_price * (1 - discount_pct/100)` as a
  new clean column (update the schema, the dictionary, and add a test).
- **Profndex parametrize:** parametrize `test_handle_outliers_caps_and_flags` across
  several boundary values (0, 1, 500, 501).
```
