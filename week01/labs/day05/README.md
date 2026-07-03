# Cordwell Home & Hardware — Week 1 Capstone (Student)

A small, tested Python project that turns a messy raw orders CSV into a clean,
validated Parquet artifact. This is the **student starter**: the constants, the
pandera schema, and every function's signature + docstring are given; **your job
is to implement the function bodies in `src/cordwell/data/cleaning.py` until
`pytest` passes**, complete three `TODO` tests, and write the data dictionary.

Work in this order (each unlocks more passing tests): `normalize_label` ->
`profile_dataframe` -> `handle_missing` -> `drop_duplicates` -> `coerce_dtypes`
-> `normalize_strings` -> `handle_outliers` -> `validate` -> `write_parquet` ->
`run_cleaning_pipeline`. Run `pytest -q` often.

## Layout
```
cordwell_capstone/
├── src/cordwell/
│   ├── __init__.py
│   └── data/
│       ├── __init__.py          # re-exports the public cleaning API
│       └── cleaning.py          # all pipeline functions + pandera schema
├── tests/
│   ├── conftest.py              # shared fixtures (incl. the messy fixture)
│   └── data/
│       └── test_cleaning.py     # AAA unit + integration tests (25)
├── scripts/
│   ├── generate_raw_data.py     # make the messy raw CSV (>= 45k rows)
│   ├── profile_raw.py           # "profile first" — inspect before cleaning
│   └── run_pipeline.py          # production entry point (CSV -> Parquet)
├── data/
│   ├── raw/                     # generated raw CSV (never overwritten by the pipeline)
│   └── processed/               # clean Parquet + data dictionary
├── notebooks/                   # exploration only (optional)
├── requirements.txt
└── setup.py                     # minimal; enables `pip install -e .`
```

## Setup (plain pip — no uv / pyproject / Ruff)
```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .                                       # makes `cordwell` importable
```

## Run it
```bash
# 1) generate the messy raw data (seeded/reproducible)
python scripts/generate_raw_data.py                    # -> data/raw/cordwell_orders_raw.csv

# 2) profile it BEFORE cleaning
python scripts/profile_raw.py

# 3) clean -> validate -> write Parquet
python scripts/run_pipeline.py                         # -> data/processed/orders_clean.parquet

# 4) run the tests
pytest -q                    # then, once green:  pytest --cov=src
```

## Notes
- **pandera import (currency):** this project uses the modern `import pandera.pandas as pa`.
  The Day-5 deck shows `import pandera as pa`, which still works but now emits a
  `FutureWarning` in pandera ≥ 0.20.
- The pipeline must be a single composed function (`run_cleaning_pipeline`) — the
  same code runs in tests and in `run_pipeline.py`. No logic lives in scripts.
- Raw data in `data/raw/` is never overwritten by the pipeline; outputs go to
  `data/processed/`.
