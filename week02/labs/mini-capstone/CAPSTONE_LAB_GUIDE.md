# Week 2 Mini-Capstone Lab Guide
## Building the Cordwell Support Assistant Fine-Tuning Pipeline

This is the capstone of Weeks 1 and 2. Everything you have practiced (pandas, NumPy,
requests, SQL extraction, cleaning, joining, validation, and profiling) comes
together into one application that produces a real deliverable: an LLM-ready
fine-tuning dataset, built from three messy sources and proven clean by its own
tests and quality gates.

You are not training or calling a model. You are building the data pipeline that
decides whether such a model could ever be good. In practice this is where AI
engineers spend most of their time, and it is where fine-tuning projects quietly
succeed or fail.

Time budget: 5 to 6 hours, in teams of three or four. Work through the tasks in
order. The test suite is your specification: a task is done when its tests are green.

## The mission

Cordwell Home and Hardware wants an assistant that drafts a first-pass reply for a
human support agent to review and send. To fine-tune it, you need a clean dataset of
real customer messages paired with good agent replies, in the format the fine-tuning
endpoint accepts. Your pipeline extracts that data from a support database, two REST
APIs, and two legacy files; transforms and cleans it; validates and profiles it; and
writes it out as `messages`-format JSONL with a train and validation split.

## Setup (once, together)

```
# From the cordwell_support_capstone folder
# Include 3.13 on the end of python/pip calls if needed
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-capstone.txt
pip install -e .
python scripts/generate_sources.py
python -m pytest -q          # 8 pass, 42 fail. The 42 are your work.
```

If those 8 pass and 42 fail cleanly (failures say `NotImplementedError`, not import
errors), your environment is good.

## How to work

Run the whole suite to see the landscape, then narrow to the file you are on:

```
python -m pytest -q tests/test_extract_api.py
```

Every function you implement has a contract docstring that tells you what "done" looks
like. The amount of help in those docstrings drops as the lab goes on, on purpose. Task 1
walks you through the mechanics. By Task 8 you get the goal and the constraints and are
trusted to work out the how, because by then you have seen the pattern four times.

When you get stuck, in this order:

1. **Read the failing test.** The tests are the specification. They are usually more
   precise about what is wanted than any prose.
2. **Read `HINTS.md`.** Each task has three levels: a nudge, an approach, and something
   close to the code. Spend one level, go back to the editor, and see if it moves.
3. **Ask a teammate, then the instructor.**

Being stuck is productive for about ten minutes and corrosive after forty. Take the hint.
Nobody is scoring you on how few you used.

When several tasks are independent, split them across the team and integrate at Task 8.

A theme runs through the tasks: validation is not one step at the end, it is a contract
enforced at every boundary the data crosses. You author a Pandera schema for the raw
tickets frame (Task 1), two Pydantic record models for the API data (Task 2), and wire
the given final-gate schemas over the assembled examples (Task 6). Notice how the
posture differs: at ingestion you drop bad records and count them, at the final gate you
refuse to proceed.

### Suggested team roles

- Extractor: Tasks 1 to 3 (SQL and its boundary schema, APIs and their record models,
  legacy files).
- Transformer: Tasks 4 and 5 (selection, imputation, balancing, PII, assembly).
- Quality and Load lead: Tasks 6 and 7 (tokens, final-gate validation, profiling,
  writing), and pairs with the Extractor on the boundary contracts in Tasks 1 and 2.
- Integrator: Task 0 data dictionary up front, then Task 8, and the three TODO tests
  throughout.

Everyone reviews the integration. The pipeline order in Task 8 is a team decision and
the most important design conversation of the day.

## Tasks

### Task 0: Understand the data (about 30 minutes)

Fill in `DATA_DICTIONARY_TEMPLATE.md`. Inspect the SQLite tables, hit the mock API,
and open both legacy files. You are answering: what is usable, what is missing, what
is malformed, and what is a distractor you should ignore. Do not skip this. Every
later decision depends on it.

Acceptance: the template is filled and the team agrees on which resolutions are
usable, which category dominates, and which table is a distractor.

### Task 1: Extract from SQL, and guard the boundary (about 45 minutes)

Implement `extract/sql.py :: read_tickets_chunked`. Stream the usable tickets from
the database in chunks and return one DataFrame. The `_TICKETS_SQL` query and the
engine are given; your job is the chunked read and the safe parameter binding.

Then author your first contract: `quality/schema.py :: tickets_schema` and
`quality/validate.py :: validate_tickets`. This is a Pandera schema over the raw
tickets frame, run the moment it leaves SQL, that proves the extraction contract
held. Model it on the given `EXAMPLE_TABLE_SCHEMA`. Think carefully about what must
be true at this boundary versus later: resolution must already be a usable value and
the reply must be non-empty, but category may still be missing here, so do not
constrain it to the taxonomy yet.

Acceptance: `tests/test_extract_sql.py` and the `tickets_schema` tests in
`tests/test_quality.py` are green. Notice the SQL injection test: it proves why the
resolution list is bound as a parameter, not pasted into the SQL.

### Task 2: Extract from the APIs, and validate each record (about 75 minutes)

Implement `extract/api.py`: `build_session`, `fetch_reviews`, `fetch_spec`,
`fetch_specs`. The reviews endpoint is paginated and injects transient 429 and 500
responses; your Retry policy must absorb them so the pagination loop only sees the
eventual 200s. The specs endpoint returns malformed JSON about a tenth of the time;
`fetch_spec` must return None on every failure mode and never raise.

Alongside this, author two Pydantic contracts and validate records at ingestion:

- `quality/schema.py :: SpecRecord` and its `material_not_blank` validator.
  `fetch_spec` validates each spec with `SpecRecord.model_validate` and treats a
  validation failure exactly like a JSON parse failure: return None. Some specs are
  valid JSON but have a blank or missing material, and this is what catches them.
- `quality/schema.py :: ReviewRecord` and its `rating_in_range` validator, plus
  `quality/validate.py :: validate_reviews`, which drops invalid reviews (an
  out-of-range rating, a missing field) and counts the drops. Some reviews are valid
  JSON but violate the contract, and if they reach the rating aggregation they corrupt
  the grounding facts.

Use the given `ChatMessage.content_not_blank` as the pattern for a `@field_validator`.

Acceptance: `tests/test_extract_api.py` and the `SpecRecord`, `ReviewRecord`, and
`validate_reviews` tests in `tests/test_quality.py` are green, including the test that
a fresh session survives the rate-limited and 5xx pages.

### Task 3: Extract from the legacy files (about 30 minutes)

Implement `extract/files.py :: load_legacy_chats`. The CSV and JSON loaders are
given; you unify them, resolve whitespace-only text to missing, drop exact-duplicate
pairs and blank rows, and record how many rows you dropped.

Acceptance: `tests/test_extract_files.py` is green, including the test proving the
CSV is not valid utf-8.

### Task 4: Select and balance (about 75 minutes)

Implement the four functions in `transform/select.py`: `normalize_sources`,
`impute_category`, `select_candidates`, `balance_categories`. This is the pandas core
of the pipeline and the densest task. Two things to get right:

- Missing categories are a policy, not an accident. Impute from keywords when you can
  be confident, exclude when you cannot, and count both so the loss is visible.
- Every filter logs how many rows it removed. When your final count surprises you,
  that log is how you find the culprit.

Acceptance: `tests/test_transform_select.py` is green. The filter log must be
monotonic (each step's remaining count never goes up) and balancing must be
deterministic for a fixed seed.

### Task 5: Scrub PII and assemble (about 45 minutes)

Implement `transform/pii.py` (`scrub_pii`, `scrub_series`, `contains_pii`) and
`transform/assemble.py` (`build_product_context`, `assemble_examples`). The PII
patterns are given; you apply them. `build_product_context` is the payoff for the API
extraction: it joins products, specs, and reviews into a per-product context string
that grounds product questions. `assemble_examples` builds the three-message array.

Acceptance: `tests/test_transform_pii.py` is green (including the Hypothesis property
test that no order id survives scrubbing).

### Task 6: Tokenize and gate the finished dataset (about 60 minutes)

Implement `quality/tokens.py` (`count_tokens`, `gate_by_tokens`),
`quality/validate.py :: validate_examples`, and `quality/profile.py`
(`profile_dataset`). You already authored the ingestion contracts in Tasks 1 and 2;
this is the final gate on the assembled examples. Character length is not token
length; the tokenizer is how you enforce the real budget. `validate_examples` runs
the given final-gate schemas (the `EXAMPLE_TABLE_SCHEMA` Pandera schema and the
`ChatExample` Pydantic model) over the assembled examples. Profiling computes the
metrics and turns them into hard pass or fail gates. Unlike the ingestion contracts,
which drop bad records and count them, this gate raises: our own pipeline should never
produce a bad example.

Acceptance: the tokenizer and `validate_examples` tests in `tests/test_quality.py`
are green.

### Task 7: Split, write, verify (about 30 minutes)

Implement `load/writer.py`: `stratified_split`, `write_jsonl`, `verify_roundtrip`,
`split_and_write`. Stratify the split by category so both shards see every category.
Write one JSON object per line. Then reload with Hugging Face `datasets` to prove the
file parses. This lab has no network access and `datasets` will reach for the hub
unless you prevent it; getting that right is part of the task.

Acceptance: the writer tests in `tests/test_load_and_pipeline.py` are green.

### Task 8: Integrate (about 30 minutes)

Implement `pipeline.py :: run_pipeline`. Call your stage functions in the right order
and thread the data through. Wire in the boundary validators you authored: the tickets
schema guards the raw frame coming out of SQL, and the review contract guards the
records coming out of the API.

Most of the order falls out of the data dependencies. One pair of steps does not. Both
orderings will run without error, and both will produce a dataset, but only one of them
is correct, and the gap between them is thousands of examples. Before you commit, read
what each check in `profile.py` measures, and ask which of your steps changes the text
that a later step compares. If you get it wrong a quality gate will fail. That is the
gate working. Find out why it fired rather than lowering the threshold.

Acceptance: `python scripts/run_pipeline.py` runs end to end, prints a green profile,
and the end-to-end test in `tests/test_load_and_pipeline.py` passes. At this point the
full suite (minus your Task T tests) should be green.

### Task T: Write your own tests (throughout)

Fill in the three tests in `tests/test_todo.py` as you reach the relevant code. Good
engineers write the test that would have caught the bug they just fixed.

## Definition of done

- `python -m pytest -q` is fully green, including your three Task T tests.
- `python scripts/run_pipeline.py` writes `data/output/` with train and validation
  JSONL, a tokenizer, and a passing profile report.
- Your data dictionary is complete.
- Your team can explain the pipeline order, including which pair of steps was order
  sensitive, how you discovered it, and what the profile report showed when you had it
  backwards.
- Your team can explain why the ingestion contracts drop and count bad records while
  the final gate raises, and which library (Pandera or Pydantic) you reached for at
  each boundary and why.

## Stretch goals (for fast finishers)

1. Harden the PII patterns. The name redaction only catches "My name is First Last".
   Real names are not that polite. Add a case for a signature line, and write a test.
   Then explain in one paragraph why regex alone is not enough and what a production
   system would use instead.
2. Make imputation auditable. Instead of silently imputing, attach the matched keyword
   to each imputed row and profile which keywords drove the most imputations. Are any
   of them risky?
3. Add a second validation split by source, so you can measure how the model would do
   on legacy-chat-style inputs specifically. Does the source mix in your training set
   match what production will actually see?
4. The balancing step throws away thousands of clean order-status examples. Is that
   the right call? Implement an alternative (for example, capping at a higher target
   or weighting instead of discarding) and argue for one over the other with numbers
   from your profile report.
5. Tighten a contract. Add a field to `SpecRecord` or `ReviewRecord` with its own
   validator (for example, `warranty_months` must be non-negative, or `created_at`
   must parse as a date), generate a record that violates it, and confirm your
   ingestion validation drops or rejects it. Then decide: should that failure drop the
   record (like reviews) or raise (like the final gate), and why?
