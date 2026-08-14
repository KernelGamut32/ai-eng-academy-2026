# Week 7 Mini-Capstone: End-to-End RAG for Cordwell Home & Hardware

A 2 to 3 day team capstone. You will build, evaluate, and serve a retrieval augmented generation system over a realistic product knowledge base: containerized Pinecone Local and MLflow, TruLens evaluation, an experiment sweep with programmatic champion selection, and a FastAPI service that exposes the populated store to other teams.

Everything runs on your Mac with no cloud accounts and no API keys. The offline backend makes the whole pipeline work with no model server at all; LM Studio and Ollama plug in when you want live generation.

## The scenario

Cordwell Home & Hardware wants a support assistant grounded in its policies, product manuals, how-to guides, and operations documents. Your team owns the full path: ingest the document corpus into a vector store, answer questions with retrieved context, measure quality with an evaluation harness, run a configuration sweep tracked in MLflow, select a champion configuration with explicit gates and criteria, and serve both raw context chunks and full answers over HTTP.

The corpus is the real world in miniature: 31 documents that include everything document collections actually do to pipelines. Your ingest code has to survive it.

## Repository layout

```
docker-compose.yml         Pinecone Local + MLflow services
requirements.txt           Pinned dependencies (dedicated venv, see below)
.env.example               Configuration template
run.sh                     CLI wrapper: ./run.sh starter ingest
corpus/                    31 Cordwell documents + MANIFEST.md
data/eval/eval_set.jsonl   18 evaluation questions with references
starter/src/cordwell_rag/  Your working package (12 TODO tasks)
solution/src/cordwell_rag/ Instructor solution
tests/                     Shared pytest suite, runs against either package
artifacts/                 Sweep outputs and champion.json land here
HINTS.md                   Progressive hints, three levels per task
HINTS_DETAILED.md          Working-core hints with line commentary
PARTICIPANT_GUIDE.md       Phases, milestones, and the day plan
EXPERIMENTS_MLFLOW_GUIDE.md  Sweep, UI comparison, champion selection
```

## Setup (do this first, takes about 10 minutes)

**0. Configuration updates.** Update line 67 of ```config.py``` to reference your local path to the ```all-MiniLM-L6-v2``` model.

**1. Dedicated virtual environment.** This capstone must NOT share the main cohort venv: mlflow 3.15.1 pins pandas below 3, so this project runs pandas 2.3.3 on purpose.

```bash
python3.13 -m venv .venv
source .venv/bin/activate
pip3.13 install -r requirements.txt
```

**2. Services.**

```bash
docker compose up -d
docker compose ps        # both services should show running
```

**3. Configuration.**

```bash
cp .env.example .env     # defaults are correct for the compose file
```

**4. Verify.**

```bash
./run.sh starter check-services
```

Pinecone Local and MLflow must show UP. LM Studio and Ollama are optional and can stay DOWN until you switch backends.

**5. Baseline the tests.**

```bash
CAPSTONE_TARGET=starter pytest tests/ -q
```

Expect roughly 33 failures and 17 passes. That is the starting position, not a problem: the failures are your task list. `CAPSTONE_TARGET=solution pytest tests/ -q` is the target state (50 passed) and is how the instructor graded the solution.

## The work

Twelve TODO tasks live in `starter/src/cordwell_rag/`, each with a contract docstring stating exactly what the tests require. The PARTICIPANT_GUIDE maps them to milestones and suggests a team split. In brief:

| Task | Where | What |
|---|---|---|
| 1 | embeddings.py | Runtime device selection (CUDA, MPS, CPU) |
| 2 | corpus_loader.py | Encoding-tolerant file reading |
| 3 | corpus_loader.py | Latest-version document selection |
| 4 | chunking.py | Size-guaranteed chunking |
| 5 | chunking.py | Content dedupe |
| 6 | vector_store.py | Guarded, batched upsert |
| 7 | vector_store.py | Query and match normalization |
| 8 | llm.py | Offline extractive answering with abstention |
| 9 | rag.py | Pipeline retrieve and generate |
| 10 | evaluation.py | The RAG triad metric implementations |
| 11 | experiments.py | Champion selection over MLflow runs |
| 12 | api.py | The /search endpoint |

## Backends

`LAB_BACKEND` in `.env` selects the answer generator:

- `offline` (default): a deterministic extractive answerer. No model server, identical results on every machine. All tests run in this mode.
- `lmstudio`: OpenAI-compatible server on port 1234.
- `ollama`: OpenAI-compatible server on port 11434.

Both live backends share one client code path and differ only by base URL. The model tag lives in `LAB_MODEL` (default `gemma4`; confirm the exact tag your machine pulled with `ollama list`). A dead server raises a clear connection error. There is no silent fallback between backends, by design: an experiment that quietly swaps models produces numbers nobody can trust.

`EMBEDDING_BACKEND` selects embeddings the same way: `st` (semantic, sentence-transformers all-MiniLM-L6-v2, 384 dimensions) or `hash` (deterministic lexical, same dimensionality, what the tests use).

## Daily driver commands

```bash
./run.sh starter check-services      # ping everything
./run.sh starter ingest              # corpus -> chunks -> vectors -> index
./run.sh starter serve               # FastAPI on http://localhost:8000
./run.sh starter experiment          # the 6-run evaluation sweep
./run.sh starter select-champion     # gates + composite -> champion.json
./run.sh starter dashboard           # TruLens dashboard on port 8501
CAPSTONE_TARGET=starter pytest tests/ -q
```

MLflow UI: http://localhost:5001. API docs once serving: http://localhost:8000/docs.

## Expected results (offline backend, hash embeddings)

The solution's sweep on this corpus and eval set produces, for reference once your own sweep runs:

| Run | hit rate | MRR | ROUGE-L | groundedness | abstain recall |
|---|---|---|---|---|---|
| baseline-k3 | 1.000 | 0.967 | 0.382 | 0.636 | 0.000 |
| baseline-k5 | 1.000 | 0.967 | 0.376 | 0.640 | 0.000 |
| grounded_strict-k3 | 1.000 | 0.967 | 0.351 | 0.749 | 0.667 |
| grounded_strict-k5 | 1.000 | 0.967 | 0.349 | 0.742 | 0.667 |
| concise-k3 | 1.000 | 0.967 | 0.431 | 0.700 | 0.667 |
| concise-k5 | 1.000 | 0.967 | 0.405 | 0.668 | 0.667 |

Champion: `grounded_strict-k3`. The baseline variants fail the abstain recall gate because they never refuse, which is the whole argument for gates: the fluent configuration is not the trustworthy one. Small numeric drift is normal if you change chunking parameters; the shape of the story should hold. With `EMBEDDING_BACKEND=st` the absolute numbers differ (semantic embeddings score differently than lexical ones) and the offline abstention threshold deserves re-checking; see the experiments guide.

## Verification ledger

Everything below was executed and verified against the pinned versions on 2026-08-13, in a Linux sandbox without Docker. Items marked "instructor verify" require a live-Docker machine and are on the pre-class checklist.

| Claim | Status |
|---|---|
| pinecone 9.1.0: `Index.upsert` is keyword-only (`vectors=`) | Verified by execution |
| pinecone 9.1.0: `create_index(..., vector_type="dense")`, `has_index`, `describe_index` | Verified by execution |
| trulens 2.12.0: OTEL mode requires dict selectors; `Metric(implementation=..., selectors={...})` | Verified by execution (full app run) |
| trulens 2.12.0: `compute_feedbacks()` before starting the next TruApp, or pending evaluations are silently dropped | Verified by execution |
| openai 3.0.0: `chat.completions.create`, `APIConnectionError` on dead server | Verified by execution |
| mlflow 3.15.1: file store rejected by default (database backend required) | Verified by execution; compose uses SQLite |
| mlflow 3.15.1 pins pandas below 3 (dedicated venv required) | Verified at install time |
| Full test suite, solution target | 50 passed, 2 live tests skipped |
| Full offline sweep + champion selection | Executed; numbers above are real output |
| Pinecone Local container behavior (index host ports, metadata limit parity) | Instructor verify before class |
| compose on Apple Silicon (image architecture, `platform:` line) | Instructor verify before class |
| sentence-transformers 5.7.0 install + model download on cohort Mac | Instructor verify before class |
| `gemma4` tag present on cohort machines (`ollama list`) | Instructor verify before class |

## Decisions table

| Decision | Choice | Why |
|---|---|---|
| MLflow host port | 5001 | macOS AirPlay Receiver squats on 5000 |
| Pinecone Local port | 5080 (indexes 5081+) | Emulator default; keeps clear of 5001 |
| Metadata guard | Client-side check against 40960 bytes before upsert | Fail loud with the chunk id instead of a mid-batch server error |
| Abstention contract | Answer must begin with `NOT ENOUGH CONTEXT` | Exact, testable, and prompt-enforceable across backends |
| Offline abstention threshold | 0.30 (hash embeddings) | Calibrated on the real eval set; see experiments guide |
| Champion artifact | MLflow tag + `artifacts/champion.json` | The pipeline is configuration, not weights; a config artifact is the honest deployment handoff. In production with registered models, the Week 7 registry alias pattern (`models:/name@alias`) is the equivalent |
| Tests vs services | Unit suite runs with zero services via a fake index enforcing the real SDK shape | The red and green signal must be available on any machine, any time |

## Currency flags

- `⚠️ CURRENCY FLAG` mlflow 3.15.1 refuses the classic `./mlruns` file store unless `MLFLOW_ALLOW_FILE_STORE=true`; database backends are the supported path. The compose file and all instructions use SQLite. If a tutorial you find online says `mlflow ui` with no backend, it predates this.
- `⚠️ CURRENCY FLAG` trulens 2.x replaced the 1.x `Feedback`/`Selector.on_input()` chain style with OTEL spans and dict selectors. Materials showing `.on_input().on(...)` are for 1.x and will raise in this environment.
- `⚠️ CURRENCY FLAG` pinecone 9.x removed positional `upsert`. Snippets showing `index.upsert(batch)` are for older clients.
- `⚠️ CURRENCY FLAG` sentence-transformers pinned at 5.7.0, current on PyPI at build time; verify against the cohort mirror before class.

## A note on responsible AI

The grounded_strict variant, the abstention contract, and the abstain recall gate exist because a retail assistant that invents a return window is worse than one that says it does not know. The champion criteria encode that value explicitly: no configuration ships unless it refuses what it cannot support. Groundedness in this lab is measured with embedding similarity, which is a proxy; the TruLens dashboard shows per-question traces so you can audit where the proxy is wrong. All corpus content is synthetic and fictional.
