# Week 5 Lab 03: Cordwell Support Assistant
## RAG pipeline with a fine-tuned adapter, evaluated end to end

Students wire the Module 01 LoRA adapter and the Module 02 vector index
into one retrieval chain, then evaluate it: format adherence,
abstention, faithfulness by claim decomposition, answer relevancy, and a
retrieval degradation diagnosis. Core lab about 3.5 hours plus a 30
minute stretch.

## Files

| File | What it is | Who gets it |
|---|---|---|
| `Lab03_RAG_Student.ipynb` | Stubbed lab notebook, 22 soft checks, opens at 3 passing | Students |
| `Lab03_RAG_Solution.ipynb` | Fully executed solution, 22 of 22 passing | Instructor, release after lab |
| `cordwell_rag_backend.py` | Given infrastructure: corpus loader, offline stand-ins, local backend, judges | Students (import, not read) |
| `data/cordwell_corpus.jsonl` | 14 synthetic Cordwell documentation chunks | Students |
| `data/eval_queries.json` | 6 answerable and 4 unanswerable evaluation questions | Students |
| `HINTS.md` | Progressive hint tier, three levels per task | Students |
| `HINTS_DETAILED.md` | Detailed hint tier, working cores with commentary | Students |
| `setup/LOCAL_MODE_SETUP.md` | Docker, adapter, and judge server setup for local mode | Students (optional) |
| `setup/bootstrap_index.py` | Rebuilds the Pinecone Local index (required after every container start) | Students (local mode) |
| `INSTRUCTOR_DEMO_SCRIPT.md` | I-do walkthrough of Parts A and B to the hand off | Instructor |
| `INSTRUCTOR_WALKTHROUGH.md` | Line by line solution walkthrough, deck corrections, verification ledger | Instructor |
| `requirements.txt` | Pinned dependencies with currency flags | Everyone |

## Quick start (offline, the default)

```bash
pip install -r requirements.txt
jupyter lab Lab03_RAG_Student.ipynb
```

Run all cells. A fresh notebook reports 3 of 22 checks passing and never
crashes; the remaining 19 come from the ten TODO functions. The offline
backend is deterministic, so every number in the lab text reproduces
exactly.

## Backends

| Variable | Values | Default |
|---|---|---|
| `RAG_BACKEND` | `offline`, `local` | `offline` |
| `JUDGE_BACKEND` | `inprocess`, `lmstudio`, `ollama` | `inprocess` |
| `JUDGE_MODEL` | any served model tag | `gemma4` (confirm the pulled tag) |

Local mode (real SmolLM2 plus adapter, Pinecone Local retrieval) is
documented in `setup/LOCAL_MODE_SETUP.md`. Check thresholds are
calibrated against offline; in local mode treat them as guidance.

**Design note for the instructor.** The Module 03 deck sketches a single
three-value selector (`local`, `cloud_judge`, `offline`). This lab
splits it into the two knobs above, because the generation backend and
the judge backend vary independently (offline generation with a live
Ollama judge is a useful class configuration, and it matches the
LangChain lab selector standard from Week 3). Flagged as an open
question: if you prefer the deck's single knob, the mapping is
mechanical.

## Environment notes

* Cohort standard is Python 3.13; this package was built and executed on
  Python 3.12.3 (sandbox). No 3.13-only features are used.
* All device selection in local mode is runtime auto-select (CUDA, then
  MPS, then CPU). Nothing assumes a GPU.
* Synthetic data only. Cordwell Home & Hardware is fictional.
