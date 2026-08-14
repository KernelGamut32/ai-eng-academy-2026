# Participant Guide: Cordwell RAG Mini-Capstone

Teams of 3. Two to three days. One system, built end to end.

This guide is the map: what you are building, in what order, with which milestone proving each phase done. The README covers setup; HINTS.md and HINTS_DETAILED.md cover getting unstuck; the experiments guide covers the sweep. Keep this file open all week.

## What you will be able to say you did

By the end, your team has: ingested a messy real-shaped document corpus into a vector store running in Docker, built a RAG pipeline with three prompt variants and three swappable LLM backends, implemented the RAG triad evaluation metrics and run them through TruLens, executed a tracked 6-configuration sweep in MLflow, selected a champion configuration with explicit quality gates, and served the result as a documented HTTP API. That sentence is a resume bullet. Every clause of it is checked by a test or produces an artifact you can show.

## The two rules

**Rule 1: the tests are the specification.** Every task's contract docstring states what the tests check. When a test and your intuition disagree, the test wins, and the disagreement is worth discussing at standup because it usually means the contract encodes a lesson.

**Rule 2: rotate the skeptic.** Each phase, one team member is the designated skeptic: they do not write the code, they try to break it and question it. Why dedupe before upsert and not after? What happens to the metadata of a dropped duplicate? Why is the fluent variant losing? The skeptic writes down the two best questions per phase; they are your standup material and they are the kind of question the Week 9 content takes seriously.

## Suggested team split

The module graph splits cleanly three ways. Own your lane, review the others.

- **Engineer A, ingestion:** Tasks 2, 3, 4, 5 (corpus_loader, chunking). The corpus is hostile; this lane's tests encode every trap.
- **Engineer B, retrieval and serving:** Tasks 1, 6, 7, 12 (embeddings device pick, vector_store, the /search endpoint).
- **Engineer C, generation and evaluation:** Tasks 8, 9, 10 (offline answerer, pipeline, triad metrics).
- **Together, day 2 to 3:** Task 11 (champion selection), the sweep, and the writeup. Champion criteria are a team decision, not a lane.

Solo or pair teams: follow the milestone order below instead; it is dependency order.

## Milestones

Each milestone has a command and an unambiguous pass state. Do not start the next phase on a red milestone.

### M1: Environment up (Day 1, first hour)

```bash
docker compose up -d
./run.sh starter check-services
CAPSTONE_TARGET=starter pytest tests/ -q
```

Pass: Pinecone Local UP, MLflow UP, pytest reports approximately 33 failed / 17 passed with zero collection errors. Read the failure list once, out loud, as a team: it is the task list in test form.

### M2: Corpus survives ingestion (Day 1 morning)

Tasks 2, 3, 4, 5, then:

```bash
CAPSTONE_TARGET=starter pytest tests/test_corpus_loader.py tests/test_chunking.py -q
```

Pass: both files fully green. The numbers to expect, and to be able to explain at standup: 31 documents load, 30 are active, about 190 chunks, at least 5 duplicate chunks dropped. Every one of those numbers has a story; the skeptic should be able to tell each one.

### M3: Vectors in the store (Day 1 afternoon)

Tasks 1, 6, 7, then:

```bash
CAPSTONE_TARGET=starter pytest tests/test_embeddings.py tests/test_vector_store.py -q
./run.sh starter ingest
```

Pass: tests green, and ingest prints a chunk count and an upserted count that match each other and M2's numbers. Spot-check retrieval before moving on: the ingest command ends with a probe query and its top hit; read it and judge whether it is sane.

### M4: Questions get answers (Day 2 morning)

Tasks 8, 9, then:

```bash
CAPSTONE_TARGET=starter pytest tests/test_llm.py tests/test_rag.py -q
```

Pass: green, including the abstention tests. Then interrogate your own system for ten minutes: ask it three questions you know the corpus answers, one it cannot answer, and one that is ambiguous. Watch what comes back. This manual session is where most teams first notice something the metrics will formalize in M5.

### M5: Evaluation runs and the sweep completes (Day 2 afternoon)

Tasks 10, 11, then:

```bash
CAPSTONE_TARGET=starter pytest tests/test_evaluation.py tests/test_experiments.py -q
./run.sh starter experiment
./run.sh starter select-champion
```

Pass: tests green; six runs appear in the MLflow UI at http://localhost:5001; select-champion prints a champion and writes `artifacts/champion.json`. Open the MLflow comparison view and the TruLens dashboard (`./run.sh starter dashboard`) and spend real time in both; the experiments guide has a tour. The interesting question is not which run won but why the baselines were ineligible.

### M6: Served and demonstrated (Day 3)

Task 12, then:

```bash
CAPSTONE_TARGET=starter pytest tests/ -q     # the full suite, green
./run.sh starter serve
```

Pass: 50 passed (2 live tests skip without their services; with Docker up they run too). `http://localhost:8000/docs` renders; `/search?q=return window` returns chunks; `/answer` returns a grounded answer with sources for a fair question and refuses an unfair one.

## The demo (30 minutes per team, Day 3)

Structure it as a story, not a feature tour:

1. **One number from ingest** (2 min): how many documents in, how many chunks out, and one thing the corpus did that your code had to survive.
2. **Live answers** (5 min): one question it answers well with sources, one it correctly refuses. Show the JSON, point at the sources array.
3. **The sweep** (10 min): MLflow comparison view on screen. Walk the gates. Name the champion and defend the criteria as if a stakeholder asked why the highest-ROUGE run did not ship.
4. **One trace** (5 min): TruLens dashboard, a single question's trace, retrieval to answer to scores. Pick one where a metric surprised you.
5. **The skeptic's best question** (5 min): present it and your best current answer.

## Acceptance criteria (done when)

- Full test suite green on the starter target: 50 passed.
- `./run.sh starter ingest` completes and reports counts consistent with M2.
- Six runs visible in MLflow; `champion.json` exists and names a run that passes all gates.
- The API serves /health, /search, and /answer per the docs page, refusing at least one deliberately unanswerable question.
- Each team member can explain one module they did not write.

## Stretch goals (pick at most one, only after M6)

- **Semantic embeddings:** switch `EMBEDDING_BACKEND=st`, re-ingest, rerun the sweep, and compare leaderboards. The offline abstention threshold was calibrated for hash embeddings; recalibrating it for st is the real work here (the experiments guide shows the method).
- **Live backend:** stand up Ollama, switch `LAB_BACKEND=ollama`, rerun the sweep, and compare the live leaderboard against offline. Budget for it being slower; that latency column is itself a finding.
- **A fourth variant:** design a prompt variant you believe beats grounded_strict on the composite without failing a gate, register it in VARIANTS, and prove it in MLflow.

## Working agreements that save teams

- Commit at every green milestone at minimum.
- When stuck, the ladder is: contract docstring, the failing test's assertion, HINTS.md level 1, then up the levels. Fifteen minutes stuck without moving up the ladder is too long.
- Pick ONE hint tier per task. HINTS.md guides you to the answer; HINTS_DETAILED.md explains the working core when you would rather read code than be led. Reading both wastes your time.
- The solution package exists in the repo. Opening it before your demo defeats the point; after the demo, diffing it against your implementation is genuinely worthwhile.
