# Week 7 Extension Lab: Scheduling the Cordwell RAG Refresh with Airflow

An orchestration extension to the Week 7 RAG mini-capstone. The RAG system you built there arrives here COMPLETE: corpus ingestion, Pinecone Local vector store, the evaluation sweep, MLflow tracking, and champion selection are all given as working code. The new work is making it run itself: an Airflow DAG that refreshes the knowledge base on a schedule, evaluates it, selects a champion against the gates, and either publishes a dated report or fails loudly.

One day of team work. Everything runs on your Mac with no cloud accounts: Pinecone Local, MLflow, and Airflow all live in docker compose.

## What changes from the capstone

The capstone asked "can you build the pipeline." This lab asks the production question that follows: "who runs it at 2:30 AM, and what happens when it goes wrong." Concretely, you will write the orchestration seam (plain Python task functions), wire them into an Airflow 3 DAG with a schedule, a branch, and explicit failure behavior, and migrate one inherited Airflow 2 DAG that the version upgrade broke.

## Repository layout

```
docker-compose.yml           Pinecone Local + MLflow + Airflow
docker-airflow/              Airflow image build (Dockerfile + pinned deps)
requirements.txt             Host venv dependencies (dedicated venv, see below)
.env.example                 Configuration template (CAPSTONE_TARGET lives here)
run.sh                       CLI wrapper: ./run.sh starter ingest
corpus/                      31 Cordwell documents + MANIFEST.md
data/eval/eval_set.jsonl     18 evaluation questions with references
starter/src/cordwell_rag/    Complete RAG system + 5 TODO task functions
starter/airflow/dags/        Refresh DAG (3 TODOs) + broken legacy DAG
solution/src/cordwell_rag/   Instructor solution (withheld until after the lab)
solution/airflow/dags/       Completed DAG + migrated legacy DAG
tests/                       Shared pytest suite, runs against either target
artifacts/                   Refresh reports and champion.json land here
HINTS.md                     Progressive hints, three levels per task
HINTS_DETAILED.md            Working-core hints with line commentary
PARTICIPANT_GUIDE.md         Phases, milestones, and the day plan
```

## Setup (about 15 minutes; the image build dominates)

**0. Configuration updates.** Update line 67 of ```config.py``` to reference your local path to the ```all-MiniLM-L6-v2``` model.

**1. Airflow image download.** From the **images** folder at <https://gamuttechnologysvcs-my.sharepoint.com/:f:/p/asanders/IgD_SIVCz8YJQYh7BL3DUy4ZAVgU8-9SO8Lo3boIy-wwV8g?e=D4X53b>, download **airflow-local-v3.3.1.tar**. Run `docker image load -i airflow-local-v3.3.1.tar`.

**2. Add execution permissions to `run.sh`.** In a terminal, navigate to the location of `run.sh` and execute `chmod +x run.sh`.

**3. Dedicated virtual environment** (same rule as the capstone: mlflow 3.15.1 pins pandas below 3, so never share the main cohort venv):

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

**4. Configuration, then services.** The compose file reads CAPSTONE_TARGET from .env to decide which DAG folder Airflow mounts, so copy the env file BEFORE starting:

```bash
cp .env.example .env       # CAPSTONE_TARGET=starter by default
docker compose up -d --build
docker compose ps          # three services; airflow needs ~30s after "running"
```

The first `up --build` builds the Airflow image with the RAG client libraries baked in (a few minutes, once). Subsequent starts are instant.

**5. Verify.**

```bash
./run.sh starter check-services            # pinecone + mlflow UP
open http://localhost:8081                 # Airflow UI (no login in lab mode)
CAPSTONE_TARGET=starter pytest tests/ -q   # expect 7 failed, 51 passed
```

The 52 passes are the entire RAG system arriving green. The 9 failures (you may see 8 failed and 1 skipped) are related to the new tasks. Tasks E6 to E9 are graded by the DAG tests, which need Airflow and therefore run inside the container:

```bash
docker compose cp tests airflow:/opt/airflow/lab_tests
docker compose exec airflow bash -c "pip install pytest && CAPSTONE_TARGET=starter python -m pytest /opt/airflow/lab_tests/test_dag_structure.py -q"
```

Expect 5 failed, 1 passed in the starter. In the Airflow UI you will also see one DAG import error banner: that is real, it is Task E9, and the error message is the lesson.

## The work

| Task | Where | What |
|---|---|---|
| E1 | pipeline_tasks.py | Preflight service check that fails with a diagnosis |
| E2 | pipeline_tasks.py | Refresh ingest with empty-index and partial-upsert guards |
| E3 | pipeline_tasks.py | Evaluation sweep tagged with the refresh date |
| E4 | pipeline_tasks.py | Dated refresh report writer |
| E5 | pipeline_tasks.py | The loud no-champion failure |
| E6 | cordwell_kb_refresh_dag.py | Schedule the DAG (cron, Airflow 3 semantics) |
| E7 | cordwell_kb_refresh_dag.py | The branch: publish or alert, from the verdict |
| E8 | cordwell_kb_refresh_dag.py | Two explicit dependency edges TaskFlow cannot infer |
| E9 | legacy_weekly_eval_dag.py | Migrate the Airflow 2 DAG the upgrade broke |

The design rule the module teaches: task functions are plain callables with JSON-serializable inputs and outputs, and the DAG file is wiring, not logic. That split is why E1 to E5 are testable in seconds without Airflow, and why the DAG file stays readable.

## The refresh DAG, once complete

```
preflight >> ingest >> evaluate >> select >> route_on_verdict >> publish_report
                                                             \\> no_champion_alert
```

Nightly at 02:30. A refresh whose sweep produces no gate-passing configuration refreshes the index but does NOT update champion.json, and surfaces as a FAILED run: silence is the one behavior a scheduled pipeline is never allowed to have.

## Expected results

With the offline backend and hash embeddings (the containerized defaults), a completed refresh run produces the same numbers as the capstone reference: ingest 31 loaded, 30 active, 190 chunks, 6 duplicates dropped; six MLflow runs; champion grounded_strict-k3 at composite 0.5251; and `artifacts/reports/refresh_<date>.md` on the host. The report date comes from the DAG's data interval, not the wall clock.

## Verification ledger

Everything below was executed against apache-airflow 3.3.1 (Python 3.12) on 2026-08-23, in a Linux sandbox without Docker; the DAG itself was executed end to end in-process via Airflow's own test runner with a fake vector store and SQLite MLflow. Items marked "instructor verify" need a live-Docker machine.

| Claim | Status |
|---|---|
| `schedule_interval=` raises TypeError at DAG construction on 3.3.1 | Verified by execution (this is the starter legacy DAG's import error) |
| Bare cron string on `schedule=` builds CronTriggerTimetable; introspection attribute is `.expression` | Verified by execution |
| `{{ execution_date }}` raises jinja UndefinedError; `data_interval_end` renders and is injectable as a TaskFlow parameter | Verified by execution |
| Solution DAG full run via dag.test(): 7 tasks, XCom dicts, branch to publish, dated report, state SUCCESS | Verified by execution; sweep numbers identical to the capstone reference |
| Failure branch: no_champion routes to alert, publish_report SKIPPED, run FAILED with gate diagnosis | Verified by execution |
| dag.test() on 3.3.1 requires the DAG serialized first (`airflow dags reserialize`) | Verified by execution; documented in the ops guide |
| DagBag.get_dag() falls back to previously serialized DAGs in the metadata DB | Verified by execution; tests read the folder-parsed dags dict instead |
| Full pytest suite, solution target, main venv | 58 passed, 3 skipped |
| Full pytest suite, starter target, main venv | 7 failed (E1 to E5), 51 passed, 3 skipped |
| DAG structure tests, Airflow venv | Solution 6 passed; starter 5 failed, 1 passed |
| docker compose three-service stack on Apple Silicon (image build, standalone startup, mounts) | Instructor verify before class |
| In-container structure test run via `docker compose exec` | Instructor verify before class |

## Decisions table

| Decision | Choice | Why |
|---|---|---|
| Airflow deployment | `airflow standalone`, one container, SQLite | Exactly enough Airflow for a lab; the ops guide names what production changes |
| Airflow UI host port | 8081 | 8080 is the gateway port elsewhere in Week 7 |
| Auth | SIMPLE_AUTH_MANAGER_ALL_ADMINS=True | Lab-only; no password hunting. Flagged in compose as never-in-production |
| Image strategy | Custom Dockerfile baking client deps | _PIP_ADDITIONAL_REQUIREMENTS reinstalls on every container start |
| DAG folder selection | CAPSTONE_TARGET in .env drives the compose mount | One switch for tests, run.sh, and Airflow alike |
| Heavy imports | Inside task functions, never at DAG module top | The scheduler re-parses DAG files continuously; parse time is a tax on everything |
| Orchestration seam | Plain functions in pipeline_tasks.py | Unit-testable without Airflow; XCom-safe by construction |
| tracking_uri | Passed explicitly at call time | run_sweep's default binds config at import time; late config changes would be ignored |
| Refresh date | From data_interval_end, never wall clock | Re-running a past interval must reproduce the same report |
| catchup | False | A week of downtime should trigger one fresh refresh, not seven stale backfills |

## Currency flags

- `⚠️ CURRENCY FLAG` Airflow 2 reached end of life April 2026. Most tutorials online still show `schedule_interval=`, `{{ execution_date }}`, and `provide_context=True`; all three are removed in Airflow 3 and the first two are planted in this lab's legacy DAG precisely because students will meet them in the wild.
- `⚠️ CURRENCY FLAG` On Airflow 3.3, a bare cron string means CronTriggerTimetable: the DAG fires AT the cron moment with a zero-width data interval, not after a full interval elapses as Airflow 2 did. Materials reasoning about "the run for yesterday's interval" predate this.
- `⚠️ CURRENCY FLAG` mlflow 3.15.1 refuses the classic filesystem store; the compose file uses SQLite (inherited flag from the capstone, still active here).
- `⚠️ CURRENCY FLAG` The Airflow image tag `apache/airflow:3.3.1-python3.12` and the standalone auth env var were current at build time; verify both against the official docs on a cohort machine before class.

## A note on responsible AI

The refresh DAG inherits the capstone's values and adds the scheduling one: the no-champion path exists so that a nightly pipeline whose quality gates fail cannot silently keep serving yesterday's champion while pretending everything is fine. The run fails, the report is withheld, and the gate failure counts land in the task log. All corpus content is synthetic and fictional.
