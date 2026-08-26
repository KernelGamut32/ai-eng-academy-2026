# Participant Guide: Scheduling the Cordwell Refresh

Teams of 3. One day. The RAG system is given; the orchestration is yours.

## What you will be able to say you did

By end of day: you took a working RAG evaluation pipeline and made it operate itself, writing the orchestration layer as testable plain functions, wiring an Airflow 3 DAG with a cron schedule, XCom data flow, a verdict branch, and deliberate failure behavior, and migrating a broken Airflow 2 DAG across a major version boundary. The last item alone is a real workplace task with a real market rate.

## The two rules (unchanged from the capstone)

**The tests are the specification.** Every task's contract docstring states what the tests check. E1 to E5 are graded by `tests/test_pipeline_tasks.py` in your host venv; E6 to E9 by `tests/test_dag_structure.py` inside the Airflow container.

**Rotate the skeptic.** One member per phase questions instead of writing: Why does the branch need both outcome tasks as DIRECT downstreams? What would silently go wrong if the report used the wall clock instead of the data interval? Why is catchup False, and when would True be right? The skeptic's two best questions are your standup material.

## Suggested team split

- **Engineer A, task functions:** E1, E2 (preflight, ingest guards). The failure-message contracts are the work: a scheduled pipeline is judged by what its 2 AM failures say.
- **Engineer B, task functions:** E3, E4, E5 (sweep tagging, report, loud failure).
- **Engineer C, the DAG:** E6, E7, E8, plus first read of the legacy DAG for E9.
- **Together, afternoon:** E9 migration, the live scheduled run, and the demo.

A and B can work entirely in the host venv with fast tests; C lives in the Airflow UI and container. Swap lanes at lunch if you finish early; the point of three lanes is three explanations at the demo.

## Milestones

### M1: Stack up, baselines read (first 30 minutes)

```bash
cp .env.example .env
docker compose up -d --build
./run.sh starter check-services
CAPSTONE_TARGET=starter pytest tests/ -q
```

Pass: three services running, check-services green, pytest shows 7 failed / 51 passed. Open the Airflow UI at http://localhost:8081 and find two things before writing any code: the `cordwell_kb_refresh` DAG (parsed, unscheduled) and the DAG import error banner. Read the import error out loud as a team; it names a keyword that no longer exists, and that sentence is Task E9's entire diagnosis.

### M2: Task functions green (mid-morning)

Tasks E1 to E5, then:

```bash
CAPSTONE_TARGET=starter pytest tests/test_pipeline_tasks.py -q
```

Pass: 8 passed. The sweep test takes about 45 seconds because it runs the real 6-configuration evaluation against a fake index; the other seven finish in two.

### M3: DAG structurally correct (around lunch)

Tasks E6, E7, E8, then run the structure tests inside the container:

```bash
docker compose cp tests airflow:/opt/airflow/lab_tests
docker compose exec airflow bash -c "pip install pytest && CAPSTONE_TARGET=starter python -m pytest /opt/airflow/lab_tests/test_dag_structure.py -q"
```

Pass: only ONE failure remains (the legacy DAG import error, which is M4's job). The schedule test, both edge tests, and the task-set test are green. Sanity-check in the UI: the DAG's graph view shows the branch fanning into both outcome tasks, and the schedule column reads your cron.

### M4: The migration (early afternoon)

Task E9. The legacy weekly DAG has TWO problems, and the second is the one that burns people: the parse error the UI showed you, and a template that will only fail at RUN time, after the parse error is fixed. The migration is minimal by design; if your diff against the legacy file is more than a few lines, you are rewriting, not migrating. Then:

Pass: the container structure tests report 6 passed, and the UI import error banner is gone.

### M5: A real scheduled run (afternoon)

From the root project folder in a terminal, run `chmod -R o+rwX artifacts` to make the artifacts directory writeable from the Airflow container.

Trigger the refresh DAG manually in the UI (the ops guide walks the buttons), watch the graph turn green task by task, then verify the outputs:

```bash
ls artifacts/reports/          # refresh_<date>.md appeared
cat artifacts/champion.json    # grounded_strict-k3
open http://localhost:5001     # six runs tagged with today's refresh_tag
```

Then break it on purpose: `docker compose stop pinecone-local`, trigger again, and watch preflight fail in seconds with a message naming the dead service. Restart pinecone, and use the UI to clear and re-run JUST the failed task. That clear-and-retry motion is half of what operating a scheduler means.

## The demo (15 minutes per team)

1. **The graph** (3 min): the completed DAG in the UI, one sentence per task, and why the two E8 edges had to be explicit.
2. **A green run and its artifacts** (4 min): the dated report, champion.json, and the tagged MLflow runs. Say where the date came from and why not the wall clock.
3. **The failure story** (4 min): show the preflight failure from M5 and the no-champion design. The claim to defend: a scheduled pipeline that cannot fail loudly is worse than no pipeline.
4. **The migration diff** (3 min): legacy vs migrated, line by line. Name which failure was parse-time and which was run-time, and why that ordering deceives.

## Acceptance criteria (done when)

- Host suite: 58 passed. Container structure tests: 6 passed. No import error banner in the UI.
- One triggered refresh run fully green, with the dated report on the host and six tagged runs in MLflow.
- One deliberately failed run demonstrating preflight, plus a clear-and-retry recovery.
- Each member can explain one task they did not write.

## Stretch goals (pick at most one)

- **Backfill semantics:** set catchup=True on a copy of the DAG with a start_date three days back, run it, and explain what happened and why the real DAG says False. Cheap to do, surprisingly deep to explain.
- **Failure notification:** add an on_failure_callback that appends a line to artifacts/refresh_failures.log with the task id and timestamp; prove it fires using the M5 breakage.
- **Sensor the corpus:** replace the fixed schedule trigger for ingest with a check that skips the refresh when no corpus file changed since the last run (hash the directory into an Airflow Variable). Discuss what this trades against "refresh nightly no matter what."
