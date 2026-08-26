# Detailed Hints: Working Cores with Commentary

The working core of each task with commentary on why each line exists, withholding function shells, return assembly, and glue. If you would rather be led stepwise, use HINTS.md; one tier per task.

---

## Task E1: check_services_or_raise

```python
required = {
    "pinecone": config.PINECONE_HOST,       # both read from config, never
    "mlflow": config.MLFLOW_TRACKING_URI,   # hard-coded: inside the Airflow
}                                            # container these resolve to
                                             # in-network service names
down: List[str] = []
status: Dict[str, str] = {}
for name, url in required.items():
    try:
        httpx.get(url, timeout=timeout_s)    # ANY response means alive; we
        status[name] = "up"                  # are probing reachability, not
    except Exception as exc:                 # correctness, so a 404 is fine
        down.append(f"{name} ({url}): {type(exc).__name__}")
```
The raise is yours: fire only after the loop, joining every entry, so one nightly failure email names the whole outage instead of the first casualty. Collecting-then-raising versus raising-on-first is the entire design decision here.

## Task E2: refresh_ingest

```python
docs = load_corpus()
active = select_active_documents(docs)
chunks, dup_dropped = chunk_corpus(active)
if not chunks:
    # Guard one, BEFORE any embedding or index work: a refresh that
    # would publish an empty index must die here, cheaply and clearly.
    raise RuntimeError("refresh produced zero chunks; refusing to publish an empty index")

embedder = _build_embedder()                 # the factories, not direct
vectors = embedder.embed([c.text for c in chunks])
store = _build_store()                       # construction: these are the
store.ensure_fresh_index()                   # seams the tests patch
written = store.upsert_chunks(chunks, vectors)
if written != len(chunks):
    # Guard two: a half-written index is worse than no refresh, because
    # nothing downstream can tell it is half-written.
    raise RuntimeError(f"partial upsert: {written} of {len(chunks)} vectors written; "
                       "index state is not trustworthy")
```
The counts dict is yours: five plain ints under the contract's exact keys. Plain ints matter because this dict rides XCom to two downstream tasks; anything fancier is a serialization bug you meet at 2:30 AM.

## Task E3: run_refresh_sweep

```python
embedder = _build_embedder()
store = _build_store()
store.connect_existing_index()      # attach, never create: a sweep against
                                    # a missing index means ingest did not
                                    # run, and THAT is the error to surface

items = load_eval_set()
config.ensure_dirs()
session = TruSession(database_url=config.TRULENS_DB_URL)

run_ids = run_sweep(
    embedder, store, items, session,
    tracking_uri=config.MLFLOW_TRACKING_URI,
    # Explicit on purpose: run_sweep's DEFAULT bound config at import
    # time. Modules import lazily inside tasks, so whether the default
    # is stale depends on what imported first: an order-dependent bug.
    # Reading config at call time deletes the whole bug class.
)

mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
client = mlflow.tracking.MlflowClient()
for run_id in run_ids:
    client.set_tag(run_id, "refresh_tag", refresh_tag)
    # Tagging after the fact keeps run_sweep refresh-agnostic: the
    # capstone CLI still uses it unchanged, and this task decorates.
```
The return dict is yours. The tag is what lets you open MLflow, filter one nightly refresh, and see exactly its six runs; without it, night three's runs and night four's are an undifferentiated pile.

## Task E4: write_refresh_report

```python
reports_dir = config.ARTIFACTS_DIR / "reports"
reports_dir.mkdir(parents=True, exist_ok=True)
path = reports_dir / f"refresh_{run_date}.md"
# run_date is an argument. The DAG derives it from data_interval_end,
# so clearing and re-running the August 14 interval on August 16 still
# writes refresh_2026-08-14.md: reproducibility over recency.

metrics = verdict.get("metrics", {})
lines = [
    f"# Cordwell KB Refresh Report: {run_date}",
    ...
]
for name in sorted(metrics):                 # sorted: two reports diff
    lines.append(f"- {name}: {metrics[name]:.4f}")   # cleanly line by line
```
Assembly and the ingest-counts block are yours; the tests name the facts that must appear. `.get` with defaults throughout, because a report writer that crashes on a missing key turns a reporting problem into a pipeline failure.

## Task E5: raise_no_champion

```python
failures = verdict.get("gate_failures", {})
detail = ", ".join(f"{gate}: {count} runs failed" for gate, count in failures.items())
raise RuntimeError(
    "refresh completed but no configuration passed the champion gates "
    f"({detail or 'no gate detail available'}); index was refreshed but "
    "champion.json was NOT updated"
)
```
That is nearly the whole task, and the design content is in what it does NOT do: no logging-and-returning, no soft warning. The branch sent the run here because quality failed; this task's job is making the schedule show red with a message that says which standard failed and what state the world was left in.

## Task E6: the schedule

```python
schedule="30 2 * * *",
```
Commentary is the value here: on Airflow 3.3 this one line builds a CronTriggerTimetable, meaning the run FIRES at 02:30 with data_interval_start == data_interval_end == 02:30. Airflow 2 would have run the "02:30 interval" at the FOLLOWING 02:30, and a decade of blog posts explains that old behavior. If your mental model says "the run for yesterday", update it: this DAG's runs are moments, not intervals.

## Task E7: the branch

```python
if verdict.get("status") == "champion_selected":
    return "publish_report"
return "no_champion_alert"
```
Two choices worth defending at standup. `.get` rather than `[...]`: a malformed verdict routes to the alert instead of crashing the router itself. And the default direction is the FAILURE path: when in doubt, a scheduled pipeline should escalate, not publish.

## Task E8: the two edges

```python
services >> counts            # ordering without data: TaskFlow cannot
                              # infer what no argument expresses
choice >> [report, alert]     # a branch skips among its DIRECT
                              # downstreams; without this edge both
                              # outcome tasks answer to select() alone
                              # and BOTH run every night
```
Placement is yours (after all task invocations). The second edge's failure mode is the sneaky one: the DAG renders, runs, and publishes a report every night INCLUDING nights the gates failed, while also raising the alert. Both-paths-ran is the bug the graph view shows in one glance.

## Task E9: the migration

The complete diff, which is the point (a migration you can read in one glance):

```python
schedule="0 6 * * 1",
# was: schedule_interval="0 6 * * 1",
#   -> TypeError at DAG construction; the UI import error you saw at M1

op_kwargs={"eval_date": "{{ data_interval_end.strftime('%Y-%m-%d') }}"},
# was: "{{ execution_date.strftime('%Y-%m-%d') }}"
#   -> parses clean, then jinja UndefinedError the first time it RUNS
```
Everything else stays. The two-stage failure ordering deserves a sentence in your demo: fixing only what the parser catches passes review and detonates on schedule, which is precisely how half-finished Airflow migrations fail in the wild.

---

## Stretch goal cores

**Failure callback.** `default_args["on_failure_callback"] = _log_failure` where the callback receives a context dict; `context["task_instance"].task_id` and a timestamp appended to a log file is the whole body. Prove it with the M5 breakage, not by reading the code.

**Corpus sensor.** Hash the corpus directory (filenames + sizes + mtimes through hashlib) in a @task.short_circuit ahead of ingest; compare against an Airflow Variable (Variable.get with a default, Variable.set after). short_circuit returning False skips everything downstream. The discussion it buys: what staleness bound did you just trade for the saved compute?
