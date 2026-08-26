# Progressive Hints

Three levels per task; escalate only after a real attempt. Level 3 shows the key line or two, never the whole body. If you would rather read the working core with commentary, use HINTS_DETAILED.md instead; one tier per task.

The final section, "When Airflow fights back," is keyed to symptoms, not tasks.

---

## Task E1: check_services_or_raise

**Level 1.** Two required services, one loop, and the contract's soul is the error message: collect ALL the failures before raising, so one message diagnoses the whole outage.

**Level 2.** Build the dict of name to URL from config. For each, httpx.get inside try; success records "up", any exception appends a formatted entry (name, URL, exception type name) to a down list. After the loop: raise RuntimeError joining the down list, or return the status dict.

**Level 3.** The entry format that makes 2 AM readable:
```python
down.append(f"{name} ({url}): {type(exc).__name__}")
```

## Task E2: refresh_ingest

**Level 1.** You composed this pipeline in the capstone CLI; the new work is the two guards and the counts dict. Use the module's _build_store and _build_embedder factories, not direct construction.

**Level 2.** load_corpus, select_active_documents, chunk_corpus. Guard one: empty chunks list raises before any embedding happens. Embed, ensure_fresh_index, upsert_chunks. Guard two: written != len(chunks) raises naming both numbers. Return the five counts as plain ints.

**Level 3.** The partial-upsert guard:
```python
if written != len(chunks):
    raise RuntimeError(f"partial upsert: {written} of {len(chunks)} vectors written; ...")
```

## Task E3: run_refresh_sweep

**Level 1.** The capstone's run_sweep does the heavy lifting; your function is setup (factories, connect_existing_index, eval set, TruSession), one explicit argument the contract insists on, and a tagging loop afterward.

**Level 2.** Pass tracking_uri=config.MLFLOW_TRACKING_URI explicitly (the contract explains the import-time-default trap). After run_sweep returns the ids, set_tracking_uri, build one MlflowClient, and set_tag(run_id, "refresh_tag", refresh_tag) per run. Return the dict with both keys.

**Level 3.** The tagging loop:
```python
client = mlflow.tracking.MlflowClient()
for run_id in run_ids:
    client.set_tag(run_id, "refresh_tag", refresh_tag)
```

## Task E4: write_refresh_report

**Level 1.** Build the path from config.ARTIFACTS_DIR and run_date, mkdir with parents, assemble markdown lines, write, return str(path). The date is an ARGUMENT; the function never asks the clock what day it is.

**Level 2.** The tests check for: the run_date in the text, the champion run_name, the chunk count from the counts dict, and every metric name with its value. A lines list joined with newlines keeps it readable; iterate sorted(verdict["metrics"]) for the metric block.

**Level 3.** The path contract:
```python
path = config.ARTIFACTS_DIR / "reports" / f"refresh_{run_date}.md"
path.parent.mkdir(parents=True, exist_ok=True)
```

## Task E5: raise_no_champion

**Level 1.** Three sentences of message assembly and one raise. The message must carry the gate failure counts and the fact that champion.json was NOT updated.

**Level 2.** Join gate_failures items into "gate: N runs failed" fragments; embed in a message that also says the index WAS refreshed. Then raise RuntimeError. Returning normally here would paint the run green; the raise IS the feature.

**Level 3.**
```python
detail = ", ".join(f"{gate}: {count} runs failed" for gate, count in failures.items())
```

## Task E6: the schedule

**Level 1.** One keyword argument on the @dag decorator, and the keyword's NAME is the lesson. The starter comment tells you the cron target: 02:30 daily.

**Level 2.** schedule="30 2 * * *". Not schedule_interval (removed in Airflow 3; raises TypeError at import). On 3.3 a bare cron string builds a CronTriggerTimetable: fires AT 02:30 with a zero-width data interval.

**Level 3.** There is genuinely one line; the structure test asserts the timetable type and its .expression.

## Task E7: the branch

**Level 1.** A branch task's return value is a STRING: the task_id of the one downstream task that should run. Everything else directly downstream gets skipped.

**Level 2.** Read verdict["status"]. Exactly one value ("champion_selected") routes to "publish_report"; every other value, including a missing status, routes to "no_champion_alert". Default-to-alert is the safe failure direction.

**Level 3.**
```python
if verdict.get("status") == "champion_selected":
    return "publish_report"
return "no_champion_alert"
```

## Task E8: the two edges

**Level 1.** TaskFlow infers dependencies from DATA flow: passing a task's return into another task is an edge. The two missing edges carry no data, which is exactly why they must be declared with >>.

**Level 2.** Edge one: the preflight result feeds nothing, but ingest must wait for it: `services >> counts`. Edge two: the branch must own BOTH outcome tasks as direct downstreams or it cannot skip the loser: `choice >> [report, alert]`.

**Level 3.** Both lines together, placed after all the task calls:
```python
services >> counts
choice >> [report, alert]
```

## Task E9: the migration

**Level 1.** Two defects, discovered in sequence. The first announces itself in the UI import error banner before you change anything. The second hides until the DAG actually RUNS, because it lives in a template string that only renders at execution.

**Level 2.** Defect one: the DAG constructor keyword; rename it to the Airflow 3 name (the cron string itself is fine). Defect two: the op_kwargs template references a variable removed in Airflow 3; the modern equivalent that marks the END of the run's data interval is the drop-in replacement. Change nothing else; a migration's diff should be readable in one glance.

**Level 3.** The two changed fragments:
```python
schedule="0 6 * * 1",
op_kwargs={"eval_date": "{{ data_interval_end.strftime('%Y-%m-%d') }}"},
```

---

## When Airflow fights back (symptom-keyed)

**Symptom: red "DAG Import Errors" banner, TypeError about an unexpected keyword argument.** An Airflow 2 idiom hit an Airflow 3 parser. The banner names the file and the keyword; the fix is Task E9's first half. Note what this failure mode teaches: the DAG never loads, so its schedule silently stops existing.

**Symptom: a task fails with jinja2.exceptions.UndefinedError: 'execution_date' is undefined.** The DAG parsed fine and then died at run time: template variables render at execution, not at parse. This is Task E9's second half, and the deceptive ordering (parse-clean, run-broken) is why upgraded clusters pass review and then fail their first scheduled night.

**Symptom: your DAG is missing from the UI entirely, no error banner.** Usually the file was saved outside the mounted dags folder for your CAPSTONE_TARGET, or you edited the other target's copy. `docker compose exec airflow ls /opt/airflow/dags` shows what the container actually sees. Changing CAPSTONE_TARGET in .env requires `docker compose up -d` to remount.

**Symptom: preflight and ingest show as running at the same time, or both outcome tasks ran on one refresh.** The E8 edges are missing. TaskFlow only infers edges from data flow; ordering-only relationships and branch fan-outs must be declared. The graph view makes both mistakes visible in one glance.

**Symptom: ModuleNotFoundError: cordwell_rag inside a task log.** The sys.path bootstrap at the top of the DAG file could not find the package: either the src mount is missing from compose (did the stack start before .env existed?) or the import got moved to module top level ahead of the bootstrap. Task logs in the UI show the full traceback; start there, not in the scheduler logs.

**Symptom: the UI shows the DAG but the toggle is off and nothing ever runs.** DAGs arrive paused. Unpausing is the toggle at the left of the DAG list; a manual trigger also works while paused for exactly one run. This is a feature (new DAGs should not surprise-execute), not a bug.

**Symptom: MLflow runs from the DAG have no refresh_tag, or the sweep logged to the wrong place.** The tracking URI was read at the wrong time, or the tagging loop never ran. Confirm E3 passes tracking_uri explicitly; the contract docstring explains the import-time-default trap that makes this bug order-dependent and therefore intermittent.
