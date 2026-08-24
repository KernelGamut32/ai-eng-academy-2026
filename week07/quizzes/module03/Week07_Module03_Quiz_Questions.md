# Week 7, Module 03 Knowledge Check
## ML Pipelines, Automation and API Design: Airflow 3 Orchestration

**Format:** 7 multiple-choice questions, one correct answer each.
**Time:** 15 to 20 minutes.
**Scope:** This module only (Airflow 3.3.1 DAGs, scheduling, XCom, DVC and MLflow inside tasks, quality gates, reliability, testing).
**Environment assumed in every question:** Apache Airflow 3.3.1, MLflow 3.15.1, the Cordwell review summarizer pipeline from the slides.

Read code questions carefully. Several of them turn on *when* something happens (parse time versus run time) or on *which* task is affected, not just on whether something breaks.

---

### Question 1 (code)

A teammate copies a DAG definition from an older internal wiki page into `dags/cordwell_retrain.py` on the cohort Airflow 3.3.1 install:

```python
from datetime import datetime
from airflow.sdk import DAG, task

with DAG(
    dag_id="cordwell_summarizer_weekly_retrain",
    schedule_interval="0 2 * * 1",
    start_date=datetime(2026, 9, 7),
    catchup=False,
) as dag:

    @task
    def pull_reviews():
        return "/workspace/cordwell/data/reviews"

    pull_reviews()
```

What is the observable result after the DAG processor picks up the file?

- **A.** The DAG appears in the UI and is scheduled for 02:00 every Monday; `schedule_interval` is still accepted as a legacy alias for `schedule`.
- **B.** The DAG appears in the UI with a deprecation banner, but it will only run when triggered manually because the legacy keyword is ignored.
- **C.** The DAG does not appear in the UI at all. The file raises `TypeError` during parsing, and the failure is recorded in the DAG processor log and in `DagBag.import_errors`.
- **D.** The DAG appears in the UI, but the first scheduled run fails at task start with `TypeError` because the schedule is validated lazily.

---

### Question 2 (code)

The same teammate fixes Question 1 by switching to `schedule="0 2 * * 1"`, then adds a BashOperator that stamps the week into the log:

```python
from airflow.providers.standard.operators.bash import BashOperator

stamp = BashOperator(
    task_id="stamp_week",
    bash_command="echo 'week of {{ execution_date | ds }}'",
)
```

The DAG now shows up in the UI. What happens when the `stamp_week` task actually executes?

- **A.** The task succeeds and prints the run's date. `execution_date` is still provided in the template context for backward compatibility.
- **B.** The task succeeds and prints `week of` followed by an empty string, because undefined Jinja variables render as blank.
- **C.** The task never starts. Undefined template variables are caught during DAG parsing, so the DAG would not have shown up in the UI.
- **D.** The task fails at run time with a Jinja `UndefinedError` because `execution_date` is no longer in the template context; the fix is to use `logical_date` or `data_interval_start`.

---

### Question 3

The weekly DAG is declared with `schedule="0 2 * * 1"` and no explicit timetable object. The scheduler creates the run whose wall-clock trigger time is **Monday, September 14, 2026 at 02:00**.

Inside a task in that run, what values do `context["logical_date"]` and `context["data_interval_start"]` hold?

- **A.** `logical_date` is Sep 7 02:00 and `data_interval_start` is Sep 7 02:00; the run is labeled with the start of the week it processes.
- **B.** `logical_date` is Sep 14 02:00 and `data_interval_start` is Sep 7 02:00; the label is the trigger time but the interval covers the prior week.
- **C.** `logical_date` is Sep 14 02:00 and `data_interval_start` is Sep 14 02:00; a bare cron string uses `CronTriggerTimetable`, which produces a zero-width interval at the trigger time.
- **D.** `logical_date` is Sep 14 02:00 and `data_interval_start` is `None`; cron-scheduled runs do not carry a data interval unless a timetable is supplied.

---

### Question 4 (code)

To make retraining idempotent, `train` checks MLflow for an existing run with the same `run_key` and skips itself if one is found. The DAG is wired as shown:

```python
from airflow.sdk import task
from airflow.sdk.exceptions import AirflowSkipException

@task
def train(data_tag: str, hparams: dict) -> dict:
    run_key = make_run_key(data_tag, hparams)
    if run_already_complete(run_key):
        raise AirflowSkipException(f"run_key {run_key} already complete")
    ...
    return {"model_path": model_path, "mlflow_run_id": run_id}

@task
def evaluate(train_out: dict) -> dict:
    ...

evaluate(train(data_tag="v3.2", hparams=HPARAMS))
```

On Monday the scheduler starts a run, `train` finds an existing run for `run_key`, and raises `AirflowSkipException`. Which statement correctly describes what happens next?

- **A.** `train` is marked `failed`, the retry policy kicks in, and after retries are exhausted `evaluate` is marked `upstream_failed`.
- **B.** `train` is marked `skipped`, and `evaluate` also ends in `skipped` without ever running, because its default `trigger_rule` is `all_success`. To make `evaluate` run against the existing model, give it `trigger_rule="none_failed"` or have `train` return the existing run id instead of skipping.
- **C.** `train` is marked `skipped`, and `evaluate` runs normally with `train_out` set to `None`, since a skip does not block downstream tasks.
- **D.** `train` is marked `success` with an empty XCom, and `evaluate` fails with `KeyError: 'model_path'` when it tries to read the return value.

---

### Question 5 (code)

Before the weekly run, the registry for `MODEL_NAME` looks like this:

| Alias | Version |
|---|---|
| `champion` | 4 |
| `challenger` | 5 |

The promotion task from the slides runs to completion:

```python
from airflow.sdk import task
from mlflow import MlflowClient

@task
def promote_challenger() -> int:
    c = MlflowClient()
    v = c.get_model_version_by_alias(MODEL_NAME, "challenger").version
    c.set_registered_model_alias(MODEL_NAME, "champion", v)
    return v
```

What is the alias table immediately after `promote_challenger` finishes, and what single call rolls back to the previous production model?

- **A.** `champion` points to 5 and `challenger` is removed automatically. Rollback: `c.delete_registered_model_alias(MODEL_NAME, "champion")` followed by re-registering version 4.
- **B.** `champion` points to 5 and `challenger` still points to 5. Rollback: `c.set_registered_model_alias(MODEL_NAME, "champion", 4)`.
- **C.** `champion` points to 5 and `challenger` is moved back to 4 so the two aliases swap. Rollback: run `promote_challenger` again.
- **D.** The call fails because version 5 must first be transitioned out of the `Staging` stage before an alias can be assigned.

---

### Question 6

A colleague wants to be paged if the weekly retrain takes longer than four hours, and writes:

```python
@task(sla=timedelta(hours=4))
def train():
    ...
```

The DAG parses and appears in the UI. On Airflow 3.3.1, what is the actual behavior, and what is the correct replacement?

- **A.** The SLA works as in Airflow 2: a miss is logged in the SLA Misses view and `sla_miss_callback` on the DAG fires. No change needed.
- **B.** The DAG file raises `TypeError` because `sla` is no longer a valid task argument; replace it with `execution_timeout=timedelta(hours=4)`.
- **C.** The argument is accepted, a removal warning is emitted at parse time, and the value is discarded. Nothing fires at four hours. Replace it with a DAG-level `DeadlineAlert` using `DeadlineReference.DAGRUN_QUEUED_AT`, `interval=timedelta(hours=4)`, and a `SyncCallback`.
- **D.** The argument is silently converted into a `DeadlineAlert` measured from `DAGRUN_LOGICAL_DATE`, so the behavior is correct but the reference point is wrong.

---

### Question 7

The team replaces the Monday cron schedule with asset-aware scheduling:

```python
reviews = Asset("file:///workspace/cordwell/data/reviews.parquet")

with DAG(
    dag_id="cordwell_summarizer_retrain_on_data",
    schedule=[reviews],
    start_date=datetime(2026, 9, 7),
) as dag:
    ...
```

Which event causes the scheduler to create a run of this consumer DAG?

- **A.** Airflow watches the file's modification time on disk and creates a run whenever `reviews.parquet` changes, regardless of who wrote it.
- **B.** A task in any DAG that declares `outlets=[reviews]` completes successfully, which records an asset event; the scheduler then creates the consumer run.
- **C.** The producer DAG run finishes with state `success`, whether or not any of its tasks declare the asset as an outlet.
- **D.** A `FileSensor` polling `reviews.parquet` in `reschedule` mode returns true; asset schedules are syntactic sugar over a filesystem sensor.

---

*End of knowledge check.*
