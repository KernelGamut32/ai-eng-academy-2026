# Module 04 Knowledge Check: MLflow Experiment Tracking

**AI Engineering Academy | Gamut Technology Services**

Five multiple choice questions covering Module 04: local tracking setup, logging runs, `mlflow.evaluate`, comparing runs and selecting a champion, and managing lifecycle with registry aliases.

All questions use the Cordwell Home and Hardware scenario from the module: a binary classifier that flags customer reviews needing safety-team escalation, recall-first, with three runs sweeping regularization strength C over 0.1, 1.0, and 10.0.

**Instructions.** Choose the single best answer for each question. Three of the five show code and ask you to reason about what it does. Read the code carefully. In one question the code runs cleanly, raises nothing, and still puts the wrong model into serving.

Closed book. Approximately 12 minutes.

---

### Question 1

Your teammate sets up MLflow locally and points the tracking URI at a bare directory rather than a database.

```python
mlflow.set_tracking_uri("file:///tmp/mlruns")
mlflow.set_experiment("cordwell_escalation_triage")
```

On the version pinned for this module, what happens?

- A. It works normally. Parameters, metrics, and the model registry all function against a local directory.
- B. Tracking works, but any `set_registered_model_alias` call later fails, because only the registry needs a database.
- C. It raises an `MlflowException` at setup. The filesystem tracking backend is in maintenance mode on MLflow 3.x, which is why the module standardizes on `sqlite:///mlflow.db`.
- D. It silently falls back to an in-memory store, so runs appear during the session and vanish when the process exits.

---

### Question 2

A teammate is debugging a training loop and adds a second `log_param` call for a key already logged in the same run.

```python
with mlflow.start_run(run_name="logreg_C1.0"):
    mlflow.log_param("C", 1.0)
    mlflow.log_param("class_weight", "balanced")

    # ... later in the same run, after some refactoring ...
    mlflow.log_param("class_weight", "balanced")   # line A
    mlflow.log_param("C", 0.5)                     # line B
```

What happens at lines A and B?

- A. Both lines raise `MlflowException`, because parameters are immutable and cannot be logged twice under any circumstances.
- B. Line A succeeds silently because the value is unchanged. Line B raises `MlflowException` because the value differs, and the stored value of `C` remains 1.0.
- C. Both lines succeed. Line B overwrites `C` with 0.5, which is why parameters are useful for mid-run configuration changes.
- D. Line A raises because the key already exists. Line B succeeds because a new value creates a new parameter revision.

---

### Question 3

The module replaces a hand-built metric harness with a single call.

```python
edf = pd.DataFrame(Xte)
edf["label"] = yte

with mlflow.start_run(run_name="eval_champion"):
    res = mlflow.models.evaluate(
        "models:/cordwell-escalation-clf@champion",
        edf,
        targets="label",
        model_type="classifier",
    )
```

The run records `accuracy_score` 0.857, `recall_score` 0.8308, `true_positives` 54, `false_negatives` 11, and `example_count` 1000, among others. Which statement about this call is correct?

- A. The first argument pins a specific model version, so the evaluation result will not change even after the team promotes a new version.
- B. `targets="label"` tells MLflow to add a column named `label` holding the model's predictions, which is then compared against the DataFrame index.
- C. The first argument resolves through the `@champion` alias to whichever version currently holds it, and one call logs the whole metric suite, including the four confusion-matrix cell counts, replacing roughly six manual `log_metric` calls.
- D. `model_type="classifier"` is optional metadata for the UI; the same metric suite is produced whether or not it is supplied.

---

### Question 4

This promotion job is meant to move the `@challenger` alias onto the single best eligible run. It runs with no error and reports success.

```python
RECALL_FLOOR, F1_FLOOR = 0.80, 0.42

runs = c.search_runs([exp.experiment_id], order_by=["metrics.f1 DESC"])

for run in runs:                                    # best F1 first
    recall = run.data.metrics.get("recall", 0)
    f1     = run.data.metrics.get("f1", 0)
    if recall >= RECALL_FLOOR and f1 >= F1_FLOOR:
        versions = c.search_model_versions(f"run_id='{run.info.run_id}'")
        v = max(versions, key=lambda m: int(m.version))
        c.set_registered_model_alias(MODEL_NAME, "challenger", v.version)
        c.set_model_version_tag(MODEL_NAME, v.version, "validation_status", "passed")
        print(f"Version {v.version} promoted to @challenger")
```

All three Cordwell runs clear both floors: F1 values are 0.4303, 0.4286, and 0.4286, with recall 0.8308 across all three. What is the defect and its consequence?

- A. There is no defect. Each assignment appends to the alias list, so `@challenger` ends up pointing at all three eligible versions and the serving layer picks the best.
- B. The loop is missing a `break`. An alias assignment is an overwrite, not an append, so `@challenger` ends on the last eligible version iterated, which is a lower-F1 model than the one the ordering was meant to select.
- C. The defect is `order_by=["metrics.f1 DESC"]`, which returns runs worst-first, so the first iteration already promotes the wrong version.
- D. The second and third `set_registered_model_alias` calls raise `MlflowException`, because an alias that is already assigned cannot be reassigned without first being deleted.

---

### Question 5

A team member finds a 2023 tutorial and proposes wiring the Cordwell promotion pipeline to `client.transition_model_version_stage(...)` with the `Production` and `Staging` stages. Based on this module, what is the correct response and the correct translation?

- A. Stages and aliases are equivalent naming choices, so either is fine as long as the team is consistent.
- B. Stages were removed in MLflow 3.x, so the tutorial code raises `AttributeError` and there is no migration path other than rewriting the pipeline.
- C. Registry stages have been deprecated since MLflow 2.9.0 and warn on every call while still functioning. The replacement is aliases: `set_registered_model_alias` in place of the stage transition, and a `models:/name@champion` URI in place of `models:/name/Production`.
- D. Stages should be kept for serving and aliases used only for experiment tracking, since the two systems address different layers.

---

*End of quiz. Five questions. Answer key is a separate file.*
