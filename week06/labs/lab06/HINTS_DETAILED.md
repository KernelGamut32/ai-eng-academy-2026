# HINTS_DETAILED.md: Detailed Tier

Pick one tier per task: this file shows the working core of each TODO with line by line commentary, `HINTS.md` nudges instead. Reading both wastes your time.

What "working core" means here: the lines that carry the concept are shown verbatim and explained. The function shell, the return assembly, and the glue between blocks are withheld, so finishing still requires reading and understanding rather than pasting. The instructor solution notebook stays withheld until after the lab.

---

## TODO 1: compute_generation_metrics

The core is three scoring blocks.

```python
bleu = sacrebleu.corpus_bleu(candidates, [references])
```

Candidates first. sacrebleu supports scoring against several reference sets at once, so the second argument is a list of reference lists; wrapping your single reference list in brackets is how you say "one reference per candidate". The result object exposes `.score`, already on the 0 to 100 scale.

```python
scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
rouge_vals = [
    scorer.score(ref, cand)["rougeL"].fmeasure
    for ref, cand in zip(references, candidates)
]
```

One scorer built once, reused per pair; building it inside the loop wastes work. `score(target, prediction)` takes the reference first, the reverse of sacrebleu, and returns a dict keyed by metric name whose values carry `.precision`, `.recall`, and `.fmeasure`. You want the fmeasure.

```python
token_f1_vals = [token_f1(ref, cand) for ref, cand in zip(references, candidates)]
```

The provided helper does the counting; you just map it over the pairs.

Withheld: the two input assertions, and assembling the return dict (three keys, each wrapped in `float`; the means come from `np.mean` over the two lists).

---

## TODO 2: run_classifier_experiment

The core is the logging sequence inside the run context.

```python
with mlflow.start_run(run_name=config["run_name"]) as run:
    pipeline, metrics, y_pred = train_classifier(config)
```

`start_run` as a context manager guarantees the run is closed even if something inside raises, so you never leave a half open run in the UI. Training happens inside the run so any failure is attributable to it.

```python
    mlflow.log_params({
        "model_type": "tfidf_logreg",
        "max_features": config["max_features"],
        "ngram_min": config["ngram_range"][0],
        "ngram_max": config["ngram_range"][1],
        ...
    })
```

Parameters are inputs you chose before the run. The ngram tuple becomes two scalar parameters because MLflow stores parameters as strings and two numbers filter better in the UI than one stringified tuple. The remaining entries mirror the contract list: the two C and max_iter values from `config`, the two split sizes from `len(X_train)` and `len(X_test)`, the seed, and the corpus version string.

```python
    mlflow.set_tags({
        "task": "intent_classification",
        "domain": "home_improvement_retail",
        "dataset": "cordwell_synth_v1",
        "owner": "student",
    })
    mlflow.log_metrics(metrics)
```

Tags are organizational labels, which is why they get their own call instead of riding along as parameters: the UI filter language addresses them as `tags.task`. The metrics dict comes back from the helper ready to log.

```python
    cm_path = save_confusion_matrix_png(y_test, y_pred, config["run_name"])
    mlflow.log_artifact(cm_path, artifact_path="plots")
```

The helper writes the PNG locally and returns its path; `log_artifact` uploads that file to the tracking server and files it under the `plots` folder of this run. The predictions CSV works identically with the other helper and the `results` folder.

```python
    mlflow.sklearn.log_model(pipeline, name="intent_classifier")
```

MLflow 3 syntax: the parameter is `name`, and `artifact_path` is deprecated. The logged model becomes its own object under the experiment's Models view (id starting `m-`), not a file in the run artifact tree, which is why the artifact browser will show only `plots` and `results`.

Withheld: the `require_mlflow()` guard, the print line, and returning `run.info.run_id`.

---

## TODO 3: run_generation_experiment

The core is the sample, answer, score sequence.

```python
sampled = sample_eval_rows(sample_size)
```

The provided helper always returns the same seeded rows, so every variant is scored on identical questions and differences between runs are attributable to the variant, not the sample. Rolling your own `df.sample` here breaks the paired comparison and the exact match check with it.

```python
with mlflow.start_run(run_name=f"gen_{variant_id}") as run:
    answers = [
        answer_for_row(row, variant_id)
        for row in sampled.to_dict("records")
    ]
```

`to_dict("records")` turns the frame into plain dicts, one per row, which is what the router expects. The router hides the backend decision: offline simulation or a live model, your harness code is identical either way. That indifference is the design lesson of the lab.

```python
    results_df = sampled.copy()
    results_df["model_answer"] = answers
    results_df["answer_tokens"] = [len(a.split()) for a in answers]

    metrics = compute_generation_metrics(
        results_df["reference_answer"].tolist(), answers
    )
    metrics["avg_answer_tokens"] = float(results_df["answer_tokens"].mean())
```

Copy before mutating so the shared sample stays clean. Whitespace split is a deliberately cheap token count; it feeds the length gate in the promotion policy later. The length metric joins the same dict so one `mlflow.log_metrics` call logs all four.

```python
    mlflow.log_params({
        "prompt_version": variant_id,
        "temperature": cfg["temperature"],
        "sample_size": len(results_df),
        "backend": BACKEND_MODE,
        "model_name": MODEL_NAME if BACKEND_MODE != "offline" else "offline_simulation",
        "corpus_version": "cordwell_synth_v1",
    })
```

`cfg` is the `PROMPT_VARIANTS[variant_id]` entry. `sample_size` logs the actual row count rather than the requested number, so the record stays honest if the request exceeded the corpus. The `model_name` conditional keeps offline runs from claiming a model they never called.

Withheld: the guard and assertion at the top, the tags call (identical shape to TODO 2 with `task` set to `answer_generation`), writing the CSV into `WORK_DIR` and logging it under `results`, the print, and the two element return.

---

## TODO 4: select_and_tag_champion

The core is the query and the gated loop.

```python
client = MlflowClient()
experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
runs = client.search_runs(
    experiment_ids=[experiment.experiment_id],
    filter_string="tags.task = 'answer_generation'",
    order_by=["metrics.rouge_l_f1 DESC"],
)
```

The fluent `mlflow.*` functions you used in TODOs 2 and 3 are for writing from inside a run; the client API is for reading and administering from outside. The filter string is the same language as the UI search box, which is the point: what you clicked in Part 5, you now script. Ordering server side means your loop meets the runs best first and needs no sorting of its own.

```python
for run in runs:
    passes, reasons = policy.evaluate(run.data.metrics)
    if passes and champion_run_id is None:
        champion_run_id = run.info.run_id
        client.set_tag(champion_run_id, "champion", "true")
        client.set_tag(champion_run_id, "promoted_date", date.today().isoformat())
        client.set_tag(champion_run_id, "promotion_policy", policy.name)
```

The `is None` guard is the whole promotion semantics: the first passer in rank order wins and nobody after it gets tagged, no matter how many also pass. Tag values are strings, hence `"true"` and the ISO date. Recording which policy promoted the run means the decision can be audited later without archaeology.

```python
    report_rows.append({
        "run_name": run.data.tags.get("mlflow.runName", run.info.run_id[:8]),
        ...
        "passes_policy": passes,
        "is_champion": run.info.run_id == champion_run_id,
        "reasons": "; ".join(reasons) if reasons else "all criteria met",
    })
```

Every run gets a row whether it passed or not; a promotion report that only shows the winner is useless in review. The run name lives in the reserved `mlflow.runName` tag, with the short id as fallback. The remaining columns pull each metric with `run.data.metrics.get(...)` per the contract list.

Withheld: the guard, the experiment assertion, initializing `champion_run_id` and `report_rows` before the loop, building the DataFrame, and the return.

---

## Stretch goals

The detailed tier covers stretch goals at the same depth in the instructor solution notebook's commentary; the traps to know before attempting:

1. **Fourth variant.** The simulator's fallback branch returns the reference nearly unchanged, so add a `friendly_v4` branch to `simulate_answer` before running it, or the variant scores close to perfect and the ranking lies. Keep a reference to the original function and delegate to it for the other variants.
2. **Semantic metric.** Fit one `TfidfVectorizer` on references plus candidates together, transform both, and take the cosine of each aligned pair only, not the full similarity matrix.
3. **Policy sweep.** Report, do not tag. Only the chosen policy promotes; a sweep that tags at every level manufactures conflicting champions.
