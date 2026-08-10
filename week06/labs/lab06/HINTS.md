# HINTS.md: Progressive Tier

Pick one tier per task and stick with it: this file nudges, `HINTS_DETAILED.md` shows the working core with commentary. Reading both wastes your time. Within this file, read level 1 first and only go deeper if you are still stuck after a real attempt.

---

## TODO 1: compute_generation_metrics

**Level 1 (approach).** Three metrics, three tools, all shown working in the cell above the stub. BLEU is one call over the whole batch. ROUGE-L and token F1 are per pair scores you average with `np.mean`. Assemble a dict with the three required keys and float values.

**Level 2 (structure).** One `sacrebleu.corpus_bleu` call taking the candidate list and the reference list, in that order, and note how the reference argument has to be wrapped. Then create one `RougeScorer` for `rougeL` with stemming, loop over `zip(references, candidates)` calling its `score` method, and collect each `.fmeasure`. A second loop (or the same one) collects `token_f1` values. Average both lists.

**Level 3 (key lines).** The two calls the gotchas live in, exactly as they appear in context:

```python
bleu = sacrebleu.corpus_bleu(candidates, [references])
...
scorer.score(ref, cand)["rougeL"].fmeasure
```

`bleu.score` is already on the 0 to 100 scale; do not rescale it.

---

## TODO 2: run_classifier_experiment

**Level 1 (approach).** Everything model related is done for you by `train_classifier`. Your function is a wrapper: open a run, call the helper, then log each category from the Part 3 table (parameters, tags, metrics, two artifacts, model), and return the run id.

**Level 2 (structure).** `require_mlflow()`, then `with mlflow.start_run(run_name=...) as run:`. Inside: one `mlflow.log_params` call with a dict built from `config` plus the fixed values (train and test sizes come from `len(X_train)` and `len(X_test)`); one `mlflow.set_tags` call; one `mlflow.log_metrics` call with the dict `train_classifier` returned. The two artifact helpers each return a file path; pass each path to `mlflow.log_artifact` with the right `artifact_path`. Log the model last. Return `run.info.run_id`.

**Level 3 (key lines).** The two calls people get wrong, in context:

```python
mlflow.log_artifact(cm_path, artifact_path="plots")
...
mlflow.sklearn.log_model(pipeline, name="intent_classifier")
```

The model call uses `name`, not `artifact_path`; the latter is deprecated in MLflow 3. The ngram parameters are logged as two scalars, `ngram_min` and `ngram_max`, pulled from the tuple by index.

---

## TODO 3: run_generation_experiment

**Level 1 (approach).** Same logging discipline as TODO 2, different content. Sample once with the provided helper, answer every row with the provided router, score the batch with your TODO 1 function, add the length metric, log everything, write the results frame to CSV and log it as an artifact, return both the run id and the frame.

**Level 2 (structure).** Sample before opening the run. Inside the run: build the answers list by looping `sampled.to_dict("records")` through `answer_for_row`. Copy the sampled frame, add `model_answer` and `answer_tokens` (split on whitespace and count). Call `compute_generation_metrics` with the reference column as a list and the answers, then add `avg_answer_tokens` to that same dict before `mlflow.log_metrics`. Parameters come from `PROMPT_VARIANTS[variant_id]` plus the module level config values; note the `model_name` value depends on whether `BACKEND_MODE` is offline. Write the CSV into `WORK_DIR` and log it under `results`.

**Level 3 (key lines).** The sampling and answering core, in context:

```python
sampled = sample_eval_rows(sample_size)
...
answers = [answer_for_row(row, variant_id) for row in sampled.to_dict("records")]
```

Using `sample_eval_rows` (not your own `df.sample`) is what makes every variant score the same rows; the exact match check depends on it. Log `sample_size` as the actual row count, `len(results_df)`, in case the requested size exceeded the corpus.

---

## TODO 4: select_and_tag_champion

**Level 1 (approach).** Query, rank, gate, tag, report. The client query can do the filtering and the ranking for you, so your loop only applies the policy to each run in order, remembers the first passer, and collects report rows.

**Level 2 (structure).** `MlflowClient()`, `get_experiment_by_name(EXPERIMENT_NAME)`, then `search_runs` with the experiment id in a list, a `filter_string` selecting the generation task tag, and `order_by` on the rouge metric descending. Loop the returned runs: `policy.evaluate(run.data.metrics)` gives `(passes, reasons)`. The first passer gets three `client.set_tag` calls and becomes the remembered champion; every run contributes one report row. Metrics for the report come from `run.data.metrics.get(...)`, the display name from the `mlflow.runName` tag.

**Level 3 (key lines).** The query, in context:

```python
runs = client.search_runs(
    experiment_ids=[experiment.experiment_id],
    filter_string="tags.task = 'answer_generation'",
    order_by=["metrics.rouge_l_f1 DESC"],
)
```

Guard the champion assignment with `champion_run_id is None` so only the first passer is tagged. `date.today().isoformat()` gives the promoted date.
