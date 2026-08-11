# HINTS.md: Progressive Tier

Three escalating levels per task. Level 1 names the approach, level 2 sketches the structure, level 3 shows the key line or two in context, never the whole solution.

**Pick one hint tier per task.** This file nudges; `HINTS_DETAILED.md` shows the working core with commentary. Reading both wastes time. Start here if you are close; go there if you are stuck on mechanics.

---

## Task 1: token_f1

**Level 1.** This is set arithmetic. Turn both strings into sets of unique tokens, measure the overlap, and compute precision and recall from the two set sizes. Two early exits return 0.0 before any division can go wrong.

**Level 2.** Four steps: build `pred_tokens` and `ref_tokens` with `set(tokenize(...))`; return 0.0 if either set is empty; compute the overlap size with `&`; return 0.0 if it is zero, otherwise precision is overlap over the prediction size, recall is overlap over the reference size, and the answer is their harmonic mean.

**Level 3.** The core arithmetic, after the guards:

```python
overlap = len(pred_tokens & ref_tokens)
precision = overlap / len(pred_tokens)
recall = overlap / len(ref_tokens)
```

The return line is the standard F1 formula on those two.

---

## Task 2: compute_text_metrics

**Level 1.** One library call handles BLEU for the whole batch at once; ROUGE and token F1 are scored pair by pair in a loop and averaged. The contract's two cautions (bracket shape for sacrebleu, argument order for rouge) are the entire difficulty.

**Level 2.** Structure: compute BLEU first from the full lists, remembering the divide by 100. Build one `RougeScorer` before the loop, not inside it. Loop with `zip(predictions, references)`, collecting `rougeL` fmeasure values and `token_f1` values into two lists. Average both with `np.mean`, wrap in `float(...)`, and return the three-key dict.

**Level 3.** The two lines people get wrong:

```python
bleu = sacrebleu.corpus_bleu(predictions, [references]).score / 100.0
rouge_result = scorer.score(ref, pred)  # reference first, prediction second
```

The fmeasure you want is `rouge_result["rougeL"].fmeasure`.

---

## Task 3: normalize_sentiment_label

**Level 1.** Lowercase once, then it is a substring scan over `VALID_LABELS` in order, with a fixed fallback when nothing matches.

**Level 2.** One `for label in VALID_LABELS:` loop containing one `if label in lowered:` test that returns the label; after the loop, return the fallback.

**Level 3.**

```python
for label in VALID_LABELS:
    if label in lowered:
        return label
```

---

## Task 4: compute_sentiment_metrics

**Level 1.** Two sklearn calls you used in Lab 1.2: one for accuracy, one that returns precision, recall, F1, and support in a single tuple. Mind the argument order sklearn uses: true labels first.

**Level 2.** `accuracy_score(references, predictions)` gives accuracy. `precision_recall_fscore_support(references, predictions, average="macro", zero_division=0)` returns four values; you need the first three. Wrap everything in `float(...)` and assemble the four-key dict.

**Level 3.**

```python
precision, recall, f1, _ = precision_recall_fscore_support(
    references, predictions, average="macro", zero_division=0
)
```

---

## Task 5: evaluate_variant

**Level 1.** An outer loop over the three task types, an inner loop over that task's rows, then one scoring call per task. The only real branching is sentiment versus the two text tasks, and it happens twice: once when extracting prediction and reference from a row, once when choosing the metric function.

**Level 2.** Set up `metrics = {}` and `prediction_rows = []`. Define the plan as a list of `(task_type, limit)` pairs in the contract's order. Per task: skip if the limit is 0, slice with `eval_df[eval_df["task_type"] == task_type].head(limit)`, and iterate with `.iterrows()`. Per row: call `run_cordwell_app`, then branch: sentiment normalizes the answer and takes `row["sentiment_label"]` as reference; the others use the raw answer and `row["reference_text"]`. Append to the predictions and references lists and to `prediction_rows`. After the row loop, call the matching metric function and copy its entries into `metrics` with the `f"{task_type}_{key}"` prefix.

**Level 3.** The prefixing step at the end of each task:

```python
for key, value in task_metrics.items():
    metrics[f"{task_type}_{key}"] = value
```

and the sentiment branch inside the row loop:

```python
if task_type == "sentiment":
    prediction = normalize_sentiment_label(raw_answer)
    reference = row["sentiment_label"]
```

---

## Task 6: log_variant_to_mlflow

**Level 1.** One `with mlflow.start_run(...) as run:` block containing three phases: params, metrics, artifact. The run id you must return lives on the `run` object the context manager gives you.

**Level 2.** Inside the block: five `mlflow.log_param` calls (two values come from the globals `BACKEND_MODE` and `MODEL_NAME`); a loop over `result["metrics"].items()` calling `mlflow.log_metric`; then build the CSV path as `LAB_DIR / f"{result['variant_name']}_predictions.csv"`, write it with `.to_csv(csv_path, index=False)`, and log it with `mlflow.log_artifact(str(csv_path), artifact_path="predictions")`. Return `run.info.run_id` from inside the block.

**Level 3.**

```python
with mlflow.start_run(run_name=result["variant_name"]) as run:
    ...
    mlflow.log_artifact(str(csv_path), artifact_path="predictions")
    return run.info.run_id
```

---

## Task 7: the two lexical judges

**Level 1.** Both are the given `lexical_context_relevance` with a different denominator, which is the point of the task. Groundedness has one extra wrinkle: its source can arrive as a list.

**Level 2.** For groundedness: if `source` is a list or tuple, join it with spaces into one string first. Then `content_tokens` on both sides, an empty-statement guard returning 0.0, and overlap divided by the statement's token count. Answer relevance is the same shape with prompt and response, dividing by the prompt's token count, no join needed.

**Level 3.** The list handling and the denominator that defines groundedness:

```python
if isinstance(source, (list, tuple)):
    source = " ".join(source)
...
return len(statement_tokens & source_tokens) / len(statement_tokens)
```

---

## Task 8: build_trulens_metrics

**Level 1.** Copy the shape of the given context relevance metric twice and change three things each time: the implementation function, the name string, and the Selector dicts per the table in the task. Neither new metric needs `.aggregate`.

**Level 2.** Groundedness: `Metric(implementation=lexical_groundedness, name="Groundedness (lexical)")`, then one `.on` mapping `"statement"` to a RECORD_ROOT Selector with attribute `RECORD_ROOT.OUTPUT`, and one `.on` mapping `"source"` to a RETRIEVAL Selector with attribute `RETRIEVAL.RETRIEVED_CONTEXTS` and `collect_list=True`. Answer relevance: same pattern with `"prompt"` on `RECORD_ROOT.INPUT` and `"response"` on `RECORD_ROOT.OUTPUT`. The dict keys must exactly match your Task 7 parameter names. Remember to remove the `None` placeholders.

**Level 3.** One complete Selector binding to pattern-match against:

```python
.on(
    {
        "source": Selector(
            span_type=SpanAttributes.SpanType.RETRIEVAL,
            span_attribute=SpanAttributes.RETRIEVAL.RETRIEVED_CONTEXTS,
            collect_list=True,
        )
    }
)
```

---

## Task 9: attach_trulens_metrics_to_mlflow

**Level 1.** Three lookups then a loop: find the leaderboard row for the app version, find the MLflow run by name, then copy each metric across with a sanitized key. Each lookup raises `ValueError` when it finds nothing.

**Level 2.** Leaderboard side: `session.get_leaderboard().reset_index()` makes `app_version` a filterable column; filter, check `.empty`, take `.iloc[0]`. MLflow side: get the experiment by `EXPERIMENT_NAME`, then `client.search_runs` with the runName filter string from the contract plus `order_by=["attributes.start_time DESC"]` and `max_results=1`. Loop over `METRIC_NAMES`, skip missing or null values with `pd.notna`, build the key with one `re.sub`, and call `client.log_metric(run_id, key, float(value))` while collecting into the dict you return.

**Level 3.** The key sanitizer and the logging call:

```python
key = "trulens_" + re.sub(r"[^a-z0-9]+", "_", metric_name.lower()).strip("_")
client.log_metric(run_id, key, float(row[metric_name]))
```

---

## Stretch goals

**Stretch 1, level 1.** No new code. Call the four functions you already built, in the order the task lists, with variant B's arguments. The only thinking is `expected_total` for the TruLens wait: it counts all records in the session, both versions.

**Stretch 2, level 1.** The hedge detector is a three-line substring scan returning 1.0 or 0.0. The Metric wraps it with a single `.on` binding `"response"` to the RECORD_ROOT output. The subtle part is the wait: filter records to the new app version before checking completeness, because older records will never grow a Hedge Rate score.

**Stretch 3, level 1.** Live backend only. Build `TruOpenAI(model_engine=MODEL_NAME)` from `trulens.providers.openai`, then reuse your Task 8 Selectors verbatim with `provider.context_relevance`, `provider.groundedness_measure_with_cot_reasons`, and `provider.relevance` as the implementations. Expect minutes, not seconds, and scores that vary run to run.
