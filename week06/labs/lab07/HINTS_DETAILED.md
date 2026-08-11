# HINTS_DETAILED.md: Detailed Tier

This tier shows the **working core** of each task verbatim, with line-by-line commentary explaining why each line is there. It withholds the function shells, return assembly, and glue, so finishing a task still means reading, understanding, and completing the code rather than pasting a function.

**Pick one hint tier per task.** If a nudge is all you need, use `HINTS.md` instead; reading both wastes time. The instructor solution notebook remains the only fully assembled, executed artifact.

---

## Task 1: token_f1

The core, minus the function shell and the final return:

```python
pred_tokens = set(tokenize(prediction))
ref_tokens = set(tokenize(reference))
```

`tokenize` gives a list; wrapping in `set` collapses duplicates because this metric is about vocabulary coverage, not counts. Saying "return" five times should not score five times.

```python
if not pred_tokens or not ref_tokens:
    return 0.0
```

Empty sets would make a later division blow up, and semantically an empty prediction earned a zero anyway. Guard first, divide later.

```python
overlap = len(pred_tokens & ref_tokens)
if overlap == 0:
    return 0.0
```

`&` is set intersection. The second guard matters for the same reason: with zero overlap, precision and recall are both 0 and the harmonic mean formula would divide 0 by 0.

```python
precision = overlap / len(pred_tokens)
recall = overlap / len(ref_tokens)
```

Precision reads "of what the prediction said, how much was in the reference." Recall reads "of what the reference said, how much did the prediction cover." Same overlap, two denominators, two questions.

You write: the F1 return line combining the two (the standard formula), and the `def` line with its docstring.

---

## Task 2: compute_text_metrics

```python
bleu = sacrebleu.corpus_bleu(predictions, [references]).score / 100.0
```

Three details packed in one line. `corpus_bleu` scores the whole batch at once because BLEU is defined over a corpus, not averaged per sentence. The second argument is a list OF reference lists, because BLEU supports multiple references per prediction; we have one reference set, hence exactly one pair of extra brackets. And `.score` is 0 to 100, so dividing by 100 puts it on the same 0 to 1 scale as every other number in this lab.

```python
scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
```

Built once, outside the loop. It is configuration, not per-pair state. `use_stemmer=True` lets "returned" match "return", which is also a preview of what Task 7's judges cannot do. Pinning this configuration is why our ROUGE numbers are comparable run to run; ROUGE numbers from different configurations are not comparable at all.

```python
for pred, ref in zip(predictions, references):
    rouge_result = scorer.score(ref, pred)  # reference first, prediction second
    rouge_vals.append(rouge_result["rougeL"].fmeasure)
    f1_vals.append(token_f1(pred, ref))
```

The comment is the whole trap: `scorer.score(target, prediction)` with the reference first, the reverse of sacrebleu's order and of the mental order most people type. Swapped arguments produce a plausible wrong number with no error. `rouge_result` is a dict keyed by metric name; each value carries precision, recall, and `fmeasure`, and fmeasure is the one we report.

You write: initializing the two accumulator lists, the `np.mean` averaging wrapped in `float(...)`, and the three-key return dict.

---

## Task 3: normalize_sentiment_label

```python
lowered = raw_answer.lower()
for label in VALID_LABELS:
    if label in lowered:
        return label
```

Lowercase once so `NEGATIVE`, `Negative`, and `negative` all match. The loop order is `VALID_LABELS` order, which makes behavior deterministic when a rambling reply mentions two labels: the first in tuple order wins. Substring matching (`in`) is deliberately forgiving because real model output wraps the label in punctuation and prose.

You write: the fallback return after the loop (the contract names it) and the function shell.

---

## Task 4: compute_sentiment_metrics

```python
accuracy = accuracy_score(references, predictions)
precision, recall, f1, _ = precision_recall_fscore_support(
    references, predictions, average="macro", zero_division=0
)
```

Both sklearn calls take true labels first, predictions second: the reverse convention from rouge, which is exactly why every contract in this lab spells the order out. `average="macro"` computes each class's score independently and averages them unweighted, so the small positive class counts as much as the larger ones; with imbalanced labels, macro is the honest choice. `zero_division=0` covers the edge where a model never predicts some class: score it 0 quietly instead of raising a warning mid-harness. The trailing `_` discards per-class support counts we do not need.

You write: the four-key return dict with each value wrapped in `float(...)` so downstream logging never sees a numpy scalar.

---

## Task 5: evaluate_variant

```python
task_plan = [
    ("faq", n_faq),
    ("summarization", n_summ),
    ("sentiment", n_sent),
]
```

The plan as data. The contract fixes this order, and keeping it as a list of pairs means one loop instead of three copies of the same code.

```python
for task_type, limit in task_plan:
    if limit == 0:
        continue
    subset = eval_df[eval_df["task_type"] == task_type].head(limit)
```

The zero-limit skip lets a live-backend user shrink or drop a task without editing the loop. `.head(limit)` takes the first rows deterministically; a random sample would break the locked check numbers.

```python
for _, row in subset.iterrows():
    result = run_cordwell_app(
        row["input_text"], task_type, top_k=top_k, prompt_style=prompt_style
    )
    raw_answer = result["answer"]
    if task_type == "sentiment":
        prediction = normalize_sentiment_label(raw_answer)
        reference = row["sentiment_label"]
    else:
        prediction = raw_answer
        reference = row["reference_text"]
```

One row, one app call, then the branch that gives each task type its own idea of prediction and reference. Sentiment predictions go through Task 3's normalizer immediately, so by the time anything is scored or saved, the label space is already clean.

```python
prediction_rows.append(
    {
        "variant_name": variant_name,
        "task_type": task_type,
        "input_text": row["input_text"],
        "reference": reference,
        "prediction": prediction,
    }
)
```

The per-example record, built in the same breath as the metrics inputs. This list becomes the predictions DataFrame, which becomes the MLflow artifact, which is what lets you later answer "which exact examples got worse."

```python
if task_type == "sentiment":
    task_metrics = compute_sentiment_metrics(predictions, references)
else:
    task_metrics = compute_text_metrics(predictions, references)
for key, value in task_metrics.items():
    metrics[f"{task_type}_{key}"] = value
```

After each task's rows: route to the right scorer, then flatten into one namespace with the task prefix. `faq_bleu` and `summarization_bleu` are different numbers about different data; the prefix keeps them from colliding in one flat dict, which is the shape MLflow logging wants.

You write: the two accumulator initializations at the top, the per-task `predictions` and `references` lists, and the final three-key return dict (the contract lists its keys, including wrapping `prediction_rows` in `pd.DataFrame`).

---

## Task 6: log_variant_to_mlflow

```python
with mlflow.start_run(run_name=result["variant_name"]) as run:
```

The context manager opens the run and guarantees it closes even if something inside raises, so you never leave a zombie run marked RUNNING in the UI. `run_name` is what you will read in the UI list, and it is also what Task 9 searches on.

```python
mlflow.log_param("variant_name", result["variant_name"])
mlflow.log_param("prompt_style", prompt_style)
mlflow.log_param("top_k", top_k)
mlflow.log_param("backend_mode", BACKEND_MODE)
mlflow.log_param("model_name", MODEL_NAME)
```

Params are the experiment's identity: everything someone would need to reproduce the run. Logging `backend_mode` and `model_name` is what stops you from comparing an offline-stub run against a live Gemma run six weeks from now without noticing.

```python
for key, value in result["metrics"].items():
    mlflow.log_metric(key, value)
```

The harness already flattened metrics into one dict of floats, so logging is a two-line loop. That was the reason for Task 5's prefixing decision.

```python
csv_path = LAB_DIR / f"{result['variant_name']}_predictions.csv"
result["predictions"].to_csv(csv_path, index=False)
mlflow.log_artifact(str(csv_path), artifact_path="predictions")
```

Artifacts are files: write locally, then upload. `index=False` keeps pandas from adding a meaningless row-number column. `log_artifact` wants a string path, hence `str(...)` around the Path. `artifact_path="predictions"` files it under a folder in the run instead of the root.

You write: the return of `run.info.run_id`, placed inside the `with` block, plus the shell.

---

## Task 7: the two lexical judges

Groundedness core:

```python
if isinstance(source, (list, tuple)):
    source = " ".join(source)
```

The Selector you will wire in Task 8 uses `collect_list=True`, which delivers all retrieved contexts as one list. Joining with spaces turns them into a single evidence pool; a claim supported by any document counts as supported.

```python
statement_tokens = content_tokens(statement)
if not statement_tokens:
    return 0.0
source_tokens = content_tokens(source)
return len(statement_tokens & source_tokens) / len(statement_tokens)
```

The denominator is the statement, and that IS the metric: "of what the answer claimed, how much appears in the evidence." Divide by the source instead and you would be measuring how much of the evidence got quoted, a completely different (and mostly useless) question. `content_tokens` strips stopwords first so "the" appearing in both sides earns no credit.

Answer relevance core:

```python
prompt_tokens = content_tokens(prompt)
if not prompt_tokens:
    return 0.0
response_tokens = content_tokens(response)
return len(prompt_tokens & response_tokens) / len(prompt_tokens)
```

Same skeleton, denominator flipped to the prompt: "of what was asked, how much did the answer engage with." Between the two functions only the denominator and the absence of the list-join differ, which is precisely the lesson.

You write: both function shells with docstrings. The cores above are complete.

---

## Task 8: build_trulens_metrics

Groundedness, fully wired:

```python
m_groundedness = (
    Metric(implementation=lexical_groundedness, name="Groundedness (lexical)")
    .on(
        {
            "statement": Selector(
                span_type=SpanAttributes.SpanType.RECORD_ROOT,
                span_attribute=SpanAttributes.RECORD_ROOT.OUTPUT,
            )
        }
    )
    .on(
        {
            "source": Selector(
                span_type=SpanAttributes.SpanType.RETRIEVAL,
                span_attribute=SpanAttributes.RETRIEVAL.RETRIEVED_CONTEXTS,
                collect_list=True,
            )
        }
    )
)
```

Line by line: `Metric(...)` binds your Task 7 function as the implementation and fixes the display name the leaderboard and the checks both expect, character for character. The first `.on` says "when scoring a record, take the RECORD_ROOT span's OUTPUT (the final answer) and pass it as the `statement` argument." The dict key must match the parameter name in your function signature exactly; a typo here fails at evaluation time, not at wiring time. The second `.on` feeds `source` from the RETRIEVAL span's context list, and `collect_list=True` means one call with the whole list, which is why your Task 7 groundedness accepts a list. No `.aggregate`, because one call produces one score per record already. Contrast with the given context relevance metric: `collect_list=False` fans out to one call per context, producing several scores that `.aggregate(np.mean)` must then fold into one.

You write: `m_answer_relevance` following the identical pattern with the table's other two bindings (`prompt` from RECORD_ROOT INPUT, `response` from RECORD_ROOT OUTPUT, no `collect_list` needed since neither attribute is fanned out), and removal of the two `None` placeholders so the guard stops raising.

---

## Task 9: attach_trulens_metrics_to_mlflow

```python
lb = session.get_leaderboard().reset_index()
match = lb[lb["app_version"] == app_version]
if match.empty:
    raise ValueError(f"No TruLens leaderboard row for app_version={app_version!r}")
row = match.iloc[0]
```

The leaderboard arrives indexed by app name and version; `reset_index()` demotes those to ordinary columns so a boolean filter works. Filtering by `app_version` is the durable join key. The old lab this one replaces selected rows by position ("the first five records are variant A"), which breaks the moment anyone re-runs a cell; matching on the version string cannot break that way. The empty check raises loudly because silently logging nothing is how dashboards lie.

```python
runs = client.search_runs(
    [experiment.experiment_id],
    filter_string=f"tags.mlflow.runName = '{run_name}'",
    order_by=["attributes.start_time DESC"],
    max_results=1,
)
```

MLflow stores the run name as a tag, so the filter string queries `tags.mlflow.runName`. Re-running the notebook creates a second run with the same name; sorting by start time descending with `max_results=1` deterministically picks the newest, so the join always lands on the run you just logged.

```python
for metric_name in METRIC_NAMES:
    if metric_name in row and pd.notna(row[metric_name]):
        key = "trulens_" + re.sub(r"[^a-z0-9]+", "_", metric_name.lower()).strip("_")
        client.log_metric(run_id, key, float(row[metric_name]))
```

`pd.notna` skips scores that never landed rather than logging NaN. The `re.sub` collapses every run of non-alphanumeric characters to one underscore, so `"Groundedness (lexical)"` becomes `groundedness_lexical` and, with the prefix, `trulens_groundedness_lexical`: a key MLflow accepts and a namespace that keeps structured scores visually separate from the harness metrics on the run page. `float(...)` because the leaderboard hands back numpy scalars.

You write: fetching the experiment with its own `ValueError` guard, extracting `run_id` from `runs[0].info.run_id`, the `logged` dict accumulation, and the return.

---

## Stretch 1: the A-B run end to end

The whole solution is four calls you already own, in order:

```python
variant_b_result = evaluate_variant("prompt_grounded_v2", prompt_style="strong_grounding", top_k=2)
variant_b_run_id = log_variant_to_mlflow(variant_b_result, prompt_style="strong_grounding", top_k=2)
records_df, feedback_cols = run_trulens_eval(
    "prompt_grounded_v2", top_k=2, prompt_style="strong_grounding",
    expected_total=2 * len(TRULENS_QUERIES),
)
joined_b = attach_trulens_metrics_to_mlflow("prompt_grounded_v2", "prompt_grounded_v2")
```

The one thing to reason about is `expected_total`: `wait_for_trulens` counts every record in the session, and the baseline's six are already there, so waiting for six again would return instantly with variant B still pending. Twelve is the number. When you read the comparison, trace both leaderboard deltas (groundedness down, answer relevance up) to the single hedge sentence the stub appends under strong grounding; the walkthrough's Stretch 1 note has the full reading if yours differs.

## Stretch 2: the Hedge Rate metric

Detector and wiring core:

```python
def hedge_detector(response: str) -> float:
    lowered = response.lower()
    hedge_markers = ["do not know", "check with an associate", "not sure"]
    return 1.0 if any(marker in lowered for marker in hedge_markers) else 0.0
```

A judge is any function from span evidence to a score; this one is three lines and binary. Wrap it exactly like the answer relevance single-input pattern, one `.on` binding `"response"` to RECORD_ROOT OUTPUT, inside a `build_hedge_metric()` factory so each TruApp gets a fresh object.

The waiting core, which is the actual lesson of this stretch:

```python
version_rows = records_df[records_df["app_version"] == version_name]
done = (
    len(version_rows) >= len(TRULENS_QUERIES)
    and "Hedge Rate" in feedback_cols
    and version_rows[hedge_metric_names].notna().all(axis=1).all()
)
```

Filter to the new version's records before checking completeness. The earlier versions in the session were recorded without a Hedge Rate metric; their rows will show NaN in that column forever, so the generic `wait_for_trulens` would poll until timeout. Waiting on the right subset is the difference between a 15-second cell and a 5-minute one.

You write: the loop over the two prompt styles creating fresh apps and TruApps per version, and the final groupby that averages Hedge Rate per version.

## Stretch 3: stock LLM judges

Live backend only. The full code ships commented in the solution notebook's Appendix A; the shape is your Task 8 factory with three substitutions:

```python
from trulens.providers.openai import OpenAI as TruOpenAI
provider = TruOpenAI(model_engine=MODEL_NAME)
```

then `provider.context_relevance`, `provider.groundedness_measure_with_cot_reasons`, and `provider.relevance` as the `implementation=` arguments, with your Selectors unchanged. The provider reads `OPENAI_BASE_URL` and `OPENAI_API_KEY` from the environment, so point those at the live backend first. Budget minutes per version and expect scores to vary between runs; the comparison against your lexical numbers on the same six queries, especially where they disagree, is the deliverable.
