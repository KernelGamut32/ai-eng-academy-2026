# Week 6 Lab 1.3 Hints: Progressive Tier

How to use this file: pick one tier per task, this one or `HINTS_DETAILED.md`, not both; reading both wastes time. Within this file, read one level at a time and go back to the notebook after each. Level 1 names the approach, level 2 sketches the structure, level 3 shows the key line or two in context. If level 3 is not enough, switch to the detailed tier for that task.

---

## Task 1: split_corpus_three_way

**Level 1.** Lab 1.1's two-call pattern at new proportions: peel off 40 percent as a pool, then halve the pool. Columns out as numpy arrays first.

**Level 2.** First call: `test_size=0.40`, `stratify=y`, `random_state=seed`, keeping the 60 percent side as train. Second call cuts the pool with `test_size=0.50`, and it stratifies on the pool's own labels, not the originals. Return the six arrays in the contract's order.

**Level 3.** The second call, the one people get wrong:

```python
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=seed
)
```

The first call has the same shape with `X, y` and `test_size=0.40`. Return order: `X_train, X_val, X_test, y_train, y_val, y_test`.

## Task 2: build_and_fit_pipeline

**Level 1.** Third build of the day, same contract: two named steps, one fit, return the pipeline. Try it from memory before opening anything.

**Level 2.** Steps `"tfidf"` and `"clf"`; `ngram_range=(1, 2)` on the vectorizer; `max_iter=1000` and `random_state=seed` on the classifier. Fit the pipeline object, not the steps. Note this lab's seed is 11, not the morning's 7; use the `seed` parameter, never a typed number.

**Level 3.**

```python
pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1, 2))),
    ("clf", LogisticRegression(max_iter=1000, random_state=seed)),
])
pipeline.fit(X_train, y_train)
```

Return it.

## Task 3: evaluate_at_threshold

**Level 1.** Four steps straight from the contract: slice the positive-class probabilities, compare against the threshold, build the four-key metrics dict with the day's conventions, build the confusion matrix. Return the dict and the matrix as a tuple.

**Level 2.** The probabilities are `predict_proba`'s column 1. The comparison `(proba >= threshold)` gives booleans; `.astype(int)` makes labels. The dict uses the exact keys `"accuracy"`, `"precision"`, `"recall"`, `"f1"`, each rounded to 3, with `zero_division=0` on all but accuracy. `confusion_matrix` takes truth first.

**Level 3.** The two lines that make this function different from Lab 1.2's evaluator:

```python
proba = pipeline.predict_proba(X)[:, 1]
y_pred = (proba >= threshold).astype(int)
```

Everything after them is the metrics dict and matrix you have now written twice today.

## Task 4: threshold_sweep

**Level 1.** A loop around your Task 3: one call per threshold, one row dict per call, DataFrame at the end with a pinned column list. Handle the default thresholds inside the function.

**Level 2.** The default is `np.round(np.linspace(0.05, 0.95, 19), 2)`, built only when the argument is None, and the `np.round` is required, not decorative: unrounded linspace values fail exact lookups later. Each row is `{"threshold": threshold, **metrics}` where metrics comes from `evaluate_at_threshold` (ignore the matrix it also returns). Finish with the five-column `columns=` list from the contract.

**Level 3.**

```python
if thresholds is None:
    thresholds = np.round(np.linspace(0.05, 0.95, 19), 2)
for threshold in thresholds:
    metrics, _ = evaluate_at_threshold(pipeline, X, y, threshold)
    rows.append({"threshold": threshold, **metrics})
```

What remains: the empty list and `pd.DataFrame(rows, columns=["threshold", "accuracy", "precision", "recall", "f1"])`.

## Task 5: select_thresholds

**Level 1.** Three lookups on the sweep table. The F1 pick is an argmax over the whole table. Each stakeholder pick is a two-step: filter the rows by their floor, then argmax their preference inside the filtered rows. Wrap every result in `float(...)`.

**Level 2.** `idxmax` on a column gives the index of its (first) largest value; `.loc[that_index, "threshold"]` reads the threshold there. For the safety team, filter `sweep_df[sweep_df["recall"] >= recall_floor]` and take `idxmax` of precision inside it. The call center is the mirror image with the precision floor and recall's argmax. Return the three-key dict from the contract.

**Level 3.** One stakeholder selection in full; the other is its mirror:

```python
safety_rows = sweep_df[sweep_df["recall"] >= recall_floor]
safety_team = float(safety_rows.loc[safety_rows["precision"].idxmax(), "threshold"])
```

The F1 pick is the same pattern with no filter: `sweep_df.loc[sweep_df["f1"].idxmax(), "threshold"]`.

## Task 6: build_operating_points_table

**Level 1.** Lab 1.2's comparison-table pattern with your evaluator folded in: loop the (name, threshold) pairs, call Task 3 per pair, collect rows, return a DataFrame with the pinned six-column list.

**Level 2.** Each row is `{"operating_point": name, "threshold": threshold, **metrics}`. The matrix from `evaluate_at_threshold` is not needed here; unpack it into `_`. Rows keep the order given; the `columns=` list pins the schema.

**Level 3.**

```python
for name, threshold in named_thresholds:
    metrics, _ = evaluate_at_threshold(pipeline, X, y, threshold)
    rows.append({"operating_point": name, "threshold": threshold, **metrics})
```

What remains: the list, and `pd.DataFrame(rows, columns=["operating_point", "threshold", "accuracy", "precision", "recall", "f1"])`.

## Task 7: final_test_report

**Level 1.** Task 3 called twice at the same locked threshold, once per split, two labeled rows, validation first, DataFrame with the pinned column list.

**Level 2.** Iterate over the two (name, X, y) triples, `("validation", X_val, y_val)` then `("test", X_test, y_test)`; each row is `{"split": split, "threshold": threshold, **metrics}`.

**Level 3.**

```python
for split, Xs, ys in (("validation", X_val, y_val), ("test", X_test, y_test)):
    metrics, _ = evaluate_at_threshold(pipeline, Xs, ys, threshold)
    rows.append({"split": split, "threshold": threshold, **metrics})
```

What remains: the list, and the DataFrame with `columns=["split", "threshold", "accuracy", "precision", "recall", "f1"]`.

---

## Stretch goals

**Stretch 1.** Both classes appeared in Lab 1.1's stretch and the imports are named in the prompt. Freeze the already fitted `alert_pipeline` first, wrap the frozen object with the locked threshold from `picks`, make the API-required `fit` call (it retrains nothing), predict on the test set, and compare against `final_report`'s test row. They should match exactly; if they do not, check which pipeline you froze and which threshold you passed.

**Stretch 2.** Three ingredients: the validation probabilities you know how to slice, `average_precision_score` for the one-number grade, `precision_recall_curve` for the plot (it returns three arrays; step-plot recall against precision). To mark your operating points on the curve, get each point's precision and recall from your own `evaluate_at_threshold`. When comparing to Lab 1.2's perfect 1.000, print the lowest score any true urgent ticket received and the highest score any routine ticket received, and let those two numbers write your explanation.

**Stretch 3.** For each threshold in your sweep, you need the confusion matrix, which `evaluate_at_threshold` already returns; `cm[0, 1]` is false alarms and `cm[1, 0]` is misses. Expected cost is one multiplication and one addition per row. Collect a cost table, `idxmax`'s sibling `idxmin` finds the winner, and then price your three Task 5 picks by looking their thresholds up in the table. Before you run it, commit to a prediction: which stakeholder does an 8-to-400 cost ratio favor, and why?
