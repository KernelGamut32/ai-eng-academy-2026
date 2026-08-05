# Week 6 Lab 1.2 Hints: Progressive Tier

How to use this file: pick one tier per task, this one or `HINTS_DETAILED.md`, not both; reading both wastes time. Within this file, read one level at a time and go back to the notebook after each. Level 1 names the approach, level 2 sketches the structure, level 3 shows the key line or two in context. If level 3 is not enough, switch to the detailed tier for that task.

---

## Task 1: build_corpus

**Level 1.** Follow the contract's five steps in order; the order is the reproducibility contract, not a suggestion. One `random.Random(seed)` powers everything. Positives first (label 1 while the index is below the positive count), then negatives, then one shuffle at the very end, then the DataFrame.

**Level 2.** `n_pos = round(n_docs * positive_fraction)`. In the loop, per document: decide the label from the index, roll `kind = pick_kind(rng, label)`, write `text = make_ticket(rng, label, kind)`, append a dict with keys `"text"`, `"label"`, `"kind"`. After the loop, `rng.shuffle(rows)` (in place, using the same rng, not the `random` module) and return `pd.DataFrame(rows)`.

**Level 3.** The loop body and the two lines after it:

```python
for i in range(n_docs):
    label = 1 if i < n_pos else 0
    kind = pick_kind(rng, label)
    rows.append({"text": make_ticket(rng, label, kind), "label": label, "kind": kind})
rng.shuffle(rows)
return pd.DataFrame(rows)
```

What remains is the setup above the loop: the rng, `n_pos`, and the empty list. If the determinism check fails with this exact structure, look for a stray second Random instance or a call to `random.shuffle` instead of `rng.shuffle`.

## Task 2: split_tickets

**Level 1.** The morning's split, simplified: pull the two columns out as numpy arrays, then a single `train_test_split` call. No temporary pool, no second cut.

**Level 2.** `test_size=0.20`, `stratify=` the label array, `random_state=seed`. The call returns the four arrays in the exact order the contract lists.

**Level 3.**

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=seed
)
```

Above it: `X = df["text"].to_numpy()` and the same for `y` from `"label"`. Return all four.

## Task 3: fit_majority_baseline

**Level 1.** Construct one `DummyClassifier`, fit it, return it. The strategy name is in the task text.

**Level 2.** `DummyClassifier(strategy="most_frequent", random_state=seed)`, then `.fit(X_train, y_train)`, then return the fitted object (not the predictions; the apply cell predicts).

**Level 3.**

```python
baseline = DummyClassifier(strategy="most_frequent", random_state=seed)
baseline.fit(X_train, y_train)
```

Return `baseline`.

## Task 4: evaluate_predictions

**Level 1.** The morning's metrics function and confusion matrix, merged: build the four-key dict, build the matrix, return both as a tuple. Same conventions as the morning: round to 3 inside the function, `zero_division=0` on everything except accuracy, truth first into `confusion_matrix`.

**Level 2.** Dict keys exactly `"accuracy"`, `"precision"`, `"recall"`, `"f1"`, each value `round(<score>(y_true, y_pred, ...), 3)`. Then `cm = confusion_matrix(y_true, y_pred)`. Return `(metrics, cm)`.

**Level 3.** One dict entry as the model for the others, plus the matrix line:

```python
"recall": round(recall_score(y_true, y_pred, zero_division=0), 3),
```

```python
cm = confusion_matrix(y_true, y_pred)
```

Accuracy takes no `zero_division`. If the fixture check fails, hand-compute the fixture on paper first; it is six items.

## Task 5: build_and_fit_pipeline

**Level 1.** This is the morning's Task 2, verbatim, from a blank cell. Two named steps, one fit, return the pipeline. Try it without opening the morning notebook first; that is the exercise.

**Level 2.** Steps named `"tfidf"` and `"clf"`. Vectorizer takes `ngram_range=(1, 2)`; classifier takes `max_iter=1000` and `random_state=seed`. Fit the whole pipeline, not the steps.

**Level 3.**

```python
pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1, 2))),
    ("clf", LogisticRegression(max_iter=1000, random_state=seed)),
])
pipeline.fit(X_train, y_train)
```

Return it.

## Task 6: build_and_fit_balanced_pipeline

**Level 1.** Copy your Task 5 body and change exactly one thing on the classifier. The task text names the argument.

**Level 2.** The changed step is `LogisticRegression(max_iter=1000, random_state=seed, class_weight="balanced")`. Everything else, including the vectorizer settings, stays identical, so the comparison is attributable to the one change.

**Level 3.** The full changed line:

```python
("clf", LogisticRegression(max_iter=1000, random_state=seed, class_weight="balanced")),
```

Construct, fit, return, exactly as in Task 5.

## Task 7: build_comparison_table

**Level 1.** Loop over the (name, metrics) pairs, build one row dict per pair with `"model"` plus the four metric keys, and return a DataFrame with an explicit `columns=` list so the order is pinned.

**Level 2.** For each pair, start the row with `{"model": name}` and copy the four metric values across from the dict by key. Collect rows in a list; finish with `pd.DataFrame(rows, columns=["model", "accuracy", "precision", "recall", "f1"])`.

**Level 3.** The loop:

```python
for name, metrics in results:
    row = {"model": name}
    for key in ("accuracy", "precision", "recall", "f1"):
        row[key] = metrics[key]
    rows.append(row)
```

What remains is the empty list above and the DataFrame line below.

---

## Stretch goals

**Stretch 1.** Everything you need is Task 3 with a different strategy string plus your existing Task 4 and Task 7 functions. The `stratified` strategy really does use `random_state`, so pass `RANDOM_SEED` or your numbers will not reproduce. Before running, commit to a prediction on its accuracy versus the majority baseline.

**Stretch 2.** Three ingredients, all from the morning or the imports: `predict_proba(X_test)[:, 1]` for the scores, `average_precision_score(y_test, scores)` for the one-number ranking grade, `precision_recall_curve` for the plot. When the average precision prints, resist assuming you made an error; instead print the lowest score any true high-risk ticket received and the highest score any routine ticket received, and compare them. Then compare both to 0.5.

**Stretch 3.** The morning's stretch 3 class, `TunedThresholdClassifierCV`, wrapped around a fresh unfitted copy of the plain pipeline with `scoring="f1"` and `cv=5`. Fit on the training data, read `best_threshold_`, evaluate on the test set with your Task 4 function. Compare the chosen threshold against the two boundary scores you printed in stretch 2, and be ready to explain in one sentence why the test set still had no vote in choosing it.
