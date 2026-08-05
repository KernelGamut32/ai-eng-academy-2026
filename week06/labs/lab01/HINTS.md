# Week 6 Lab 01 Hints: Progressive Tier

How to use this file: pick one tier per task, this one or `HINTS_DETAILED.md`, not both; reading both wastes time. Within this file, read one level at a time and go back to the notebook after each. Level 1 names the approach, level 2 sketches the structure, level 3 shows the key line or two in context. If level 3 is not enough, switch to the detailed tier for that task.

---

## Task 1: split_corpus

**Level 1.** `train_test_split` cuts one dataset into exactly two pieces, so getting three pieces takes two calls. First separate 70 percent train from a 30 percent pool, then cut the pool in half. Pull the columns out first with `df["text"].to_numpy()` and `df["label"].to_numpy()`.

**Level 2.** The first call takes `X, y` with `test_size=0.30` and returns four things: `X_train, X_temp, y_train, y_temp`. The second call takes `X_temp, y_temp` with `test_size=0.50` and returns the validation and test pieces. Both calls need `stratify=` and `random_state=seed`. Ask yourself which label array each call should stratify on: the one it is actually splitting.

**Level 3.** The second call, which is the one people get wrong, looks like this; note it stratifies on the pool's labels, not the originals:

```python
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=seed
)
```

The first call has the same shape with `X, y`, `test_size=0.30`, and `stratify=y`. Return all six arrays in the order the contract lists.

## Task 2: build_and_fit_pipeline

**Level 1.** A `Pipeline` takes a list of `(name, object)` pairs. You need two: the vectorizer named `"tfidf"` and the classifier named `"clf"`. Build it, fit it on the training data, return it.

**Level 2.** The skeleton is:

```python
pipeline = Pipeline([
    ("tfidf", ...),
    ("clf", ...),
])
```

The vectorizer needs `ngram_range=(1, 2)`. The classifier needs `max_iter=1000` and `random_state=seed`. Then one `fit` call on the whole pipeline, then return.

**Level 3.** The two step objects, fully configured:

```python
("tfidf", TfidfVectorizer(ngram_range=(1, 2))),
("clf", LogisticRegression(max_iter=1000, random_state=seed)),
```

After constructing: `pipeline.fit(X_train, y_train)` and `return pipeline`. Fitting the pipeline fits both steps in order; do not fit the steps individually.

## Task 3: compute_metrics

**Level 1.** Four library functions, one dict. Each function takes `(y_true, y_pred)`. Round each result to 3 decimals with `round(value, 3)`.

**Level 2.** The dict keys are exactly `"accuracy"`, `"precision"`, `"recall"`, `"f1"`. Three of the four calls, all except accuracy, need one extra keyword argument that controls what happens when a denominator is zero. The task text names it.

**Level 3.** One entry as a model for the other three:

```python
"precision": round(precision_score(y_true, y_pred, zero_division=0), 3),
```

Recall and f1 follow the same pattern with their own functions; accuracy is the same without `zero_division`.

## Task 4: predict_and_confusion

**Level 1.** Two steps: get predictions from the pipeline, then hand truth and predictions to `confusion_matrix`. Return both.

**Level 2.** `pipeline.predict(X)` gives the prediction array. `confusion_matrix` takes two arguments and their order matters: truth first, predictions second. Swapping them transposes the matrix and every check about specific cells fails.

**Level 3.**

```python
y_pred = pipeline.predict(X)
cm = confusion_matrix(y, y_pred)
```

Return them as the tuple `(y_pred, cm)`.

## Task 5: threshold_sweep

**Level 1.** Get the positive class probabilities once, before any loop. Then for each threshold, one comparison turns probabilities into 0 and 1 labels, and your own `compute_metrics` does the scoring. Collect rows, build a DataFrame at the end.

**Level 2.** `pipeline.predict_proba(X)` returns two columns; you want column index 1. `(proba >= threshold)` gives booleans; `.astype(int)` makes them labels. Each loop iteration appends a dict with keys `threshold`, `precision`, `recall`, `f1`, pulling the last three out of the `compute_metrics` result. Build the DataFrame with an explicit `columns=` list so the order is fixed.

**Level 3.** The two lines that do the real work:

```python
proba = pipeline.predict_proba(X)[:, 1]
```

and inside the loop:

```python
y_pred = (proba >= threshold).astype(int)
```

Everything after that is calling `compute_metrics`, appending row dicts, and `pd.DataFrame(rows, columns=["threshold", "precision", "recall", "f1"])`.

## Task 6: majority_baseline_metrics

**Level 1.** Three steps: find which label is most common in the training labels, build a test-length array containing only that label, score it with your `compute_metrics`. Add the majority label to the returned dict under the key `"majority_label"`.

**Level 2.** The contract itself names the two numpy tools: `np.bincount(...).argmax()` for the most frequent label and `np.full_like(...)` for the constant array. Wrap the label in `int(...)` because the contract promises a plain int and bincount returns a numpy integer.

**Level 3.**

```python
majority_label = int(np.bincount(y_train).argmax())
y_pred = np.full_like(y_test, majority_label)
```

Then score with `compute_metrics(y_test, y_pred)`, add the extra key to that dict, and return it.

## Task 7: train_and_evaluate

**Level 1.** This task is assembly, not new code. Three functions you already wrote, called in sequence: train a pipeline, get predictions and a matrix, compute metrics. If you find yourself typing `TfidfVectorizer` or `confusion_matrix`, stop; you are re-doing work your own functions already do.

**Level 2.** Call order: `build_and_fit_pipeline` with the training data and seed, then `predict_and_confusion` with the fitted pipeline and the test data, then `compute_metrics` with the truth and the predictions that came back. Return the metrics dict and the matrix as a tuple.

**Level 3.**

```python
pipeline = build_and_fit_pipeline(X_train, y_train, seed)
y_pred, cm = predict_and_confusion(pipeline, X_test, y_test)
```

The last line computes metrics from `y_test` and `y_pred` and returns `(metrics, cm)`.

---

## Stretch goals

**Stretch 1.** The entire change is one keyword argument on the LogisticRegression inside a rebuilt pipeline. The task text names it. Retrain on the imbalanced training data, rescore on the imbalanced test set, and add a third row to the comparison.

**Stretch 2.** Both classes you need are named in the task text. The wrapping order is: freeze the fitted pipeline first, then hand the frozen object to the threshold wrapper with `threshold=0.3`. The wrapper still requires one `fit` call before `predict`; with a frozen inner model that call retrains nothing.

**Stretch 3.** Construct `TunedThresholdClassifierCV` around a fresh, unfitted pipeline with `scoring="f1"` and `cv=5`, fit it on the training data, and read the chosen cutoff from its `best_threshold_` attribute after fitting. Then predict on the test set and score as usual. When you compare its F1 to the default threshold's F1, think carefully before declaring a winner; the recap discussion depends on it.
