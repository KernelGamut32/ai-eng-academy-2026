# Week 6 Lab 1.2 Hints: Detailed Tier

How to use this file: pick one tier per task, this one or `HINTS.md`, not both; reading both wastes time. This tier shows the working core of each task with a comment on why each line is there. It deliberately withholds the function shell, the return statement assembly, and the glue, so finishing still means reading and understanding the code, not pasting a function. The instructor solution notebook remains the only fully assembled version.

---

## Task 1: build_corpus

The working core:

```python
# One seeded generator for the entire corpus. Every pick_kind and
# make_ticket call consumes randomness from this stream, so the
# contract's call order IS the reproducibility guarantee: reorder the
# calls or add a second Random instance and you get a different,
# equally plausible corpus with different downstream numbers.
rng = random.Random(seed)

# round, not int: int truncates, and exact positive counts are part of
# the contract (a check asserts exactly 50).
n_pos = round(n_docs * positive_fraction)

for i in range(n_docs):
    # All positives first, then all negatives: trivially exact counts,
    # and it makes the shuffle below load-bearing instead of cosmetic.
    label = 1 if i < n_pos else 0
    # Difficulty plumbing is provided; you just roll it per ticket.
    kind = pick_kind(rng, label)
    rows.append({"text": make_ticket(rng, label, kind), "label": label, "kind": kind})

# In place, with the SAME rng, as the last step before the DataFrame.
# Without this, the first 50 rows are all high risk: head() lies, and
# anything that splits positionally downstream inherits a disaster.
# random.shuffle (the module function) would draw from the global
# generator and break the determinism check.
rng.shuffle(rows)
```

What you still write: the function shell, the empty list above the loop, and the `pd.DataFrame(rows)` return.

## Task 2: split_tickets

The working core:

```python
# Columns out as plain numpy arrays, same idiom as the morning.
X = df["text"].to_numpy()
y = df["label"].to_numpy()

# One cut this time, not the morning's two: the main path of this lab
# tunes nothing, so no validation set needs to absorb tuning decisions.
# stratify=y matters more at 90-10 than it did at 50-50: an
# unstratified 100-row draw could carry 5 positives or 15, and every
# recall number downstream would inherit that coin flip. Stratification
# pins the test set at exactly 10.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=seed
)
```

What you still write: the function shell and the return of the four arrays in the contract's order.

## Task 3: fit_majority_baseline

The working core:

```python
# strategy="most_frequent" is your hand-rolled bincount baseline from
# this morning, wearing the standard estimator interface: it fits on
# (X, y) and predicts from X like everything else in sklearn, which is
# what lets it drop into pipelines, harnesses, and tracked runs. It
# ignores X entirely. random_state is a no-op for this strategy but is
# passed uniformly; the stratified strategy in stretch 1 consumes it.
baseline = DummyClassifier(strategy="most_frequent", random_state=seed)
baseline.fit(X_train, y_train)
```

What you still write: the function shell and returning the fitted object (the apply cell does the predicting).

## Task 4: evaluate_predictions

The working core:

```python
# The morning's two evaluation tasks merged into the one function that
# grades every model today. Shared grader means a difference between
# two comparison rows can only come from the models.
metrics = {
    "accuracy": round(accuracy_score(y_true, y_pred), 3),
    # zero_division=0 stops being theoretical today: the dummy never
    # predicts positive, precision's denominator is zero, and this
    # gives a clean explicit 0.0 instead of a warning flood.
    "precision": round(precision_score(y_true, y_pred, zero_division=0), 3),
    "recall": round(recall_score(y_true, y_pred, zero_division=0), 3),
    "f1": round(f1_score(y_true, y_pred, zero_division=0), 3),
}

# Truth first. Swapped arguments transpose the matrix and turn every
# false alarm into a miss in your reading.
cm = confusion_matrix(y_true, y_pred)
```

What you still write: the function shell and the return of `(metrics, cm)` as a tuple in that order.

## Task 5: build_and_fit_pipeline

This is the morning's pipeline, and the detailed tier declines to hand it over twice in one day: build it from memory first, and if genuinely stuck, the level 3 hint in the progressive tier shows it whole. The three settings the checks assert: steps named `"tfidf"` and `"clf"`, `ngram_range=(1, 2)` on the vectorizer, `max_iter=1000` plus the seed on the classifier. Fit the pipeline object, not the steps.

## Task 6: build_and_fit_balanced_pipeline

The working core is the one changed line:

```python
# class_weight="balanced" rescales each class's weight in the training
# loss inversely to its frequency: at 10 percent positives, each
# high-risk ticket now costs about 9 times as much to get wrong as a
# routine one, so hedging toward the majority stops being the cheap
# way to minimize loss. Every other setting stays identical to Task 5
# on purpose: one variable changed, so the before-and-after comparison
# is attributable to it.
("clf", LogisticRegression(max_iter=1000, random_state=seed, class_weight="balanced")),
```

What you still write: the pipeline around it (identical to Task 5's), the fit, and the return.

## Task 7: build_comparison_table

The working core:

```python
for name, metrics in results:
    row = {"model": name}
    # Copy by explicit key rather than dict-splatting the whole metrics
    # dict: extra keys a future run might carry (a timestamp, a label)
    # stay out of the table, and a missing key fails loudly with a
    # KeyError instead of producing a ragged frame.
    for key in ("accuracy", "precision", "recall", "f1"):
        row[key] = metrics[key]
    rows.append(row)

# Explicit columns= pins both the column order the checks assert and
# the row order as given, so the table reads the same on every machine.
pd.DataFrame(rows, columns=["model", "accuracy", "precision", "recall", "f1"])
```

What you still write: the function shell, the empty list, and the return.

---

## Stretch goals

The detailed tier covers the stretch goals at the same depth: working core with commentary, assembly withheld. Fully assembled stretch solutions live only in the instructor solution notebook.

### Stretch 1: the chance baseline

The working core:

```python
# The strategy that genuinely consumes random_state: each prediction is
# drawn at random in proportion to the training base rates, about
# 90-10. Where most_frequent is the floor of silence, stratified is the
# floor of guessing. Omit the seed and your numbers will not reproduce.
strat_clf = DummyClassifier(strategy="stratified", random_state=RANDOM_SEED)
```

What you still write: fitting it, predicting on the test set, grading with your Task 4 function, and rebuilding the comparison with your Task 7 function as a four-row table. When you read the result, look at two things: what guessing did to accuracy relative to the silent dummy, and how close chance-level recall sits to the base rate, then re-read the plain model's 0.200 recall in that light.

### Stretch 2: the ranking behind the labels

The working core:

```python
# Column 1 of predict_proba is the score for the high-risk class, the
# same slice as the morning's threshold sweep.
proba_lr = lr_pipeline.predict_proba(X_test)[:, 1]

# Average precision grades the RANKING across all thresholds at once,
# summarizing the precision-recall curve into one number. Two
# calibration facts: its random-ranker floor is the positive rate
# (0.10 here), not 0.5, and it never consults a cutoff, so it measures
# what the model knows, not how you act on it.
ap_lr = average_precision_score(y_test, proba_lr)

# The two boundary scores that explain whatever ap_lr turns out to be:
proba_lr[y_test == 1].min()   # worst score given to a real high-risk ticket
proba_lr[y_test == 0].max()   # best score given to a routine ticket
```

What you still write: the imports from `sklearn.metrics`, the prints, and the `precision_recall_curve` plot (three return values; step-plot recall against precision). When the average precision prints, compare the two boundary scores to each other and then both to 0.5, and write down in one sentence where the plain model's failure actually lives. That sentence is the hand-off to stretch 3.

### Stretch 3: the other lever

The working core:

```python
# Wraps a FRESH, unfitted copy of the plain pipeline. During fit, 5-fold
# cross validation inside the training data searches cutoffs and keeps
# the F1-maximizing one. The test set has no vote, which answers "where
# did the validation set go" in its strongest form: when you cannot
# afford a third split, tuning happens by CV inside train.
tuned_model = TunedThresholdClassifierCV(fresh_pipeline, scoring="f1", cv=5)
```

What you still write: constructing `fresh_pipeline` (Task 5's recipe, unfitted, not your already-fitted `lr_pipeline`), fitting on the training data, reading `best_threshold_` off the fitted object, predicting on the test set, and grading with Task 4. Compare the chosen threshold to stretch 2's boundary scores; the alignment is not a coincidence. When the test metrics come out suspiciously clean, the question to answer before reading the solution is not "did I cheat" (you did not; check what data chose the threshold) but "what property of this synthetic corpus makes perfection reachable, and why should I not expect it on a real queue".
