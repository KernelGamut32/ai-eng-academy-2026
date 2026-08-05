# Week 6 Lab 1.3 Hints: Detailed Tier

How to use this file: pick one tier per task, this one or `HINTS.md`, not both; reading both wastes time. This tier shows the working core of each task with a comment on why each line is there. It deliberately withholds the function shell, the return statement assembly, and the glue, so finishing still means reading and understanding the code, not pasting a function. The instructor solution notebook remains the only fully assembled version.

---

## Task 1: split_corpus_three_way

The working core:

```python
# Columns out as plain numpy arrays, the day's standard idiom.
X = df["text"].to_numpy()
y = df["label"].to_numpy()

# Call 1: peel off 40 percent as a temporary pool. The 60 percent side
# is train. Stratify so the pool carries the same 15 percent urgent rate.
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.40, stratify=y, random_state=seed
)

# Call 2: halve the pool into validation and test. The classic mistake
# is stratify=y here; the pool has its own labels, y_temp, and
# stratifying on the wrong array raises a length error at best and
# silently lies at worst.
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=seed
)
```

What you still write: the function shell and the return of all six arrays in the contract's order: `X_train, X_val, X_test, y_train, y_val, y_test`.

## Task 2: build_and_fit_pipeline

Third build of the day, and the detailed tier declines to hand it over again: build it from memory, and if genuinely stuck, the progressive tier's level 3 shows it whole. The settings the checks assert: steps named `"tfidf"` and `"clf"`, `ngram_range=(1, 2)`, `max_iter=1000` plus the seed. One warning specific to this lab: the seed parameter defaults to 11 here, not the morning's 7; use the parameter, never a typed number, and your numbers will match the target output.

## Task 3: evaluate_at_threshold

The working core:

```python
# Column 1 of predict_proba is the urgent-class score; this slice is
# the same one Lab 1.1's sweep used.
proba = pipeline.predict_proba(X)[:, 1]

# The whole lab in one comparison: booleans from >=, labels from
# astype(int). predict() is exactly this line with threshold hardwired
# to 0.5, which is why this function supersedes it.
y_pred = (proba >= threshold).astype(int)

# The day's grading conventions, unchanged: exact keys, rounded to 3
# inside the function so checks compare dicts exactly, zero_division=0
# everywhere it is accepted (at high thresholds this model predicts no
# positives at all, and precision would otherwise warn and NaN).
metrics = {
    "accuracy": round(accuracy_score(y, y_pred), 3),
    "precision": round(precision_score(y, y_pred, zero_division=0), 3),
    "recall": round(recall_score(y, y_pred, zero_division=0), 3),
    "f1": round(f1_score(y, y_pred, zero_division=0), 3),
}

# Truth first, as always; swapped arguments transpose the matrix.
cm = confusion_matrix(y, y_pred)
```

What you still write: the function shell and the return of `(metrics, cm)` as a tuple in that order.

## Task 4: threshold_sweep

The working core:

```python
# Build the default only when none was given. The np.round is
# load-bearing: raw linspace emits values like 0.15000000000000002,
# which print as 0.15 but fail exact equality in row lookups and in
# this lab's checks. Rounding makes each threshold the nearest double
# to its printed value, so == against literals behaves.
if thresholds is None:
    thresholds = np.round(np.linspace(0.05, 0.95, 19), 2)

for threshold in thresholds:
    # Pure reuse of your Task 3; the sweep has no use for the matrix,
    # so the underscore discards it (stretch 3 will want it and calls
    # the evaluator itself).
    metrics, _ = evaluate_at_threshold(pipeline, X, y, threshold)
    rows.append({"threshold": threshold, **metrics})
```

What you still write: the function shell, the empty list, and the return: `pd.DataFrame(rows, columns=["threshold", "accuracy", "precision", "recall", "f1"])`, whose explicit column list pins the schema the checks assert.

## Task 5: select_thresholds

The working core:

```python
# idxmax returns the index label of the FIRST maximal row, which makes
# tie handling deterministic by construction. This sweep has unique
# winners, but the function should not depend on luck.
f1_optimal = float(sweep_df.loc[sweep_df["f1"].idxmax(), "threshold"])

# The stakeholder pattern: a constraint becomes a FILTER, a preference
# becomes an ARGMAX inside the filter. The safety team's floor keeps
# only rows with recall >= recall_floor; among those, the fewest wasted
# callbacks means the largest precision.
safety_rows = sweep_df[sweep_df["recall"] >= recall_floor]
safety_team = float(safety_rows.loc[safety_rows["precision"].idxmax(), "threshold"])

# The call center is the mirror image: precision floor as the filter,
# recall's argmax as the preference.
center_rows = sweep_df[sweep_df["precision"] >= precision_floor]
call_center = float(center_rows.loc[center_rows["recall"].idxmax(), "threshold"])
```

The `float(...)` wrappers honor the contract's plain-floats requirement: numpy scalars print with baggage and serialize awkwardly once picks get logged.

What you still write: the function shell and the return of the three-key dict exactly as the contract names it.

## Task 6: build_operating_points_table

The working core:

```python
for name, threshold in named_thresholds:
    # One evaluator grades every operating point, so a difference
    # between two rows can only come from the thresholds.
    metrics, _ = evaluate_at_threshold(pipeline, X, y, threshold)
    rows.append({"operating_point": name, "threshold": threshold, **metrics})
```

What you still write: the function shell, the empty list, and the return with the pinned six-column list: `["operating_point", "threshold", "accuracy", "precision", "recall", "f1"]`. Rows keep the order given, which is why the apply cell's table reads f1_optimal, safety_team, call_center top to bottom.

## Task 7: final_test_report

The working core:

```python
# The one place in the lab the test set is touched, and both splits are
# graded at the SAME locked threshold: the comparison is only honest if
# nothing but the data changes between the rows.
for split, Xs, ys in (("validation", X_val, y_val), ("test", X_test, y_test)):
    metrics, _ = evaluate_at_threshold(pipeline, Xs, ys, threshold)
    rows.append({"split": split, "threshold": threshold, **metrics})
```

What you still write: the function shell, the list, and the return with `columns=["split", "threshold", "accuracy", "precision", "recall", "f1"]`, validation row first because the tuple order above puts it first.

---

## Stretch goals

The detailed tier covers the stretch goals at the same depth: working core with commentary, assembly withheld. Fully assembled stretch solutions live only in the instructor solution notebook.

### Stretch 1: ship the cutoff with the model

The working core:

```python
# Inside out: FrozenEstimator wraps the ALREADY FITTED pipeline and
# turns fit into a no-op, so the exact model you validated is the model
# being shipped. FixedThresholdClassifier then makes plain predict
# apply the locked cutoff, so downstream code inherits the business
# decision without knowing predict_proba exists.
production_model = FixedThresholdClassifier(
    FrozenEstimator(alert_pipeline), threshold=picks["f1_optimal"]
)
production_model.fit(X_train, y_train)  # API-required; retrains nothing
```

What you still write: the imports (`FixedThresholdClassifier` from `sklearn.model_selection`, `FrozenEstimator` from `sklearn.frozen`), predicting on the test set, computing the four metrics with the day's conventions, and the comparison against `final_report.iloc[1]`. It should match exactly; if it does not, check which pipeline you froze (the fitted one, not a fresh one) and which threshold you passed.

### Stretch 2: the sweep's continuous twin

The working core:

```python
proba_val = alert_pipeline.predict_proba(X_val)[:, 1]

# Average precision summarizes the ENTIRE precision-recall curve, every
# distinct score treated as a cutoff, into one threshold-free number.
# Its random-ranker floor is the positive rate (0.15 here), not 0.5.
ap_val = average_precision_score(y_val, proba_val)

# precision_recall_curve returns three arrays; the third (thresholds)
# is one shorter than the other two and is not needed for the plot.
precision_curve, recall_curve, _ = precision_recall_curve(y_val, proba_val)

# The two boundary scores that explain whatever ap_val turns out to be:
proba_val[y_val == 1].min()   # quietest real emergency's score
proba_val[y_val == 0].max()   # loudest routine ticket's score
```

What you still write: the imports, the prints, the step plot of recall against precision, and the loop that marks your three operating points on the curve (each point's coordinates come from your own `evaluate_at_threshold` at that threshold: x is recall, y is precision). When you compare to Lab 1.2's 1.000, look at the order of your two boundary scores; there, the lowest positive outscored the highest negative, and here it does not. Write the one-sentence consequence: which lab's failure could a threshold fully fix, and which cannot be fixed by any threshold at all?

### Stretch 3: the number the discussion was missing

The working core:

```python
COST_FALSE_ALARM = 8          # coordinator time per pointless callback
COST_MISSED_EMERGENCY = 400   # expected damage and escalation per miss

for threshold in sweep_df["threshold"]:
    # The sweep discarded the confusion matrices; this loop wants them,
    # so it calls the evaluator directly. In the 2x2 layout, row 0 is
    # true-routine and row 1 is true-urgent, so cm[0, 1] counts false
    # alarms and cm[1, 0] counts misses.
    _, cm = evaluate_at_threshold(alert_pipeline, X_val, y_val, threshold)
    fp = int(cm[0, 1])
    fn = int(cm[1, 0])
    expected_cost = fp * COST_FALSE_ALARM + fn * COST_MISSED_EMERGENCY
```

What you still write: collecting the rows into a cost DataFrame, `idxmin` to find the cost-minimizing row (the mirror of Task 5's `idxmax`), the lookup that prices each of your three Task 5 picks, and optionally the cost-versus-threshold plot. Before running, commit to a prediction about which stakeholder an 8-to-400 ratio favors; after running, the two questions worth a written sentence each: how many pointless callbacks does one missed emergency buy at these costs, and what would the false-alarm cost have to become before the call-center pick stops losing?
