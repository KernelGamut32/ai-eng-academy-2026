# Week 6 Lab 01 Hints: Detailed Tier

How to use this file: pick one tier per task, this one or `HINTS.md`, not both; reading both wastes time. This tier shows the working core of each task with a comment on why each line is there. It deliberately withholds the function shell, the return statement assembly, and the glue, so finishing still means reading and understanding the code, not pasting a function. The instructor solution notebook remains the only fully assembled version.

---

## Task 1: split_corpus

The working core:

```python
# Columns out of the DataFrame as plain numpy arrays. to_numpy() is the
# current idiom; arrays keep every downstream sklearn call simple.
X = df["text"].to_numpy()
y = df["label"].to_numpy()

# Cut 1 of 2: peel off 30 percent as a temporary pool. stratify=y forces
# both sides to mirror the label balance of the full dataset, so a small
# sample cannot drift away from 50-50 by luck. random_state pins the
# shuffle so your numbers match the worked target output.
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, stratify=y, random_state=seed
)

# Cut 2 of 2: halve the pool into validation and test. The stratify
# argument is y_temp, the labels of the data actually being cut here.
# Stratifying on the original y is the classic mistake; the array lengths
# would not even match.
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=seed
)
```

What you still write: the function shell and the return of all six arrays in the exact order the contract specifies. Check that order against the docstring, not your memory; the apply cell unpacks positionally.

## Task 2: build_and_fit_pipeline

The working core:

```python
# A Pipeline is a list of (name, estimator) steps that acts as one model.
# Step names matter here: the checks look them up by "tfidf" and "clf".
Pipeline([
    # ngram_range=(1, 2) indexes single words and adjacent word pairs.
    # Pairs let the model tell "installation cost" (a pricing question)
    # apart from "installation, two of the doors will not close".
    ("tfidf", TfidfVectorizer(ngram_range=(1, 2))),

    # max_iter=1000 gives the iterative solver room to converge on a
    # text-sized feature space; the default budget of 100 can end early
    # with a ConvergenceWarning. random_state pins solver randomness.
    ("clf", LogisticRegression(max_iter=1000, random_state=seed)),
])
```

And the fitting idea: one `fit(X_train, y_train)` call on the pipeline object runs both steps in order. The vectorizer learns its vocabulary from the training text only; later `predict` calls reuse that vocabulary on new text, which is what keeps held-out data from leaking into the features.

What you still write: the function shell, assigning the pipeline to a variable, the fit call, and the return.

## Task 3: compute_metrics

The working core:

```python
# Each metric is one library call on (truth, predictions), rounded to the
# lab's 3-decimal display convention. Rounding inside the function lets
# the checks compare whole dicts for exact equality.
round(accuracy_score(y_true, y_pred), 3)

# precision, recall, and f1 all take zero_division=0. The edge case it
# governs: a model that never predicts positive gives precision a zero
# denominator. Without the argument sklearn warns and substitutes; with
# it the answer is an explicit, silent 0.0. Task 6 scores exactly such a
# model, so this choice is load-bearing, not cosmetic.
round(precision_score(y_true, y_pred, zero_division=0), 3)
round(recall_score(y_true, y_pred, zero_division=0), 3)
round(f1_score(y_true, y_pred, zero_division=0), 3)
```

What you still write: the function shell and the dict literal that maps the keys `"accuracy"`, `"precision"`, `"recall"`, `"f1"` to those four expressions. Key spelling is checked exactly.

## Task 4: predict_and_confusion

The working core:

```python
# The fitted pipeline turns raw message strings into label predictions.
y_pred = pipeline.predict(X)

# Argument order is truth first, predictions second. Swapping them
# transposes the matrix: every false alarm would read as a miss and the
# exact-cell check would fail. With classes 0 and 1 the layout is
# [[TN, FP], [FN, TP]]: rows are the true class, columns the predicted.
cm = confusion_matrix(y, y_pred)
```

What you still write: the function shell and the return of the two objects as a tuple in the contract's order.

## Task 5: threshold_sweep

The working core:

```python
# predict_proba returns one column per class in ascending class order;
# column 1 is P(installation_issue). Compute it once, outside the loop:
# the model's opinion never changes across thresholds, only our cutoff
# does. This is also how threshold tuning is done on real systems:
# score once, decide many times.
proba = pipeline.predict_proba(X)[:, 1]

for threshold in thresholds:
    # This one comparison IS the decision rule that predict was hiding.
    # Booleans from >=, cast to ints so they are labels, not True/False.
    y_pred = (proba >= threshold).astype(int)

    # Reuse Task 3 rather than calling the sklearn scorers again; the
    # rounding convention comes along for free.
    metrics = compute_metrics(y, y_pred)

    # One row dict per threshold. Pull precision, recall, and f1 out of
    # the metrics dict; the accuracy key is simply not used here.
    {"threshold": threshold, "precision": metrics["precision"],
     "recall": metrics["recall"], "f1": metrics["f1"]}
```

And the assembly idea: append each row dict to a list, then build the result with `pd.DataFrame(rows, columns=["threshold", "precision", "recall", "f1"])`. The explicit `columns=` pins the column order the checks expect.

What you still write: the function shell, the list, the append, and the return.

## Task 6: majority_baseline_metrics

The working core:

```python
# bincount over integer labels returns counts by value: index 0 holds
# how many zeros, index 1 how many ones. argmax gives the index with the
# largest count, which is the majority label. int() converts the numpy
# integer to the plain int the contract promises (and plain ints
# serialize cleanly when metrics get logged later this week).
majority_label = int(np.bincount(y_train).argmax())

# The entire "model": an array shaped and typed like y_test, filled with
# one constant. This is the competitor every real model must beat.
y_pred = np.full_like(y_test, majority_label)
```

And the scoring idea: run your `compute_metrics(y_test, y_pred)`, then add `"majority_label"` as an extra key on the returned dict before handing it back. Because Task 3 set `zero_division=0`, the all-negative predictions score a clean precision of 0.0 with no warning.

What you still write: the function shell, the compute_metrics call, the extra key, and the return.

## Task 7: train_and_evaluate

The working core:

```python
# Assembly of your own Tasks 2, 4, and 3, in that order. If you are
# typing any sklearn class name in this task, back up: the point is that
# evaluation primitives, once written as functions, make a new
# experiment a three-line recipe.
pipeline = build_and_fit_pipeline(X_train, y_train, seed)
y_pred, cm = predict_and_confusion(pipeline, X_test, y_test)
metrics = compute_metrics(y_test, y_pred)
```

What you still write: the function shell and the return of `(metrics, cm)`.

---

## Stretch goals

The detailed tier covers the stretch goals at the same depth: working core with commentary, assembly withheld. Fully assembled stretch solutions live only in the instructor solution notebook.

### Stretch 1: class_weight="balanced"

The working core is a rebuilt classifier step:

```python
# class_weight="balanced" rescales each class's weight in the training
# loss inversely to its frequency. At 12 percent positives, each positive
# example counts roughly 7 times as much as each negative, so ignoring
# the rare class stops being the cheap way to minimize loss. Everything
# else about the pipeline is unchanged, which is what makes the
# before-and-after comparison clean.
LogisticRegression(max_iter=1000, random_state=seed, class_weight="balanced")
```

What you still write: the pipeline around it, fitting on `X_train_imb, y_train_imb`, scoring on the imbalanced test set with your existing functions, and a third row in the comparison DataFrame. Predict what will happen to recall before you run it, then check yourself.

### Stretch 2: FixedThresholdClassifier with FrozenEstimator

The working core:

```python
from sklearn.model_selection import FixedThresholdClassifier
from sklearn.frozen import FrozenEstimator

# FrozenEstimator wraps the ALREADY FITTED pipeline from Task 2 and turns
# fit into a no-op, so the exact model you evaluated is the model being
# wrapped, with no silent retrain. FixedThresholdClassifier then makes
# plain predict apply your cutoff, so every downstream consumer inherits
# the business decision without knowing predict_proba exists.
FixedThresholdClassifier(FrozenEstimator(baseline_pipeline), threshold=0.3)
```

What you still write: assigning it, the one `fit(X_train, y_train)` call the sklearn API contract requires before predicting (harmless here, the frozen inner model does not retrain), predicting on the test set, and comparing the resulting confusion matrix and metrics against your Task 5 row for 0.3. They should match exactly; if they do not, check which pipeline you froze.

### Stretch 3: TunedThresholdClassifierCV

The working core:

```python
from sklearn.model_selection import TunedThresholdClassifierCV

# Wraps a FRESH, unfitted pipeline: during fit it runs 5-fold cross
# validation on the training data, tries a range of cutoffs on each
# fold's held-out slice, and keeps the one that maximizes the named
# metric. The test set has no vote in the choice, which is the entire
# discipline of this lab expressed as an estimator.
TunedThresholdClassifierCV(fresh_pipeline, scoring="f1", cv=5)
```

What you still write: constructing the fresh pipeline to hand it, fitting on `X_train, y_train`, reading `best_threshold_` off the fitted object, predicting on the test set, and scoring. When its test F1 lands below the default threshold's F1, resist calling that a defeat; work out why picking the winner by test score would itself break the rules this lab taught, and bring that argument to the recap discussion.
