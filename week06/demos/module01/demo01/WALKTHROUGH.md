# Demo 1 Walkthrough: every line, in plain terms

This is the teaching companion to `Demo01_Read_the_Matrix_SOLUTION.ipynb`.
It explains each line of code the way you would to a strong software engineer
who has never touched ML: no assumed vocabulary, every choice justified.
Read it before you teach; steal phrasing freely.

The audience already trusts arrays, functions, and unit tests. The move
throughout is to map every new idea onto one of those.

---

## Cell: Setup

```python
import numpy as np
import pandas as pd
```

Standard tools. `numpy` is fast arithmetic on arrays of numbers. `pandas`
reads the CSV and gives us a table object. Nothing ML specific yet.

```python
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
```

scikit-learn is the standard Python toolkit for classical ML. We import only
from `sklearn.metrics`, the grading submodule. Worth saying out loud: we are
not importing anything that trains a model. This entire demo is grading
homework that a model already turned in. Evaluation is a separate activity
from training, with its own tools, and you can be excellent at it without
training anything.

Each imported name, one breath each:

- `confusion_matrix`: counts the four kinds of right and wrong.
- `ConfusionMatrixDisplay`: the same four counts as a colored picture.
- `classification_report`: a formatted table of the per class metrics.
- `accuracy_score`, `precision_score`, `recall_score`, `f1_score`: one
  function per metric from the slides. Each takes (truth, predictions) and
  returns a single float.

```python
df = pd.read_csv("cordwell_test_set.csv")
```

Loads the test set into a table called a DataFrame. 2,000 rows, one per
product review. This file is the held out test set from the slides: the
model never saw these rows during training, which is the only reason the
grades we compute today mean anything.

```python
y_test = df["y_true"].to_numpy()
y_score = df["y_score"].to_numpy()
```

Pull two columns out of the table into plain numpy arrays.

- `y_test` is the answer key. A human labeled every review: 1 means genuine
  safety issue, 0 means routine. In ML this is called ground truth, and the
  naming convention `y` for "the thing being predicted" is universal enough
  that fighting it costs more than adopting it.
- `y_score` is what the model produced: a number between 0 and 1 per review.
  Higher means the model leans toward safety issue. Resist calling it a
  probability; it is a score. The distinction becomes a whole slide this
  afternoon.

Why `.to_numpy()`: pandas objects carry an index, a set of row labels that
tag along with the data, and some operations select rows by label where
numpy selects by position. For today's arithmetic, plain positional arrays
are simpler and remove a class of subtle indexing bugs. The bootstrap
function later today does exactly this coercion for exactly this reason, so
planting the habit here pays off twice.

```python
print(f"{len(df)} reviews loaded")
print(f"{int(y_test.sum())} genuine safety escalations")
print(f"Base rate: {y_test.mean():.4f}")
```

Three sanity prints. Summing an array of 0s and 1s counts the 1s, so
`y_test.sum()` is the number of genuine safety issues: 67. The mean of an
array of 0s and 1s is the fraction of 1s: 0.0335, which is the base rate,
the frequency of the thing we care about. 3.35 percent. Every trap in
today's module is downstream of that small number, so it goes on the screen
before anything else.

`df.head(3)` shows the first three rows so the room sees real(istic) review
text and believes the data exists.

---

## Cell: Beat 1, score to decision

```python
THRESHOLD = 0.50
```

A named constant, upper case, at the top: this is a configuration decision,
not a fact of nature. The whole point of Beat 1 is that this number exists
and someone chose it.

```python
y_pred = (y_score >= THRESHOLD).astype(int)
```

The most important line of the morning, and it is one comparison.

Reading inside out: `y_score >= THRESHOLD` compares every score in the array
against 0.50 at once (numpy applies the comparison elementwise) and produces
an array of 2,000 booleans, True where the model leans safety issue.
`.astype(int)` converts True to 1 and False to 0, matching the vocabulary of
the answer key so the two arrays can be compared.

The engineer framing: the model outputs an analog signal; this line is the
comparator that digitizes it. `predict()` in sklearn does the same
comparison internally at 0.50 and never shows you. Writing it by hand makes
the hidden decision visible, and this afternoon the decision gets
interrogated: 0.50 is a default, not a recommendation.

```python
print(f"Reviews flagged for escalation: {y_pred.sum()} of {len(y_pred)}")
```

160 of 2,000: the model asks humans to read 8 percent of the queue. This is
the first appearance of the operational lens (how much work does this policy
create), which becomes the capacity constraint conversation in Section 6.

---

## Cell: Beat 2, the four counts

```python
tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()
```

One line, three ideas.

**Idea 1: what the function counts.** For every review there are only four
possible stories, given by crossing what was true with what was predicted:

| | Predicted escalate | Predicted routine |
|---|---|---|
| **Actually a safety issue** | true positive (caught it) | false negative (missed it) |
| **Actually routine** | false positive (false alarm) | true negative (correctly ignored) |

`confusion_matrix` reads both arrays and tallies how many reviews fell into
each cell. Every metric this morning is arithmetic on these four tallies and
nothing else. That is why the demo is called Read the Matrix: learn to read
these four numbers and the metrics stop being formulas to memorize and
become sentences about the four cells.

**Idea 2: why `labels=[0, 1]`.** sklearn has to decide which class gets the
first row. By default it infers the ordering from the values present in the
data. Passing `labels=[0, 1]` pins it: row and column order is 0 then 1,
full stop. On this dataset the default happens to agree, so this argument
changes nothing today; it is a defensive habit. The day your data slice
contains only one class, or your labels are strings, the explicit list is
the difference between a correct matrix and a wrong or misshapen one. Pin
your label order the way you would pin a dependency version.

**Idea 3: what `.ravel()` does and why the order matters.** The function
returns a 2x2 numpy grid. `.ravel()` flattens it row by row into four
numbers, and with labels pinned to `[0, 1]` that flattened order is exactly:

```
tn, fp, fn, tp
```

Top row first: actually routine (true negatives, then false positives), then
the bottom row: actually safety (false negatives, then true positives).

Say this plainly: **if you unpack these four names in the wrong order,
nothing crashes.** Python happily assigns four integers to four names. Your
precision and recall silently trade places, and 0.33 and 0.79 are both
plausible numbers, so nothing looks wrong. This is the best kind of bug to
show a room of engineers: type safe, exception free, and completely wrong.
The checks cell in the student notebook catches exactly this, which is a
nice moment to point out that evaluation code deserves tests like any other
code.

```python
print(f"TP={tp}  FN={fn}  FP={fp}  TN={tn}")
```

TP=53 FN=14 FP=107 TN=1826. Printed in story order (the interesting cells
first) rather than ravel order, which is worth a one liner: the unpacking
order is sklearn's contract, the printing order is ours.

---

## Cell: Beat 2, the picture

```python
ConfusionMatrixDisplay.from_predictions(
    y_test, y_pred,
    labels=[0, 1],
    display_labels=["routine", "escalate"],
    colorbar=False,
)
```

Same four counts as a heatmap. `from_predictions` recomputes the matrix from
the two arrays (there is a sibling, `from_estimator`, that takes a fitted
model instead; we do not have or need one). `labels=[0, 1]` pins the order
exactly as before, and `display_labels` replaces 0 and 1 with human words in
that same order, so "routine" names class 0 and "escalate" names class 1.
`colorbar=False` drops a legend that adds nothing for a 2x2.

Teaching note on the color: the giant dark cell is the 1,826 true negatives.
That visual imbalance IS the class imbalance. Most of this dataset is easy
routine reviews, and any metric that averages over the whole square inherits
that dominance. You are literally looking at the reason accuracy misleads,
one beat before proving it with numbers.

---

## Cell: Beat 3, the four metrics

```python
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
```

Four calls, same signature: (answer key, predictions). In matrix terms:

- **Accuracy** = (TP + TN) over all 2,000. "What fraction of all calls were
  right." Result 0.9395. Sounds excellent; the demo spends its final two
  beats dismantling that impression.
- **Precision** = TP over (TP + FP) = 53 over 160. "Of everything we
  escalated, what fraction was genuinely a safety issue." Result 0.3312, so
  two thirds of the escalation queue is noise. Precision is the cost you pay
  per alert; the triage team feels this number directly.
- **Recall** = TP over (TP + FN) = 53 over 67. "Of the genuine safety
  issues, what fraction did we catch." Result 0.7910, so one report in five
  slips past unread. Recall is the coverage you get; the customer feels this
  number.
- **F1** = the harmonic mean of precision and recall, 0.4670. The harmonic
  mean is the strict average: it stays low unless both inputs are decent,
  so it cannot be gamed by maxing one and abandoning the other. What it
  hides, and the slides say this sharply, is that it weights the two
  equally, which quietly claims a false alarm and a missed safety report
  cost the same. At Cordwell they do not. F1 is on screen today as the
  common vocabulary, not as a recommendation.

```python
print(f"Recall by hand: TP / (TP + FN) = {tp} / {tp + fn} = {tp / (tp + fn):.4f}")
```

The demystification line. It recomputes recall from the raw counts of Beat 2
and prints the identical 0.7910. The library function and the slide formula
are the same arithmetic; there is nothing behind the curtain. For engineers
new to ML this one line buys a lot of trust in everything that follows.

---

## Cell: Beat 3, the one call report

```python
print(classification_report(
    y_test, y_pred,
    target_names=["routine", "escalate"],
    digits=3,
    zero_division=0,
))
```

The everyday tool. One call prints precision, recall, and F1 for each class
plus averages. Argument by argument:

- `target_names` replaces 0 and 1 with words, in class order, same
  convention as the display cell.
- `digits=3` widens the formatting from the default 2. With 67 positives,
  even three digits flatters the true certainty, which is Section 5's topic;
  for now it just matches the slides.
- `zero_division=0` answers a question that will otherwise ambush you in
  production: what should precision be when the model predicts a class zero
  times? The formula becomes 0 divided by 0. Without this flag sklearn
  prints a warning and substitutes 0.0 anyway; inside a logging pipeline
  that warning scrolls past and you get a quietly odd number. With the flag
  you have made the choice explicitly and the warning disappears. It is the
  same discipline as handling a division by zero in any service code:
  decide the edge case on purpose.

Two things to point at in the output:

- The `escalate` row reproduces exactly the numbers computed one cell ago:
  0.331, 0.791, 0.467. Same arithmetic, batch form.
- The `support` column: 1,933 and 67. Support is the number of true examples
  of each class, and 67 is the real sample size behind every safety metric
  on screen. When Section 5 asks how trustworthy 0.4670 is, support is where
  the answer starts.

The `routine` row also reads well: precision 0.992 means the stream of
reviews the model waves through is very clean, which is exactly what a high
volume triage system wants from its negative channel.

---

## Cell: Beat 4, the flip

```python
fn_idx = np.where((y_test == 1) & (y_pred == 0))[0]
```

Find the misses. `(y_test == 1)` is a boolean array marking genuine safety
issues; `(y_pred == 0)` marks reviews the model called routine. `&` is
elementwise AND (note: `&`, not the Python keyword `and`, because we are
combining whole arrays position by position). True in both means a missed
safety report. `np.where(...)` returns the positions of the True entries,
wrapped in a one element tuple, hence the trailing `[0]` to take the array
itself out of the tuple. Result: the 14 row numbers of the 14 false
negatives.

Fourteen is small enough for a person to read every one, which the slides
turn into an error analysis discipline later today. The demo just reads one:

```python
print(f"  {df['review_text'].iloc[fn_idx[0]]!r}")
```

`.iloc` is pandas by position (versus `.loc`, by label), matching the
positional row numbers numpy just handed us. The `!r` prints quotes around
the text. The review that comes out describes fasteners vibrating loose and
a part flying off, with none of the obvious keywords (fire, shock, injury).
Read it to the room and ask whether it is obviously a safety report. That
borderline feel is what a miss usually looks like; the model missed it for
roughly the same reason a keyword filter would.

```python
y_flipped = y_pred.copy()
y_flipped[fn_idx[0]] = 1
```

The experiment. Copy first, because numpy assignment without `.copy()` would
alias the same memory and mutate `y_pred` behind our backs, and later cells
still need the original. Then set one position, the first miss, to 1. We are
simulating a counterfactual: same model, same test set, but this one report
got caught. Exactly one prediction out of 2,000 differs.

---

## Cell: Beat 4, the delta table

```python
def metric_row(name, fn, before, after):
    b, a = fn(y_test, before), fn(y_test, after)
    print(f"{name:<10} {b:.4f} -> {a:.4f}   ({(a - b) * 100:+.2f} points)")
```

A tiny helper: take a metric function, score the original and the flipped
predictions against the same answer key, print both plus the change in
percentage points (`:+.2f` forces a sign so gains read as gains). Passing
the metric function as an argument is an idiom the room already knows from
callbacks; it also quietly rehearses the shape of the bootstrap function
they will meet in Section 5, which takes a metric argument the same way.

The output:

```
Accuracy   0.9395 -> 0.9400   (+0.05 points)
Precision  0.3312 -> 0.3354   (+0.42 points)
Recall     0.7910 -> 0.8060   (+1.49 points)
F1         0.4670 -> 0.4737   (+0.67 points)
```

The reading, denominator by denominator:

- **Recall +1.49 points.** Its denominator is the 67 genuine safety issues.
  One rescued report out of 67 is a meaningful chunk: 1 over 67 is about
  1.5 percent, and there it is.
- **Accuracy +0.05 points.** Its denominator is all 2,000 reviews. The same
  rescued report is 1 over 2,000. The 1,826 easy true negatives dilute the
  event almost to invisibility.
- **Precision +0.42 points.** The flip added a true positive without adding
  a false alarm, so the escalation queue got one review purer: 54 of 161.
- **F1 +0.67 points**, dragged up by recall, damped by precision.

One sentence to land it: recall moved thirty times more than accuracy for
the same single event, and the event was the most important thing that can
happen on this test set. A metric whose denominator is dominated by the easy
class is structurally unable to notice the rare class. That is not a flaw in
the formula; it is the formula.

---

## Cell: Beat 5, the bridge

```python
always_routine = np.zeros_like(y_test)
```

The laziest possible model: an array of 2,000 zeros, same shape and dtype as
the answer key, meaning predict routine for absolutely everything. No
training, no inference cost, no safety reports read, ever.

```python
print(f"...{accuracy_score(y_test, always_routine):.4f}")
print(f"...{accuracy_score(y_test, y_pred):.4f}")
```

0.9665 versus 0.9395. The do nothing model beats the real model on accuracy
by 2.7 points, purely because 96.65 percent of the test set is routine and
agreeing with the majority is cheap. Meanwhile its recall is exactly zero:
all 67 safety reports sit unread.

This closes the loop opened on slide 8 (the question about what the all
routine model scores) and hands the baton to Section 4: if accuracy can
rank a useless model above a useful one, we need metrics that cannot be
gamed by a constant predictor. Stop the demo on that sentence; naming those
metrics is the next slide's job, and the reveal lands better from the deck.

---

## Cell: Checks

```python
def check(name, fn):
    ...
    try:
        ok = bool(fn())
    except Exception:
        ok = False
```

A twelve assertion micro test suite over the notebook's variables. Each
check is a lambda evaluated inside try except, so a stub that is still
`None` reads as "not yet" rather than a red stack trace; students can Run
All at any stage and get a progress report instead of a crash. Frame it for
the room in their own vocabulary: this is unit testing for an analysis, and
the habit generalizes; evaluation code guards release decisions, so it
deserves tests at least as much as the service code it gates.

Two checks worth a comment if someone reads them closely: the ST-2 check
pins the exact tuple `(1826, 107, 14, 53)`, which is how a swapped ravel
unpacking gets caught, and the ST-5 pair verifies both that `y_flipped` has
exactly one extra positive and that `y_pred` still has 160, proving the
`.copy()` actually protected the original.

---

## Instructor appendix cell

```python
from sklearn.metrics import balanced_accuracy_score, matthews_corrcoef
```

Held out of the live demo on purpose. Balanced accuracy (0.8678) averages
the recall of each class, so the all routine model scores exactly 0.5 by
construction. MCC (0.4880) is a correlation between predictions and truth
using all four cells, and any constant predictor scores exactly 0.0, which
the third print demonstrates. These are Section 4's payoff; use this cell
one on one with fast finishers, not on the projector before the slides make
the argument.

---

## Numbers cross reference

Everything the demo prints, against the deck. All execution verified.

| Quantity | Demo output | Deck |
|---|---|---|
| Rows, positives, base rate | 2000, 67, 0.0335 | 2000, 67, 3.35 percent |
| TP, FN, FP, TN at 0.50 | 53, 14, 107, 1826 | 53, 14, 107, 1826 |
| Accuracy | 0.9395 | 0.9395 |
| Precision | 0.3312 | 0.3312 |
| Recall | 0.7910 | 0.7910 |
| F1 | 0.4670 | 0.4670 |
| Flip: recall after | 0.8060 | recall jumps, accuracy barely twitches |
| Flip: accuracy after | 0.9400 | same |
| All routine accuracy | 0.9665 | 0.9665 |
| Appendix: balanced accuracy | 0.8678 | 0.8678 |
| Appendix: MCC | 0.4880 | 0.4880 |
