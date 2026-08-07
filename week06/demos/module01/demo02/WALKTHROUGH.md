# Demo 2 Walkthrough: every line, in plain terms

Teaching companion to `Demo02_Watch_the_Interval_Collapse_SOLUTION.ipynb`.
Every line explained the way you would to a strong software engineer who has
never touched statistics beyond a unit test's pass rate. Read before
teaching; steal phrasing freely.

The one idea underneath the whole demo, stated up front because everything
else is implementation: **a metric computed on a sample is itself a
measurement with noise**, and the bootstrap is a way to measure that noise
using nothing but the sample you already have. The engineering analogy that
lands: you would never report a latency benchmark from one run; you run it
many times and report the spread. The test set is one run. The bootstrap
manufactures the reruns.

---

## Cell: Setup

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
```

Only one new import versus Demo 1: `train_test_split`, and it is not being
used to split anything today. It reappears later as a convenient stratified
sampler for the sweep. `f1_score` is the metric under the microscope, but
nothing in this demo is F1 specific; the function built in Beat 1 takes the
metric as an argument.

```python
df = pd.read_csv("cordwell_test_set.csv")
y_test = df["y_true"].to_numpy()
y_score = df["y_score"].to_numpy()
y_pred = (y_score >= 0.50).astype(int)
```

The identical file from Demo 1, byte for byte, and the identical threshold.
Continuity is the point: the 0.4670 this demo interrogates is the same
0.4670 the room computed by hand this morning. The `.to_numpy()` coercions
matter even more today than they did in Demo 1, for a reason that gets its
own paragraph in Beat 1.

---

## Cell: Beat 1, the bootstrap function

This is the intellectual center of the demo, so it gets the slow treatment.

### The idea before the code

The question: our F1 is 0.4670 on the 2,000 reviews we happened to collect.
If the universe handed us a different 2,000 reviews from the same stream,
what number would we get? If the answer is "somewhere between 0.46 and
0.47," the four decimals are earned. If it is "somewhere between 0.39 and
0.55," they are not.

The obvious experiment, collecting fifty more test sets, costs fifty times
the labeling budget. The bootstrap fakes it: build each pretend test set by
drawing 2,000 rows **from our own test set, with replacement**. With
replacement is the crux, and worth a full beat of explanation: each draw
picks uniformly from all 2,000 rows, so a given review can be picked twice,
three times, or never. Every resample therefore differs from the original
in exactly the way a freshly collected sample would: some kinds of examples
over represented, some under, purely by luck. Score each pretend test set,
collect 2,000 scores, and the spread of those scores estimates the spread
you would have seen across real re collections. It is a simulation of
sampling luck, run on the only sample you have.

Why the middle 95 percent of the scores, rather than the min and max: the
extreme tails are dominated by the flukiest resamples, so by convention we
report the range that covers 95 percent of the simulated outcomes, cutting
2.5 percent off each end. Hence the percentiles 2.5 and 97.5.

### The code, line by line

```python
def bootstrap_ci(y_true, y_pred, metric, n_boot=2000, seed=0):
```

The signature makes three promises. `metric` is a function argument, the
same callback idiom as Demo 1's delta table helper, so this one tool puts
an interval on F1 today, on precision tomorrow, on Cohen's kappa in
Section 7, unchanged. `n_boot=2000` is how many pretend test sets to score;
more gives smoother percentiles at linear cost. `seed=0` pins the random
draws so every run of this notebook, on every machine in the room, prints
identical numbers. Determinism is a courtesy in a demo and a requirement in
an experiment log.

```python
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
```

The slide says these two lines are not decoration, and here is the plain
version of why. A pandas Series remembers row labels. After a shuffling
operation like `train_test_split`, those labels are scrambled: position 0
of the Series might carry label 1847. numpy arrays have no labels; position
0 is just position 0. The line `y_true[idx]` below hands over an array of
positions. Give that to a numpy array and you get the rows at those
positions, which is what the bootstrap means. Give it to a pandas Series
and pandas tries to treat them as labels, which either raises a `KeyError`
or, worse, silently returns different rows. `np.asarray` converts a Series
to a plain positional array and costs nothing if the input is already one.
Coerce at the boundary, then reason in one indexing dialect inside: that is
a habit worth naming for a room of engineers, because it is the same
discipline as validating types at an API boundary.

```python
    rng = np.random.default_rng(seed)
```

The modern numpy random API: build a generator object with a seed and draw
from it, rather than mutating hidden global state via the legacy
`np.random.*` calls. Two demos in two weeks have now quietly modeled this;
if someone asks, the one line answer is that a passed around generator
makes randomness local, testable, and reproducible, like dependency
injection for luck.

```python
    n, vals = len(y_true), []
```

`n` rows in the real test set; `vals` will collect one metric value per
pretend test set.

```python
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
```

The resample in one line: draw `n` integers, each uniformly from 0 up to
but not including `n`. Those are row positions, duplicates welcome, and
duplicates are the whole point. `idx` IS the pretend test set, expressed as
positions into the real one.

```python
        if y_true[idx].sum() == 0:
            continue
```

The degenerate resample guard. By luck a resample can contain zero
positives; F1 involves dividing by counts of positives, so the metric is
meaningless there and the resample is skipped. Two honest footnotes worth
having in your pocket. First, how often it fires depends on scale: with 67
positives in 2,000 rows the odds of drawing none are astronomically small,
but at the small end of Beat 4's sweep (3 positives in 100 rows) roughly
one resample in twenty is degenerate, and the function quietly runs on the
rest. Second, skipping the worst case resamples is a mild optimism bias:
the real spread includes those disasters and the reported interval does
not. At this severity it is a footnote; the deeper fix (stratified
bootstrap) is a fine answer to hold for a sharp question rather than volunteer.

```python
        vals.append(metric(y_true[idx], y_pred[idx], zero_division=0))
```

Score the pretend test set: index both truth and predictions by the same
positions, so each resampled row keeps its own prediction paired with its
own label, and hand both to the metric. `zero_division=0` is passed through
because a rare resample can produce a zero denominator inside the metric
even when positives exist (all of them predicted negative, say), and Demo 1
already established what this flag does: decide the edge case explicitly,
silence the warning.

```python
    return np.percentile(vals, [2.5, 97.5])
```

Sort the 2,000 scores conceptually and read off the values below which 2.5
percent and 97.5 percent of them fall. What comes back is a two element
array: the 95 percent confidence interval.

One framing sentence for the room, because the term "confidence interval"
carries scar tissue from statistics courses: this one requires no formulas,
no distribution tables, and no assumptions beyond "my test set resembles
the stream it came from." It is thirteen lines and a for loop. The price of
that simplicity was 2,000 metric evaluations, roughly four seconds of CPU,
which in 2026 is free.

---

## Cell: Beat 2, the full set interval

```python
ci_full = bootstrap_ci(y_test, y_pred, f1_score)
width_full = ci_full[1] - ci_full[0]
```

Nothing new mechanically; everything new rhetorically. The output:

```
F1 point estimate: 0.4670
95% interval:      [0.3871, 0.5487]
Width:             0.1616
```

Three readings to offer, in escalating order of usefulness:

1. The honest sentence: "F1 is 0.47, give or take 0.08."
2. The decimal place audit: the first decimal is trustworthy, the second is
   in doubt, the third and fourth communicate nothing but false confidence.
3. The engineering translation: this is the error bar every benchmark
   number deserves. A team that would laugh at a latency report from one
   run has been nodding at 0.4670 all morning.

Worth pre empting a sharp question here: the interval endpoints themselves
are Monte Carlo estimates from 2,000 resamples, so a different bootstrap
seed moves them by a few thousandths. That is why the demo pins the seed,
and it is one more reason not to worship third decimals, including these.

---

## Cell: Beat 3, the slice

```python
SLICE_SEED = 1066   # pinned: reproduces the slice on the slide exactly

slice_df = df.sample(n=200, random_state=SLICE_SEED)
```

`DataFrame.sample` draws 200 rows without replacement: a random subset,
which is exactly what "we evaluated on a quick 200 review sample" means in
practice. The seed is pinned for the usual two reasons, reproducibility
across the room and agreement with the slide table, and the comment says so
rather than leaving 1066 to look like numerology.

```python
y_slice = slice_df["y_true"].to_numpy()
p_slice = (slice_df["y_score"].to_numpy() >= 0.50).astype(int)
```

The same two moves as the setup cell, on the slice: coerce to positional
arrays, cut the scores at the same 0.50. Note that predictions are recut
from the slice's own scores rather than sliced out of `y_pred`; the two are
equivalent here, and recomputing keeps the cell self contained.

```python
ci_slice = bootstrap_ci(y_slice, p_slice, f1_score)
```

Same function, no changes: the payoff of writing it generically. Output:

```
Rows: 200   Positives: 6
F1 point estimate: 0.3636
95% interval:      [0.0952, 0.6000]
Width:             0.5048
```

The two sentences that matter. First: an interval from 0.10 to 0.60 spans
essentially the entire useful range for this task, so this evaluation can
distinguish a model from a coin and nothing finer; and note it cannot be
rescued by more resamples, because `n_boot` controls the smoothness of the
estimate, not the amount of information in six positives. Second: the
governing quantity was never the 200 rows. It was the 6 positives. Rare
positive problems size their test sets in positives, and a proposal to
"evaluate on a quick sample of N rows" should be immediately translated to
"evaluate on roughly 0.03 times N positives" and judged in that currency.

The slide table's slice row and this cell agree exactly, so the room can
check the deck against the screen.

---

## Cell: the subset helper

```python
def subset(frame, n_rows, seed=0):
    if n_rows >= len(frame):
        return frame
    sub, _ = train_test_split(
        frame,
        train_size=n_rows,
        stratify=frame["y_true"],
        random_state=seed,
    )
    return sub
```

Pre written plumbing for the sweep, three decisions inside it worth being
able to defend:

- `train_test_split` as a sampler: asking it for a stratified "train" part
  of size N and discarding the rest is the idiomatic one liner for "give me
  a random N row subset that preserves class balance." The underscore is
  the discarded remainder.
- `stratify=frame["y_true"]`: the morning's split slide said stratify is
  not optional when positives are rare, and here is the payoff. Stratified
  sampling pins the positive count at each size to 3, 7, 17, 34 instead of
  letting it wander, which makes the sweep a clean experiment about size
  rather than a noisy experiment about size and luck simultaneously.
- The early return: sampling 2,000 of 2,000 is the full set, and
  `train_test_split` would refuse anyway (it must leave at least one row
  behind), so the top size short circuits.

---

## Cell: Beat 4, the sweep

```python
SWEEP_SIZES = [100, 200, 500, 1000, 2000]
sweep_rows = []
```

Five pretend labeling budgets and a list to collect one result row per
budget.

```python
for N in SWEEP_SIZES:
    sub = subset(df, N)
    yt = sub["y_true"].to_numpy()
    yp = (sub["y_score"].to_numpy() >= 0.50).astype(int)
    lo, hi = bootstrap_ci(yt, yp, f1_score)
```

Each pass is Beat 2 in miniature, on a smaller world: take a stratified
subset, coerce, cut at 0.50, bootstrap. `lo, hi` unpacks the two element
array the function returns. The whole loop is 2,000 resamples times five
sizes, about 14 seconds, and the print inside the loop makes the wait
legible: rows land one at a time.

```python
    sweep_rows.append({
        "N": N,
        "positives": int(yt.sum()),
        "f1": f1_score(yt, yp),
        "lo": lo,
        "hi": hi,
        "width": hi - lo,
    })
```

A dict per size: rows of a small results table. `int(yt.sum())` records the
positive count, which is about to become the x axis, because the entire
thesis is that positives, not rows, are the currency.

The executed table:

```
    N  pos     F1      lo      hi   width
  100    3  0.462   0.167   0.750   0.583
  200    7  0.345   0.095   0.556   0.460
  500   17  0.459   0.281   0.600   0.319
 1000   34  0.462   0.348   0.564   0.216
 2000   67  0.467   0.387   0.549   0.162
```

Two columns to read aloud. The width column collapses monotonically: 0.58,
0.46, 0.32, 0.22, 0.16. And the F1 column wobbles: 0.46, 0.35, 0.46, 0.46,
0.47. Same model, same distribution, honest samples of different sizes.
That wobble is not a bug in the small samples; it IS sampling noise, the
very quantity the interval reports. A team comparing two models on 200 row
test sets would be ranking coin flips.

---

## Cell: the plot

```python
xs = [r["positives"] for r in sweep_rows]
ws = [r["width"] for r in sweep_rows]
```

Positives on x, width on y. Not N on x: the deck says plot against positive
count, and the reason is the lesson itself.

```python
ax.plot(xs, ws, marker="o", linewidth=2)
for r in sweep_rows:
    ax.annotate(f"N={r['N']}", ...)
```

The curve, with each point labeled by its row count so the room can hold
both scales at once: 2,000 rows is the point labeled N=2000 sitting at
x=67.

```python
ax.axvline(67, linestyle="--", color="gray")
ax.annotate("your test set\n(67 positives)", ...)
```

The memorable moment the deck names. The dashed line is the room's own test
set, placed on the curve. Everything to the left is a worse evaluation than
the one they have; the flattening slope to the right is what more labeling
budget would buy.

```python
fig.savefig("interval_collapse.png", dpi=150)
```

Writes the fallback plot the deck calls for. The shipped copy in this
folder came from exactly this line.

The square root framing for the closing beat, in plain terms: statistical
noise shrinks with the square root of the sample size, so quadrupling the
positives roughly halves the width. Check it against the table: 17
positives gave 0.32, and four times that, 67 (near enough), gave 0.16.
Halving the width again means finding roughly 200 more genuine safety
reports, which at a 3.35 percent base rate means labeling roughly 6,000
more reviews. That arithmetic, done on this curve, is how you turn "we
should improve our eval" into a budget line.

---

## Cell: Checks

Same soft harness as Demo 1: each check evaluates inside try except, `None`
stubs read as "not yet," and the pinned seeds make exact value checks
possible. Two worth pointing at if a student reads them: the fixture check
runs `bootstrap_ci` on an eight element toy so ST-1 is validated
independently of everything downstream, and the final check asserts the
sweep's 2,000 row interval equals Beat 2's interval to the last bit, which
is only true because the same seeded generator makes the same draws, a
quiet proof that the determinism actually holds.

---

## Instructor appendix cells

**Appendix 1** makes the next slide's hypothetical real. The comparison
slide posits model A at 0.467 and model B at 0.489 with overlapping
intervals; cutting the same scores at 0.60 instead of 0.50 produces
exactly that B. The cell then does the thing the slide recommends: instead
of comparing two separate intervals, it bootstraps **the difference**,
scoring both prediction sets on the same resampled rows and taking the
percentile interval of B minus A. Pairing on the same rows cancels the
shared luck (a hard resample is hard for both models), which is why the
difference interval comes out at width 0.079 versus 0.162 for either model
alone: the promised "tighter." And the verdict still straddles zero, so
even the sharper instrument finds no real difference between 0.467 and
0.489 on 67 positives. Both halves of the slide's claim, executed. Two
cautions: do not run it live before the comparison slide, and if asked why
B is "a different model" when it is the same scores at a different cut, the
honest answer is that it is a stand in with exactly the pedagogical
properties needed, and threshold choice becomes its own topic in Section 6.

**Appendix 2** is ammunition for Formative Check 3, which is the very next
thing after this demo: someone reports F1 to three decimals from 40
positives. The first 1,201 rows of the shipped file contain exactly 40
positives, and the executed interval width is 0.214, inside the 0.15 to
0.25 band the answer slide claims. If a student challenges the claim, this
cell settles it in four seconds.

---

## Numbers cross reference

| Quantity | Demo output | Deck |
|---|---|---|
| Full set F1 | 0.4670 | 0.467 |
| Full set 95% interval | [0.3871, 0.5487] | [0.383, 0.543] (see errata note in README) |
| Full set width | 0.1616 | 0.160 (same errata note) |
| Slice rows, positives | 200, 6 | 200, 6 |
| Slice F1 | 0.3636 | 0.364 |
| Slice interval | [0.0952, 0.6000] | [0.095, 0.600] |
| Slice width | 0.5048 | 0.505 |
| Sweep sizes | 100, 200, 500, 1000, 2000 | same |
| Sweep runtime | 13 to 14 s measured | roughly 15 s |
| 40 positive width (appendix) | 0.214 | 0.15 to 0.25 claimed |
| Paired comparison A, B (appendix) | 0.4670, 0.4898 | 0.467, 0.489 posed as hypothetical |
