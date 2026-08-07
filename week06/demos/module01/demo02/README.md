# Week 6, Module 01, Demo 2: Watch the Interval Collapse

Eight minute live demo for Section 5 of the Module 01 deck. The thirteen
line bootstrap, the interval on the morning's F1, the 200 review slice that
cannot tell a good model from a bad one, and the closing plot of interval
width against positive count with the room's own test set marked on it.

## Contents

| File | What it is |
|---|---|
| `demo_script.md` | Instructor run book: beat by beat, verbatim expected output, talk track, timing, interaction hooks, fallback, recovery table. |
| `Demo02_Watch_the_Interval_Collapse_SOLUTION.ipynb` | Instructor notebook, fully executed. Doubles as the fallback. Two appendix cells: the paired model comparison and the Formative Check 3 support. |
| `Demo02_Watch_the_Interval_Collapse_STUDENT.ipynb` | Follow along notebook. Four YOUR TURN stubs; plumbing pre written. Cold runs clean at 1 of 12 checks. |
| `WALKTHROUGH.md` | Line by line teaching companion in plain terms. |
| `interval_collapse.png` | Pre rendered closing plot: the deck's named fallback for this demo. |
| `cordwell_test_set.csv` | Byte identical to the Demo 1 file (md5 6337bbe5a2c18aed35a7306c4750e8d6). One dataset, all day. |
| `make_dataset.py` | Regenerates the CSV bit for bit. |
| `requirements.txt` | Pinned dependencies with currency notes. |

## Quick start

```
pip install -r requirements.txt
jupyter lab Demo02_Watch_the_Interval_Collapse_SOLUTION.ipynb
```

Run All: about 40 seconds total (the sweep is 14 of them), finishing at 12
of 12 checks. No GPU, no network, no Docker services.

## Deck accuracy review (Section 5 scope)

Reviewed against sklearn 1.8.0 and numpy 2.4.4 by execution:

**Correct as printed.** The bootstrap function on the worked code slide is
current and correct: the coercion rationale, the resample with replacement
idiom, the degenerate resample guard, and the percentile call all check
out, and the function's exact semantics were reproduced here verbatim. The
slice row of "The Answer, Verified" reproduces exactly on the shipped
dataset with a pinned 200 row sample (seed 1066): 6 positives, F1 0.364,
interval [0.095, 0.600], width 0.505. The Formative Check 3 answer's claim
(40 positives gives an interval 0.15 to 0.25 wide) verifies at width 0.214
on this data. The demo box's runtime claim (roughly 15 seconds) measures at
13 to 14 seconds.

**Errata 1: full set interval values.** The deck prints the full set
interval as [0.383, 0.543], width 0.160, including as the inline result
comment on the worked code slide. The shipped dataset, run through the
deck's own function at its own default seed, gives [0.387, 0.549], width
0.162. Both are draws of the same measurement: across bootstrap seeds 0
through 49 on the shipped data, the lower endpoint ranges 0.378 to 0.389,
the upper 0.537 to 0.549, and the width 0.151 to 0.169, so the deck's
numbers sit inside the Monte Carlo spread of the identical quantity. They
cannot, however, be reproduced at the pinned seed on the shipped file.
Recommended fix on the next slide pass: update the three numbers (table row
and code comment) to [0.387, 0.549] and 0.162 so deck and screen agree
character for character. All demo materials in this package quote the
executed values.

**Errata 2 (previously flagged, restated).** The worked code slide calls
the bootstrap "thirteen lines"; the Section 8 recap says "twelve lines."

## Decision needed before the Section 6 demo is built

While verifying this demo, the Section 6 threshold sweep was checked
against the shipped dataset, and it does not match: the dataset was tuned
to reproduce the Section 3, 4, and 5 headline numbers (confusion matrix at
0.50, all derived metrics, ROC-AUC, average precision), which do not
constrain behavior at other thresholds. Examples: at threshold 0.60 the
deck's sweep table says precision 0.565, recall 0.582, F1 0.574, queue 34
per 1,000, while the shipped data gives 0.372, 0.716, 0.490, 32. The F1
optimal threshold on the shipped data is 0.798 (F1 0.559), not the deck's
0.601 (0.5821). The cost table and Brier numbers diverge similarly.

More decisively: parts of the deck's own sweep table are internally
inconsistent, so no dataset can reproduce them. At threshold 0.70 the table
implies 36 flagged reviews (queue 18 per 1,000) with recall 0.388, which
forces TP = 26 and therefore precision 26 over 36 = 0.722, not the printed
0.743. The rows at 0.60 and 0.80 fail the same integer arithmetic.

Recommendation: do not attempt to retune the dataset to Section 6 (its
table is unreachable by construction, and retuning would invalidate the
executed Demo 1 and Demo 2 outputs). Instead, when the Section 6 demo or
lab is built, recompute the sweep, the F1 optimal threshold, the cost
table, the capacity constrained threshold, and the calibration numbers from
the shipped CSV, and update the Section 6 slides to those executed values,
which is the same verification first discipline applied to Errata 1. Flag
kept here so the decision is yours before that build starts.

## Verification ledger

Execution confirmed on Python 3.12.3, numpy 2.4.4, pandas 3.0.2,
scikit-learn 1.8.0, matplotlib 3.10.8:

- Solution notebook executes end to end via nbclient, 12 of 12 checks, in
  about 40 seconds.
- Student notebook cold runs with zero errors, 1 of 12 checks.
- Completability: filling the four stubs with solution bodies and re
  executing yields 12 of 12.
- Full set interval [0.3871, 0.5487] width 0.1616; slice interval [0.0952,
  0.6000] width 0.5048; sweep positives 3, 7, 17, 34, 67 with strictly
  decreasing widths 0.583, 0.460, 0.319, 0.216, 0.162; sweep wall clock 13
  to 14 seconds; paired comparison interval [-0.0196, 0.0591]; 40 positive
  width 0.214. All recomputed from the shipped CSV.
- `interval_collapse.png` is the figure saved by the executed solution
  notebook.

Estimates pending instructor dry run: per beat wall clock timings in
`demo_script.md`.
