# Week 6, Module 01, Demo 1: Read the Matrix

Ten minute live demo for Section 3 of the Module 01 deck (Model Evaluation
Metrics and Experiment Tracking). Confusion matrix anatomy, the four core
metrics, and the single prediction flip that shows why accuracy cannot see
what Cordwell cares about.

## Contents

| File | What it is |
|---|---|
| `demo_script.md` | Instructor run book: beat by beat commands, verbatim expected output, talk track, timing, interaction hooks, fallback, recovery table. |
| `Demo01_Read_the_Matrix_SOLUTION.ipynb` | Instructor notebook, fully executed. Doubles as the fallback if anything dies live. Includes an optional appendix cell (balanced accuracy and MCC) for fast finishers. |
| `Demo01_Read_the_Matrix_STUDENT.ipynb` | Follow along notebook. Six YOUR TURN stubs (one to three lines each); all boilerplate pre written. Cold runs clean with zero errors at 1 of 12 checks. |
| `WALKTHROUGH.md` | Line by line teaching companion in plain terms, written for engineers new to AI. |
| `cordwell_test_set.csv` | 2,000 synthetic Cordwell reviews: id, text, ground truth label, model score. |
| `make_dataset.py` | Regenerates the CSV bit for bit. Seeds pinned. |
| `requirements.txt` | Pinned dependencies with currency notes. |

## Quick start

```
pip install -r requirements.txt
jupyter lab Demo01_Read_the_Matrix_SOLUTION.ipynb
```

Run All. The final checks cell should report 12 of 12 passing in under one
second. No GPU, no network, no Docker services: this demo is deliberately
dependency free so it can never fail on stage, and that design choice is
itself a teaching point (the deck's own rule that an evaluation harness must
not depend on a call that can fail mid class).

## The dataset

Synthetic and clearly fictional. `y_score` is the output of a simulated
classifier and is the source of truth for every metric; `review_text` is
templated flavor so error analysis moments have something human to read. The
generator was tuned so the file reproduces the deck's published numbers
exactly, including the score based metrics later demos in this module will
use:

- Confusion matrix at threshold 0.50: TP=53, FN=14, FP=107, TN=1826
- Accuracy 0.9395, precision 0.3312, recall 0.7910, F1 0.4670, F2 0.6192
- Balanced accuracy 0.8678, MCC 0.4880
- ROC-AUC 0.9621, average precision 0.6040, base rate 0.0335

Note on seeds: the deck's scenario slide says "seed=42". In this generator
seed 42 drives the text templating and row shuffle; the score distribution
uses its own pinned internal seed (documented in `make_dataset.py`) because
it was tuned specifically to land on the deck's numbers. Do not change
either seed if the deck and the notebooks need to agree. The same CSV should
be reused by Demo 2 and the Section 6 threshold material so every number in
the room stays consistent.

## Accuracy review notes (deck, Demo 1 scope)

Reviewed the Demo 1 relevant slides against sklearn 1.8.0 by execution:

- The ravel order claim (tn, fp, fn, tp), the zero_division behavior, the
  majority baseline precision and recall claims, and every quoted metric
  value are correct as printed. No corrections required in Demo 1 scope.
- One improvement folded into the demo rather than the deck: the demo passes
  `labels=[0, 1]` explicitly to `confusion_matrix` and the display call. The
  deck's snippet omits it and is correct on this data; the explicit form is
  the defensive habit worth teaching (single class slices, string labels).
- Cosmetic deck inconsistency outside Demo 1 scope, flagged for a future
  slide pass: the Section 5 worked code slide calls the bootstrap "thirteen
  lines" while the Section 8 recap says "twelve lines".

## Verification ledger

Execution confirmed on Python 3.12.3, numpy 2.4.4, pandas 3.0.2,
scikit-learn 1.8.0, matplotlib 3.10.8:

- Solution notebook executes end to end via nbclient, 12 of 12 checks.
- Student notebook cold runs with zero errors, 1 of 12 checks.
- Completability: filling the six stubs with solution bodies and re
  executing yields 12 of 12.
- CSV regeneration is deterministic; all metric values above recomputed from
  the shipped file.

Estimates pending instructor dry run: per beat wall clock timings in
`demo_script.md`.
