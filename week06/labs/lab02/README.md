# Week 6 Lab 1.2: Accuracy vs F1 on Imbalanced Data

Cordwell Home and Hardware's support queue tags a small fraction of tickets as high risk: a completed installation actively causing safety or property damage. Students build the classifier that catches them, watch accuracy fail to notice whether it works, and fix both the metric and the model.

**Duration:** 120 minutes plus a 12 to 15 minute instructor demo.
**Position:** afternoon of Week 6 Day 1, immediately after Lab 1.1 (confusion matrix and core metrics). Assumes Lab 1.1's pipeline, metrics functions, and split discipline are fresh.
**Audience:** engineers new to AI.
**Hardware:** cohort MacBooks, CPU only. scikit-learn does not use the GPU, so no device selection is needed; the full solution notebook executes in well under one minute.
**Services:** none. No Docker containers, no local LLM server, no network calls. The MLflow tracking server (Docker) arrives later in Week 6.

## Files

| File | Audience | Purpose |
|---|---|---|
| `demo_week6_lab02.md` | Instructor | I-do demo script (the accuracy trap and DummyClassifier on a 20-row toy) with expected outputs, timing marks, recovery notes, and anticipated questions |
| `week6_lab02_student.ipynb` | Students | Stubbed lab notebook. Cold Run All completes with zero errors at 3 of 26 checks passing |
| `week6_lab02_solution.ipynb` | Instructor only | Fully implemented and executed, 26 of 26 checks passing, including all three stretch goal solutions. Withhold until after the lab |
| `WALKTHROUGH_week6_lab02_solution.md` | Instructor | Line by line commentary on every solution and provided cell, with verified outputs and a verification ledger |
| `HINTS.md` | Students | Progressive tier: three escalating levels per task |
| `HINTS_DETAILED.md` | Students | Detailed tier: working core of each task with commentary, assembly withheld |
| `requirements.txt` | Everyone | Pinned dependencies with currency notes (identical environment to Lab 1.1) |

Distribute to students: the student notebook, both hint files, and `requirements.txt`. Students pick one hint tier per task; both files open with a self-selection note.

## Setup

Machines configured for Lab 1.1 need nothing new. Otherwise:

```bash
pip install -r requirements.txt
```

Then open `week6_lab02_student.ipynb` in JupyterLab and Run All once. Expected: zero errors, `Checks passing: 3/26`, and TODO messages marking every unimplemented task. The cold-run pass count is lower than Lab 1.1's (3, not 6) because this lab's corpus is student work (Task 1), not provided plumbing; the notebook says so where the count is stated.

## Lab shape

Seven tasks with a soft check harness (26 checks total), a worked target output section showing every number students code toward, a group discussion block, and a written summary. Tasks:

1. Assemble the imbalanced corpus (500 tickets, 10 percent high risk; sentence pools and ticket builders provided, dataset engineering is the work)
2. Stratified 80-20 split (two-way, with the notebook explaining why no validation set today)
3. Majority baseline via DummyClassifier, the production idiom for the baseline built by hand in Lab 1.1
4. One shared evaluation function (metrics dict plus confusion matrix) that grades every model in the lab
5. The Lab 1.1 pipeline rebuilt from a blank cell, where the accuracy trap springs: 0.920 accuracy, 0.200 recall, decoy precision of 1.000
6. class_weight="balanced": recall 0.200 to 1.000 for the price of one false alarm
7. The comparison table that puts the whole argument in three rows

Verified headline numbers (seeded, exact): majority baseline [[90, 0], [10, 0]] at 0.900 accuracy and 0.000 everything else; plain logistic regression [[90, 0], [8, 2]] at 0.920, 1.000, 0.200, 0.333; balanced [[89, 1], [0, 10]] at 0.990, 0.909, 1.000, 0.952.

Stretch goals (solutions in the instructor notebook only): the chance-level stratified dummy; average precision and the PR curve, where students discover the plain model's ranking of the test set is perfect and the 0.5 cutoff was the entire failure; TunedThresholdClassifierCV, which finds the 0.199 cutoff by cross validation inside the training data and closes the loop with Lab 1.1's threshold lesson.

## Design notes for the instructor

- **The source lab's corpus leaked labels into the text.** Its class-specific resolution sentences literally wrote the label into every ticket ("The ticket is marked as routine with no immediate safety or property risk noted"), which makes any classifier perfect and leaves no errors to dissect. The rebuilt corpus shares one neutral resolution pool across both classes and adds two saboteur pools: hard negatives (genuine complaints, dense with installation vocabulary, zero danger) and hard positives (real risk reported in a calm voice: a faint gas smell, a warm outlet cover). The hard positives are what the plain model misses, which is what gives the lab its errors and its argument.
- **Same seed as Lab 1.1, chosen empirically.** `RANDOM_SEED = 7` with a 10 percent positive rate and hard-case rates of 0.4 (positives) and 0.25 (negatives) is the configuration, out of a swept grid, where the plain model catches 2 of 10 rather than 0 of 10 (most configurations collapse it into the dummy, too extreme to teach the middle ground) and the balanced model catches all 10 at one false alarm. The walkthrough's verification ledger records the sweep.
- **Task ordering mirrors Lab 1.1 on purpose.** The pipeline task is deliberately identical (spaced repetition from a blank cell); the metrics and confusion matrix tasks merge into one shared evaluator, which is the design idea (one grader for all models) that MLflow runs will inherit later in the week.
- **All numbers are execution-verified.** Every metric quoted in the notebooks, walkthrough, hints, and demo script comes from executing the solution notebook on the pinned stack, with warnings promoted to errors during verification. The ledger at the end of the walkthrough classifies every claim.

## Currency flags

- **seaborn dropped (change from the source notebook).** The source lab imported seaborn solely for confusion matrix heatmaps. The lab now uses sklearn's `ConfusionMatrixDisplay`, matching Lab 1.1, and the dependency is removed from the stack. If a student installs seaborn anyway nothing breaks; it simply is not needed.
- **scikit-learn threshold and imbalance tooling.** `class_weight="balanced"` is long-stable API. Stretch 3 teaches `TunedThresholdClassifierCV` (scikit-learn 1.5 and later); Lab 1.1's stretch goals covered `FixedThresholdClassifier` and `FrozenEstimator` (1.6), and the two labs cross-reference so students see loss reweighting and threshold moving as the two levers for imbalance.
- **Resampling libraries deliberately out of scope.** If students ask about SMOTE or oversampling, the demo script's anticipated questions carry the answer: same family of fix, lives in imbalanced-learn, comes with fold-leakage traps, flagged as further reading rather than a live detour.
- **Week 6 deck dependency, not this lab.** The Week 6 outline names OpenAI evals for eval harnesses. The OpenAI Evals platform API shuts down November 30, 2026, and the `gpt-4` legacy family shuts down October 23, 2026. This lab has no OpenAI dependency; the flag is repeated here from Lab 1.1's README so it stays visible at the start of the week.
