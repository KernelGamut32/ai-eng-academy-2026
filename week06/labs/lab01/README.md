# Week 6 Lab 01: Confusion Matrix and Core Metrics with scikit-learn

Cordwell Home and Hardware support triage: build and evaluate a text classifier that routes customer messages to the installation issue queue or the general inquiry queue, then learn to read every number that evaluation produces.

**Duration:** 120 minutes plus a 20 to 25 minute instructor demo.
**Audience:** engineers new to AI. Assumes Weeks 1 through 5.
**Hardware:** cohort MacBooks, CPU only. scikit-learn does not use the GPU, so no device selection is needed; total runtime for the full solution notebook is well under one minute.
**Services:** none. No Docker containers, no local LLM server, no network calls. The MLflow tracking server (Docker) and TruLens arrive later in Week 6; today is deliberately dependency-light so the concepts carry the time.

## Files

| File | Audience | Purpose |
|---|---|---|
| `demo_week6_lab01.md` | Instructor | I-do demo script from cold environment to the Task 1 hand-off, with expected outputs, timing marks, recovery notes, and anticipated questions |
| `week6_lab01_student.ipynb` | Students | Stubbed lab notebook. Cold Run All completes with zero errors at 6 of 28 checks passing |
| `week6_lab01_solution.ipynb` | Instructor only | Fully implemented and executed, 28 of 28 checks passing, including all three stretch goal solutions. Withhold until after the lab |
| `WALKTHROUGH_week6_lab01_solution.md` | Instructor | Line by line commentary on every solution and provided cell, in plain terms, with verified outputs and a verification ledger |
| `HINTS.md` | Students | Progressive tier: three escalating levels per task |
| `HINTS_DETAILED.md` | Students | Detailed tier: working core of each task with line by line commentary, assembly withheld |
| `requirements.txt` | Everyone | Pinned dependencies with currency notes |

Distribute to students: the student notebook, both hint files, and `requirements.txt`. Students pick one hint tier per task; both files open with a self-selection note.

## Setup

```bash
pip install -r requirements.txt
```

Then open `week6_lab01_student.ipynb` in JupyterLab and Run All once. Expected: zero errors, `Checks passing: 6/28`, and TODO messages marking every unimplemented task.

## Lab shape

Seven tasks with a soft check harness (28 checks total), each task preceded by its concept in the notebook and followed by an apply cell and a checks cell. A Worked target output section shows the exact numbers students are coding toward before any task begins. Tasks:

1. Stratified train, validation, and test split (70-15-15)
2. TF-IDF plus Logistic Regression pipeline
3. Reusable compute_metrics function, applied to validation
4. Confusion matrix and metrics on the held out test set
5. Decision threshold sweep and the precision-recall trade
6. Majority class baseline on an imbalanced corpus
7. Logistic Regression on the imbalanced corpus: accuracy versus F1

Stretch goals (solutions in the instructor notebook only): class_weight balanced, FixedThresholdClassifier with FrozenEstimator, TunedThresholdClassifierCV.

## Design notes for the instructor

- **The corpus is deliberately imperfect.** About 20 percent of messages are borderline, mixing installation service questions with hedged concerns, and their labels reflect the disagreement real labelers would have. This puts a ceiling below 100 percent on any classifier and guarantees the confusion matrix contains both error types. A student who claims the borderline messages are mislabeled has found the discussion, not a bug; the demo script's recovery notes cover it.
- **One seed everywhere.** `RANDOM_SEED = 7` drives corpus generation, both splits, and the model, so every student's numbers match the worked target output exactly and the checks can assert exact values. The seed was chosen so the trained classifier makes a realistic mix of both error types.
- **All numbers are execution-verified.** Every metric quoted in the notebooks, walkthrough, hints, and demo script comes from executing the solution notebook on the pinned stack. The verification ledger at the end of the walkthrough classifies every claim.

## Currency flags

- **scikit-learn threshold tooling.** The lab teaches thresholding by slicing `predict_proba` by hand because that is the transparent version. Current scikit-learn (1.5 and later) provides `FixedThresholdClassifier` and `TunedThresholdClassifierCV` as the production idiom, and 1.6 added `FrozenEstimator` for wrapping already fitted models. All three are taught in the stretch goals rather than flagged and skipped.
- **Week 6 deck dependency, not this lab.** The Week 6 outline names OpenAI evals for eval harnesses. The OpenAI Evals platform API shuts down November 30, 2026, and the `gpt-4` family shuts down October 23, 2026. This lab has no OpenAI dependency, but the eval harness lab later this week and the corresponding slides need replacements before those dates. Flagged here so it is visible at the start of the week.
