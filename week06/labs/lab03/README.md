# Week 6 Lab 1.3: Threshold Tuning for Cordwell Safety Alerts

Cordwell Home and Hardware wants urgent safety tickets, a gas smell, a shifting railing, a hot wall plate, flagged for immediate triage. Students train the classifier in minutes and then spend the lab on the decision the model cannot make: where the cutoff goes. They sweep thresholds on validation, select operating points for competing stakeholders, lock one, and grade it exactly once on the test set.

**Duration:** 120 minutes plus a 10 to 12 minute instructor demo.
**Position:** closing lab of Week 6 Day 1, after Lab 1.1 (confusion matrix and core metrics) and Lab 1.2 (accuracy vs F1 on imbalanced data). Adds zero new API surface on purpose; the new content is the tuning discipline.
**Audience:** engineers new to AI.
**Hardware:** cohort MacBooks, CPU only. scikit-learn does not use the GPU, so no device selection is needed; the full solution notebook executes in well under one minute.
**Services:** none. No Docker containers, no local LLM server, no network calls. The MLflow tracking server (Docker) arrives later in Week 6.

## Files

| File | Audience | Purpose |
|---|---|---|
| `demo_week6_lab03.md` | Instructor | I-do demo script (thresholding as arithmetic on ten hand-written scores) with expected outputs, timing marks, recovery notes, and anticipated questions |
| `week6_lab03_student.ipynb` | Students | Stubbed lab notebook. Cold Run All completes with zero errors at 6 of 27 checks passing |
| `week6_lab03_solution.ipynb` | Instructor only | Fully implemented and executed, 27 of 27 checks passing, including all three stretch goal solutions. Withhold until after the lab |
| `WALKTHROUGH_week6_lab03_solution.md` | Instructor | Line by line commentary on every solution and provided cell, with verified outputs and a verification ledger |
| `HINTS.md` | Students | Progressive tier: three escalating levels per task |
| `HINTS_DETAILED.md` | Students | Detailed tier: working core of each task with commentary, assembly withheld |
| `requirements.txt` | Everyone | Pinned dependencies with currency notes (identical environment to Labs 1.1 and 1.2) |

Distribute to students: the student notebook, both hint files, and `requirements.txt`. Students pick one hint tier per task; both files open with a self-selection note.

## Setup

Machines configured for the morning labs need nothing new. Otherwise:

```bash
pip install -r requirements.txt
```

Then open `week6_lab03_student.ipynb` in JupyterLab and Run All once. Expected: zero errors, `Checks passing: 6/27` (3 environment checks plus 3 corpus checks, since the corpus is provided), and TODO messages marking every unimplemented task.

## Lab shape

Seven tasks with a soft check harness (27 checks total), a worked target output section carrying every number students code toward including the full 19-row sweep table, provided plot cells for the sweep curves and side-by-side confusion matrices, a group discussion block, and an individual reflection. Tasks:

1. The three-way split returns (60-20-20 this time, with the notebook arguing why: today we tune, and tuning needs somewhere to land that is not the test set)
2. The day's pipeline, third build from a blank cell, with a predict-the-flag-count exercise before the apply cell
3. `evaluate_at_threshold`: the reusable instrument; every later artifact is this function in a loop
4. The 19-threshold sweep on validation (Lab 1.1 sampled 3 thresholds; this is the production version)
5. `select_thresholds`: business constraints as filter-then-argmax (safety team recall floor 0.95, call center precision floor 0.90, plus the F1 optimum)
6. The three operating points side by side: three different products from one model
7. Lock the threshold, open the test drawer once, read the validation-to-test gap

Verified headline numbers (seeded, exact): default 0.5 flags 7 of 100 validation tickets, [[85, 0], [8, 7]], F1 0.636; the sweep peaks at threshold 0.25 with F1 0.903; picks are f1_optimal 0.25, safety_team 0.15 (recall 1.000, 14 false alarms), call_center 0.30 (precision 1.000, 4 misses); the locked 0.25 scores 0.98 accuracy, 1.000 precision, 0.867 recall, 0.929 F1 on test, moving one ticket in each direction from validation.

Stretch goals (solutions in the instructor notebook only): shipping the cutoff inside `FixedThresholdClassifier` plus `FrozenEstimator` and verifying it reproduces the Task 7 test row exactly; the precision-recall curve with the three operating points marked, where average precision comes out 0.974, not 1.000, because the loudest routine ticket (score 0.273) outranks the quietest emergency (0.187), the deliberate contrast with Lab 1.2's perfect ranking; and cost-based selection at 8 dollars per false alarm and 400 per miss, where the arithmetic lands on the safety team's threshold at 112 dollars against the F1 optimum's 416 and the call center's 1600.

## Design notes for the instructor

- **The source lab's positives all screamed.** Every urgent ticket carried exactly five safety sentences, so probability scores pile up near the extremes and every threshold from 0.1 to 0.9 behaves identically, which flattens the sweep the lab exists to teach. The rebuilt generator puts urgent tickets on a loudness spectrum (1 to 4 safety sentences, weighted quiet, padded with routine chatter) and gives about one routine ticket in six a single safety-flavored sentence. Scores now spread across the range, the sweep has texture, and the quiet emergencies are exactly what the default 0.5 misses.
- **The source lab seeded the global generators.** `random.seed` plus `np.random.seed` at import time means re-running the corpus cell without re-running imports produces a different corpus. Rebuilt to the course convention: a local `random.Random(seed)` inside `build_corpus`, making the corpus a pure function of its arguments.
- **The seed is 11, not the morning's 7, and the choice is documented.** Seeds 3 through 13 were swept; the lab's requirement was three distinct stakeholder picks with unique winners inside every selection filter (seed 9, among others, collapsed the call-center pick onto the F1 optimum). The imports cell tells students each lab declares its own empirically chosen seed, and the demo's recovery notes flag the muscle-memory-types-7 failure mode.
- **Corpus provided, not a task.** Assembly was Lab 1.2's Task 1; today it is plumbing, per the cognitive-load budget. The intro markdown still walks students through the one generator decision that makes the sweep work.
- **Selection logic gets a fixture.** Task 5's first check grades `select_thresholds` on a hand-checkable four-row synthetic sweep whose correct answers differ from the real ones, so filter-then-argmax bugs are caught independently of Tasks 2 through 4.
- **All numbers are execution-verified.** Every metric quoted in the notebooks, walkthrough, hints, and demo script comes from executing the solution notebook on the pinned stack, with warnings promoted to errors during verification. The ledger at the end of the walkthrough classifies every claim, including the seed sweep and the corpus design tuning.

## Currency flags

- **scikit-learn threshold tooling.** The lab's manual sweep is the pedagogical path; the deployment idioms are `FixedThresholdClassifier` (sklearn 1.5 and later, taught in stretch 1 with `FrozenEstimator`, 1.6 and later) and `TunedThresholdClassifierCV` (1.5 and later, taught in Lab 1.2's stretch). Together the three labs cover the manual, frozen, and cross-validated variants of the same decision.
- **`np.round` on the threshold grid.** Raw `np.linspace` values fail exact float equality against literals (0.15000000000000002 vs 0.15); the contract's `np.round(..., 2)` is required for the row-lookup checks and is called out in both hint tiers as a general floats-as-keys lesson.
- **Week 6 deck dependency, not this lab.** The Week 6 outline names OpenAI evals for eval harnesses. The OpenAI Evals platform API shuts down November 30, 2026, and the `gpt-4` legacy family shuts down October 23, 2026. This lab has no OpenAI dependency; the flag is repeated here from the earlier READMEs so it stays visible at the start of the week.
