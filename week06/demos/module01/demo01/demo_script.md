# Demo 1: Read the Matrix

Module 01, Section 3. Ten minutes, live, CPU only. No network, no model call,
no Docker services. That is not a limitation, it is the design: this demo can
never die on stage, and you should say so out loud, because "the evaluation
harness must not depend on a network call that can fail mid class" is itself
a slide later today.

## What this demo proves

1. Every metric in Section 3 is arithmetic on four counts: tn, fp, fn, tp.
2. The `.ravel()` order matters, and getting it backwards fails silently.
3. Accuracy is structurally unable to see the thing Cordwell cares about.
   You prove it by flipping a single missed safety report to caught and
   watching recall jump 1.49 points while accuracy moves 0.05.
4. Bridge to Section 4: the do nothing baseline outscores the model on
   accuracy, 0.9665 to 0.9395.

## Files

| File | Purpose |
|---|---|
| `Demo01_Read_the_Matrix_SOLUTION.ipynb` | Your copy. Fully executed. This is also the fallback. |
| `Demo01_Read_the_Matrix_STUDENT.ipynb` | Follow along copy. Six short YOUR TURN stubs, everything else pre written. |
| `cordwell_test_set.csv` | 2,000 synthetic reviews, 67 positives, scores from a simulated classifier. |
| `make_dataset.py` | Regenerates the CSV bit for bit. Seeds are pinned; do not change them. |

## Pre-flight checklist (5 minutes, before class)

- [ ] `cordwell_test_set.csv` sits in the same directory as both notebooks.
- [ ] Fresh kernel, Run All on your SOLUTION copy: last check cell reports
      `12 of 12 checks passing`. Under one second of compute.
- [ ] Run All on an untouched STUDENT copy: no red cells, checks report
      `1 of 12 checks passing`. If a student's cold notebook shows anything
      else, they have a stale file.
- [ ] Projector font size: the confusion matrix print line and the delta
      table are the two moments the back row must be able to read.
- [ ] Decide now which machine drives. If you type live, type in the STUDENT
      copy so the room sees exactly what they see.

## Room setup

Students open `Demo01_Read_the_Matrix_STUDENT.ipynb` and Run All before you
start. Tell them: the setup cell and every plotting or table cell is pre
written on purpose. The six stubs marked YOUR TURN are the only typing, and
each is one to three lines. The checks cell at the bottom is theirs to rerun
as often as they like.

## Beat by beat

Timings assume you type the stub lines live in the student copy while
narrating. Total 10 minutes with about a minute of slack.

### Beat 0: Frame it (30 seconds, talk only)

> "The slides just gave you a 2x2 grid and four metrics. Ten minutes of code
> to make that concrete, on the Cordwell test set from this morning. Watch
> for one thing: at the end I will change exactly one prediction out of two
> thousand, and I want you to predict which numbers move."

### Beat 1: A score is not a decision (1.5 min)

Run the setup cell (pre written), then type ST-1:

```python
THRESHOLD = 0.50
y_pred = (y_score >= THRESHOLD).astype(int)
```

Expected output after the guarded print:

```
Reviews flagged for escalation: 160 of 2000
```

Talk track:

> "The model gives us a score per review. Nobody can act on a 0.71. The
> comparison against a threshold is where a score becomes a decision, and I
> am writing it out by hand so you can see the decision exists. If you call
> `predict()` instead, this exact line still runs, you just do not get to see
> it, and the 0.50 is chosen for you. This afternoon we will have strong
> opinions about that number. For now we take the default."

Point at `160 of 2000`: the model wants humans to look at 8 percent of the
queue. Hold that thought for Section 6.

### Beat 2: The four counts (2.5 min)

Type ST-2:

```python
tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()
```

Expected output:

```
TP=53  FN=14  FP=107  TN=1826
```

Interaction hook, before running: "Which of the four cells will be the
biggest number?" Everyone says TN; confirm and explain why that is exactly
the imbalance problem in one picture.

Talk track on the two things worth memorizing:

> "First: `.ravel()` flattens the grid in a fixed order, tn, fp, fn, tp.
> Memorize it. If you unpack these backwards, nothing crashes. You get four
> plausible looking numbers and your precision and recall silently swap.
> Second: I passed `labels=[0, 1]` explicitly. Without it, sklearn infers the
> order from the data. On this dataset that happens to be the same thing, but
> on the day your test slice contains only one class, explicit labels are the
> difference between a correct matrix and a shape error."

Run the pre written display cell. Read the four cells in Cordwell terms,
slowly, pointing: 53 caught, 14 safety reports sitting unread, 107 wasted
triage reviews, 1,826 correctly ignored. The 14 is the cell with a cost.

### Beat 3: Four metrics off the matrix (2 min)

Type ST-3:

```python
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
```

Expected output:

```
Accuracy : 0.9395
Precision: 0.3312
Recall   : 0.7910
F1       : 0.4670
Recall by hand: TP / (TP + FN) = 53 / 67 = 0.7910
```

Talk track:

> "Accuracy 94 percent, feels great, park that feeling. Precision 0.33: two
> thirds of what we escalate turns out to be routine. Recall 0.79: we still
> miss one genuine safety report in five. And the last line is the point of
> this beat: the sklearn function and the formula on the slide are the same
> thing. There is no magic in the library, it is dividing 53 by 67."

Then type ST-4 and run the one call report:

```python
print(classification_report(y_test, y_pred,
    target_names=["routine", "escalate"], digits=3, zero_division=0))
```

Point at the `escalate` row reproducing the numbers just computed, and at
`support`: 67. That column is foreshadowing for Section 5.

> "`zero_division=0` is the flag you will thank yourself for on the day a
> class gets zero predictions. Without it: a warning and a silent nan in a
> pipeline that keeps running."

### Beat 4: Flip one prediction (2.5 min, the memorable moment)

Type ST-5:

```python
fn_idx = np.where((y_test == 1) & (y_pred == 0))[0]
y_flipped = y_pred.copy()
y_flipped[fn_idx[0]] = 1
```

Expected output:

```
14 false negatives. Flipping the first one, row 8:
  'The power strip vibrates so hard the fasteners back out. One piece flew off near my face.'
```

Read the review aloud. Ask: "Is it obvious this is a safety report? Sort of.
No fire, no shock, no injury keyword. That is what a miss looks like."

Interaction hook, before running the pre written delta cell: "One prediction
out of two thousand changes. Shout out: which metrics move, and roughly how
much?" Take three answers. Then run:

```
metric     before     after
Accuracy   0.9395 -> 0.9400   (+0.05 points)
Precision  0.3312 -> 0.3354   (+0.42 points)
Recall     0.7910 -> 0.8060   (+1.49 points)
F1         0.4670 -> 0.4737   (+0.67 points)
```

Talk track, pointing at the two rows:

> "Recall moved thirty times more than accuracy. Same single rescued safety
> report. Recall's denominator is the 67 reviews Cordwell actually cares
> about. Accuracy's denominator is all two thousand, so it is drowning in
> easy true negatives. Accuracy did not notice the most important thing that
> happened on this test set. Keep that in your head for the next section."

The precision row is a bonus teaching point if asked: precision also rose
because the flip added a true positive without adding a false positive.

### Beat 5: The bridge (1 min)

Type ST-6:

```python
always_routine = np.zeros_like(y_test)
```

Expected output:

```
Accuracy of a model that never escalates: 0.9665
Accuracy of our actual model:             0.9395
```

Talk track:

> "Slide 8 asked what accuracy a model gets if it predicts routine for every
> single review. There it is: 96.65 percent. It beats our real model by 2.7
> points while reading zero safety reports. Accuracy is not broken, it is
> answering a question nobody at Cordwell asked. The next section is about
> the metrics that do not fall for this."

Stop there. Do not show MCC or balanced accuracy yet; they are the payoff of
the next slides. If a fast finisher asks, the instructor appendix in the
solution notebook has both, verified.

### Wrap (30 seconds)

Have everyone rerun the checks cell. The room should be at or near
`12 of 12 checks passing`. Anyone stuck: the checks name the stub to revisit.

## Fallback path

If a machine, projector, or kernel dies: open
`Demo01_Read_the_Matrix_SOLUTION.ipynb`, already executed, and walk the same
beats pointing at the captured outputs. Every expected output block above is
verbatim from that file, so the narrative is identical. There is no network
dependency to fail in the first place.

## What breaks, and the recovery

| Symptom | Cause | Recovery |
|---|---|---|
| `FileNotFoundError` on the CSV | Notebook opened from a different working directory | `%pwd` in a cell; move the CSV next to the notebook or `cd` the kernel. Ten seconds. |
| A student's checks disagree with the projector | Stale or edited CSV | Rerun `python make_dataset.py`; seeds are pinned, the file is bit reproducible. |
| Student unpacked ravel as tp first | The classic | Do not fix it silently. Show it: their precision and recall swapped and both still look plausible. This mistake is a better teacher than the slide. |
| Checks cell shows `not yet` on ST-3 with correct looking numbers | They computed metrics from a hand built matrix with swapped cells | Same teaching moment as above. |

## Timing summary

| Beat | Minutes |
|---|---|
| 0 Frame | 0.5 |
| 1 Score to decision | 1.5 |
| 2 Four counts | 2.5 |
| 3 Metrics and report | 2.0 |
| 4 The flip | 2.5 |
| 5 Bridge | 1.0 |
| Slack | 1.0 (interaction hooks absorb it) |

## Verification ledger

Execution confirmed, this environment (Python 3.12.3, numpy 2.4.4,
pandas 3.0.2, scikit-learn 1.8.0, matplotlib 3.10.8):

- All expected output blocks above are verbatim from the executed solution
  notebook, including the confusion matrix counts, all four metrics, the
  classification report, the delta table, and the baseline comparison.
- Every number matches the slide deck to four decimals: TP=53, FN=14,
  FP=107, TN=1826, accuracy 0.9395, precision 0.3312, recall 0.7910,
  F1 0.4670, baseline accuracy 0.9665, base rate 0.0335. The appendix values
  (balanced accuracy 0.8678, MCC 0.4880) and the score based values used by
  later demos (ROC-AUC 0.9621, average precision 0.6040) also match the deck.
- Student notebook cold runs with zero errors at 1 of 12 checks; filling the
  six stubs with the solution bodies yields 12 of 12 (completability
  confirmed by automated re execution).
- Total demo compute is under one second on CPU.

Estimates, not measured: the per beat wall clock timings above assume live
typing plus narration and should be confirmed in one instructor dry run.
