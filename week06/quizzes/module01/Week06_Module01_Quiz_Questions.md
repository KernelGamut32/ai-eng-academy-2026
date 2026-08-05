# Module 01 Knowledge Check: Model Evaluation Metrics

**AI Engineering Academy | Gamut Technology Services**

Five multiple choice questions covering Module 01: the confusion matrix and core metrics, evaluation under class imbalance, uncertainty on a reported metric, threshold selection, and evaluating an LLM judge.

All questions use the Cordwell Home and Hardware scenario from the module: 2,000 reviews in the test set, 67 genuine safety escalations, a base rate of 3.35 percent.

**Instructions.** Choose the single best answer for each question. Three of the five show code and ask you to reason about what it does. Read the code carefully. Several distractors are true statements pointed at the wrong setting, and in one question the code runs cleanly and still reports the wrong number.

---

### Question 1

The Cordwell model is evaluated at threshold 0.50 alongside a model that predicts "routine" for every single review.

| Metric | Always predict "routine" | The actual model |
|---|---|---|
| Accuracy | 0.9665 | 0.9395 |
| Recall (safety) | 0.0000 | 0.7910 |
| MCC | 0.0000 | 0.4880 |
| Balanced accuracy | 0.5000 | 0.8678 |

A teammate points at the accuracy column and concludes the useful model has a defect, since it scores below the trivial baseline. What is the correct reading of this table?

- A. The teammate is right. Any model scoring below the majority baseline on accuracy has a defect and should be rejected.
- B. Accuracy is dominated by the 1,933 negatives, so it rewards the constant predictor for being right about the majority class. MCC and balanced accuracy score the constant predictor at 0.000 and 0.500, which is what exposes it as useless.
- C. The accuracy gap means the model is overfitting the positive class, and the fix is to raise the threshold above 0.50.
- D. The two models are statistically indistinguishable on accuracy, so any of the four metrics could serve as the ship criterion.

---

### Question 2

This is the confusion matrix code from the module.

```python
from sklearn.metrics import confusion_matrix

tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
print(f"TP={tp} FN={fn} FP={fp} TN={tn}")
# TP=53 FN=14 FP=107 TN=1826
```

A teammate rewrites the unpacking line as `tp, fp, fn, tn = confusion_matrix(y_test, y_pred).ravel()` and reruns the pipeline. What happens?

- A. It raises a `ValueError`, because the four returned counts are name-bound and cannot be reordered.
- B. Nothing changes, because `.ravel()` returns a dictionary-like object that binds each count to its correct name.
- C. It runs without error, but `tp` now holds 1,826 and `tn` holds 53, so every downstream metric is computed from swapped counts and still returns a plausible-looking number.
- D. It runs and reports identical metrics, because precision and recall are symmetric under a relabeling of the positive class.

---

### Question 3

This code evaluates the model's ranking quality.

```python
from sklearn.metrics import roc_auc_score, average_precision_score

roc_auc_score(y_test, y_pred)              # y_pred = hard 0/1 labels
average_precision_score(y_test, y_pred)
```

`y_pred` holds the hard predictions from `clf.predict(X_test)`, not the continuous scores. What is the consequence?

- A. Both calls raise a `ValueError`, since these metrics require continuous scores and validate their input.
- B. Both calls run and return numbers, but the metrics are computed from a two-point curve rather than the full ranking, so they silently understate the model's ranking quality. The fix is to pass the continuous scores.
- C. Both calls return exactly the same values as passing scores, since the ranking induced by hard labels is equivalent.
- D. `roc_auc_score` raises, but `average_precision_score` succeeds, because only ROC requires continuous input.

---

### Question 4

The module reports Cordwell's F1 as 0.4670 with a 95 percent bootstrap interval of [0.383, 0.543], a width of 0.160 on 67 positives. A 200-row slice with 6 positives gives an interval of [0.095, 0.600].

Your team compares two candidate models on the full test set. Model A scores F1 0.467 and model B scores F1 0.489, and both intervals are roughly 0.16 wide and overlap heavily. What have you established?

- A. Model B is better, because 0.489 exceeds 0.467 and both were measured on the same test set.
- B. Nothing about which model is better. Overlapping intervals of that width mean the 0.022 gap is within measurement noise. Bootstrapping the paired difference on the same test set is the tighter test.
- C. Model B is better, but only if the difference exceeds one half of the interval width.
- D. The comparison is invalid because bootstrap intervals cannot be computed on F1, only on accuracy.

---

### Question 5

The module evaluates an LLM judge against human labels on an imbalanced task and reports two numbers: raw agreement of 0.9105 and Cohen's kappa of 0.3334. A stakeholder sees the 91 percent figure and calls the judge validated. Why is that conclusion wrong?

- A. Raw agreement is miscalculated whenever labels are imbalanced, so the 0.9105 figure is simply an arithmetic error.
- B. Kappa is the pessimistic bound and raw agreement is the optimistic one, so the true agreement is their average, roughly 0.62.
- C. When one class is rare, two annotators agree most of the time purely by both saying "not a safety issue." Kappa corrects for that chance agreement, and 0.3334 shows the judge captures only a modest share of agreement beyond chance.
- D. Raw agreement measures the judge against itself across runs, while kappa measures it against humans, so the two numbers describe different experiments.

---

*End of quiz. Five questions. Answer key is a separate file.*
