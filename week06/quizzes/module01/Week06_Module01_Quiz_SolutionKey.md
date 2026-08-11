# Module 01 Knowledge Check: Solution Key

**AI Engineering Academy | Gamut Technology Services | Instructor-facing. Do not distribute to students.**

Answer distribution: B appears 3 times, C 2 times. On a five-question set some clustering is unavoidable; reshuffle option order if you reuse the set.

Question types: code questions are 2 and 3, and Question 4 reasons about the output of the module's bootstrap code without reprinting it. Concept questions are 1 and 5.

**Coverage note.** The five questions map one-to-one onto the deck's own closing slide, "The Five Things to Take Away." Nothing here is drawn from a corner of the deck.

| Q | Answer | Type | Maps to | Deck takeaway |
|---|--------|------|---------|---------------|
| 1 | B | concept | Section 4, imbalance and the majority baseline | #1 |
| 2 | C | code | Section 3, confusion matrix and ravel order | #1 |
| 3 | B | code | Section 4, ROC-AUC and average precision | #3 |
| 4 | B | applied | Section 5, bootstrap confidence intervals | #2 |
| 5 | C | concept | Section 7, LLM judge and Cohen's kappa | #5 |

**Takeaway #4 (threshold as a business decision) is not covered here.** It is the single largest idea in the deck, spanning all of Section 6, and it deserves its own item. Adding a sixth question is the cleanest fix if you want full coverage. A ready-to-use item is supplied at the end of this key.

---

### Question 1 - Answer: B

The test set holds 1,933 negatives against 67 positives, so accuracy's denominator is dominated by the majority class. A constant "routine" predictor gets 1,933 of 2,000 correct and scores 0.9665 while catching zero safety reports. The real model gives up 2.7 points of accuracy to catch 53 of 67 escalations. The metrics that expose the fraud are the ones built to resist it: MCC scores any constant predictor at exactly 0.000 regardless of class distribution, and balanced accuracy, the mean of per-class recall, scores it at exactly 0.500 by construction.

Verified: reconstructing the deck's confusion matrix (TP 53, FN 14, FP 107, TN 1826) reproduces every figure in the table exactly, including the constant predictor's MCC of 0.0 and balanced accuracy of 0.5.

Why the distractors are wrong. A is the rule the deck explicitly corrects, and it is the most important wrong answer on the sheet. Applied literally, it tells you to discard the model that works. Expect some students to pick it, because "beat the baseline" is a habit carried in from balanced-data problems. C invents overfitting as the diagnosis and prescribes raising the threshold, which would trade away recall, the one thing Cordwell most needs. D misreads a 2.7-point gap as statistical noise and then draws the wrong conclusion anyway, since the metrics are not interchangeable. The gap is real and it points the opposite direction from what the teammate thinks.

---

### Question 2 - Answer: C (code)

`confusion_matrix(...).ravel()` flattens the 2 by 2 into a plain numpy array in a fixed order: tn, fp, fn, tp. The names on the left of the assignment are just positional bindings. Rewriting them as `tp, fp, fn, tn` binds `tp` to the first slot, which holds 1,826 true negatives, and `tn` to the last slot, which holds 53. Nothing raises. Every metric computed downstream from those variables is then built from swapped counts, and the results still look like plausible numbers, which is exactly what makes the bug survive review.

Verified: `confusion_matrix(y_test, y_pred).ravel()` on the reconstructed Cordwell data returns 1826, 107, 14, 53 in that order.

Why the distractors are wrong. B invents dictionary-like behavior. `.ravel()` returns an ordinary flat array with no name binding at all, which is the root of the hazard. A assumes validation that does not exist. Both `fp` and `fn` land in their correct positions under the rewrite, so even a shape check would pass. D is the subtlest option and worth discussing: precision and recall do swap under a genuine relabeling of the positive class, which makes the claim feel principled. But this is not a relabeling. It is a mislabeling of variables in one line of Python, and the metric functions downstream never see it. The deck's own takeaway on this slide says to memorize the order for precisely this reason.

---

### Question 3 - Answer: B (code)

Both functions accept any numeric array and will happily consume hard 0/1 labels. With only two distinct values, the ROC and PR curves collapse to a single operating point plus endpoints rather than sweeping the full ranking, so both metrics measure something narrower than intended and report a worse number. No exception is raised at any point.

Verified on a synthetic set matching the Cordwell shape (2,000 rows, 67 positives): passing continuous scores gave ROC-AUC 0.9965 and average precision 0.9453, while passing hard predictions on the identical data gave 0.9856 and 0.6996. Average precision fell by roughly 25 points. Neither call raised.

Note the asymmetry, which is the useful engineering detail: ROC-AUC barely moved while average precision collapsed. On a rare-positive problem, average precision is both the more honest metric and the more sensitive one to this bug, so the number that matters most is the number that degrades most.

Why the distractors are wrong. A and D both assume input validation. There is none, and D adds a false distinction between the two functions. C claims the values are identical, which the verification directly contradicts. The deck names this the most common bug in this code, and its defining property is that it is silent. A student who expects an exception will never catch it.

Debrief hook worth 30 seconds: ask how you would notice this in a real pipeline. The answer is that average precision would sit implausibly close to the threshold-dependent F1, since both would be measuring the same single operating point.

---

### Question 4 - Answer: B

With 67 positives the F1 interval is 0.160 wide. Two models scoring 0.467 and 0.489 differ by 0.022, roughly one seventh of that width, with heavily overlapping intervals. That is not a measured difference, it is noise. The module's rule is direct: no model comparison without intervals. The sharper instrument is bootstrapping the paired difference on the same test set, which controls for the specific examples both models got wrong and yields a tighter interval on the quantity you actually care about, the gap, rather than comparing two independent intervals.

Why the distractors are wrong. A is the default engineering instinct, that a bigger number on the same test set wins, and it is the habit the entire section exists to break. C invents a half-width decision rule with no statistical standing. It is attractive because it sounds like a rigorous threshold, which makes it the strongest distractor. D is false on the facts: the module's own bootstrap function takes any metric callable and is demonstrated on F1 specifically.

Worth landing in the debrief: sample size here is governed by the positive count, not the row count. 2,000 rows sounds ample and yields 67 positives, and the 200-row slice with 6 positives produces an interval spanning essentially the whole useful range.

---

### Question 5 - Answer: C

When 96.65 percent of items are negative, two annotators who both mostly say "not a safety issue" will agree most of the time by accident. Raw agreement counts that accidental agreement as success, which is why 0.9105 is not evidence of a good judge. Cohen's kappa rescales agreement against what chance alone would produce given each annotator's marginal rates. A kappa of 0.3334 says the judge captures only a modest share of the agreement available beyond chance. The module's framing is the one to repeat: an LLM judge is a binary classifier, and every imbalance lesson from the morning applies to it unchanged.

Verified: a judge-versus-human table on 2,000 items (TP 51, FP 0, FN 179, TN 1770) yields raw agreement 0.9105 and kappa 0.3352, confirming the deck's two figures are jointly consistent rather than drawn from different setups.

Why the distractors are wrong. A calls the raw agreement an arithmetic error. It is computed correctly and is simply the wrong statistic for the job, which is a sharper and more useful distinction than "wrong math." B invents an averaging rule with no basis. Kappa and raw agreement are not bounds on a common quantity, and the midpoint of 0.62 is meaningless. D fabricates a difference in what each number measures. Both are computed from the same judge-versus-human comparison on the same items.

---

## Scoring and use

Suggested cut line is 4 of 5. On a five-item check the resolution is coarse, so treat a miss as a signal about which section to revisit rather than an overall verdict.

The two questions most likely to separate the room are 2 and 3, both of which describe code that runs cleanly and reports wrong numbers. That silent-failure pattern is the habit this module is built to instill, and it is worth debriefing both even if the room scores well.

Fast debrief order if time is short: 3, 1, 5, then the rest.

---

## Optional sixth question (threshold as a business decision, takeaway #4)

Add this if you want all five closing takeaways covered. Answer is C.

> The module reports three candidate thresholds for the Cordwell escalation model: 0.60 chosen by maximizing F1 at a total cost of $7,120, 0.50 from the `predict()` default at $3,928, and 0.38 chosen by minimizing cost at $1,876. The unconstrained cost optimum flags 20.4 percent of all reviews, but the triage team can absorb only 10 percent, which moves the final choice to 0.49. What does this sequence establish?
>
> - A. F1 maximization is the correct default, and the cost model is a refinement to apply only when cost data is available.
> - B. The cost-minimizing threshold of 0.38 is the right answer, and the capacity constraint is an operational detail to resolve separately.
> - C. Maximizing F1 implicitly asserts that a false alarm and a missed safety report cost the same, which is false here and makes it the most expensive of the three choices. The defensible threshold comes from an explicit cost model bounded by the capacity that actually exists.
> - D. Because 0.49 sits close to the 0.50 default, the default was effectively correct and the analysis confirms it.

D is the trap worth watching for: 0.49 does land near 0.50, but by coincidence of this cost model and this capacity cap, not because the default encodes anything. Change either input and it moves. A student who picks D has learned the number instead of the method.

---

## Verification ledger

Every quantitative claim was recomputed rather than taken from the deck.

| Claim | How verified | Result |
|---|---|---|
| All Section 3 and 4 metrics (Q1, Q2) | Reconstructed the test set from TP 53, FN 14, FP 107, TN 1826 and recomputed with sklearn | Accuracy 0.9395, precision 0.3312, recall 0.7910, F1 0.4670, F2 0.6192, balanced accuracy 0.8678, MCC 0.4880. All match the deck exactly |
| Majority baseline figures (Q1) | Scored a constant zero predictor on the same data | Accuracy 0.9665, MCC 0.0, balanced accuracy 0.5. All match |
| `.ravel()` ordering (Q2) | Called `confusion_matrix(...).ravel()` directly | Returns 1826, 107, 14, 53 in tn, fp, fn, tp order |
| Hard labels into ranking metrics (Q3) | Ran both functions on scores and on hard predictions over matched synthetic data | Scores 0.9965 and 0.9453; hard labels 0.9856 and 0.6996. No exception raised in either call |
| Judge agreement and kappa consistency (Q5) | Solved for a 2 by 2 table reproducing both deck figures | TP 51, FP 0, FN 179, TN 1770 gives agreement 0.9105 and kappa 0.3352 |

Stack used for verification: scikit-learn 1.8.0, numpy 2.4.x, pandas 3.0.2, Python 3.12.3 (sandbox). Nothing in this quiz depends on a version-specific API. The deck's `TunedThresholdClassifierCV` reference, which requires scikit-learn 1.5 or newer, appears only in the optional sixth question's stem as reported figures and is not itself tested.

One separate note carried from the deck review, not tested here: the deck's slide 46 bootstrap function relies on the two `np.asarray` calls to work on the output of `train_test_split`. Confirmed live that omitting them raises `KeyError` when the split returns a pandas Series, because pandas indexes by label while the resampler generates positions. Worth a mention in the debrief if a student asks why those lines are there, since the deck flags them as load-bearing rather than decorative.
