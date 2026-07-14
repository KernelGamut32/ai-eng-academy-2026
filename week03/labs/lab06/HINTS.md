# Lab 06 Hints (withholdable)

One nudge per task. These are direction, not solutions. Read the contract in the notebook first;
open a hint only when you are stuck for more than a few minutes. Every function is small.

**A note on the model.** `simulated_llm` is a fixed classifier that reads the examples in your
prompt. It is not random. If a variant does not improve, the problem is almost always in what you
fed it, not in the model.

---

### Part A

**tokenize.** Lowercase before you split, and decide which characters count as part of a word. The
apostrophe matters for words like `won't`.

**jaccard.** Two set operations and a division. The only thing that can go wrong is the empty case.

---

### Part B

**build_prompt.** One function has to serve both the zero-shot and the few-shot prompt. What is
different between them, and how do you make that difference conditional on the argument?

**parse_output.** The model does not hand you clean JSON. Look at the raw text it returns and ask
what you must strip before a JSON loader will accept it.

**score.** Count outcomes per label first (a confusion matrix), then derive the three metrics from
those counts. Guard every division. Decide what a missing prediction counts as before you compute
recall.

---

### Part C

**curate.** The quota is per role, so group before you rank. Rank inside each group by the same
similarity idea from Part A, measured against the whole eval vocabulary. When similarities tie, you
need a deterministic tie-break or your selection will not be reproducible.

---

### Part D

**order_easy_to_hard.** A sort. You only need a numeric rank for the three roles.

**order_similar_first.** You have already written this ranking once in this lab. Reuse the idea.

**order_interleave.** This one is not a sort. It is a rotation across the three roles. Think about
what you do when one role empties before the others.

---

### Part E

**validate_output.** Fail loud, not quiet. Check the shape, then the count (and cross-check the
count the model reported against the count it actually returned), then every label and every reasons
field. Raise on the first violation.

---

### Stretch

**curate_with_coverage.** Read the Part E residual first. One eval item stays wrong. The example
that fixes it is in the pool but did not get curated. Start from your curated set and make sure that
one example is included.

**pick_by_budget.** Greedy, in order, stop before you overspend. The token estimate is in the
contract. Then ask which examples get cut at a tight budget and whether they were the ones doing the
work.

**minimal_pair_probe.** You already have a zero-shot path. Feed it your list of short reviews and
map each back to its position. The interesting part is choosing four reviews that differ by one word
and land on four labels.
