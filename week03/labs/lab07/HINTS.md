# Lab 07 Hints (open only when stuck)

These are deliberately terse. Try the contract in the docstring first. Each TODO has three levels. Read one, go back to the notebook, and only come back for the next level if you are still stuck.

---

## TODO 1, normalize_answer

1. The comparison later is exact equality, so "54" and 54 must end up identical. Decide a single canonical form for numbers and one for labels.
2. Booleans are a trap in Python. Think about where `True` lands if you test for `int` first.
3. To detect a numeric string, strip it, then check whether it is all digits after removing one optional minus and one optional decimal point.

## TODO 2, score

1. Loop over the gold keys, not the prediction records. That way a missing prediction counts against you instead of vanishing.
2. Build an id to answer lookup from the records once, then index it.
3. Normalise both sides before comparing, and return the count correct, the total, and a per-item list you can print.

## TODO 3, self_consistency

1. This is a vote. One tally per task id, one increment per run.
2. Coerce each answer to a string before counting, or 54 and "54" will split the vote.
3. Ties need a deterministic rule or the whole notebook stops being reproducible. Pick the tied answer that appeared first, and remember to attach an agreement value of winning votes over total votes.

## TODO 4, guard_private

1. Private mode means no reasoning in the output. What key would a leaked rationale live under?
2. Return the offending ids, not a boolean, so a failure tells you which records leaked.
3. One comprehension is enough.

## TODO 5, acceptance

1. Check structure before content, and return early if the top-level shape is wrong so later lines can assume it.
2. Accumulate problems into a list instead of raising, so a caller sees every issue at once.
3. Use `.get` on the stats count so a missing count is a reported mismatch rather than an error.

## TODO 6, private CoT prompt

1. Two independent switches: one turns reasoning on, one hides the rationale. You need both set the right way.
2. Reasoning stays on only if a step-by-step trigger is present. Do not delete it while you are removing the rationale.
3. Start from the exposed prompt. Keep the trigger, add an instruction to think privately, and drop the rationale field from the contract.
