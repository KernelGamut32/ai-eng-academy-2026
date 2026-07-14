# Lab 03 Hints

Use these only after you have wrestled with a task for a few minutes. They are
nudges, not solutions. Each task has one or two.

## Task 1 — build_prompt
- Whoever reads your `<INPUTS>` block later keys off a very specific line shape.
  Look at what the mock searches for and match it exactly.
- Context needs numbers because evidence has to point somewhere real.

## Task 2 — validate_triage
- Report every violation, not the first. Accumulate.
- Two things can go wrong before you ever reach a record. Guard the shape first.
- A "line number" with a leading zero is not a line number. Anchor your match.

## Task 3 — answer_then_verify
- The draft is text, not JSON. Do not parse it as JSON.
- The order of your two repairs matters. One of them depends on the other having
  already happened.
- Do not trust the draft's label to be in the allowed set, and do not trust its
  action to be the right length. Assume both are suspect and prove otherwise.

## Task 4 — auto_rubric_score
- One criterion is not "fewer defects is better." It only earns full marks when a
  specific safe behavior was actually required and delivered. When it was not
  required at all, it is neutral, not full.
- Three of the five criteria share the same count-to-score shape. Write it once.

## Task 5 — compare_determinism
- Comparing dicts across runs can miss equal outputs whose keys are ordered
  differently. Make a canonical string first.
- One temperature should collapse to a single distinct output. The other should
  not. If both collapse, check that you are actually passing the temperature and
  varying the seed.

## Stretch 1 — ambiguity-aware refusal
- Stop letting keyword order decide. Count how many categories match, then act on
  the count.

## Stretch 2 — strict template
- Do not rewrite the base validator. Call it, then add exactly one rule on top.

## Stretch 3 — batch drift report
- A set per input, filled across runs, and you want its size at the end.
