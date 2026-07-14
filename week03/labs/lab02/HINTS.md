# HINTS (optional, withholdable)

Use only if you are stuck. One concept-level nudge per task, no code. If you want the full
answer, that is the solution notebook, not this file.

**Part A - pick_frame.** Read what the output is. A fixed bucket from a list is one frame. A
few named fields pulled out is another. A shorter version of the input is a third. Text made
from nothing is the fourth. A tie between two labels is still a decision over a fixed set.

**Part B - prompt.** The check lists exactly what must appear. Every ticket id, every label,
the contract, the policy, the word rationale, the tie-break line. Build the string so all of
them are present.

**Part B - models.** Three concerns map to three tools: a value that must be in a set, a value
whose word count must sit in a range, and a count that must match a length across two fields.
Pydantic has a place for each. Unknown keys are a model config setting.

**Part B - validate.** Once the models carry the rules, the validator is almost nothing. Let
the models do the work.

**Part C - _normalize_date.** Two accepted input shapes, one output shape. A parse-then-format
round trip handles both. None passes through. Anything else is an error.

**Part C - Invoice.** Only three fields are required. The rest default to absent. Date fields
need their normalization to run before the type check. Email is checked only when present.

**Part C - validate_invoices.** The top level is an array. Reject it if it is not, then validate
each element.

**Part D - prompt.** Same idea as Part B. Make everything the check names appear in the string.

**Part D - validate_summary.** The output is prose, so parse lines yourself. Reject quotes first.
Check the header line. Split each bullet into a label and a value. Two of the bullets are lists
with their own caps. Enforce the word budget on each value.

**Stretch.** Each stretch function has a one-paragraph contract in its docstring. Read the check
if you want the exact expected shape.
