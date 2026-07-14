# HINTS - Lab 01 (optional, instructor may withhold)

These are deliberately terse. One nudge per task, concept only, no code. Hand them out only if a learner is stuck, not by default. The goal this week is to make students reason from the contract rather than pattern-match a hint.

**Task 1 - build_prompt.** The four section headers are fixed strings and their order is part of the contract. Think about how to join four pieces with a blank line between them, and how a list of constraints becomes one bullet per line.

**Task 2 - count_shots.** You are counting lines, not parsing structure. Which single line marker did the contract say identifies an example? Map the count to a name at the end.

**Task 3 - contract models.** Two `@field_validator` methods on `Record` and one `@model_validator(mode="after")` on `ClassificationOutput`. A validator enforces a rule by raising when the rule is broken and returning the value otherwise. For the count rule, you need the whole object, which is why it runs after construction.

**Task 4 - validate_output.** Two failure modes, two `try` blocks: parsing can fail, and validation can fail. Each returns a report rather than raising. Pull the field location out of each `pydantic` error so the message says where the problem is.

**Task 5 - lint_prompt.** Four independent checks, one `Finding` each, appended to a list you sort by code at the end. Lowercase the prompt once and search for substrings. The only fiddly one is the numeric length constraint, which wants a small regex.

**Task 6 - build_few_shot.** Guard before you build: check for leakage and raise first, so a contaminated prompt is never constructed. Then render each example so that Task 2 can still detect it, which means the `Example ...:` line matters.
