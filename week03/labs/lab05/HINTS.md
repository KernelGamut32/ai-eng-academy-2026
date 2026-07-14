# Lab 05 Hints

Use these only after you have tried a task. Each task has three levels: a nudge, a pointed hint, and a near-answer. Read the smallest level that unblocks you.

---

## TODO 1 -- The output contract

**Level 1.** You are writing two Pydantic v2 models. Every rule in the markdown (id shape, the two enums, non-empty reasons, the policy-line range, no extra fields, the count invariant) maps to a field type, a validator, or a config setting.

**Level 2.** Reject unknown fields with `model_config = ConfigDict(extra="forbid")`. A number-or-null field is `Optional[float] = None`. Per-field rules use `@field_validator("name")` above a `@classmethod`. The count invariant compares two fields, so it cannot live in a single-field validator.

**Level 3.** For a cross-field rule like `stats.count == len(records)`, override `model_post_init(self, __context)` on the report model and raise `ValueError` when they disagree. Every `@field_validator` must return the value it checked.

---

## TODO 2 -- The prompt linter

**Level 1.** Loop over `REQUIRED_SECTIONS`. For each tag, look for `<TAG>...</TAG>` in the text and record whether it is missing, empty, or present.

**Level 2.** A pattern like `<TAG>(.*?)</TAG>` needs to match across newlines. Use `[\s\S]*?` so you do not depend on a flag. Record each found tag's position with the match's `start()`.

**Level 3.** Sort found tags by position to get the order they appear. For `order_ok`, do not compare against the full required list. Compare your found order against the required list filtered down to only the tags you found. They should be equal.

---

## TODO 3 -- The disciplined prompt

**Level 1.** Start from `WEAK_SECTIONS`. Ask what the weak prompt never tells the model to do. Look at the four bullets in the task description.

**Level 2.** The model needs an explicit output contract that says return only JSON with no fences, plus constraints that pin the fee to a number, require a non-empty grounded reason per record, and require cited policy line numbers. Fill every canonical section so the linter is happy too.

**Level 3.** In CONSTRAINTS, write one sentence per rule: fee_percent must be a JSON number not a string; reasons must be non-empty and policy-grounded; cite the policy line numbers in policy_lines. In OUTPUT_CONTRACT, show the JSON shape and end with "Return ONLY JSON. No markdown fences."

---

## TODO 4 -- The acceptance suite

**Level 1.** Build a list of `(name, passed, detail)` tuples. Never raise; a failing check is a tuple with `passed=False`, not an exception.

**Level 2.** Parse with strict `json.loads` first. If it fails, append one failing result and return immediately. Otherwise validate the parsed object against `TriageReport`, then index the records by `input_id` to write the per-case checks.

**Level 3.** A small local helper `add(name, cond, detail="")` keeps the code flat. For C-104, require action `warranty_repair` and `"6"` in its policy_lines rather than an exact line set. End with two global checks: every record has non-empty reasons, and every record cites at least one line in the range 1 to 6.

---

## TODO 5 -- Tone variants

**Level 1.** Copy `STRONG_SECTIONS` and change as little as possible.

**Level 2.** For the concise variant, add one CONSTRAINT line capping reason length. For the supportive variant, warm up the ROLE and add one CONSTRAINT line keeping reasons factual. Do not touch OUTPUT_CONTRACT.

**Level 3.** `dict(STRONG_SECTIONS)` makes a copy. Reassign one key on the copy. Because the contract is unchanged, both variants still pass every acceptance check.

---

## Stretch 1 -- Failure-driven repair

**Level 1.** Read the failing check names from the acceptance results and turn them into concrete instructions.

**Level 2.** Group the fixes by concept: a numeric-fee reminder, a grounded-reasons reminder, a cite-the-lines reminder. Emit them inside a `<REPAIR_NOTES>` block.

**Level 3.** Append the returned block to the weak prompt string and re-run the model. If your reminders name the numeric fee, the grounded reasons, and the cited lines, the next output will pass every check.

---

## Stretch 2 -- Robust extraction

**Level 1.** The bare-prompt output has JSON inside a code fence surrounded by prose. Find the JSON and parse just that.

**Level 2.** Try a regex for a fenced block first. If there is none, fall back to the raw string, then slice from the first brace to the last brace.

**Level 3.** `raw.find("{")` and `raw.rfind("}")` bound the object; `json.loads` the slice between them. Remember the caveat: the durable fix is a better prompt, not a more forgiving parser.
