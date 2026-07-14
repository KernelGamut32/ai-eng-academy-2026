# Lab 04 - HINTS (escape hatch)

Use these only when you are stuck. Each task has three levels: a nudge, a shape, then the key line.
Try the contract in the notebook docstring first. The point of this lab is to reason from a contract to
an implementation, so spend a few minutes before opening a level.

---

## TODO 1 - CLASSIFY_PROMPT
- **Level 1.** A reliable prompt has five parts: role, delimited context, task, constraints, output contract, and a self-check. The structural check looks for those, plus all five labels and a `{policy}` placeholder.
- **Level 2.** Use `###` section headers and a `<<<POLICY>>> {policy} <<<END POLICY>>>` block so the model cannot confuse instructions with data. State the two statuses and when each applies.
- **Level 3.** Include the literal token `needs_proof`, the word `JSON`, and a final "self-check before answering" line. Keep the policy out of line by writing `{policy}`, not the pasted text.

## TODO 2 - EXTRACT_PROMPT
- **Level 1.** Same skeleton as TODO 1, but the contract is a JSON array and the body is a field-by-field schema description.
- **Level 2.** List each field with its type and whether it is required. Add an explicit "do not invent values" rule and a "numbers are numeric, no symbols" rule.
- **Level 3.** The check looks for the tokens `schema`, `json`, `invent`, `required`, and `self`, plus a delimiter and `{policy}`.

## TODO 3 - SUMMARIZE_PROMPT
- **Level 1.** Audience first: managers. Then scope, then length limits, then the Markdown contract.
- **Level 2.** Constrain to at most 6 bullets and at most 14 words each. Forbid PII and blaming language.
- **Level 3.** Include the words `manager`, `total spend`, `proof`, and `bullet`, and mention the numbers 6 and 14.

## TODO 4 - EXPENSE_SCHEMA
- **Level 1.** Top level is `{"type": "array", "items": { ... }}`. The item is an object with `required`, `properties`, and `additionalProperties: false`.
- **Level 2.** `input_id` uses `"pattern": "^E-\\d{3}$"`. `category` and `status` use `"enum": [...]`. `amount_usd` is a number with `"minimum": 0`. `tip_percent` is `"type": ["number", "null"]` with min 0 and max 20.
- **Level 3.** Nullable fields take a list type, for example `"type": ["string", "null"]`. Setting `additionalProperties` to `false` is what makes an unexpected extra key fail.

## TODO 5 - validate
- **Level 1.** Build a `Draft202012Validator(schema)` and iterate its errors over `data`.
- **Level 2.** `iter_errors` yields `ValidationError` objects. Sort them for stable output before formatting.
- **Level 3.** Sort with `key=lambda e: e.json_path` (a clean string path). Format each as `f"{e.json_path}: {e.message}"`. Do not sort by `e.path`, a deque that can raise when index types differ across errors.

## TODO 6 - make_repair_prompt
- **Level 1.** Return one string. It names the contract, lists repair rules, then embeds the errors.
- **Level 2.** Restate the same rules the schema and policy enforce, in imperative voice, so a model could act on them without seeing the schema file.
- **Level 3.** Join the error list with newlines. If the list is empty, embed the literal `(none)`. End with "Emit ONLY the corrected JSON array".

## TODO 7 - repair_records
- **Level 1.** Loop the records, build a fresh dict each time so you never mutate the input, and apply the transforms in the contract order.
- **Level 2.** Drop keys not in `ALLOWED_KEYS`. Pull the digits out of `input_id` and reformat. Coerce numbers by stripping non-numeric characters. Map `category` through `CATEGORY_ALIASES`.
- **Level 3.** For numbers, `re.sub(r"[^0-9.\-]", "", str(val))` then `float(...)`, guarding the empty case to return `None`. Recompute status last: `"needs_proof" if (amount >= 25.0 and not has_receipt) else "ok"`. This is what fixes the `E-105` status that the schema let through.

## TODO 8 - build_manager_summary
- **Level 1.** Three outputs: a rounded total, a de-duplicated category list in first-appearance order, and the needs_proof ids.
- **Level 2.** Round the total to 2 decimals. For the split, track a `seen` set while appending unseen categories to a list so order is preserved.
- **Level 3.** `needs = [r["input_id"] for r in clean if r["status"] == "needs_proof"]`. The verified total is `1236.79`.

---

## Stretch S1 - compare_prompt_variants
- Call `validate(simulate_model("extract", few_shot=False))` and the same with `few_shot=True`. Return the two lengths. The zero-shot output carries six flaws, the few-shot output carries one.

## Stretch S2 - policy_violations
- Recompute the expected status the same way `repair_records` does, and compare it to the record's status. Also flag any non-null `tip_percent` outside 0 to 20. Return an empty list when everything complies.
