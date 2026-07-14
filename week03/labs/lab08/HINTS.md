# Hints

Three levels per task. Read only as far as you need. Level 1 is a nudge, level 2
is the approach, level 3 is the key line.

---

## Task 1. build_zero_shot_prompt

1. You are returning one big string. Build a list of lines and join them.
2. Three sections in order: instruction, output contract, inputs. Put the allowed
   labels into the instruction and list each item as `id: text`.
3. `return "\n".join(lines)` where lines includes the tag markers, a literal JSON
   skeleton string for the contract, and one `f'{it["id"]}: {it["text"]}'` per
   item.

## Task 2. build_few_shot_prompt

1. Start from your Task 1 prompt and add an examples block before the query.
2. Each example is one line. A compact form like `[label] text` reads well.
3. Loop the examples with `f'[{ex["label"]}] {ex["text"]}'`, wrap them in an
   examples section, then reuse your item loop for the query section.

## Task 3. parse_records

1. The response may be wrapped in a code fence and may have a stray sentence
   around it. Strip the fence, then find the JSON.
2. A regex with `re.DOTALL` can pull the inside of a triple-backtick block. After
   that, slice from the first open brace to the last close brace.
3. `json.loads(text[text.find("{"):text.rfind("}") + 1])["records"]`. Raise
   `ValueError` when no braces are found.

## Task 4. score_classification

1. Build the confusion matrix first, then read precision, recall, and F1 off it.
2. Use `defaultdict(Counter)` keyed `cm[true][pred]`. True positives are the
   diagonal, false positives are the rest of the column, false negatives are the
   rest of the row.
3. For label `l`: `tp = cm[l][l]`, `fp = sum(cm[t][l] for t in labels if t != l)`,
   `fn = sum(cm[l][p] for p in labels if p != l)`. Guard every division against a
   zero denominator. Macro is the plain mean of each metric across labels.

## Task 5. build_cot_prompt

1. One function, a boolean flag. The only differences are the instruction and
   whether the contract mentions a rationale.
2. Branch on `use_cot` to pick the instruction line and the contract skeleton.
3. When `use_cot` is true, the contract includes a `rationale` field. When false,
   it does not. Both loop the items into an inputs section.

## Task 6. normalize_answer

1. The goal is that `1.6` and `"1.6"` compare equal, and `" Deny "` equals
   `"deny"`.
2. Numbers become float. Strings that look numeric become float. Other strings
   are stripped and lower-cased.
3. Check `bool` first, since `bool` is a subclass of `int`. Then try
   `float(s)` inside a `try`, and on `ValueError` return the cleaned string.

## Task 7. score_cot

1. Compare each prediction to its gold, but normalize both sides first.
2. Count how many `normalize_answer(pred) == normalize_answer(gold)`.
3. Accuracy is correct over total. Return the per-item rows too so you can print
   which ones missed.

## Task 8. consensus

1. For each id, tally the answers across all runs and take the most common.
2. `defaultdict(Counter)` keyed by id. Vote on the stripped string form of each
   answer so `28` and `"28"` count together.
3. `{_id: ctr.most_common(1)[0][0] for _id, ctr in votes.items()}`.

## Task 9. estimate_tokens

1. Roughly one token per four characters.
2. Round it, and never return less than one.
3. `return max(1, round(len(text) / 4))`.

## Task 10. check_no_leakage

1. A record leaks if it has any field beyond id and final answer, or if the
   answer text reads like reasoning.
2. Compare each record's keys to the allowed set. Also scan the answer string for
   markers like `step`, `because`, `therefore`.
3. `extra = set(r.keys()) - {"id", "final_answer"}`. Collect the id when `extra`
   is non-empty or any marker is in the lower-cased answer string. Return
   `{"clean": not offenders, "offenders": offenders}`.
