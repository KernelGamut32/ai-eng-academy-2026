# Lab 2.2 Hints: Progressive Tier

Pick one hint tier per task. This file escalates in three levels per task: level 1 names the approach, level 2 sketches the structure, level 3 shows the key line or two in context. If you want line-by-line commentary on the working core instead, use HINTS_DETAILED.md. Reading both tiers for the same task wastes your time.

Reveal levels one at a time. Try the task again after each level before reading the next.

---

## Task 1: split_sentences

**Level 1.** The regex object has a split method that does the heavy cut. Your work is cleanup: whitespace and empties. A list comprehension handles both in one pass.

**Level 2.** Three steps in one function: call `SENTENCE_SPLIT_REGEX.split(text)` to get raw pieces, strip each piece, keep only the pieces that are still non empty after stripping. A stripped empty string is falsy in Python, which makes the filter condition short.

**Level 3.** The cleanup comprehension has this shape:

```python
[p.strip() for p in parts if p.strip()]
```

where `parts` came from the regex split. Return that list.

---

## Task 2: summarize_lead_k

**Level 1.** Split, slice, join. Python slicing already handles the short document case for you.

**Level 2.** Call your Task 1 function, take a slice of the first `max_sentences` items, join with a single space. Note that `some_list[:5]` on a three item list returns the three items without complaint, so no length check is needed.

**Level 3.** The core is one line:

```python
" ".join(sentences[:max_sentences])
```

---

## Task 3: summarize_tfidf

**Level 1.** Four phases: split, early return if already short enough, rank sentence indices by score, reassemble the winners in document order. The provided `sentence_scores` does all the vector math; you never touch the matrix yourself.

**Level 2.** For the ranking with the tie rule, sort the indices `range(len(sentences))` with a key that sorts primarily by score descending and secondarily by index ascending. A tuple key of the negated score and the index does both at once. Take the first `max_sentences` indices from that ranking, then sort those chosen indices ascending so the final summary follows document order, not score order.

**Level 3.** The two sorting lines:

```python
ranked = sorted(range(len(sentences)), key=lambda i: (-scores[i], i))
chosen = sorted(ranked[:max_sentences])
```

Then join `sentences[i]` for each `i` in `chosen`.

---

## Task 4: compute_rouge_scores

**Level 1.** One call to `scorer.score` with the arguments in the documented order, then unpack three Score objects into nine dictionary keys. Passing the arguments by keyword makes the order mistake impossible.

**Level 2.** The scorer returns a dict keyed by `"rouge1"`, `"rouge2"`, `"rougeL"`. Each value has `.precision`, `.recall`, and `.fmeasure` attributes. Loop over the three pairs of library name and short prefix (`rouge1` and `r1`, and so on) instead of writing nine assignment lines.

**Level 3.** The scoring call and the key construction:

```python
result = scorer.score(target=reference, prediction=candidate)
```

and inside the loop, keys built like `f"{short}_precision"`.

---

## Task 5: score_corpus

**Level 1.** Iterate the rows, score each one with your Task 4 function, collect the renamed dicts in a list, build one DataFrame at the end. Building the DataFrame once at the end is much faster than appending rows to a DataFrame in a loop.

**Level 2.** `df.iterrows()` yields an index and a row; the row supports column access by name for `reference_summary` and the candidate column. Rename each score dict with a dict comprehension that prepends the prefix. When constructing the final DataFrame, pass `index=df.index` so the later concat aligns rows instead of duplicating them.

**Level 3.** The renaming comprehension:

```python
{f"{prefix}_{k}": v for k, v in scores.items()}
```

and the constructor call `pd.DataFrame(records, index=df.index)`.

---

## Task 6: build_summary_tables and plot_rouge_l_histogram

**Level 1.** The summary table is a dict of dicts turned into a DataFrame: one outer entry per metric, one inner entry per statistic. The layout table is a single groupby line. The histogram is two `plt.hist` calls on the same axes.

**Level 2.** Loop over the provided `F_COLUMNS` mapping to build the outer dict, computing `.mean()` and `.std()` on the two columns each metric names. `pd.DataFrame.from_dict(rows, orient="index")` turns the nested dict into the required shape with metrics as the index. The layout table is `scores_df.groupby("layout")[["A_rl_f", "B_rl_f"]].mean()`. For the histogram, pass the same explicit `bins` array to both calls so the bars align, and an `alpha` below 1.0 so the overlap region stays readable.

**Level 3.** The from_dict call:

```python
summary_table = pd.DataFrame.from_dict(rows, orient="index")
```

and a shared bin edge array such as `np.linspace(0.0, 1.0, 41)` passed to both hist calls.

---

## Task 7: find_extreme_examples and show_example

**Level 1.** For each of the four extremes: find the extreme value in the score column, then select every row holding that value, then apply the tie rule. For show_example, locate the row by its doc_id column, not by positional index, and print the fields with labels.

**Level 2.** The tie safe pattern: compute `scores_df[col].max()` (or min), then `scores_df.loc[scores_df[col] == extreme, "doc_id"]` gives the doc_ids of every tied row, and `.min()` of that selection applies the smallest doc_id rule. Wrap in `int(...)` because the check requires plain ints. A loop over four tuples of key, column, and kind avoids writing the pattern four times.

**Level 3.** The selection line for one extreme:

```python
matching = scores_df.loc[scores_df[col] == extreme, "doc_id"]
result[key] = int(matching.min())
```

---

## Task 8: build_variant_scores

**Level 1.** Two nested loops: outer over the selected doc ids, inner over the three variants of that document's base summary. Build each variant as a string operation on the base, score it, append a flat record dict, and make one DataFrame at the end.

**Level 2.** Locate each document's row with the same doc_id lookup you used in Task 7. The variants are pure string assembly: the hallucinated variant is the base, a space, and the three hallucination sentences joined by spaces; the redundant variant is `" ".join([base] * 3)`. Iterate the inner loop over a list of name and candidate pairs so the record building code is written once.

**Level 3.** The variant construction lines:

```python
hallucinated = base + " " + " ".join(HALLUCINATION_SENTENCES)
redundant = " ".join([base] * 3)
```

---

## Stretch 1: compute_oracle_ceiling

**Level 1.** Structurally this is a one column version of Task 5: iterate rows, score `oracle_summary` against `reference_summary`, keep only the `rl_f` value each time.

**Level 2.** Collect the values in a plain list during the loop, assign the list to `scores_df["oracle_rl_f"]` after the loop, and return `float(scores_df["oracle_rl_f"].mean())`. The float call matters because the contract asks for a plain float, not a numpy scalar.

**Level 3.** The per row extraction:

```python
values.append(compute_rouge_scores(row["reference_summary"], row["oracle_summary"])["rl_f"])
```
