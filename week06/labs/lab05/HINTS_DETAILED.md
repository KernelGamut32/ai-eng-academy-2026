# Lab 2.2 Hints: Detailed Tier

Pick one hint tier per task. This file shows the working core of each task verbatim with a comment on why each line is there. It withholds the function shell, the return assembly, and the glue, so finishing still requires reading and understanding the code rather than pasting a function. If you want lighter nudges instead, use HINTS.md. Reading both tiers for the same task wastes your time.

---

## Task 1: split_sentences

The working core:

```python
parts = SENTENCE_SPLIT_REGEX.split(text)
```

The regex uses a lookbehind, so the split point is after the punctuation mark and the punctuation stays attached to its sentence. `parts` may contain pieces with stray whitespace, and can contain empty strings if the text ends in whitespace.

```python
[p.strip() for p in parts if p.strip()]
```

One pass does both cleanup jobs. The condition calls `strip` on the raw piece: a piece that is only whitespace strips down to an empty string, which is falsy, so it is dropped. The output expression strips again so the kept sentences carry no leading or trailing whitespace. Calling strip twice on short strings costs nothing and keeps the comprehension to one line.

You supply: the function shell and the return.

---

## Task 2: summarize_lead_k

The working core:

```python
sentences = split_sentences(text)
```

Reuse Task 1. Never re-implement splitting inline; if the splitter changes, every summarizer should change with it.

```python
sentences[:max_sentences]
```

Python slicing never raises on a short list: slicing five from a list of three returns the three. That single property satisfies the short document clause of the contract with no if statement.

You supply: the function shell, the space join, and the return.

---

## Task 3: summarize_tfidf

The working core:

```python
sentences = split_sentences(text)
if len(sentences) <= max_sentences:
    return " ".join(sentences)
```

The early return handles the degenerate case before any scoring work happens. Joining and returning here also guarantees the output format matches the normal path: a single space separated string.

```python
scores = sentence_scores(sentences, vectorizer)
```

The provided helper returns one float per sentence, aligned by position. Position alignment is what lets plain integer indices stand in for sentences during ranking.

```python
ranked = sorted(range(len(sentences)), key=lambda i: (-scores[i], i))
```

This sorts indices, not sentences. The key is a tuple: negating the score makes the default ascending sort put high scores first, and the second element, the index itself, settles ties in favor of the earlier sentence, exactly as the contract requires. One line encodes both the ranking and the tie rule.

```python
chosen = sorted(ranked[:max_sentences])
```

`ranked[:max_sentences]` takes the winners, but they arrive in score order. Sorting them ascending restores document order, so the assembled summary reads front to back like prose rather than jumping around by importance.

You supply: the function shell and the final join over `sentences[i] for i in chosen`.

---

## Task 4: compute_rouge_scores

The working core:

```python
result = scorer.score(target=reference, prediction=candidate)
```

Keyword arguments make the trap structurally impossible: the reference is named as the target and the candidate as the prediction, and no reader ever has to remember positional order. This is the single most important line in the lab.

```python
for metric_key, short in [("rouge1", "r1"), ("rouge2", "r2"), ("rougeL", "rl")]:
    score = result[metric_key]
```

The loop pairs the library's metric names with the short prefixes the lab's column naming uses. Looping instead of writing nine literal assignments means the naming scheme lives in exactly one place.

```python
    out[f"{short}_precision"] = score.precision
    out[f"{short}_recall"] = score.recall
    out[f"{short}_f"] = score.fmeasure
```

Each Score object exposes the three numbers as attributes. Note the library spells it `fmeasure` while our flat keys spell it `_f`; the translation happens here and nowhere else.

You supply: the function shell, initializing `out`, and the return.

---

## Task 5: score_corpus

The working core:

```python
records = []
for _, row in df.iterrows():
    scores = compute_rouge_scores(row["reference_summary"], row[candidate_col])
```

`iterrows` yields an index and a row Series; the underscore discards the index because the row's own data is all we need. The candidate column name arrives as a parameter, which is what lets the same function score System A and System B.

```python
    records.append({f"{prefix}_{k}": v for k, v in scores.items()})
```

The dict comprehension renames all nine keys in one expression. Appending plain dicts to a list and building one DataFrame at the end is the fast pattern; growing a DataFrame row by row inside a loop is quadratic and a known pandas antipattern.

```python
pd.DataFrame(records, index=df.index)
```

Carrying the source index through is what makes the later `pd.concat(..., axis=1)` align score rows with corpus rows. Without it, concat aligns on default integer indices, which happens to work here but breaks the moment anyone filters the corpus first. Aligning explicitly is the habit worth building.

You supply: the function shell and the return.

---

## Task 6: build_summary_tables and plot_rouge_l_histogram

The working core of the tables:

```python
rows = {}
for metric, (col_a, col_b) in F_COLUMNS.items():
    rows[metric] = {
        "A_mean": scores_df[col_a].mean(),
        "A_std": scores_df[col_a].std(),
        "B_mean": scores_df[col_b].mean(),
        "B_std": scores_df[col_b].std(),
    }
```

The provided F_COLUMNS mapping drives the loop, so metric names and column names are never retyped. Each outer entry becomes one row of the final table; each inner key becomes a column.

```python
summary_table = pd.DataFrame.from_dict(rows, orient="index")
```

`orient="index"` reads the outer dict keys as the row index, which is exactly the rouge1, rouge2, rougeL index the contract requires.

```python
layout_table = scores_df.groupby("layout")[["A_rl_f", "B_rl_f"]].mean()
```

Selecting the two columns before the mean keeps the result to exactly the required shape. This one groupby line is the analytical heart of the lab: it is the slice that exposes what the corpus means hide.

The working core of the plot:

```python
bins = np.linspace(0.0, 1.0, 41)
plt.hist(scores_df["A_rl_f"], bins=bins, alpha=0.6, label="System A (Lead-2)")
plt.hist(scores_df["B_rl_f"], bins=bins, alpha=0.6, label="System B (TF-IDF)")
```

Passing the same explicit bin edges to both calls is what makes the two histograms visually comparable; letting each call choose its own bins produces offset bars that mislead. The alpha keeps the overlap region readable, and the labels feed the legend.

You supply: both function shells, the axis labels, title, legend, `plt.show()`, and the paired return.

---

## Task 7: find_extreme_examples and show_example

The working core of the extremes:

```python
for key, col, kind in [("A_best", "A_rl_f", "max"), ("A_worst", "A_rl_f", "min"),
                       ("B_best", "B_rl_f", "max"), ("B_worst", "B_rl_f", "min")]:
    extreme = scores_df[col].max() if kind == "max" else scores_df[col].min()
```

Four extremes differ only by column and direction, so the loop is driven by a table of tuples. First find the extreme value itself.

```python
    matching = scores_df.loc[scores_df[col] == extreme, "doc_id"]
    result[key] = int(matching.min())
```

Selecting all rows that equal the extreme, then taking the minimum doc_id, implements the tie rule explicitly. The shortcut `idxmax` returns the first positional match, which coincides with the smallest doc_id only while the frame stays sorted; this version stays correct even if someone sorts the frame first. The int call satisfies the plain int requirement in the contract.

The working core of show_example:

```python
row = scores_df.loc[scores_df["doc_id"] == doc_id].iloc[0]
```

Look up by the doc_id column, not by positional index. The two coincide in this lab, but treating an identifier column as a position is a habit that fails the first time a frame gets filtered or reordered. `iloc[0]` unwraps the one row selection into a Series for clean field access.

You supply: the show_example print layout (header, document, reference, both candidates with their `A_rl_f` and `B_rl_f` formatted to four decimals), the result dict initialization, and the return.

---

## Task 8: build_variant_scores

The working core:

```python
for doc_id in SELECTED_DOC_IDS:
    row = scores_df.loc[scores_df["doc_id"] == doc_id].iloc[0]
    base = row["candidate_B"]
    hallucinated = base + " " + " ".join(HALLUCINATION_SENTENCES)
    redundant = " ".join([base] * 3)
```

Same doc_id lookup as Task 7. The variants are plain string assembly, and following the construction rules exactly matters here: the checks verify the class means to four decimal places, and a doubled space or a missing separator shifts the tokenization and the numbers.

```python
    for variant_name, candidate in [("base", base),
                                    ("hallucinated", hallucinated),
                                    ("redundant", redundant)]:
        scores = compute_rouge_scores(row["reference_summary"], candidate)
```

Iterating over name and candidate pairs writes the scoring and record building once for all three variants. Every variant is scored against the same reference, which is the point of the experiment: only the candidate changes.

You supply: the function shell, the record dict with the six required columns pulled from `scores`, the append, and the final `pd.DataFrame(records)`.

---

## Stretch 1: compute_oracle_ceiling

The working core:

```python
values = []
for _, row in scores_df.iterrows():
    scores = compute_rouge_scores(row["reference_summary"], row["oracle_summary"])
    values.append(scores["rl_f"])
```

Same iteration shape as Task 5, reduced to one candidate column and one retained value per row. The oracle goes in the candidate position: the question is how well a perfect content summary scores against the reference, so the reference stays the target.

```python
scores_df["oracle_rl_f"] = values
```

Assigning the list after the loop adds the column in one operation, aligned by position with the rows just iterated.

You supply: the function shell and the return of the column mean as a plain float.
