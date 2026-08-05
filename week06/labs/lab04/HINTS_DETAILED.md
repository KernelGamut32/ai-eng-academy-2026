# Week 6 Lab 2.1: Detailed Hints

Pick one hint tier per task, not both. This file shows the working core of each task with line-by-line commentary explaining why each line is there. The function shells, variable assembly, display calls, and print formatting are left to you, so completing a task still means reading, understanding, and writing code rather than pasting. If you would rather work from nudges, use `HINTS.md` instead.

---

## Task 1: Corpus-level BLEU

The working core is two extraction-and-score moves:

```python
references = cordwell_df["reference"].tolist()
```

`corpus_bleu` wants plain Python lists of strings, not pandas Series. `.tolist()` does that conversion. Repeat for both candidate columns.

```python
bleu_a = corpus_bleu(candidates_a, [references])
```

Two things happening here. First, argument order: candidates come first, references second. Second, the brackets: BLEU as a metric supports scoring against several alternative references per segment, so sacrebleu's type for the second argument is a list of reference lists. We have one reference per row, so we wrap our single list in one outer list. Forgetting the brackets is the most common error in this task, and the error message it produces is not obvious.

You still need to: repeat the call for System B, and print both `.score` values to two decimals. The returned objects are `BLEUScore` instances; keep them whole because Task 2 reads their internals.

---

## Task 2: Score components

The working core is one helper that flattens a BLEUScore into a dict:

```python
p1, p2, p3, p4 = bleu_result.precisions
```

`.precisions` is always a four-element list, one entry per n-gram order from unigram to 4-gram, already on the 0 to 100 scale. Tuple unpacking gives them clean names.

```python
return {"bleu": bleu_result.score, "p1": p1, "p2": p2, "p3": p3, "p4": p4,
        "bp": bleu_result.bp, "sys_len": bleu_result.sys_len, "ref_len": bleu_result.ref_len}
```

Every value here is a plain attribute read; no computation. `bp` is the brevity penalty multiplier between 0 and 1, and the two lengths are total token counts across the whole corpus after sacrebleu's tokenization.

```python
components_df = pd.DataFrame([row_a, row_b], index=["System A", "System B"])
```

A list of dicts becomes rows; the dict keys become columns; the explicit `index=` gives the rows the names the check harness looks up with `.loc`.

You still need to: write the function shell, call it on both score objects, and display the frame rounded to 3 decimals. When it renders, before moving on, find the number in the table that should surprise you.

---

## Task 3: Sentence-level BLEU columns

The working core is one comprehension per system:

```python
[
    sentence_bleu(cand, [ref]).score
    for cand, ref in zip(cordwell_df["candidate_A"], cordwell_df["reference"])
]
```

`zip` walks the two columns in lockstep so each candidate meets its own reference. `sentence_bleu` takes the candidate first and a plain list of reference strings second; note this wrapping is one level shallower than `corpus_bleu`'s, because here it is one segment with possibly several references, not several segments. `.score` pulls the float out immediately so the column holds numbers, not objects.

You still need to: assign each comprehension to its DataFrame column (`bleu_A`, `bleu_B`). Expect about a second for all 1000 scores.

---

## Task 4: Histogram function

The working core:

```python
fig, ax = plt.subplots()
ax.hist(df["bleu_A"], bins=30, alpha=0.6, label="System A")
ax.hist(df["bleu_B"], bins=30, alpha=0.6, label="System B")
```

`plt.subplots()` hands back the figure and axes as separate objects, which matters because the contract says return both. Two `hist` calls on the same `ax` overlay the distributions; `alpha=0.6` makes each translucent so the overlap region reads as a blend instead of one bar hiding the other; `bins=30` is fine enough that System B's isolated spike at 100 stays visibly separate from the rest of its mass; the `label=` values are what `ax.legend()` will display.

You still need to: the axis labels, title, legend call, the `return fig, ax` line, and one call to the function below the definition. The check harness verifies the legend, the labels, and the returned tuple.

---

## Task 5: Where B beats A

The working core is a delta, a filter, and an equality count:

```python
cordwell_df["delta_B_minus_A"] = cordwell_df["bleu_B"] - cordwell_df["bleu_A"]
```

Column arithmetic in pandas is element-wise, so this is the per-row score gap in one line. Positive values mean System B outscored System A on that row.

```python
b_wins_df = (
    cordwell_df[cordwell_df["delta_B_minus_A"] >= 5.0]
    .sort_values("delta_B_minus_A", ascending=False)
)
```

The inner comparison produces a boolean mask; indexing with it keeps only the rows where B's advantage is at least 5 BLEU points. Sorting descending puts the most extreme disagreements with our intuition on top, which is where you want to start reading.

```python
(b_wins_df["candidate_B"] == b_wins_df["reference"]).sum()
```

Comparing two string columns with `==` yields a boolean Series that is True exactly where System B's output is character-identical to the reference. Summing booleans counts the Trues. This one expression is the whole diagnosis: it measures how many of B's "wins" involved doing literally nothing.

You still need to: display the top 5 rows with the listed columns, wrap the count in `int()` and store it as `n_passthrough_wins`, and print the summary sentence from the worked target output. Then read two of the winning rows and let the result sink in.

---

## Task 6: Where A crushes B

The working core is Task 5's pattern with the direction flipped:

```python
cordwell_df["delta_A_minus_B"] = cordwell_df["bleu_A"] - cordwell_df["bleu_B"]
a_wins_df = (
    cordwell_df[cordwell_df["delta_A_minus_B"] >= 15.0]
    .sort_values("delta_A_minus_B", ascending=False)
)
```

Same mask-filter-sort chain; the threshold of 15 is deliberately larger because we are asking for decisive wins, and there are still hundreds of them.

You still need to: the count print in the target output's phrasing and the top-3 display with the listed columns.

---

## Task 7: N-grams and clipped precision

The working core of `ngrams`:

```python
[tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
```

A sliding window. For each start position `i`, slice out `n` consecutive tokens. The range bound is the subtle part: the last window must fit entirely inside the list, so the last valid start is `len(tokens) - n`, and `range` needs one more than that. The `tuple(...)` conversion matters because lists are not hashable and the very next step needs to count these with `Counter`.

The working core of `modified_precision`:

```python
cand_counts = Counter(ngrams(candidate_tokens, n))
ref_counts = Counter(ngrams(reference_tokens, n))
clipped = sum(min(count, ref_counts[gram]) for gram, count in cand_counts.items())
```

Two frequency tables, then the clip. For each distinct n-gram the candidate produced, it earns credit for the smaller of "how many times the candidate said it" and "how many times the reference contains it." `min` is the entire anti-cheat mechanism: a candidate repeating one common word gets capped at the reference's count. `Counter` returns 0 for missing keys rather than raising, so n-grams the reference never contains silently earn 0 credit with no special casing.

The denominator is `sum(cand_counts.values())`, the total number of n-grams the candidate produced, counting repeats. Clipped credit divided by total production is precision.

You still need to: the function shells, the empty-candidate guard returning 0.0, tokenizing the example row's candidate A and reference with `.split()`, computing `manual_p1` and `manual_p2`, and the two formatted prints.

---

## Task 8: Brevity penalty and manual BLEU-2

The working core of the penalty:

```python
if c_len >= r_len:
    return 1.0
return math.exp(1 - r_len / c_len)
```

Two branches, exactly as sacrebleu implements it. At or above reference length there is no penalty, because BLEU already punishes verbosity through diluted precision. Below reference length, the exponent `1 - r_len / c_len` is negative and grows more negative as the candidate shrinks, so `exp` of it decays smoothly from 1 toward 0. At exactly equal lengths the formula gives `exp(0) = 1`, so the two branches agree at the boundary.

The combination:

```python
bp * math.exp(0.5 * (math.log(p1) + math.log(p2))) * 100
```

The middle factor is the geometric mean of the two precisions, written in log space: average the logs, exponentiate back. Geometric rather than arithmetic because BLEU wants failing any order to hurt badly; a zero precision sends the geometric mean to zero no matter how good the other order is. The `* 100` converts to sacrebleu's 0 to 100 scale so your number and its number are comparable at a glance.

You still need to: the shell, `manual_bp` and `manual_bleu2` for candidate A from your Task 7 variables, the same five-step calculation on candidate B's own tokens, the two `sentence_bleu` comparison calls, and the printed layout from the worked target output.

---

## Stretch goals

Full assembled stretch solutions live only in the instructor solution notebook, but here is the same working-core treatment.

**Stretch 1, System C.** The core is three lines of list processing:

```python
words = [w.strip(",.!?;") for w in reference.split()]
content = [w for w in words if len(w) > 4]
rng.shuffle(content)
```

Strip punctuation so "checkout." and "checkout" are the same word, filter to longer words as a crude "content word" proxy, and shuffle in place with the provided RNG so the run is reproducible. Join up to 40 of them with spaces, add a period, and you have a system that emits only reference vocabulary in a garbage order. Then score it exactly as in Task 1 and read the component split exactly as in Task 2. Before you look: predict `precisions[0]` and `precisions[3]`.

**Stretch 2, chrF.** The core is one import and three mirror-image calls:

```python
from sacrebleu import corpus_chrf
chrf_a = corpus_chrf(cordwell_df["candidate_A"].tolist(), [references])
```

Identical calling convention to `corpus_bleu`, including the reference wrapping. Put both metrics' `.score` values for all three systems into one small DataFrame so the rankings sit side by side. The deliverable insight is not any number; it is which pair of systems trades places between the two columns, and your explanation of why in terms of what each metric counts.
