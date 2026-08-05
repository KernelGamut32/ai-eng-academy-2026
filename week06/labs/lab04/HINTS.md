# Week 6 Lab 2.1: Progressive Hints

Pick one hint tier per task, not both. This file gives three escalating levels per task: level 1 names the approach, level 2 sketches the structure, level 3 shows a key line or two in context. If you would rather read the working core of a task with commentary and then write your own version, use `HINTS_DETAILED.md` instead.

---

## Task 1: Corpus-level BLEU

**Level 1.** Everything you need is in the Section 1 worked example, part 2. Pull the three text columns out of the DataFrame as lists, then call `corpus_bleu` twice with the same reference list.

**Level 2.** Three list extractions with `.tolist()`, then two calls shaped like the toy example: candidates as the first argument, and the references wrapped in an outer list as the second. Store the whole returned objects, then print `.score` from each with an f-string format of `:.2f`.

**Level 3.** The wrapping is the part people miss:

```python
bleu_a = corpus_bleu(candidates_a, [references])
```

One set of brackets around `references`, because sacrebleu supports multiple reference streams and we have exactly one.

---

## Task 2: Score components

**Level 1.** No new computation. You are unpacking attributes that already exist on `bleu_a` and `bleu_b` (`.score`, `.precisions`, `.bp`, `.sys_len`, `.ref_len`) into a two-row DataFrame.

**Level 2.** Write a small helper that takes one BLEUScore object and returns a dict with the eight required keys. `.precisions` is a list of four floats you can unpack in one line. Then build the DataFrame from a list of two dicts and pass `index=["System A", "System B"]`.

**Level 3.** The unpack and the constructor:

```python
p1, p2, p3, p4 = bleu_result.precisions
...
components_df = pd.DataFrame([row_a, row_b], index=["System A", "System B"])
```

---

## Task 3: Sentence-level BLEU columns

**Level 1.** One `sentence_bleu(candidate, [reference]).score` call per row, per system. A list comprehension over paired columns is the cleanest shape.

**Level 2.** `zip` the candidate column and the reference column, score each pair, and assign the resulting list straight into a new DataFrame column. Then repeat for the other system. Note the reference wrapping again: a plain list of strings this time, `[ref]`.

**Level 3.** One of the two columns:

```python
cordwell_df["bleu_A"] = [
    sentence_bleu(cand, [ref]).score
    for cand, ref in zip(cordwell_df["candidate_A"], cordwell_df["reference"])
]
```

---

## Task 4: Histogram function

**Level 1.** Standard matplotlib: `plt.subplots()` for the figure and axes, two `ax.hist(...)` calls on the same axes, labels, legend, and return the pair.

**Level 2.** Each `hist` call needs the column, `bins=30`, `alpha=0.6`, and a `label=`. After both, set `ax.set_xlabel`, `ax.set_ylabel`, `ax.set_title`, call `ax.legend()`, and `return fig, ax`. Then call the function once below the definition and unpack the result.

**Level 3.** The core pair of calls:

```python
fig, ax = plt.subplots()
ax.hist(df["bleu_A"], bins=30, alpha=0.6, label="System A")
```

---

## Task 5: Where B beats A

**Level 1.** Three moves: a subtraction column, a boolean filter plus sort, and a count of rows where two string columns are exactly equal.

**Level 2.** `cordwell_df["delta_B_minus_A"] = cordwell_df["bleu_B"] - cordwell_df["bleu_A"]`, then filter with a `>= 5.0` comparison and `.sort_values(..., ascending=False)`. For the passthrough count, compare the two text columns of the filtered frame with `==` and `.sum()` the boolean result; wrap in `int()` for a clean print.

**Level 3.** The diagnostic count:

```python
n_passthrough_wins = int((b_wins_df["candidate_B"] == b_wins_df["reference"]).sum())
```

---

## Task 6: Where A crushes B

**Level 1.** Same recipe as Task 5 with the subtraction flipped and a threshold of 15. No equality count this time.

**Level 2.** New delta column, filter `>= 15.0`, sort descending, print `len(a_wins_df)` in the required sentence, then display the head with the listed columns.

**Level 3.** The filter and sort in one chain:

```python
a_wins_df = (
    cordwell_df[cordwell_df["delta_A_minus_B"] >= 15.0]
    .sort_values("delta_A_minus_B", ascending=False)
)
```

---

## Task 7: N-grams and clipped precision

**Level 1.** `ngrams` is a sliding window: slice `tokens[i : i + n]` for each valid start index and convert each slice to a tuple. `modified_precision` is the Section 1 clipping demo generalized: count n-grams on both sides with `Counter`, clip with `min`, divide.

**Level 2.** For `ngrams`, the last valid start index is `len(tokens) - n`, so the range runs to `len(tokens) - n + 1`. For `modified_precision`: build `Counter(ngrams(cand, n))` and `Counter(ngrams(ref, n))`, sum `min(candidate count, reference count)` across the candidate's distinct n-grams, and divide by the total candidate n-gram count, which is `sum(cand_counts.values())`. Return 0.0 if the candidate has no n-grams.

**Level 3.** The clipping line, which is the whole idea:

```python
clipped = sum(min(count, ref_counts[gram]) for gram, count in cand_counts.items())
```

`ref_counts[gram]` is safely 0 for n-grams the reference never contains, because `Counter` returns 0 for missing keys instead of raising.

---

## Task 8: Brevity penalty and manual BLEU-2

**Level 1.** `brevity_penalty` is a two-branch function straight from the formula in the task intro. BLEU-2 is the brevity penalty times the geometric mean of p1 and p2, and a geometric mean of two numbers is `exp(0.5 * (log(p1) + log(p2)))`.

**Level 2.** Branch on `c_len >= r_len` returning 1.0, else `math.exp(1 - r_len / c_len)`. For candidate A you already have tokens and precisions from Task 7, so `manual_bp` and `manual_bleu2` are two lines; remember the final `* 100` to land on sacrebleu's scale. For candidate B, redo the same five steps on its own tokens: split, p1, p2, bp, combine. Then call `sentence_bleu` on each candidate and print both sets side by side.

**Level 3.** The combination line for candidate A:

```python
manual_bleu2 = manual_bp * math.exp(0.5 * (math.log(manual_p1) + math.log(manual_p2))) * 100
```

---

## Stretch goals

**Stretch 1, level 1.** Strip punctuation from each word, keep words longer than 4 characters, `rng.shuffle` the list, join the first 40 with spaces. Then it is Task 1 again with a third candidate column, and Task 2's attribute reads to see the component breakdown.

**Stretch 1, level 2.** Before running the corpus score, write down your prediction for `precisions[0]` and `precisions[3]`. Every emitted word came from the reference, and almost no shuffled pair of words is a real reference bigram. Now check yourself.

**Stretch 2, level 1.** `from sacrebleu import corpus_chrf`, and its call signature mirrors `corpus_bleu`: candidates first, then the wrapped reference list. Score all three systems, put BLEU and chrF side by side in a small DataFrame, and compare rankings rather than magnitudes.

**Stretch 2, level 2.** The interesting question is why the B and C ranking flips between metrics. Think about what a character n-gram survives that a word 4-gram does not, and about the fact that chrF is an F-score, meaning recall of the reference content counts against System B's truncation.
