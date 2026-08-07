# Module 02 Knowledge Check: Evaluating Generated Text

**AI Engineering Academy | Gamut Technology Services**

Five multiple choice questions covering Module 02: why surface-overlap metrics break on generated text, ROUGE configuration and reproducibility, LLM-as-a-judge design, and deciding whether a metric delta is real.

All questions use the Cordwell Home and Hardware scenario from the module: Task A is support ticket summarization with a gold reference set, and Task B is catalog blurb generation with no references.

**Instructions.** Choose the single best answer for each question. Three of the five show code and ask you to reason about what it does. Read the code carefully. Every code question in this set describes something that runs without raising an exception and still reports a wrong or misleading number.

Closed book. Approximately 12 minutes.

---

### Question 1

The module scores one Cordwell ticket summary two ways. The reference says the customer wants a **replacement** shipped before Saturday.

| System | BLEU | chrF++ | ROUGE-1 | ROUGE-L |
|---|---|---|---|---|
| A: promises a **refund** | 69.97 | 79.55 | 83.87 | 83.87 |
| B: correct paraphrase, promises a **replacement** | 6.89 | 36.71 | 34.29 | 34.29 |

Separately, on the reference "Use the large drill bit for masonry," the candidate using "big" (synonym) and the candidate using "small" (antonym) both score BLEU 59.46 and ROUGE-1 85.71.

What do these two results, taken together, establish?

- A. BLEU is poorly implemented for this task, and switching to chrF++ or ROUGE-L resolves the ranking.
- B. The reference set is too small. Adding more reference summaries per ticket would let these metrics rank System B correctly.
- C. Every metric in this family is a function of token overlap, so it is structurally unable to represent meaning. Catching System A requires a check that reads the source ticket and applies a rule, not a better overlap metric.
- D. The metrics are correct and System A is genuinely the better output, since it matches the reference wording more closely.

---

### Question 2

A teammate computes ROUGE for the Cordwell summarizer.

```python
from rouge_score import rouge_scorer

REF  = "the customer wants a replacement pressure washer unit shipped before Saturday morning"
CAND = "customer wants a replacement shipped Saturday"

scorer = rouge_scorer.RougeScorer(["rouge1"], use_stemmer=True)

scores = scorer.score(CAND, REF)          # <-- note the argument order
r1 = scores["rouge1"]
print(f"P={r1.precision:.4f} R={r1.recall:.4f} F={r1.fmeasure:.4f}")
# P=0.5000 R=1.0000 F=0.6667
```

The teammate reports that the summarizer achieves perfect recall. What actually happened?

- A. Nothing is wrong. Recall of 1.0000 is correct, because every candidate token appears in the reference.
- B. `RougeScorer.score` takes the target first and the prediction second, so the arguments are reversed. Precision and recall are swapped: true recall is 0.5000, not 1.0000. The F-measure is unchanged by the swap, so reporting F alone would have hidden the bug entirely.
- C. The call raises a `ValueError` on reversed arguments, so the printed output could not have come from this code.
- D. `use_stemmer=True` inflated recall by collapsing distinct tokens to shared stems, and setting it to `False` would return the true value.

---

### Question 3

This code scores a three-sentence Cordwell summary against a three-sentence reference. The content is identical and only the sentence order differs.

```python
from rouge_score import rouge_scorer

scorer = rouge_scorer.RougeScorer(["rougeL", "rougeLsum"], use_stemmer=False)

single_ref  = "The installer mounted the brackets. The shelving was secured to the studs. The customer signed off on the work."
single_cand = "The shelving was secured to the studs. The customer signed off on the work. The installer mounted the brackets."

s = scorer.score(single_ref, single_cand)
print(s["rougeL"].fmeasure, s["rougeLsum"].fmeasure)
# 0.7368 0.7368
```

Passing the identical text with sentences separated by newline characters instead yields `rougeL` 0.7368 and `rougeLsum` 1.0000. What explains this?

- A. `rougeLsum` is nondeterministic and should be averaged across several runs.
- B. Newline characters are counted as tokens, which inflates the `rougeLsum` denominator and coincidentally produces 1.0000.
- C. `rougeLsum` computes the longest common subsequence per sentence and aggregates, so it needs newline-separated sentences to find sentence boundaries. Given a single-line string it silently degrades to `rougeL`, reporting a number that looks valid.
- D. `rougeL` is the variant that requires newlines, and the single-line result of 0.7368 is the degraded one.

---

### Question 4

A team stands up an LLM judge for the Cordwell blurb pipeline.

```python
system_prompt = """
You are an expert evaluator. Rate the following summary
on a scale of 1 to 10 for overall quality.

Respond with: {"score": }
"""

JUDGE_MODEL     = "gpt-4o"
GENERATOR_MODEL = "gpt-4o"

# Pairwise bake-off: system_a is always presented first
for ref, a, b in eval_pairs:
    result = judge.evaluate(ref, a, b)
```

The judge returns scores and the pipeline runs end to end. Which set of defects is present?

- A. The rubric lacks a named criterion and anchored scale, the judge shares a model family with the generator, and the pairwise order is never swapped. These activate noise, self-preference bias, and position bias respectively.
- B. The only defect is the missing `temperature=0` setting, which makes the scores nondeterministic.
- C. The scale should be 1 to 100 rather than 1 to 10 to give the judge more resolution, and no other defect is present.
- D. The judge is correctly configured. Pairwise comparison already controls for position and self-preference bias by design.

---

### Question 5

You run the module's paired bootstrap in sacrebleu on a 200-item Cordwell blurb set and get this:

```
baseline  BLEU = 70.64  ci = +/- 4.91
candidate BLEU = 71.31  ci = +/- 4.80  p = 0.333  delta = +0.67
```

A colleague writes in the pull request: "Candidate improves BLEU by 0.67. Small but potentially meaningful, recommend shipping." What is the correct assessment?

- A. The colleague is right. A positive delta on a 200-item set is weak evidence but still points the right direction, so shipping is defensible.
- B. The delta sits well inside a confidence interval of roughly plus or minus 4.9 with p equal to 0.333. This is noise, and "small but potentially meaningful" is the wrong label for it. No difference has been measured.
- C. The result is invalid because BLEU cannot be bootstrapped, only chrF++ and ROUGE can.
- D. The delta is real but too small to matter operationally, so the decision should rest on inference cost instead.

---

*End of quiz. Five questions. Answer key is a separate file.*
