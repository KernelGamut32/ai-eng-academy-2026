# Module 02 Knowledge Check: Solution Key

**AI Engineering Academy | Gamut Technology Services | Instructor-facing. Do not distribute to students.**

Answer distribution: A appears 1 time, B 2 times, C 2 times. On a five-question set some clustering is unavoidable; reshuffle option order if you reuse the set.

Question types: code questions are 2, 3, and 4. Concept questions are 1 and 5.

**Coverage note.** The five questions map onto the deck's four Key Takeaways plus Segment 4, "The Break," which is the module's thesis slide.

| Q | Answer | Type | Maps to | Deck takeaway |
|---|--------|------|---------|---------------|
| 1 | C | concept | Segment 4, the metric break | Surface metrics are tripwires, not verdicts |
| 2 | B | code | Segment 3, ROUGE argument order | A metric without its config is not a number |
| 3 | C | code | Segment 3, Trap 2 (rougeL vs rougeLsum) | A metric without its config is not a number |
| 4 | A | code | Segment 6, rubric anatomy and the four biases | Judged evaluation needs calibration |
| 5 | B | concept | Segment 8, paired bootstrap | No interval, no claim |

**Two segments are not covered.** Segment 5 (semantic and learned metrics, including the BERTScore `rescale_with_baseline` point) and Segment 7 (reference-free evaluation and the RAGAS callback) get no dedicated item. Question 1's correct answer gestures at Segment 7 by naming the source-reading check, but neither segment is tested directly. A ready-to-use sixth question on BERTScore rescaling is supplied at the end of this key if you want fuller coverage.

---

### Question 1 - Answer: C

Two independent results point at the same conclusion. First, on the Cordwell ticket, every metric in the family ranks the factually wrong output above the correct one, BLEU by roughly a factor of ten. System A promises a refund and the customer receives a pressure washer. Second, the minimal pair shows that swapping "large" for its synonym "big" and for its antonym "small" produces identical BLEU and identical ROUGE-1. Meaning preserved and meaning inverted are indistinguishable. That is not weak correlation with meaning, it is structural inability to represent it, because every metric in Segments 2 and 3 is a function of token overlap and none of them reads the source document or applies a rule.

Verified against sacrebleu 2.6.0 and rouge-score 0.1.2: the synonym and antonym candidates both score BLEU 59.46 and ROUGE-1 85.71, differing only on chrF++ by 0.4 points (75.27 against 74.87), which is noise rather than signal about meaning.

Why the distractors are wrong. A proposes swapping metrics, but the table shows chrF++ and ROUGE-L making the same error as BLEU, so the fix cannot live inside the family. This is the most instructive wrong answer, because "use a better metric" is the natural engineering reflex. B proposes more references, which is the move the module explicitly rules out in Discussion 1. More references would help with paraphrase, the System B problem, but the System A failure is factual, and no quantity of correct references makes an overlap statistic notice that "refund" and "replacement" are different promises. D accepts the metric's verdict at face value, which is the failure the whole module exists to prevent.

---

### Question 2 - Answer: B (code)

`rouge_scorer.RougeScorer.score` takes the target first and the prediction second, which is the reverse of the usual mental order and the reverse of sacrebleu's convention. The call here passes `CAND` first, so the library treats the six-token candidate as the reference and the twelve-token reference as the candidate. Precision and recall come back swapped. The reported recall of 1.0000 is really the precision, and the true recall is 0.5000. The summarizer covers half the reference content, not all of it.

The detail that makes this dangerous: the F-measure is the harmonic mean of precision and recall, which is symmetric, so it reads 0.6667 either way. A team that reports only F, which is what papers report, would never see the bug.

Verified: with target and prediction correctly ordered, P is 1.0000 and R is 0.5000, F is 0.6667. With them reversed, P is 0.5000 and R is 1.0000, F is 0.6667. Confirmed programmatically that P and R swap exactly and F is identical to machine precision.

Why the distractors are wrong. A accepts the output as correct and offers a rationale that actually describes precision, not recall, which is exactly the confusion the swap creates. C assumes input validation that does not exist. Both arguments are strings of the right type, so nothing raises. D blames the stemmer, which is a real ROUGE hazard from the adjacent slide and therefore a tempting misattribution, but stemming affects which tokens match, not which argument is treated as the reference.

Note on the deck's own example: the "Run It" slide uses a pair where P and R are equal, so the swap is invisible there. Worth mentioning live that the deck's inline comment flagging the argument order is doing real work even though its own example cannot demonstrate the failure.

---

### Question 3 - Answer: C (code)

`rougeLsum` is the variant summarization papers report. It computes the longest common subsequence per sentence and aggregates, which is what lets it recognize reordered but identical content. It finds sentence boundaries by splitting on newline characters. Handed a single-line string it finds exactly one sentence, computes one LCS over the whole text, and produces the same number as `rougeL`. No warning, no error, just a valid-looking score that silently understates a summary whose content is complete.

Verified: on the three-sentence reordered pair, single-line input gives `rougeL` 0.7368 and `rougeLsum` 0.7368, identical. Newline-separated input gives `rougeL` 0.7368 and `rougeLsum` 1.0000. The deck demonstrates the same structural behavior with a different example at 0.5600 against 0.9600.

Why the distractors are wrong. A invents nondeterminism. Both variants are fully deterministic. B offers a mechanical-sounding explanation involving newline tokenization, which is attractive because it correctly identifies newlines as the operative detail while getting the mechanism backwards. Newlines are sentence delimiters here, not counted tokens, and they do not inflate a denominator. D inverts the two variants, claiming `rougeL` needs newlines. `rougeL` treats the input as one sequence regardless, which is why its score is stable at 0.7368 across both formats. That stability is the tell, and a student who traces both columns can rule D out from the numbers alone.

The module calls this the most common silent ROUGE bug in production code, which is worth repeating in the debrief.

---

### Question 4 - Answer: A (code)

Three defects, each mapping to a control the module names:

1. **Vague criterion and unanchored scale.** "Overall quality, 1 to 10" with no named criterion, no anchors on the scale points, and no worked examples in the prompt. The module's rubric anatomy requires a named criterion, a defined scale with anchors, structured output, and at least one passing and one failing example. An unanchored 1 to 10 produces noise.
2. **Self-preference bias.** `gpt-4o` serves as both generator and judge. A judge scores its own model family higher. The control is never using the same family for both roles.
3. **Position bias.** The loop always presents `system_a` first. In pairwise comparison, slot A wins more often than chance, roughly 10 to 15 points of win-rate swing on close calls. The control is running every pair twice with the order swapped and counting order-flips as ties.

Also worth noting live, though not required for the answer: the structured output field `{"score": }` is malformed and has no rationale field, and the rubric offers no criterion for what the score even measures.

Why the distractors are wrong. B names `temperature=0`, which is real and good practice, and the module's judge code does set it, but determinism is not what makes this prompt unusable. A perfectly deterministic judge applying a vague rubric returns consistent noise. This is the strongest distractor because it identifies a genuine omission and mistakes it for the main problem. C proposes more scale resolution, which moves in the wrong direction. The module notes that binary pass or fail is more reliable than 1 to 10, since finer scales without anchors add variance rather than information. D claims pairwise comparison inherently controls for position bias, which inverts the truth. Pairwise is where position bias lives, and it is the reason the order-swap control exists.

---

### Question 5 - Answer: B

The delta of plus 0.67 sits well inside a confidence interval of roughly plus or minus 4.9, and p equals 0.333. Nothing has been measured. The module's standard is direct: no interval, no claim. The phrase "small but potentially meaningful" is specifically the wrong label, because it treats a point estimate as directional evidence when the interval says the sign of the true difference is not even established.

Worth landing in the debrief: the colleague did the harder half correctly. They computed a relative delta against an identified baseline rather than quoting an absolute score, which is genuinely better practice. The failure is stopping one step short. The module's reporting standard asks for five items in a pull request: the metric with its full config signature, the delta against a named baseline, the confidence interval, the test type with its p-value, and the test set identifier with its size.

Why the distractors are wrong. A treats a positive point estimate as weak evidence in the right direction, which is precisely the reasoning the interval refutes. It is the most common real-world version of this mistake and the one most likely to be selected. C is false on the facts, since the module runs the paired bootstrap on BLEU directly through `sacrebleu.significance.PairedTest`. D concedes the delta is real and then argues about whether it matters, which skips the actual finding: it has not been shown to exist.

---

## Scoring and use

Suggested cut line is 4 of 5. On a five-item check the resolution is coarse, so treat a miss as a signal about which segment to revisit rather than an overall verdict.

The two questions most likely to separate the room are 2 and 3. Both describe code that runs cleanly and reports a plausible number that is wrong, which is the habit this module is built to instill. Question 2 is the more valuable of the two, because the F-measure symmetry means the standard reporting practice actively conceals the bug.

Fast debrief order if time is short: 1, 2, 5, then the rest. Those three carry the module's thesis, its sharpest silent failure, and its reporting standard.

---

## Optional sixth question (BERTScore rescaling, Segment 5)

Add this if you want Segment 5 covered. Answer is B.

> A teammate runs BERTScore with `roberta-large` on a Cordwell blurb and reports an F1 of 0.83, describing it as "solid, roughly a B grade." The call omitted `rescale_with_baseline=True`. Why is that reading wrong?
>
> - A. 0.83 is a failing score for BERTScore, and the teammate has the direction inverted.
> - B. Raw BERTScore with this model compresses into a narrow band regardless of quality, roughly 0.85 to 0.95, so an unrescaled 0.83 sits near the floor rather than in the middle. Rescaling applies a linear transform against a random-pairs baseline so the number reads on an interpretable range.
> - C. BERTScore F1 is only interpretable relative to a second system, so no single value carries meaning under any configuration.
> - D. The score is inflated because `lang` was not set, and setting it alone would correct the range.

D is the trap worth watching for: `lang` genuinely does matter, since it selects the baseline used for rescaling, but setting it without enabling `rescale_with_baseline` changes nothing about the reported range.

---

## Verification ledger

Every quantitative claim in this quiz was recomputed against the pinned libraries rather than taken from the deck.

| Claim | How verified | Result |
|---|---|---|
| Synonym and antonym score identically (Q1) | Scored both candidates against the reference with sacrebleu and rouge-score | Both BLEU 59.46, both ROUGE-1 85.71; chrF++ 75.27 against 74.87. Matches the deck |
| ROUGE argument swap behavior (Q2) | Called `score()` in both orders on an asymmetric 12-token / 6-token pair | Correct order P=1.0000 R=0.5000 F=0.6667; reversed P=0.5000 R=1.0000 F=0.6667. P and R swap exactly, F identical |
| rougeLsum newline dependence (Q3) | Scored a three-sentence reordered pair as single-line and as newline-separated | Single-line: both variants 0.7368. Newline-separated: rougeL 0.7368, rougeLsum 1.0000 |
| Deck's BLEU, smoothing, chrF, TER figures | Ran the deck's own "Run It" and smoothing slides | BLEU 37.99 with precisions 83.3/60.0/25.0/16.7 and BP 1.000; smoothing none 0.00, floor 8.03, exp 37.99; chrF 61.93, chrF++ 64.37, TER 16.67. All match exactly |
| Deck's stemmer swing | Ran the deck's installer sentence pair at both flag values | `use_stemmer=False` gives 50.00 / 9.09 / 50.00; `True` gives 100.00 / 100.00 / 100.00. Matches exactly |

Stack used for verification: sacrebleu 2.6.0, rouge-score 0.1.2, Python 3.12.3 (sandbox). These are the versions the deck itself cites, and the reported signature `nrefs:1|case:mixed|eff:no|tok:13a|smooth:exp|version:2.6.0` reproduced exactly.

Not independently verified here: the Segment 8 paired bootstrap figures (Q5), which depend on the 200-item Cordwell blurb set rather than on library behavior, and the four judge biases and their magnitudes (Q4), which are literature claims carried from the deck. The bias magnitude cited in Q4's rationale, 10 to 15 points of win-rate swing on close calls, is the deck's own figure. The Goyal et al. (2022) finding on slide 27 is not tested in this quiz.
