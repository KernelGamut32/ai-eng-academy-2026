# Week 3 Knowledge Check: Solution Key

**AI Engineering Academy | Gamut Technology Services | Instructor-facing. Do not distribute to students.**

Answer distribution: A appears 3 times, B 2 times, C 3 times, D 2 times. No positional pattern.

Question types: code questions are 3, 4, 6, 7, and 9. Concept questions are 1, 2, 5, 8, and 10.

| Q | Answer | Type | Maps to |
|---|--------|------|---------|
| 1 | C | concept | Module 01, prompting patterns and in-context learning |
| 2 | A | concept | Module 01, pattern selection heuristic |
| 3 | D | code | Module 02, structured output, provider differences |
| 4 | B | code | Module 02, schema validation |
| 5 | A | concept | Module 03, sampling parameters and determinism |
| 6 | C | code | Module 03, sampling parameter ranges |
| 7 | D | code | Module 06, chain-of-thought and self-consistency |
| 8 | B | concept | Module 04, role and persona prompting |
| 9 | C | code | Module 08, conversation memory, LangChain currency |
| 10 | A | concept | Module 02, classification design and abstention |

---

### Question 1 - Answer: C

Few-shot examples are consumed at inference. The model conditions its next-token predictions on the examples in the context window and adapts its behavior for that one response. This is in-context learning. No gradients are computed, no weights are updated, and nothing carries over to the next call unless you send the examples again.

Why the distractors are wrong. A and D both invoke real machine learning mechanisms, a gradient update and adapter weights, and attach them to the wrong process. Those are training-time and fine-tuning ideas, not prompt-time ones. This is the exact misconception the question is built to catch, since an engineer who half-remembers LoRA will find D attractive. B conflates two unrelated levers. Temperature affects sampling randomness and has nothing to do with whether the model reads the examples.

---

### Question 2 - Answer: A

The labels are already correct, so the reasoning is fine. The only failure is output shape. Few-shot examples that show the exact JSON key you want are the most direct lever for pinning format. This is the escalation heuristic from Module 01: reach only as far as the failure forces you.

Why the distractors are wrong. B and D both push chain-of-thought. They are tempting because the word "inconsistent" pattern-matches to reasoning problems, but the reasoning here is trivial and correct, so CoT adds cost and latency without touching the actual defect. C moves in the wrong direction. Raising temperature increases variability, which makes format drift worse.

---

### Question 3 - Answer: D (code)

`response_format` is an OpenAI Chat Completions parameter. It is not part of the Anthropic Messages API signature, so passing it raises a `TypeError` at call time rather than being accepted or quietly dropped. On Anthropic you steer structured output through prompt design and the current structured output support, not this parameter.

Why the distractors are wrong. A assumes the two providers share a signature, which is the root error being tested. B is plausible because APIs sometimes ignore unknown fields, but this one rejects it. C is a true statement about OpenAI JSON mode, which does require the literal word JSON in the prompt, attached to the wrong provider. That cross-provider trap is the point of the question.

---

### Question 4 - Answer: B (code)

By default `validate()` treats `format` as an annotation, not an assertion. Unless you pass a format checker, `format: date` does not reject a malformed value, so `2026-13-45` passes as long as it is a string. The `pattern` constraint, by contrast, is enforced by default, which is why `ticket_id` is genuinely validated while `opened` is not. This asymmetry is the teachable contrast.

Verified against jsonschema 4.26.0: the record above passes with default `validate()`, and flipping `ticket_id` to a value that violates the regex does raise a `ValidationError`.

Why the distractors are wrong. A is false. The `pattern` on `ticket_id` proves property constraints are checked. C is invented. jsonschema does not mutate the instance. D is technical-sounding noise. `format: date` is not draft-gated in the way described, and the reason it does not fire is the default annotation behavior, not the draft.

Follow-up to raise live: ask how to make it strict. Answer, pass a `FormatChecker` to the validator.

---

### Question 5 - Answer: A

`temperature=0` biases the sampler toward the top-probability token but does not promise identical output across calls on hosted endpoints. Providers document their chat endpoints as non-deterministic by default, and `seed` is best effort, not a guarantee. This matters for engineers who want to write exact-match assertions against model output.

Why the distractors are wrong. B and C are the reassuring misconception, that zero temperature or a seed buys byte-identical reproducibility. Both overstate what hosted inference guarantees. D fabricates a mechanism. The softmax is not disabled.

---

### Question 6 - Answer: C (code)

The OpenAI API accepts `frequency_penalty` in the range -2.0 to 2.0, so 2.5 is out of range and the request is rejected.

Why the distractors are wrong. B is the strongest trap and the most commonly chosen wrong answer. 0.0 to 2.0 is the slider range in the Playground UI, not the API range. Students who learned the parameter through the web console will pick it. The true API range includes negative values, which is why C is correct. A misses the range violation entirely, and D invents an integer-only rule that does not exist. Values like 0.5 are valid.

---

### Question 7 - Answer: D (code)

The vote is taken over raw model strings. `"Yes"`, `"yes"`, `"YES."`, and `"Yes "` are four spellings of the same answer, so `Counter` records four separate buckets of one vote each plus one `"No"`. That is a five-way tie at count one. `most_common(1)` breaks the tie by insertion order, so it happens to return `"Yes"`. The output looks correct, but the true four to one affirmative majority was never actually counted. Normalize before voting, for example lowercase, strip whitespace, and strip trailing punctuation, then tally.

Verified: the buggy function prints `Yes` from a five-way tie, and normalizing first yields `yes` with a clean four to one margin.

Why the distractors are wrong. A is false. `Counter` tallies any hashable, including strings. B is false. `most_common` returns the most frequent, not the least. C invents a parity rule that self-consistency does not require. The trap in D is that the code appears to work, which is why "it printed the right answer" is not evidence the logic is sound.

---

### Question 8 - Answer: B

Persona or role lines reliably shape tone, vocabulary, register, and format. The evidence that they improve factual accuracy is weak and inconsistent, and several studies find no reliable accuracy gain from adding an expert persona. Treat role prompting as a style and format control, not an accuracy control.

Why the distractors are wrong. A is the overclaim this question targets. Many engineers assume "act like an expert" makes the model more correct. C over-corrects in the other direction. The persona does measurably affect tone and format. D invents a refusal behavior that a persona line does not impose.

---

### Question 9 - Answer: C (code)

On the LangChain 1.x line the legacy `langchain.memory` classes are removed, and the `LLMChain` plus `chain.run()` pattern is legacy and no longer the supported path. The modern approach uses `langchain_core` primitives and LCEL, with memory handled through explicit message history rather than the old memory objects. The exact failure a student sees depends on the installed sub-packages, but the honest expectation is that this does not run as a current supported pattern.

Why the distractors are wrong. A treats a 2023 API as stable, which is the misconception under test. B is the softer version of the same error, "it is only a warning," which understates a removed import path. D is close enough to sound right, since you often do need `langchain-openai`, but the failure here is the legacy import and run pattern, not a missing provider package.

Currency note for the instructor: re-verify the exact import behavior against the version pinned in the cohort environment before class, since this line moves quickly.

---

### Question 10 - Answer: A

An explicit `other` or abstain label gives the model a legitimate destination for inputs that do not fit the defined classes. Without it, a forced-choice classifier will assign a wrong in-set label to out-of-scope or ambiguous text, which quietly pollutes downstream data. This is the abstention design point from task-to-prompt mapping.

Why the distractors are wrong. B is irrelevant. Adding a label does not meaningfully cut prompt cost. C is an overclaim. An escape hatch reduces forced errors but guarantees nothing. D describes multi-label output, which is a different design choice and not what a single `other` label provides.

---

## Scoring and use

Suggested cut line for "solid understanding" is 8 of 10. The two questions most likely to separate the room are Question 6 (the Playground range trap) and Question 7 (code that prints the right answer for the wrong reason). Both reward reading carefully rather than pattern-matching, which is the habit this check is meant to reinforce.

Fast debrief order if time is short: 7, 6, 3, 4, then the rest. Those four carry the most durable lessons about not trusting surface behavior and about provider and version specifics.
