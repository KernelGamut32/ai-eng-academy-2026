# Lab 03 Hints, Detailed Tier

Two hint files ship with this lab and both are yours. Pick **one tier
per task**: use `HINTS.md` when you want a nudge toward the approach, or
this file when you want to see the working core of a task with line by
line commentary. Reading both for the same task wastes your time.

This file shows the load-bearing lines of each task and explains why
each one is there. It withholds the function shells, return assembly,
and glue: you still write the function and wire these pieces into it,
which is where the learning is. The instructor solution notebook remains
the only fully assembled version.

---

## Part B: build_chain

The core is three constructions in sequence.

```python
prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM + "\n\nContext:\n{context}"),
    ("human", "{input}"),
])
```

The `{context}` slot is not decoration: `create_stuff_documents_chain`
renders the retrieved documents into exactly that variable and raises at
construction time if the prompt lacks it. `{input}` is the chain's
input key, so the human turn is the raw question.

```python
doc_prompt = PromptTemplate.from_template("[{source}] {page_content}")
```

This controls how each retrieved document is rendered before stuffing.
`page_content` is the document body; any other variable name, here
`source`, is looked up in `document.metadata`. Without this line the
default rendering is bare content, the model never sees a source id, and
citations are impossible no matter what the system prompt demands.

```python
combine_docs = create_stuff_documents_chain(llm, prompt, document_prompt=doc_prompt)
```

The stuff chain owns generation: render documents, fill `{context}`,
call the model. Note `document_prompt` is a keyword argument.

You wrap `combine_docs` and the retriever with `create_retrieval_chain`
(retriever first) and return the result.

---

## Part C: format_check

The parse attempt and the guards:

```python
try:
    obj = json.loads(answer.strip())
except json.JSONDecodeError:
    return False, None
```

Strip first, because trailing whitespace is legal but leading prose is
not, and `json.loads` conveniently rejects the prose-wrapped case for
you: "Sure! Here is the JSON: {...}" is not valid JSON at position 0.

```python
if set(obj) != {"answer", "citations", "confidence"}:
    return False, None
```

Iterating a dict yields keys, so `set(obj)` is the key set, and `!=`
enforces **exactly** these keys: a missing key fails, an extra key
fails. A subset test would let extra keys through, and the downstream
parser at Cordwell does not tolerate surprises.

```python
if obj["confidence"] not in {"high", "medium", "low"}:
    return False, None
```

The enum guard: "definitely" is enthusiasm, not a confidence level.

You still need the is-it-a-dict guard, the two type guards on answer
and citations, and the single success return of the tuple.

## Part C: abstention_check

```python
ok, obj = format_check(answer)
text = obj["answer"] if ok else answer
```

Reuse, do not reimplement: format_check already did the parsing work.
The second line picks what to scan: the inner answer field when the JSON
parsed, the raw string when it did not.

```python
if ok and obj["confidence"] == "low" and not obj["citations"]:
    return True
```

The structured dialect of refusal: the adapter was taught to signal
"no answer" as low confidence with nothing cited. Both conditions
matter; low confidence with a citation is a hedged answer, not a
refusal.

The prose dialect is a pattern scan you write yourself: `any` over
`re.search` of each ABSTAIN_PATTERNS entry against the lowercased text.

---

## Part D: decompose_claims

```python
raw = judge(CLAIM_PROMPT.format(answer=answer))
```

The judge is just a callable; injection means this same line works for
the scripted judge, LM Studio, or Ollama.

```python
[c.strip() for c in raw.split("\n") if c.strip()]
```

The contract says one claim per line, so cleanup is split, strip, drop
empties. The `if c.strip()` filter runs on the stripped value, which is
what drops whitespace-only lines.

## Part D: verify_claim and faithfulness

```python
out = judge(VERIFY_PROMPT.format(context=context, claim=claim))
return out.strip().upper().startswith("YES")
```

Normalize before testing: judges emit "YES", "yes.", or "Yes, because"
depending on model and mood. Strip, uppercase, prefix test handles all
of them and treats anything else, including waffle, as NO. Conservative
by design.

```python
if not claims:
    return 0.0
supported = sum(1 for c in claims if verify_claim(c, context, judge))
```

The empty case is a policy decision made explicit: an answer with no
checkable claims earns zero trust, not a free pass. The sum-over-
generator idiom counts True verdicts without building a list. You write
the decompose call above these lines and the division below them.

## Part D: answer_relevancy

```python
out = judge(RELEVANCY_PROMPT.format(question=question, answer=answer))
```

Same shape as verify_claim. The one trap: your function receives
`(answer, question)` but the prompt slots are named `question` and
`answer`, so format them by keyword, as here, and the order cannot bite
you.

---

## Part E: evaluate_set

The per-query core, inside your loop:

```python
res = chain.invoke({"input": q["question"]})
answer, docs = res["answer"], res["context"]
context = docs_to_context(docs)
fmt, _ = format_check(answer)
abst = abstention_check(answer)
```

Everything downstream is derived from these five lines: the chain gives
you the answer text and the retrieved documents, `docs_to_context`
rebuilds the exact context string the model saw, and your two Part C
verdicts come next because the faithfulness entry depends on the
abstention verdict:

```python
"faithfulness": None if abst else faithfulness(answer, context, judge),
```

None, not 0.0: an abstention asserts nothing, so it has no faithfulness
to measure, and averaging in a fake zero would punish the model for
honesty.

```python
if "gold_source" in q:
    row["context_hit"] = q["gold_source"] in row["sources"]
```

Conditional column: unanswerable queries have no gold chunk, so their
frame simply lacks the column. Membership in the sources list is the
whole recall test at this granularity.

You assemble the row dict with the remaining literal fields and finish
with `pd.DataFrame(rows)`.

## Part E: summarize

```python
faith = df["faithfulness"].dropna()
```

dropna is the reason the None convention above works: the mean is taken
only over rows that actually asserted something.

```python
"faithfulness": float(faith.mean()) if len(faith) else None,
```

The length guard covers the all-abstained frame, where mean of nothing
is NaN and you want None. The `float(...)` wrappers on every rate turn
numpy scalars into plain Python numbers so the dict serializes cleanly.
The other three rates are one-line column means you write yourself.

---

## Part F: abstention_report

The inner helper that computes one side:

```python
def side(df_ans, df_un):
    return {
        "abstention_rate_unanswerable": float(df_un["abstained"].mean()),
        "false_answer_rate_unanswerable": float((~df_un["abstained"]).mean()),
        "abstention_rate_answerable": float(df_ans["abstained"].mean()),
    }
```

Line by line: the mean of a boolean column is the fraction True, so the
first rate is literally "how often it declined when it should". The
tilde negates the boolean column elementwise, so the second is "how
often it answered when it should not have", the false answer rate; the
inner parentheses matter because `~` must apply to the column before
`.mean()`. The third catches the symmetric failure, refusing good
questions. You call this helper once per model side and assemble the
nested dict.

---

## Part G (stretch): run_degraded

```python
set_adapter(True)
chain_k1 = build_chain(llm, make_retriever(1))
df = evaluate_set(chain_k1, queries, judge)
```

The whole experiment is re-composition: same model, same questions, same
metrics, one knob turned. `make_retriever(1)` is the injected factory so
this works on either backend.

```python
hit = float(df["context_hit"].mean())
```

Mean of the boolean column, exactly as in summarize.

```python
if hit < 1.0:
    diagnosis = "fix_retrieval"
elif mean_faith is not None and mean_faith < 1.0:
    diagnosis = "fix_generation"
else:
    diagnosis = "healthy"
```

The order encodes the diagnostic table's logic: a retrieval miss makes
faithfulness numbers unreliable evidence about generation (a model can
be perfectly faithful to the wrong chunk), so retrieval is checked, and
fixed, first. You compute mean_faith with the dropna pattern from
summarize and assemble the returned frame and dict.

Stretch solutions live at full depth in the instructor solution
notebook, released after the lab.
