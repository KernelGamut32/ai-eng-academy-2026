# Lab 03 Hints, Progressive Tier

Two hint files ship with this lab and both are yours. Pick **one tier
per task**: use this file when you want a nudge toward the approach, or
`HINTS_DETAILED.md` when you want to see the working core of a task with
commentary. Reading both for the same task wastes your time. Levels
below escalate; stop at the first level that unsticks you.

---

## Part B: build_chain

**Level 1.** Two helpers, composed. `create_stuff_documents_chain` gets
the llm and your chat prompt (plus one keyword argument). Its output is
the second argument to `create_retrieval_chain`, after the retriever.

**Level 2.** Build the chat prompt with
`ChatPromptTemplate.from_messages` from two tuples, a system message and
a human message. The system string is SYSTEM plus a Context section that
ends in the `{context}` slot; the human message is just the `{input}`
slot. Then make a `PromptTemplate` that renders one document as its
source in square brackets followed by its content, and hand it to the
stuff chain as `document_prompt=`.

**Level 3.** The document prompt is one line:
`PromptTemplate.from_template("[{source}] {page_content}")`. Those two
variable names are special: `page_content` is the document body and any
other name is looked up in the document's metadata. If your citations
come back empty, this is the piece you are missing.

---

## Part C: format_check

**Level 1.** `json.loads` inside try except, then a series of guard
checks that each return the failure tuple. Only one success exit.

**Level 2.** Guard order: parse (catch `json.JSONDecodeError`), is it a
dict, is the key set **exactly** the three required keys (compare sets
with `==`, not subset), are the answer and citations the right types, is
confidence in the allowed set. Strip the string before parsing.

**Level 3.** The exact-keys guard is `set(obj) != {"answer",
"citations", "confidence"}`. Iterating a dict yields its keys, so
`set(obj)` is the key set. A subset check would wrongly accept extra
keys.

## Part C: abstention_check

**Level 1.** Reuse format_check. The text you scan for refusal phrasing
depends on whether the answer parsed.

**Level 2.** If it parsed: the JSON dialect check is confidence low plus
empty citations, and the text to scan is the parsed answer field. If it
did not parse, scan the raw string. Either way finish with `any` over
`re.search` of the patterns against the lowercased text.

**Level 3.** `text = obj["answer"] if ok else answer` is the whole
branching. Check the low-confidence-empty-citations dialect first and
return True early; fall through to the pattern scan.

---

## Part D: decompose_claims

**Level 1.** One judge call, then plain string cleanup. No JSON here,
the judge replies with lines of text.

**Level 2.** `CLAIM_PROMPT.format(answer=answer)`, pass the result to
`judge`, split the reply on newline characters, strip each piece, keep
the non-empty ones.

**Level 3.** The whole body is two lines: the judge call, then a list
comprehension with an `if c.strip()` filter. If you get empty strings in
your claim list, your filter runs after the strip in the wrong order.

## Part D: verify_claim and faithfulness

**Level 1.** verify_claim is one judge call and one string test.
faithfulness composes your two previous functions and divides.

**Level 2.** For the YES test, normalize the reply first: strip, then
uppercase, then `.startswith("YES")`. For faithfulness, get the claims,
return 0.0 immediately when the list is empty, otherwise count the
claims that verify and divide by the total.

**Level 3.** The supported count reads
`sum(1 for c in claims if verify_claim(c, context, judge))`. Dividing
two ints gives the float you want in Python 3; no cast needed.

## Part D: answer_relevancy

**Level 1.** Structurally identical to verify_claim with a different
prompt and two format arguments.

**Level 2.** `RELEVANCY_PROMPT.format(question=question, answer=answer)`
then the same strip, uppercase, startswith YES test.

**Level 3.** If this one fails while verify_claim passes, check the
argument order in your format call: the prompt names its slots
`question` and `answer`, and your function receives them as
`(answer, question)`.

---

## Part E: evaluate_set

**Level 1.** One loop over queries, one chain invoke per query, one row
dict per query built from your Part C and D functions, then
`pd.DataFrame(rows)` at the end.

**Level 2.** Per query: invoke with the input key, pull the answer and
context list from the result, render the context string with
`docs_to_context`. The row needs the abstention verdict **before** the
faithfulness entry, because faithfulness is None when abstained. Add
context_hit only when the query dict carries gold_source; test with
`"gold_source" in q`.

**Level 3.** The conditional column is added after building the row:
`if "gold_source" in q: row["context_hit"] = q["gold_source"] in
row["sources"]`. Building sources first as a list of
`d.metadata["source"]` makes that membership test one expression.

## Part E: summarize

**Level 1.** Three column means plus one mean over a filtered column,
all wrapped in float.

**Level 2.** `df["faithfulness"].dropna()` gives the non-null values;
take its mean only when it is non-empty, otherwise use None. Wrap every
mean in `float(...)` so the dict holds plain Python numbers.

**Level 3.** The guard reads
`float(faith.mean()) if len(faith) else None` where faith is the
dropna result. Without the length guard, an all-abstained frame gives
you NaN instead of None and the JSON dump downstream looks wrong.

---

## Part F: abstention_report

**Level 1.** The same three rates computed twice, once per model side.
An inner helper keeps it to a few lines.

**Level 2.** All three rates are means of the abstained column or its
negation: unanswerable abstention is the mean over the unanswerable
frame, false answer rate is the mean of NOT abstained over that same
frame, answerable abstention is the mean over the answerable frame.

**Level 3.** The negation of a boolean pandas column is `~df["abstained"]`,
and `float((~df_un["abstained"]).mean())` is the false answer rate.
Note the tilde binds after the indexing, hence the inner parentheses.

---

## Part G (stretch): run_degraded

**Level 1.** You are composing things you already built: set_adapter,
make_retriever with k of 1, build_chain, evaluate_set, then a small dict
with one if elif else.

**Level 2.** Compute the three floats first (context hit mean, format
mean, mean of non-null faithfulness), then pick the diagnosis string by
the rule in the lab text, testing context hit before faithfulness.

**Level 3.** The decision reads: hit below 1.0 gives fix_retrieval,
else faithfulness below 1.0 gives fix_generation, else healthy. If your
G1 check fails on the hit rate, print your k equal to 1 sources column
and look at which chunk won the tie for the breaker question.
