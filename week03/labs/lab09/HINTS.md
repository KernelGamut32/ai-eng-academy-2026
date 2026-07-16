# Lab 09 Hints

Use these only when you are stuck. Try the check output and the TODO contract first. Level 1
is a nudge, Level 2 is a stronger nudge, Level 3 is close to the answer.

---

## TODO 1 - extraction chain

- **Level 1.** You already saw the shape in the demo. Three things join together.
- **Level 2.** The order matters: format the message, then call the model, then parse the
  reply. Left to right.
- **Level 3.** `extract_chain = EXTRACT_PROMPT | model | ticket_parser`.

## TODO 2 - validate

- **Level 1.** Pydantic already knows how to check a dict. You only need to catch what it
  throws and reshape it.
- **Level 2.** `TicketExtract.model_validate(d)` raises `ValidationError` on bad input. That
  exception has an `.errors()` method returning a list of dicts.
- **Level 3.** Try to validate, return `[]` on success, and in the except block return a
  list built from `err['loc']` and `err['msg']` for each `err` in `e.errors()`.

## TODO 3 - repair loop

- **Level 1.** Build the repair chain the same way you built extraction. The rest is plain
  control flow.
- **Level 2.** Validate first. If there are no errors, return the ticket unchanged. Otherwise
  invoke the repair chain with the two variables its prompt expects.
- **Level 3.** `repair_chain = REPAIR_PROMPT | model | ticket_parser`. In
  `validate_or_repair`, call `repair_chain.invoke({"errors": json.dumps(errs), "original":
  json.dumps(ticket)})`, revalidate the result, and return the original if it still fails.

## TODO 4 - full pipeline

- **Level 1.** The summarize prompt does not take a raw ticket. It takes a variable. Look at
  `SUMMARIZE_PROMPT` to see which one.
- **Level 2.** Between validate_or_repair and summarize you need a small step that turns the
  ticket dict into `{"ticket": <json string>}`. A `RunnableLambda` lifts a plain function
  into the pipe.
- **Level 3.** `summarize_chain = SUMMARIZE_PROMPT | model | brief_parser`, then
  `pipeline = extract_chain | RunnableLambda(validate_or_repair) | RunnableLambda(lambda t:
  {"ticket": json.dumps(t)}) | summarize_chain`.

## TODO 5 - parallel enrichment

- **Level 1.** Two chains, run together, results collected under two keys.
- **Level 2.** `RunnableParallel(title=..., risk=...)` returns a dict with those keys when you
  invoke it. Build each branch as its own small chain first.
- **Level 3.** `title_chain = TITLE_PROMPT | model | StrOutputParser()`,
  `risk_chain = RISK_PROMPT | model | risk_parser`,
  `enrich_parallel = RunnableParallel(title=title_chain, risk=risk_chain)`. In `enrich`,
  invoke it with `{"ticket": json.dumps(ticket)}` and merge the branch results back onto the
  ticket.

## TODO 6 - resilience

- **Level 1.** Runnables have a method that returns a retrying version of themselves.
- **Level 2.** It takes a keyword argument for the maximum number of attempts.
- **Level 3.** `robust_extract = extract_chain.with_retry(stop_after_attempt=3)`.

---

## Stretch A - batch the set

- **Level 1.** You built `pipeline` and `enrich` already. Apply them across all records.
- **Level 2.** `pipeline.batch(SUPPORT_EMAILS)` gives all briefs at once. For enrichment,
  extract and repair each record, then enrich.
- **Level 3.** `all_briefs = pipeline.batch(SUPPORT_EMAILS)`; `all_enriched = [enrich(
  validate_or_repair(extract_chain.invoke({"email": r["email"]}))) for r in SUPPORT_EMAILS]`.

## Stretch B - deterministic tool

- **Level 1.** Some decisions do not need a model. A few keyword rules settle priority.
- **Level 2.** Write `priority_policy(text)` returning "low", "medium", or "high" from simple
  substring checks. Then extract, repair, and overwrite the priority field with the policy.
- **Level 3.** In `extract_with_policy`, run the extraction and repair, then set
  `ticket["priority"] = priority_policy(record["email"])` before returning.
