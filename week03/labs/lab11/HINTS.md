# Week 3, Lab 11: HINTS

Use these only after you have tried the contract in the docstring. Each task has three levels. Level 1 is a nudge, Level 2 is the approach, Level 3 is almost the answer. Stop at the earliest level that unblocks you.

---

## TODO 1: as_json

- **Level 1.** The reply is sometimes wrapped in a fenced block. Strip the fence before you parse.
- **Level 2.** Check whether the stripped text starts with three backticks. If so, remove the opening fence (which may be three backticks or three backticks plus json) and the closing fence, then parse. Wrap the parse in try and except and return an empty dict on failure.
- **Level 3.** Use two regular expressions: one that removes a leading fence, one that removes a trailing fence. Then `json.loads`. Any exception returns `{}`. Never let this function raise.

## TODO 2: schema_errors

- **Level 1.** The library already knows how to find every violation. You just format them.
- **Level 2.** Build a `Draft202012Validator(LEAD_SCHEMA)`. It has a method that yields one error object per violation without raising.
- **Level 3.** Iterate `validator.iter_errors(obj)`. For each error, join its `path` and its `message` into a readable string. Return the list. An empty list means the object is valid.

## TODO 3 to 5: extract_step, repair_step, summarize_step

- **Level 1.** Each step is the same shape: format a prompt, invoke MODEL, parse or strip the reply, return an updated dict. Repair is the only conditional one.
- **Level 2.** For extract, format `EXTRACT_TPL` with the schema, id, and email, then parse with `as_json`. For repair, call `schema_errors` first. If empty, return early with `repaired` False. Otherwise call the model with the errors and the original, parse, and mark `repaired` True. For summarize, format `SUMMARY_TPL` with the lead and strip the reply into `md`.
- **Level 3.**
  - extract: `msg = EXTRACT_TPL.format_messages(schema=json.dumps(LEAD_SCHEMA), id=rec["id"], email=rec["email"])`, then `return {"rec": rec, "json": as_json(MODEL.invoke(msg).content)}`.
  - repair: compute `errs = schema_errors(state["json"])`. If not errs, `return {**state, "errors": [], "repaired": False}`. Else format `REPAIR_TPL` with `errors=json.dumps(errs)` and `original=json.dumps(state["json"])` and `id=state["rec"]["id"]`, then `return {**state, "json": as_json(MODEL.invoke(msg).content), "errors": errs, "repaired": True}`.
  - summarize: `md = MODEL.invoke(SUMMARY_TPL.format_messages(lead=json.dumps(state["json"]))).content.strip()`, then `return {**state, "md": md}`.

## TODO 6: build_lead_chain

- **Level 1.** LCEL uses one operator to connect runnables.
- **Level 2.** Wrap each step function in `RunnableLambda` and connect them with the pipe operator, in order. Return the result. Do not invoke it.
- **Level 3.** `return RunnableLambda(extract_step) | RunnableLambda(repair_step) | RunnableLambda(summarize_step)`.

## TODO 7: BufferMemory

- **Level 1.** A list of messages, three tiny methods.
- **Level 2.** `load` returns a copy of the list. `save` appends a HumanMessage then an AIMessage. `reset` sets the list back to empty.
- **Level 3.** In `__init__` set `self.messages = []`. `save` does `self.messages += [HumanMessage(content=user), AIMessage(content=ai)]`. `load` returns `list(self.messages)`.

## TODO 8: SummaryMemory

- **Level 1.** Keep the last few messages as they are. Everything older becomes one condensed summary produced by the model.
- **Level 2.** On `save`, append the pair. If the number of messages now exceeds `keep_last`, take the messages beyond the last `keep_last`, feed the previous summary plus those evicted messages to `CONDENSE_TPL` and the model, store the result as the new summary, and keep only the last `keep_last` messages. `load` prepends a SystemMessage holding the summary when one exists.
- **Level 3.** The eviction slice is `self.messages[:-self.keep_last]`. The blob is `self.summary + " " + " ".join(m.content for m in evicted)`. Set `self.summary = self.model.invoke(CONDENSE_TPL.format_messages(blob=blob)).content.strip()` and then `self.messages = self.messages[-self.keep_last:]`. In `load`, `head = [SystemMessage(content=self.summary)] if self.summary else []` and return `head + list(self.messages)`.

## TODO 9: run_session

- **Level 1.** Loop the user turns, let the model answer, save each pair. Then ask the final question and measure that one call.
- **Level 2.** The user turns are the even indices of `ONBOARDING["turns"]`. Reset the memory first. For each user turn, format `CHAT_TPL` with `history=memory.load()` and `input=<user text>`, invoke, and save. Honor `reset_before_final`. Time only the final call. Sum approximate tokens over the loaded history, the final question, and the reply.
- **Level 3.** `for i in range(0, len(turns), 2):` pulls the user turns. After the loop, optionally reset, then `history = memory.load()`. Wrap the final invoke in `time.perf_counter()` before and after for `latency`. Tokens: `approx_tokens(" ".join(m.content for m in history)) + approx_tokens(ONBOARDING["final_q"]) + approx_tokens(final)`. Return the four keys `final`, `loaded_msgs` (which is `len(history)`), `approx_tokens`, `latency`.

## TODO 10: the reset control

- **Level 1.** There is no new function here. `run_session` already accepts `reset_before_final`. This cell just runs it with the reset engaged and lets you watch recall disappear.
- **Level 2.** If your `run_session` calls `memory.reset()` just before loading the history for the final question when `reset_before_final` is True, this cell works with no changes.
- **Level 3.** Expect `loaded_msgs` of 0 and the reply "I do not have that in memory yet." If you still see recall, your reset branch is running at the wrong time, for example after the history was already loaded.

---

## Stretch 1: prenormalize

- **Level 1.** Facts a regex can nail should not be left to the model. Phone and seat counts are regex facts.
- **Level 2.** One pattern for a phone number. One `findall` for pairs of a number followed by a role word (buyer, editor, viewer, user, account) then the word seat. Case-insensitive checks for SOC 2, SAML, and Okta feed a security list. Only include keys you actually found.
- **Level 3.** Phone: `re.search(r"(\+?\d[\d\-\s]{7,}\d)", email)`. Seats: `re.findall(r"(\d+)\s+(buyer|editor|viewer|user|account)s?", email, re.I)` then build a dict keyed by the lowercased role. Guard each security token with its own `re.search` and use `out.setdefault("security", []).append(...)`.

## Stretch 2: WindowMemory

- **Level 1.** Same three methods as BufferMemory, but load only returns a tail.
- **Level 2.** Keep everything in a list on save. On load, return the last `2 * k` messages.
- **Level 3.** `load` returns `self.messages[-2 * self.k:]`. Everything else matches BufferMemory. When you run the check, notice it expects the SSO fact to be gone. The SSO turn is older than the window, so it is dropped. That is the lesson.
