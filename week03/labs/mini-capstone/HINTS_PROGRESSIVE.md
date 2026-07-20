# HINTS: Progressive

For students who want a challenge. Each task has three levels. Read level 1 first,
try again, and only drop to the next level if you are still stuck. A capable
engineer should be able to keep moving on level 2 without opening level 3 or the
detailed hints.

- **Level 1** names the approach.
- **Level 2** sketches the structure.
- **Level 3** shows the key line in context, but never the whole function.

---

## Part A

### A1: the state
- **L1.** Four of the five fields are ordinary typed keys. Only `messages` needs
  special treatment so history accumulates.
- **L2.** Use `Annotated[..., ...]` on `messages` to attach a reducer. The reducer
  you want is the one imported at the top of the file. The other fields are just
  `name: type`.
- **L3.** The messages line looks like `messages: Annotated[list, add_messages]`.
  Add the four plain fields below it.

### A2: classify
- **L1.** Two provided helpers do the heavy lifting. You just connect them and
  return a partial update.
- **L2.** Get the latest message text with `last_user_text(...)`, then pass it to
  `classify_text(...)`. Return a dict with a single `category` key.
- **L3.** The body is two lines: `text = last_user_text(state["messages"])` then
  `return {"category": classify_text(text)}`.

---

## Part B

### B1: draft_reply
- **L1.** Look the draft up by category, and start the attempt counter.
- **L2.** Use `first_draft_for(state["category"])` for the draft, and set
  `attempts` to 0 in the same returned dict.
- **L3.** `return {"draft": first_draft_for(state["category"]), "attempts": 0}`.

### B2 and B3: escalate and finalize
- **L1.** Each returns an `outcome` string and appends one assistant message.
- **L2.** Appending means returning a `messages` list with one dict shaped like
  `{"role": "assistant", "content": ...}`. The reducer does the appending. For
  `escalate` the content is a handoff note. For `finalize` the content is the
  final `draft`.
- **L3.** `finalize` returns
  `{"outcome": "handled automatically", "messages": [{"role": "assistant", "content": state["draft"]}]}`.
  `escalate` is the same shape with outcome `"routed to a human agent"` and a
  handoff sentence as the content.

### B4: route_by_category
- **L1.** A router returns a node name as a string.
- **L2.** Check membership in `AUTO_CATEGORIES`. In-set goes to the draft path,
  everything else goes to a human.
- **L3.** `return "draft_reply" if state["category"] in AUTO_CATEGORIES else "escalate"`.

### B5: build_graph, branch portion
- **L1.** Register all five nodes, set the entry point, and add the classify
  branch.
- **L2.** `add_node` for each of the five, `add_edge(START, "classify")`, then
  `add_conditional_edges` from `classify`. Send `escalate` to `END`.
- **L3.** The branch line is
  `builder.add_conditional_edges("classify", route_by_category, ["draft_reply", "escalate"])`.

---

## Part C

### C1: revise
- **L1.** Shorten the draft and record that this pass happened.
- **L2.** Use `drop_last_sentence(state["draft"])` for the new draft, and add one
  to `attempts`.
- **L3.** `return {"draft": drop_last_sentence(state["draft"]), "attempts": state["attempts"] + 1}`.

### C2: review_gate
- **L1.** Two conditions decide the loop: is the draft still too long, and do you
  still have attempts left.
- **L2.** Compute both booleans. Return `"revise"` only when both are true.
  Otherwise return `"finalize"`. The second condition is your infinite-loop guard.
- **L3.** The condition is `if count_words(state["draft"]) > MAX_WORDS and state["attempts"] < MAX_ATTEMPTS: return "revise"`.

### C3: build_graph, cycle portion
- **L1.** The first draft and every revision ask the same gate the same question.
- **L2.** Add a conditional edge from `draft_reply` through `review_gate`, and the
  same one from `revise`. Then send `finalize` to `END`.
- **L3.** The looping edge is
  `builder.add_conditional_edges("revise", review_gate, ["revise", "finalize"])`.
  Notice `revise` is in its own destination list, which is what makes the cycle
  legal.

---

## Part D

### D1: persistence
- **L1.** You do not add nodes. Persistence is a compile option.
- **L2.** Your `build_graph` already receives a `checkpointer` argument. Make sure
  the compile call actually uses it.
- **L3.** `return builder.compile(checkpointer=checkpointer, interrupt_before=interrupt_before)`.

---

## Part E

### E1: interrupt
- **L1.** Also a compile option, not a node.
- **L2.** Thread the `interrupt_before` argument into the same compile call as the
  checkpointer. Interrupts require a checkpointer, which the runner supplies.
- **L3.** Same line as D1:
  `return builder.compile(checkpointer=checkpointer, interrupt_before=interrupt_before)`.
  The runner then calls `graph.invoke(None, config)` to resume after the pause.
