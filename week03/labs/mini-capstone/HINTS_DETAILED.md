# HINTS: Detailed

For students who want the hints to do more. Each task below gives the key line or
two and explains why it works. This does not include the whole finished file. If
you want to be nudged more gently, one level at a time, use `HINTS_PROGRESSIVE.md`
instead.

Read the task in `LAB.md`, try it, then come here if you are stuck.

---

## Part A

### A1: the state
Attach the reducer to `messages` with `Annotated`, and leave the other four fields
as plain typed keys:

```python
class TriageState(TypedDict):
    messages: Annotated[list, add_messages]
    category: str
    draft: str
    attempts: int
    outcome: str
```

**Why.** `Annotated[list, add_messages]` tells LangGraph to merge new messages by
appending rather than overwriting. Every other field uses the default overwrite
behavior, which is exactly what you want for a single current `category`, `draft`,
`attempts`, and `outcome`.

### A2: classify
```python
def classify(state: TriageState) -> dict:
    text = last_user_text(state["messages"])
    return {"category": classify_text(text)}
```

**Why.** The node reads the newest message text through the provided helper, hands
it to the provided classifier, and returns only the one key it sets. Returning a
partial update is the whole node contract.

---

## Part B

### B1: draft_reply
```python
def draft_reply(state: TriageState) -> dict:
    return {"draft": first_draft_for(state["category"]), "attempts": 0}
```

**Why.** The draft is looked up by category. Resetting `attempts` to 0 here means
the revise loop always starts its count from a known place.

### B2: escalate
```python
def escalate(state: TriageState) -> dict:
    return {
        "outcome": "routed to a human agent",
        "messages": [{"role": "assistant",
                      "content": "This ticket needs a specialist. I have routed it to a human agent."}],
    }
```

**Why.** Returning a one-item `messages` list does not replace the history. The
`add_messages` reducer appends it, so the customer-facing note is added after the
user's original message.

### B3: finalize
```python
def finalize(state: TriageState) -> dict:
    return {
        "outcome": "handled automatically",
        "messages": [{"role": "assistant", "content": state["draft"]}],
    }
```

**Why.** The final compliant draft becomes the assistant's reply, appended to the
conversation the same way.

### B4: route_by_category
```python
def route_by_category(state: TriageState) -> str:
    if state["category"] in AUTO_CATEGORIES:
        return "draft_reply"
    return "escalate"
```

**Why.** A router returns the name of the next node as a string. Anything not in
`AUTO_CATEGORIES`, including `refund` and `unknown`, defaults to a human. Defaulting
uncertainty to a person is the safe choice.

### B5: build_graph, branch portion
```python
builder = StateGraph(TriageState)
builder.add_node("classify", classify)
builder.add_node("draft_reply", draft_reply)
builder.add_node("revise", revise)
builder.add_node("escalate", escalate)
builder.add_node("finalize", finalize)

builder.add_edge(START, "classify")
builder.add_conditional_edges("classify", route_by_category, ["draft_reply", "escalate"])
builder.add_edge("escalate", END)
```

**Why.** Register every node once, enter at `classify`, then branch on the router.
The destination list must contain the two names the router can return.

---

## Part C

### C1: revise
```python
def revise(state: TriageState) -> dict:
    return {"draft": drop_last_sentence(state["draft"]), "attempts": state["attempts"] + 1}
```

**Why.** Each pass removes one sentence and records that it ran. The incrementing
`attempts` is what eventually satisfies the loop guard even if the draft never gets
short enough.

### C2: review_gate
```python
def review_gate(state: TriageState) -> str:
    over_limit = count_words(state["draft"]) > MAX_WORDS
    can_retry = state["attempts"] < MAX_ATTEMPTS
    if over_limit and can_retry:
        return "revise"
    return "finalize"
```

**Why.** Two conditions decide the loop. Keep revising only while the draft is too
long AND you have attempts left. The `can_retry` half is the guard that prevents an
infinite loop if a draft could never comply.

### C3: build_graph, cycle portion
```python
builder.add_conditional_edges("draft_reply", review_gate, ["revise", "finalize"])
builder.add_conditional_edges("revise", review_gate, ["revise", "finalize"])
builder.add_edge("finalize", END)
```

**Why.** The first draft and every revision both flow through the same gate. The
edge from `revise` back through the gate, which can send it to `revise` again, is
the cycle. `finalize` is the loop's only exit besides the attempt guard.

---

## Part D

### D1: pass the checkpointer through
Your `build_graph` already accepts `checkpointer`. Make sure the compile call uses
it:

```python
return builder.compile(checkpointer=checkpointer, interrupt_before=interrupt_before)
```

**Why.** The runner passes `InMemorySaver()` and a `thread_id`. With a checkpointer
compiled in, LangGraph saves state per thread, so the second call sees the first
call's messages. The `add_messages` reducer then appends the new turn, giving four
messages total.

---

## Part E

### E1: pass interrupt_before through
This is the same compile line as Part D. Confirm `interrupt_before` is threaded in:

```python
return builder.compile(checkpointer=checkpointer, interrupt_before=interrupt_before)
```

**Why.** With `interrupt_before=["escalate"]`, the graph runs up to `escalate` and
stops. `graph.get_state(config).next` reports `('escalate',)`, the pending node.
Calling `graph.invoke(None, config)` resumes from the saved checkpoint. Passing
`None` as the input is what tells LangGraph to continue rather than start fresh.
Interrupts only work because state is checkpointed, which is why Part E always uses
a checkpointer too.
