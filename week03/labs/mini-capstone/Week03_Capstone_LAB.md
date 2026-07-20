# LAB: Build the Cordwell Triage Agent

**Module 11 · LangGraph Fundamentals · target time 120 to 150 minutes**

You will build a support triage agent as a LangGraph state machine, one primitive
at a time, in `cordwell_triage/agent.py`. Read `README.md` first if you have not.

Every task below shows its **expected output**, so you always know the target
before you write code. You are coding toward a visible result, not guessing the
shape from a failing test.

## How to work

1. Edit only `cordwell_triage/agent.py`.
2. After each task, run the matching tests. The command is listed with each Part.
3. When all 19 tests pass, run `python -m cordwell_triage.app` to see it end to end.

## Time budget

| Part | Focus | Minutes |
|---|---|---|
| A | State and the first node | 20 to 25 |
| B | Nodes and the conditional branch | 30 to 35 |
| C | The revise cycle | 30 to 35 |
| D | Persistent memory | 15 to 20 |
| E | Human in the loop | 15 to 20 |
| Stretch | Optional extensions | remaining time |

## Definition of done

`pytest -q` reports **19 passed**, and `python -m cordwell_triage.app` runs from
top to bottom and prints the four Part banners with the output shown at the end of
this file.

---

## Part A: State and the classify node (20 to 25 min)

**Goal.** Declare the shared state, then write the node that classifies a ticket.

**Task A1 (TODO in `TriageState`).** Declare five fields. `messages` must use the
`add_messages` reducer so turns accumulate. The rest are plain typed fields:
`category: str`, `draft: str`, `attempts: int`, `outcome: str`.

**Task A2 (TODO in `classify`).** Read the newest message with
`last_user_text(state["messages"])`, classify it with `classify_text(...)`, and
return `{"category": <result>}`.

**Worked target output.**

```python
>>> from cordwell_triage.helpers import classify_text
>>> classify_text("How do I reset a tripped breaker?")
'how_to'
>>> classify_text("I want a refund for a broken drill.")
'refund'
>>> classify_text("Where is my order?")
'order_status'
>>> classify_text("Is the DeWalt drill in stock?")
'product_info'
>>> classify_text("my dog is purple")
'unknown'
```

Your `classify` node wraps that lookup and returns a partial update:

```python
>>> classify({"messages": [{"role": "user", "content": "How do I reset a tripped breaker?"}]})
{'category': 'how_to'}
```

**Check it.** `pytest -q -k "state or classify"` (3 tests).

---

## Part B: Nodes and the conditional branch (30 to 35 min)

**Goal.** Write the remaining action nodes, the category router, and wire the
branch into `build_graph`.

**Task B1 (`draft_reply`).** Return the canned draft for the category using
`first_draft_for(state["category"])`, and reset `attempts` to 0.

**Task B2 (`escalate`).** Return `outcome` of `"routed to a human agent"` and
append one assistant message announcing the handoff.

**Task B3 (`finalize`).** Return `outcome` of `"handled automatically"` and append
one assistant message whose content is `state["draft"]`.

**Task B4 (`route_by_category`).** Return `"draft_reply"` when the category is in
`AUTO_CATEGORIES`, otherwise `"escalate"`.

**Task B5 (`build_graph`, branch portion).** Add all five nodes, wire
`START -> classify`, and add the conditional edge from `classify` using
`route_by_category` with destinations `["draft_reply", "escalate"]`. Wire
`escalate -> END`. (The `draft_reply` path is completed in Part C.) Compile and
return the graph.

**Worked target output.**

```python
>>> draft_reply({"category": "how_to"})["attempts"]
0
>>> route_by_category({"category": "how_to"})
'draft_reply'
>>> route_by_category({"category": "refund"})
'escalate'
>>> route_by_category({"category": "unknown"})
'escalate'
>>> escalate({})["outcome"]
'routed to a human agent'
>>> finalize({"draft": "hello"})["outcome"]
'handled automatically'
```

Once the branch is wired, a refund ticket runs straight to a human:

```python
>>> graph = build_graph()
>>> graph.invoke({"messages": [{"role": "user", "content": "I want a refund for a broken drill."}]})["outcome"]
'routed to a human agent'
```

**Check it.** `pytest -q -k "draft or route or escalate or finalize or refund"`.

---

## Part C: The revise cycle (30 to 35 min)

**Goal.** Add the loop that shortens an over-long draft until it complies or the
attempt budget runs out.

**Task C1 (`revise`).** Return a shorter `draft` using
`drop_last_sentence(state["draft"])` and `attempts` incremented by one.

**Task C2 (`review_gate`).** Return `"revise"` when the draft is over `MAX_WORDS`
words AND `attempts` is still below `MAX_ATTEMPTS`. Otherwise return `"finalize"`.
Use `count_words(...)`.

**Task C3 (`build_graph`, cycle portion).** Add a conditional edge from
`draft_reply` using `review_gate` with destinations `["revise", "finalize"]`. Add
**the same** conditional edge from `revise`. Wire `finalize -> END`.

**Why revise routes through the same gate:** the gate is the single place that
decides "still too long, or done." Both the first draft and every revision ask it
the same question. The edge from `revise` back through the gate is the cycle.

**Worked target output.** With `MAX_WORDS = 20`, a `how_to` ticket drafts at 46
words and is trimmed one sentence per pass:

```python
>>> graph = build_graph()
>>> for step in graph.stream(
...     {"messages": [{"role": "user", "content": "How do I reset a tripped breaker?"}]},
...     stream_mode="updates"):
...     for node, update in step.items():
...         print(node, "->", update)
classify -> {'category': 'how_to'}
draft_reply -> {'draft': '... 46 words ...', 'attempts': 0}
revise -> {'draft': '... 30 words ...', 'attempts': 1}
revise -> {'draft': '... 20 words ...', 'attempts': 2}
finalize -> {'outcome': 'handled automatically', 'messages': [...]}
```

The word count falls 46 then 30 then 20, `attempts` climbs 0 then 1 then 2, and
the gate finalizes as soon as the draft is at or under 20 words. The final state:

```python
>>> result = graph.invoke({"messages": [{"role": "user", "content": "How do I reset a tripped breaker?"}]})
>>> result["outcome"], result["attempts"]
('handled automatically', 2)
>>> len(result["draft"].split()) <= 20
True
```

**Check it.** `pytest -q -k "revise or review or loop or auto_ticket"`.

---

## Part D: Persistent memory (15 to 20 min)

**Goal.** Prove the agent remembers across separate calls on the same thread.

**Task D1.** Confirm your `build_graph` passes `checkpointer` straight into
`compile(...)`. No new nodes. The runner supplies `InMemorySaver()` and a
`thread_id`.

**Worked target output.** Two calls on one thread, and `messages` holds both
exchanges because of the `add_messages` reducer:

```python
>>> from langgraph.checkpoint.memory import InMemorySaver
>>> graph = build_graph(checkpointer=InMemorySaver())
>>> config = {"configurable": {"thread_id": "cust-42"}}
>>> graph.invoke({"messages": [{"role": "user", "content": "How do I reset a tripped breaker?"}]}, config)
>>> final = graph.invoke({"messages": [{"role": "user", "content": "Where is my order?"}]}, config)
>>> len(final["messages"])
4
```

Four messages: two user turns and two assistant turns, in order. Without the
reducer you would see 2, because the second turn would overwrite the first.

**Check it.** `pytest -q -k memory`.

---

## Part E: Human in the loop (15 to 20 min)

**Goal.** Pause before the agent hands a ticket to a human, so a person can sign
off first.

**Task E1.** Confirm your `build_graph` passes `interrupt_before` straight into
`compile(...)`. No new nodes. The runner compiles with
`interrupt_before=["escalate"]` and a checkpointer.

**Worked target output.** A refund ticket runs up to the pause and stops. The
pending node is visible, and resuming finishes the run:

```python
>>> graph = build_graph(checkpointer=InMemorySaver(), interrupt_before=["escalate"])
>>> config = {"configurable": {"thread_id": "cust-99"}}
>>> graph.invoke({"messages": [{"role": "user", "content": "I want a refund for a broken drill."}]}, config)
>>> graph.get_state(config).next
('escalate',)
>>> final = graph.invoke(None, config)     # None means "resume from the pause"
>>> final["outcome"]
'routed to a human agent'
```

**Check it.** `pytest -q -k interrupt`.

---

## Stretch goals (optional)

1. **A third category rule.** Add a `warranty` category to the classifier by
   extending `_CATEGORY_KEYWORDS` in a copy of `helpers.py` you own, then confirm
   your graph routes it to `draft_reply` or `escalate` as you intend. Which side
   of `AUTO_CATEGORIES` should it fall on, and why.
2. **Tighten the policy.** Set `MAX_WORDS` to 12 and predict how many revise
   passes a `how_to` ticket takes before you run it. Then run it and check.
3. **Edit at the pause.** In Part E, after the interrupt, use
   `graph.update_state(config, {...})` to change the pending state before you
   resume, and observe the effect. This models a human editing a draft during
   review.

---

## Full expected app output

When everything is complete, `python -m cordwell_triage.app` prints:

```
====================================================================
PART B - conditional branch
====================================================================

--- ticket: 'How do I reset a tripped breaker?' ---
category: how_to
attempts: 2
outcome : handled automatically
draft   : (20 words) Thanks for contacting Cordwell Home and Hardware. To reset a tripped breaker, switch it fully off and then back on.
messages:
   [human] How do I reset a tripped breaker?
   [ai] Thanks for contacting Cordwell Home and Hardware. To reset a tripped breaker, switch it fully off and then back on.

--- ticket: 'I want a refund for a broken drill.' ---
category: refund
attempts: None
outcome : routed to a human agent
messages:
   [human] I want a refund for a broken drill.
   [ai] This ticket needs a specialist. I have routed it to a human agent.

====================================================================
PART C - the revise cycle (watch attempts climb)
====================================================================
ticket: 'How do I reset a tripped breaker?'

   ran classify
   ran draft_reply  draft now 46 words
   ran revise       draft now 30 words
   ran revise       draft now 20 words
   ran finalize

====================================================================
PART D - persistence across turns (checkpointer + thread_id)
====================================================================
After two turns on the same thread, messages holds BOTH exchanges:
category: order_status
attempts: 2
outcome : handled automatically
draft   : (20 words) Thanks for contacting Cordwell Home and Hardware. You can track any order from the Orders page using your order number.
messages:
   [human] How do I reset a tripped breaker?
   [ai] Thanks for contacting Cordwell Home and Hardware. To reset a tripped breaker, switch it fully off and then back on.
   [human] Where is my order?
   [ai] Thanks for contacting Cordwell Home and Hardware. You can track any order from the Orders page using your order number.

====================================================================
PART E - human-in-the-loop interrupt before escalate
====================================================================
paused before: ('escalate',)   (graph is waiting for a human)

... human approves, resuming ...

resumed to completion:
category: refund
attempts: None
outcome : routed to a human agent
messages:
   [human] I want a refund for a broken drill.
   [ai] This ticket needs a specialist. I have routed it to a human agent.
```

Note the `attempts: None` on the refund path is expected. That path never enters
the draft or revise nodes, so `attempts` is never set.
