# LAB: The Cordwell Triage Agent (LangGraph)

You are building a small but real support agent as a **stateful graph**. It
classifies an incoming support ticket for the fictional retailer **Cordwell Home
and Hardware**, either drafts a reply or routes the ticket to a human, revises an
over-long draft in a loop, remembers the conversation across turns, and can pause
for human approval before it escalates.

This is a Python application, not a notebook. You run it and test it from a
terminal, the way you would ship it.

---

## What is in this packet

| File | What it is | Do you edit it |
|---|---|---|
| `cordwell_triage/agent.py` | The LangGraph state machine. **This is your work.** | **Yes** |
| `cordwell_triage/helpers.py` | Provided deterministic logic (classifier, drafts, string tools). | No |
| `cordwell_triage/app.py` | Provided runner that exercises every part. | No |
| `tests/test_agent.py` | Provided tests. Run them to check your work. | No |
| `LAB.md` | The assignment. Parts A through E, with expected output for every task. | No |
| `HINTS_DETAILED.md` | Generous hints: the key lines plus full explanation. | No |
| `HINTS_PROGRESSIVE.md` | Three escalating hint levels per task, for a bigger challenge. | No |
| `requirements.txt` | Pinned dependencies. | No |

You only ever edit `agent.py`. If you find yourself wanting to change a helper or
a test, re-read the task first. The lab is designed so the answer always lives in
`agent.py`.

---

## Setup

From the packet root:

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Confirm the environment:

```bash
python -c "import langgraph; from importlib.metadata import version; print(version('langgraph'))"
# expect: 1.2.9  (any langgraph 1.x is fine for this lab)
```

## How to run

Check your work at any time:

```bash
pytest -q                 # run all tests
pytest -q -k classify     # run just the classify tests
```

See the whole agent run once you have built it:

```bash
python -m cordwell_triage.app
```

At the start, every test fails. That is expected. Each Part you complete turns
more of them green. When all 19 pass, the app runs top to bottom.

---

## The graph you are building

```
                 +------------+
        START -> |  classify  |
                 +------------+
                    |      \
      route_by_category     route_by_category
                    |         \
             (auto ticket)   (refund / unknown)
                    v             v
             +------------+   +----------+
             | draft_reply|   | escalate |----> END
             +------------+   +----------+
                    |
              review_gate  <-------------------+
                /       \                       |
          (over limit)  (ok / gave up)          |
              v              v                   |
          +--------+    +----------+             |
          | revise |    | finalize |----> END    |
          +--------+                             |
              |                                  |
              +--- review_gate (loop) ----------+
```

Two decision points drive everything. `route_by_category` picks the branch after
classification. `review_gate` runs after `draft_reply` and after each `revise`,
looping until the draft is short enough or the attempt budget runs out.

---

## New constructs explained

Everything below appears in the module slides. This is your quick reference while
you code. Each item names the one idea and shows the shape.

### TypedDict state
`State` is the shared, typed object that flows through the graph. It is a plain
`TypedDict`: Python type hints, nothing more. Every node reads from it, and every
node returns only the keys it changed.

```python
from typing_extensions import TypedDict

class State(TypedDict):
    question: str
    answer: str
```

### Reducers (the add_messages reducer)
By default, a key a node returns **overwrites** the old value. A **reducer**
changes how the update merges. The built-in `add_messages` reducer **appends**,
so conversation history accumulates instead of getting replaced. You attach a
reducer with `Annotated`:

```python
from typing import Annotated
from langgraph.graph.message import add_messages

class State(TypedDict):
    messages: Annotated[list, add_messages]   # appends, does not overwrite
```

This is the single most common "why is my state wrong" surprise. Overwrite versus
append. In this lab, `messages` uses the reducer and every other field uses the
default overwrite behavior.

### Nodes are plain functions
A node takes the state in and returns a dict of updates. No framework magic. You
can call it directly and test it like any function.

```python
def answer_node(state: State) -> dict:
    return {"answer": f"You asked: {state['question']}"}
```

Return **only the keys you change**. The graph merges your partial update into the
shared state.

### Edges, START, and END
Edges wire nodes together. `START` and `END` are the entry and exit points, not
nodes you write. A plain edge always goes to the same next node.

```python
from langgraph.graph import StateGraph, START, END

builder = StateGraph(State)
builder.add_node("answer", answer_node)
builder.add_edge(START, "answer")
builder.add_edge("answer", END)
graph = builder.compile()
```

### Conditional edges and routers
A **conditional edge** picks the next node at runtime. The **router** is a plain
function that returns the **name of the next node as a string**.

```python
def route(state: State) -> str:
    return "escalate" if state["needs_human"] else "auto_reply"

builder.add_conditional_edges("triage", route, ["escalate", "auto_reply"])
```

The third argument lists the possible destinations. The router must return one of
those exact names. Returning a name that is not a node raises `KeyError`.

### Cycles and the attempt guard
An edge, including a conditional one, can point back to an earlier node. That is a
**cycle**, the loop a linear chain cannot express. Because a graph can loop, you
are responsible for a termination condition. In this lab the guard is an
`attempts` counter checked against `MAX_ATTEMPTS`, so `revise` can never loop
forever.

### invoke and stream
`graph.invoke(input)` runs the graph and returns the **full final state** as a
dict (not just the keys the last node touched). `graph.stream(input,
stream_mode="updates")` yields one entry per node as it runs, keyed by node name.
Streaming is how you watch which path a run actually took.

### Checkpointer and thread_id (persistence)
Compile with a **checkpointer** and every step is saved automatically. Pass a
`thread_id` in the config and the graph resumes that exact conversation on the
next call. This is memory without a separate store.

```python
from langgraph.checkpoint.memory import InMemorySaver

graph = builder.compile(checkpointer=InMemorySaver())
config = {"configurable": {"thread_id": "cust-42"}}
graph.invoke({"messages": [{"role": "user", "content": "first"}]}, config)
graph.invoke({"messages": [{"role": "user", "content": "second"}]}, config)
# with the add_messages reducer, state now holds BOTH turns
```

`InMemorySaver` keeps state in process and loses it on restart, which is fine for
this lab. File-backed and database-backed savers exist for production.

### interrupt_before (human in the loop)
Compile with `interrupt_before=["some_node"]` and the graph **pauses** right
before that node runs. You inspect the paused state, and when a human approves you
resume. Pausing only works because state is checkpointed, so `interrupt_before`
requires a checkpointer.

```python
graph = builder.compile(checkpointer=InMemorySaver(), interrupt_before=["escalate"])
graph.invoke(input, config)             # runs up to the pause and stops
graph.get_state(config).next            # -> ('escalate',)  the pending node
graph.invoke(None, config)              # resume: passing None means "continue"
```

---

## Responsible AI, built in

Two things you build are also governance features, not extras.

- **The escalate branch is an approval path.** Uncertain or refund tickets go to a
  human rather than being auto-answered. In Part E you turn that into a hard pause
  with `interrupt_before`, so a person signs off before the handoff.
- **The checkpointer is an audit trail.** Every step the agent takes is saved, so
  after the fact you can see exactly what it read, decided, and did.

---

## Stuck?

Open `HINTS_DETAILED.md` if you want the key lines explained. Open
`HINTS_PROGRESSIVE.md` if you want to be nudged one level at a time. A capable
engineer should be able to keep moving on the progressive level 2 hint without
opening the detailed set. Reach for your instructor before you reach for anyone
else's `agent.py`.
