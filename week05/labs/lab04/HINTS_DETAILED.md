# Lab 04 Hints: Detailed Tier

Pick one hint tier per task, not both; reading both wastes time. This
file shows the working core of each task verbatim, with commentary on why
each line is there. It withholds the function shell, return assembly, and
glue, so finishing a task still means reading and understanding the code
rather than pasting a function. Stretch tasks are covered at the same
depth; fully assembled stretch solutions are instructor-only, released
after the lab.

---

## Part A: `do_hello_run`

The core is the run lifecycle with a snapshot taken at the right moment:

```python
run = wandb.init(
    project=PROJECT,
    name="hello-world",
    config={"purpose": "setup check"},
    tags=["setup"],
)
run.log({"setup/ok": 1.0})
```

`wandb.init` starts the recording and hands back the run object every
later call hangs off. The four arguments: `project` files the run where
every other run today goes, `name` overrides the auto-generated one with
something a human can scan, `config` records why this run exists even
though it is trivial, and `tags` let setup runs be filtered out of real
comparisons later. The `log` call carries a prefix even for a throwaway
metric, because the prefix habit is the point.

```python
{"id": run.id, "config": dict(run.config), "summary": dict(run.summary)}
```

This dict is built while the run is open, because the live run object is
the only view that works on all three backends; offline there is no
server to query afterward. `dict(...)` copies wandb's live wrapper
objects into inert plain dicts that remain valid after the run closes.

You write: the finish call, the return, and the ordering that puts the
snapshot before both.

---

## Part B1: `make_config`

The complete key set, grouped the way the slides group it:

```python
# generation
"base_model": BASE_MODEL,
"adapter": ADAPTER_REF,
"adapter_active": adapter_active,
"temperature": TEMPERATURE,
"max_new_tokens": MAX_NEW_TOKENS,
# retrieval
"embedding_model": ls.EMBEDDING_MODEL_NAME,
"chunk_size": chunk_size,
"chunk_overlap": CHUNK_OVERLAP,
"top_k": top_k,
# evaluation
"eval_set": ls.EVAL_SET_VERSION,
"judge_model": JUDGE_MODEL,
```

Three values come from the function arguments because they are the three
knobs this lab turns; everything else comes from constants and goes into
the record anyway, because constant today is not constant next month and
the config is what proves it. `eval_set` and `judge_model` are the two
keys most configs omit, and they are exactly the omissions Formative
Check 1 punished: without them the number cannot be compared to any
future number.

You write: the function shell and the return that wraps this literal.

---

## Part B2: `log_eval_metrics`

```python
run.log(
    {
        "retrieval/context_recall": aggregate["context_recall"],
        "retrieval/context_precision": aggregate["context_precision"],
        "generation/faithfulness": aggregate["faithfulness"],
        "generation/answer_relevancy": aggregate["answer_relevancy"],
        "generation/abstention_rate": aggregate["abstention_rate"],
    }
)
```

One call, five metrics, one step. The prefixes are Module 03's stage
attribution as a naming convention: retrieval owns what was fetched,
generation owns what was written, and when a number moves later its name
already says which half of the pipeline to blame. Logging all five
together keeps them on the same step, which is what makes them line up
on any chart.

```python
run.summary["headline/faithfulness"] = aggregate["faithfulness"]
```

`log` writes the time series; `summary` holds the one number a
comparison table sorts on. wandb copies last-logged values into the
summary automatically, but "last logged" is an accident and "headline"
is a decision, so the decision is made explicit by item assignment.

You write: the function shell around these two statements.

---

## Part C1 and C2: the artifact builders

The adapter core:

```python
art = wandb.Artifact(
    name="cordwell-adapter",
    type="model",
    metadata={"lora_r": 16, "lora_alpha": 32, "base_model": BASE_MODEL},
)
art.add_dir(adapter_path)
```

`name` is the artifact family; every future adapter logs under it and
wandb assigns v0, v1, v2 as contents change. `type` is the coarse UI
grouping: things you ship are `model`. `metadata` is the label on the
box, the facts a comparison table can show without downloading anything,
which is why the LoRA rank lives here and not only inside the config
json in the directory. `add_dir` stages and checksums every file, which
is what makes versions content-addressed: identical bytes logged again
cost nothing.

The eval set core differs in three deliberate places:

```python
type="dataset",
metadata={
    "semantic_version": ls.EVAL_SET_VERSION,
    "num_queries": len(ls.EVAL_QUERIES),
},
...
art.add_file(eval_path)
```

Questions are data, so the type is `dataset`. The metadata carries the
semantic version, `cordwell-eval:v2`, which is a different number with a
different job than the artifact counter W&B will assign; one names the
curated dataset revision, the other names exact bytes. `add_file` stages
the single JSONL where the adapter needed `add_dir` for a directory.

You write: both function shells and returns. Neither builder calls
`log_artifact`; producing the record of *which run* logged the bytes
requires a run, and the orchestration cell owns the run.

---

## Part C3: `record_input_artifacts`

```python
if BACKEND == "offline":
    run.config.update(
        {
            "input_adapter_artifact": "cordwell-adapter:v0",
            "input_eval_artifact": "cordwell-eval:v0",
        }
    )
```

The branch exists because of a verified property of the tool, not a lab
preference: `run.use_artifact("cordwell-adapter:v0")` raises `TypeError:
Cannot use artifact when in offline mode`, since resolving a version
name to bytes requires asking a server which bytes those are. Offline,
the same information goes to the one durable place available, the
config. Adding new keys after init is allowed; changing existing ones is
not, and these are additions.

```python
run.use_artifact("cordwell-adapter:v0")
run.use_artifact("cordwell-eval:v0")
```

On a server backend, these two calls draw the consumer edges of the
lineage graph, pairing with the producer edges `log_artifact` drew. The
pair is what makes "the score dropped, what changed?" a two-minute walk
backward through the graph.

You write: the function shell, the two return strings in their branches
(`"config-lineage"` and `"use_artifact"`), and the branch structure that
puts each path where it belongs.

---

## Part D1: `tracked_eval`

The core sequence, which is Parts B and C composed in the order that
makes the record honest:

```python
tags = ["rag-eval", "part-d"] + list(extra_tags or [])
run = wandb.init(
    project=PROJECT,
    name=run_name,
    config=make_config(adapter_active, top_k, chunk_size),
    tags=tags,
)
record_input_artifacts(run)
results = ls.run_rag_eval(
    adapter_active=adapter_active, top_k=top_k, chunk_size=chunk_size
)
log_eval_metrics(run, results["aggregate"])
```

The record opens before the work runs, with the complete config: log the
config before you run, not after you get a number you like. The inputs
are declared immediately after. The eval itself is untouched Module 03
plumbing, and the same three parameters feed both `make_config` and
`run_rag_eval` from the same arguments, so the record and the reality
cannot drift. The tags line builds a fresh list per call; `extra_tags or
[]` turns the `None` default into an empty list without the mutable
default argument trap.

You write: the function shell, the six-key snapshot dict (name, id,
config, summary, aggregate, per_query, with the dict-copy discipline
from Part A), the finish call, and the return. Do not drop `per_query`
from the snapshot; Part F consumes it.

---

## Part D2: `run_experiment`

The three calls, which are the experimental design itself:

```python
tracked_eval("adapter-off-topk5", adapter_active=False, top_k=5, chunk_size=384)
tracked_eval("adapter-on-topk5", adapter_active=True, top_k=5, chunk_size=384)
tracked_eval("adapter-on-topk1", adapter_active=True, top_k=1, chunk_size=384)
```

Rows one and two differ only in `adapter_active`; rows two and three
differ only in `top_k`; chunk size is pinned throughout. Every pair you
will compare has exactly one suspect, and checks D3 and D4 grade the
design, not the metrics. The names encode the config so the comparison
table reads honestly without opening anything.

You write: the function shell and the list assembly, in this order.

---

## Part E1: `build_comparison`

The loop body:

```python
agg = snap["aggregate"]
rows.append(
    {
        "run": snap["name"],
        "adapter_active": snap["config"]["adapter_active"],
        "top_k": snap["config"]["top_k"],
        "chunk_size": snap["config"]["chunk_size"],
        "faithfulness": agg["faithfulness"],
        "context_recall": agg["context_recall"],
        "context_precision": agg["context_precision"],
        "answer_relevancy": agg["answer_relevancy"],
        "abstention_rate": agg["abstention_rate"],
    }
)
```

The knobs are read from the config, not typed from memory, so the table
is built from the record and cannot disagree with it. This frame is the
UI comparison view in programmatic form, and it is the same shape a
Week 7 CI gate would consume to fail a build on a regression.

You write: the function shell, the `rows` list, the loop, and the
`pd.DataFrame(rows)` return.

---

## Part E2: `RECOMMENDATION`

The evidence fields, read off your own comparison table:

```python
"promote": "adapter-on-topk5",
"compared_to": "adapter-off-topk5",
"variable_changed": "adapter_active",
"evidence_metric": "generation/faithfulness",
"delta": 0.5,
```

Promote against one named baseline differing in one named key; that is
what makes the delta attributable. The metric keeps its prefix because
the prefix is part of its identity. The delta is arithmetic from the
table, 1.0 minus 0.5, and it clears the pre-stated 0.1 threshold by a
factor of five. The abstention rise in the promoted run is not a
counterargument: `correct_abstention` went to 1.0, meaning the increase
is the two unanswerable questions being correctly refused.

You write: the rejection fields (`rejected_run`, `rejected_because`),
which come from the other one-variable pair: name the run, name the
retrieval metric it damaged and by how much. Then the prose writeup in
the markdown cell, three to five sentences in the claim, comparison,
evidence format.

---

## Part F1: `build_per_query_table`

```python
table = wandb.Table(columns=TABLE_COLUMNS)
for r in per_query:
    table.add_data(
        r["query_id"],
        r["question"],
        r["answer"],
        ", ".join(r["sources"]),
        r["faithfulness"],
        r["answer_relevancy"],
        r["abstained"],
    )
```

Columns declared first, then positional `add_data` in exactly that
order; a list of dicts is not a valid data argument and fails with an
unhelpful error, which is the trap in older examples the slides called
out. `sources` is the one transformed value: cells want scalars, so the
list becomes a comma-joined string.

You write: the function shell and the return.

---

## Part F2: `find_worst_queries`

```python
ranked = sorted(per_query, key=lambda r: (r["answer_relevancy"], r["query_id"]))
```

Ascending sort on the metric is the code form of clicking a column
header in the UI. The tuple key is the deliberate detail: ties on
relevancy break alphabetically on the query id, so every machine ranks
identically and the check can assert exact ids. This baseline run has no
abstentions, so every relevancy is a real number and no `None` handling
is needed.

You write: the function shell and the comprehension that maps the first
`n` ranked rows to their `query_id` values.

---

## Part G1 (stretch): `run_manual_sweep`

The loop core:

```python
for chunk_size in [256, 384, 512]:
    for top_k in [1, 3, 5]:
        snap = tracked_eval(
            f"sweep-cs{chunk_size}-k{top_k}",
            adapter_active=True,
            top_k=top_k,
            chunk_size=chunk_size,
            extra_tags=["sweep"],
        )
```

Nine combinations, nine calls to the Part D function, which is the
lesson: a sweep is the controlled experiment in a loop, not new
machinery. The name encodes the coordinates; the `sweep` tag isolates
the nine runs in any filter. This grid varies only retrieval
parameters, so it costs nine two-second evaluations rather than nine
fine-tunes, which is why it is feasible on this hardware at all.

You write: the function shell, the row dict per iteration (run name, the
two knobs, and `context_recall`, `context_precision`,
`abstention_rate` from `snap["aggregate"]`), and the DataFrame return.

---

## Part G2 (stretch): `pick_best_config`

```python
best = sweep_df.sort_values(
    by=["context_recall", "top_k", "context_precision"],
    ascending=[False, True, False],
).iloc[0]
```

The whole three-step rule as one sort: recall descending because it is
the swept metric, then among the recall ties the smallest `top_k`
because fewer chunks per query is cheaper for the same recall, then
precision descending to break what remains. `.iloc[0]` is the winner.
The rule was stated before looking, which is what separates a decision
from a rationalization.

You write: the function shell and the return dict, remembering that
pandas hands back numpy scalars and the check compares plain Python
ints, so cast with `int(...)`.
