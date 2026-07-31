# Lab 04 Hints: Progressive Tier

Pick one hint tier per task, not both. This file escalates in three levels
per task: level 1 names the approach, level 2 sketches the structure,
level 3 shows a key line or two in context but never the whole solution.
If level 2 is not enough for a task, consider `HINTS_DETAILED.md` for that
task instead of level 3 here. The instructor solution notebook is released
after the lab.

---

## Part A: `do_hello_run`

**Level 1.** The docstring is the recipe, in order: init, log, snapshot,
finish, return. The only subtlety is that the snapshot happens while the
run is still open.

**Level 2.** Five statements. `wandb.init(...)` with the four keyword
arguments listed in the docstring, assigned to `run`. One `run.log` call
with a one-entry dict. Build the snapshot dict from `run.id`,
`dict(run.config)`, and `dict(run.summary)`. Then `run.finish()`, then
`return`.

**Level 3.** The snapshot line most people get wrong is the wrapping:

```python
"config": dict(run.config),
```

`run.config` is a live wandb object; `dict(...)` copies it into something
that survives after `run.finish()`.

---

## Part B1: `make_config`

**Level 1.** The whole function is one `return {...}` with eleven keys.
The docstring lists every key and where its value comes from: three from
the function arguments, eight from constants already defined in the
notebook or in `lab_support`.

**Level 2.** Group the literal with comments the way the slides did:
five generation keys, four retrieval keys, two evaluation keys. The three
argument-driven keys are `adapter_active`, `chunk_size`, and `top_k`.

**Level 3.** The two keys people forget, because they feel like they
belong to the harness rather than the experiment:

```python
"eval_set": ls.EVAL_SET_VERSION,
"judge_model": JUDGE_MODEL,
```

Formative Check 1 this morning was about exactly these.

---

## Part B2: `log_eval_metrics`

**Level 1.** Two statements: one `run.log` with a five-entry dict, then
one summary assignment. The metric names are given verbatim in the
docstring and the checks compare them exactly.

**Level 2.** Left side of each dict entry: the prefixed name, a string.
Right side: a lookup into the `aggregate` dict by its unprefixed key.
Recall and precision go under `retrieval/`; the other three go under
`generation/`.

**Level 3.** The summary line is an item assignment, not a method call:

```python
run.summary["headline/faithfulness"] = aggregate["faithfulness"]
```

---

## Part C1 and C2: the two artifact builders

**Level 1.** Each builder is three statements: construct
`wandb.Artifact(...)` with name, type, and metadata from the docstring;
stage the files; return the artifact. Do not call `log_artifact` here;
the orchestration cell does that inside a run.

**Level 2.** The adapter stages a whole directory, the eval set stages a
single file. One of `add_dir` and `add_file` is right for each; the
argument is the path the function received.

**Level 3.** The metadata is a plain dict passed at construction time:

```python
metadata={"lora_r": 16, "lora_alpha": 32, "base_model": BASE_MODEL},
```

---

## Part C3: `record_input_artifacts`

**Level 1.** One `if` on the global `BACKEND`. The docstring gives both
branches almost completely, including the two return strings. Your job
is mostly to understand why the branch exists: `use_artifact` needs a
server to resolve a version name to bytes.

**Level 2.** Offline branch: one `run.config.update({...})` with the two
input references, then return `"config-lineage"`. Server branch: two
`run.use_artifact(...)` calls with the version-suffixed names, then
return `"use_artifact"`.

**Level 3.** The offline branch's update call:

```python
run.config.update({"input_adapter_artifact": "cordwell-adapter:v0", ...})
```

Config keys can be added after init; they cannot be changed once set.
Adding lineage keys is an addition, so no special flags are needed.

---

## Part D1: `tracked_eval`

**Level 1.** Nothing here is new. It is Parts B and C called in the order
the docstring numbers them, with the results captured into the snapshot
dict before `finish`. The three knob arguments are used twice: once in
`make_config`, once in `ls.run_rag_eval`.

**Level 2.** Skeleton: build `tags` from the two fixed tags plus
`extra_tags`; `wandb.init` with project, name, config, tags;
`record_input_artifacts(run)`; call the eval; `log_eval_metrics`;
assemble the six-key snapshot; finish; return.

**Level 3.** The tags line, which trips people on the `None` default:

```python
tags = ["rag-eval", "part-d"] + list(extra_tags or [])
```

And remember the snapshot must include `"per_query"`; Part F needs it.

---

## Part D2: `run_experiment`

**Level 1.** Three calls to your `tracked_eval`, returned as a list, with
the exact names and settings from the Part D table. Chunk size is 384 in
all three.

**Level 2.** The design constraint to check before running: the first and
second calls must differ only in `adapter_active`; the second and third
only in `top_k`. Checks D3 and D4 grade exactly this.

**Level 3.** First call:

```python
tracked_eval("adapter-off-topk5", adapter_active=False, top_k=5, chunk_size=384),
```

---

## Part E1: `build_comparison`

**Level 1.** Loop over the snapshots, build one flat dict per run, wrap
the list in `pd.DataFrame`. Knobs come from `snap["config"]`, metrics
from `snap["aggregate"]`, the name from `snap["name"]`.

**Level 2.** Nine keys per row, named exactly as the docstring lists
them. Read the three knobs from the config even though you know their
values; the table should be built from the record, not from memory.

**Level 3.**

```python
"faithfulness": agg["faithfulness"],
```

where `agg = snap["aggregate"]` at the top of the loop body.

---

## Part E2: `RECOMMENDATION`

**Level 1.** Every field is readable off the comparison table plus the
Part D design. The promoted run beat which baseline, differing in which
single key, on which metric, by how much, against the pre-stated 0.1?
And which run is rejected, on the evidence of which metric drop?

**Level 2.** `delta` is promoted value minus baseline value for the
evidence metric, as a number from the table, not a guess. The rejected
run is the one whose single changed variable damaged a retrieval metric.

**Level 3.** The evidence metric is named with its prefix:

```python
"evidence_metric": "generation/faithfulness",
```

---

## Part F1: `build_per_query_table`

**Level 1.** Construct the table with `columns=TABLE_COLUMNS`, then loop
over `per_query` calling `add_data` once per row with the seven values in
column order. The only transformation is joining `sources` into a string.

**Level 2.** `add_data` takes positional arguments matching the column
order exactly. Six of the seven come straight out of the row dict; the
seventh is `", ".join(r["sources"])`.

**Level 3.**

```python
table = wandb.Table(columns=TABLE_COLUMNS)
```

A list of dicts is not a valid data argument; declare columns, then fill.

---

## Part F2: `find_worst_queries`

**Level 1.** Sort the rows ascending by relevancy and take the first n
ids. The docstring adds one requirement: a deterministic tiebreak.

**Level 2.** `sorted` with a `key` that returns a tuple: the metric
first, the query id second. Then a list comprehension over the first n.

**Level 3.**

```python
key=lambda r: (r["answer_relevancy"], r["query_id"])
```

---

## Part G1 (stretch): `run_manual_sweep`

**Level 1.** Two nested loops over the value lists in the docstring, one
`tracked_eval` call per combination with a formatted name and the sweep
tag, one row dict appended per run, `pd.DataFrame` at the end.

**Level 2.** The run name is an f-string built from both loop variables.
The row has six keys: the name, the two knobs, and three metrics read
from the snapshot's aggregate.

**Level 3.**

```python
snap = tracked_eval(f"sweep-cs{chunk_size}-k{top_k}", adapter_active=True,
                    top_k=top_k, chunk_size=chunk_size, extra_tags=["sweep"])
```

---

## Part G2 (stretch): `pick_best_config`

**Level 1.** The three-step rule in the docstring maps onto a single
`sort_values` with three columns and three sort directions, then take
the first row.

**Level 2.** Recall descending, top_k ascending, precision descending.
`.iloc[0]` gives the winning row; build the return dict from its
`chunk_size` and `top_k`.

**Level 3.** The return needs plain ints, because pandas hands back
numpy scalars:

```python
return {"chunk_size": int(best["chunk_size"]), "top_k": int(best["top_k"])}
```
