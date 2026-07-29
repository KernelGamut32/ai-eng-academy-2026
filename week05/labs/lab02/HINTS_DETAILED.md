# HINTS: Detailed Tier

**Week 5, Lab 02. Cordwell Support-Doc Retrieval.**

> **Pick one hint tier per task.** This file shows the working core of each task
> with commentary explaining why each line is there, and withholds the function
> shell, the return assembly, and the glue. `HINTS.md` instead walks you toward
> the answer in three escalating steps. Reading both wastes your time. Neither is
> cheating, and switching tiers between tasks is fine.
>
> You still have to read this and understand it to finish the task. Nothing here
> pastes into your notebook and works on its own.

---

## Task A1: `fraction_under`

The working core:

```python
n_under = sum(1 for d in docs if count_tokens(d["text"]) <= size)
```

`sum(1 for ...)` counts matches without building an intermediate list. You could
write `len([d for d in docs if ...])`, which is identical in effect and
allocates a throwaway list.

`count_tokens` operates on `d["text"]`, the document body, not on `d`, the whole
dict. This is the kind of thing that produces a `TypeError` several frames away
from where you made the mistake.

You still need: the empty-list guard so you never divide by zero, and the
division that turns the count into a fraction.

---

## Task B1: `chunk_fixed`

The working core:

```python
step = max(1, size - overlap)
return [tokens[i:i + size] for i in range(0, len(tokens), step)]
```

**Line 1, the step.** With no overlap the step equals the size, and chunks sit
end to end. With overlap of 50 and size of 512, the step is 462, so each chunk
starts 462 tokens after the previous one and therefore shares its first 50
tokens with it. That is what overlap means mechanically.

`max(1, ...)` is the guard. If a caller passes `overlap >= size`, the raw step
is zero or negative. `range` with a zero step raises `ValueError`, and with a
negative step silently produces nothing. Neither is a useful failure. Clamping to
1 turns a caller bug into degraded but working behaviour, and the check in the
notebook verifies it does not hang.

**Line 2, the slice.** Python slicing clamps at the end of the sequence rather
than raising, so `tokens[8:12]` on a 10 element list returns 2 elements. That is
why the final short chunk needs no special handling.

You still need: the empty input guard. Think about what the comprehension above
returns for `[]` and whether that is already correct or not.

---

## Task B2: `build_metadata`

The two lines that carry the lesson:

```python
"chunk_id": f"{doc['document_id']}#{chunk_index}",
"created_at": to_epoch(doc["created_at"]),
```

**The chunk ID.** `{document_id}#{chunk_index}` is not cosmetic. It is the reason
Part F can delete a document in two calls. The `#` separator is arbitrary but it
must not appear inside any document ID, or a prefix query for `doc_1#` would also
match `doc_1#extra#0`. Look at the corpus IDs and satisfy yourself that they are
all lowercase letters, digits, and underscores.

**The timestamp.** `to_epoch` turns `"2026-03-20T14:45:00Z"` into `1774017900`.
Pinecone's `$gte` and `$lte` need numeric operands. Pass a string and the
comparison does not raise, it evaluates to false for every record, and your query
returns nothing while looking completely healthy. Part F has a cell that
demonstrates exactly this failure, and it is worth running twice.

Note that `doc["created_at"]` is already an ISO string in the corpus, so you are
converting, not parsing by hand.

**The text cap.** `chunk_text[:1200]`. Metadata is stored per vector, counts
against storage, and increases query latency. Carrying the text at all is a
deliberate trade so results are displayable without a second lookup. Carrying the
whole document would not be.

You still need: the other eight keys, all of which are direct copies from `doc`
or from the arguments, and the dict assembly.

---

## Task B3: `chunk_corpus`

The dispatch, showing two of the four branches:

```python
if strategy == "fixed":
    pieces = [" ".join(c) for c in chunk_fixed(tokenize(doc["text"]), **kwargs)]
elif strategy == "paragraph":
    pieces = chunk_paragraphs(doc["text"], **kwargs)
```

**Why the two branches look different.** The token based chunkers take a list of
tokens and return lists of tokens, so you tokenize going in and `" ".join`
coming out. The paragraph chunker takes raw text and returns strings, because it
splits on structure that only exists in the raw text. Blank lines do not survive
tokenization.

This asymmetry is real and it is worth noticing. Tokenizing destroys the very
boundaries that Part E argues you should be chunking on.

**`**kwargs` passthrough.** The caller says `size=1024, overlap=0` and it lands
on the chunker unchanged. That is what lets one function serve four strategies
with different parameter names, `overlap` for some and `stride` for others.

The ordering constraint:

```python
total = len(pieces)
for i, piece in enumerate(pieces):
    ...
```

`total_chunks` is a per document count, so it is not knowable until that
document has been fully chunked. Build the whole `pieces` list, take its length,
then loop. Trying to compute it inside a single pass is where people tie
themselves in knots.

**The forward reference.** Your dispatch mentions `chunk_sliding_v1`,
`chunk_sliding_v2`, and `chunk_paragraphs`, none of which exist yet. Write them
into the dispatch anyway. Python resolves a global name when the line executes,
not when the function is defined, so a branch you never take cannot fail.

You still need: the record dict shape, the outer loop, and the accumulator.

---

## Task C1: `make_search`

The three lines that matter:

```python
query_vector = emb.encode([query_text])[0]

response = index.query(
    top_k=top_k * 4,
    vector=query_vector.tolist(),
    namespace=L.NAMESPACE,
    filter=flt,
    include_metadata=True,
)
```

**`emb.encode([query_text])[0]`.** Note the brackets in both directions. `encode`
takes a sequence and returns a 2D array of shape `(n, dimension)`, because it is
built to batch. One string in means one row out, and `[0]` unwraps it. Passing a
bare string would iterate its characters.

More importantly: `emb` is the embedder passed into this closure, which is the
same object that embedded the corpus. This is the single most common silent bug
in retrieval. Two different models produce plausible-looking scores and
meaningless results with no error at all. Part G demonstrates the loud version,
where the dimensions differ and it raises. The quiet version, two models sharing
a dimension, has no defence except discipline.

**`.tolist()`.** The index wants plain Python floats, not a numpy array.

**`top_k * 4`.** This one is easy to skip and it changes your numbers. The index
returns chunks; `evaluate` scores documents, deduplicating chunks up to their
parent. If a single manual contributes four of the top five chunks, asking for
five leaves you with two distinct documents and recall@5 that is quietly
measuring recall@2. Over-fetching gives deduplication room to work.

**`filter=flt`.** Passing `None` here is fine and means no filter. That is why
the same factory serves both the filtered and unfiltered cases in Part F.

**`include_metadata=True`.** Without it the matches come back with empty
metadata and `m.metadata["document_id"]` raises a `KeyError`. Metadata is not
returned by default because it costs bandwidth.

You still need: the closure structure, and reshaping `response.matches` into the
list of dicts with the four required keys.

---

## Task D1: `index_coverage`

The comparison:

```python
covered = {r["metadata"]["document_id"] for r in records}
missing = sorted(d["document_id"] for d in docs if d["document_id"] not in covered)
```

**Why a set.** You are doing a membership test once per document across 202
documents. Against a list that is a linear scan each time. Against a set it is a
hash lookup. At this scale either finishes instantly, but the set is also the
clearer statement of intent: you care about presence, not order or count.

**Why this catches what recall cannot.** Recall asks whether the right document
came back. It has no way to distinguish "ranked poorly" from "not in the index
at all", because both look like absence in the result list. This function asks a
different question entirely, upstream of any query, and that is why it is the
thing that finds the Part D bug.

**Why sort the missing list.** Set iteration order is not guaranteed to be
stable in a way you should rely on. Sorting makes the output reproducible, which
matters when you are diffing two runs to see whether a fix worked.

You still need: the four-key return dict, with counts derived from what you
built above.

---

## Task D2: `chunk_sliding_v2`

Start by working out what the broken version does, because the fix is obvious
once you have:

```python
# broken
for i in range(0, len(tokens) - size, stride):
```

For a 300 token document with `size=512`, that is `range(0, -212, 256)`. A range
whose stop is below its start with a positive step is **empty**. The loop body
never executes, the function returns `[]`, and `chunk_corpus` then loops zero
times over that document. No exception is raised anywhere. The document simply
ceases to exist, and nothing in any log will tell you.

The two corrections:

```python
if len(tokens) <= size:
    return [tokens]
```

**Correction one.** A document that fits in a single chunk should be a single
chunk. This is the fix that recovers 196 documents.

```python
    chunks.append(tokens[i:i + size])
    if i + size >= len(tokens):
        break
```

**Correction two.** Loop to `len(tokens)` rather than `len(tokens) - size`, and
break once a window has reached the end. Without the break you keep stepping and
emit a run of ever shorter trailing chunks: at 1000 tokens with size 512 and
stride 256 you would get windows starting at 0, 256, 512, 768, and the last two
are almost entirely redundant with the third. The break stops at three.

You still need: the empty input guard, the accumulator, and the return.

---

## Task E1: `chunk_paragraphs`

The accumulator, which is the straightforward half:

```python
for para in paragraphs:
    n = count_tokens(para)
    if current_tokens and current_tokens + n > size:
        chunks.append("\n\n".join(current))
        # ... carry logic goes here ...
    current.append(para)
    current_tokens += n
```

**`if current_tokens and ...`.** Read this guard carefully, because without it
the function has a genuine infinite loop. Consider a single paragraph of 600
tokens with `size=384`. Without the guard, the condition fires while `current`
is empty, you append an empty chunk, and nothing has advanced. With the guard,
the oversized paragraph is simply appended and becomes its own chunk on the next
emit.

That behaviour is correct, not a compromise. A paragraph longer than your chunk
size becoming one oversized chunk is better than splitting it mid-sentence,
which is the thing this whole function exists to avoid.

The carry logic, which is the fiddly half:

```python
carry, carry_tokens = [], 0
for prev in reversed(current):
    prev_n = count_tokens(prev)
    if carry_tokens + prev_n > overlap:
        break
    carry.insert(0, prev)
    carry_tokens += prev_n
```

**`reversed(current)`.** Overlap means the *end* of the previous chunk repeats at
the *start* of the next one, so you take from the back.

**`carry.insert(0, prev)`.** You are walking backwards but building a list that
must end up in forward order, so each item goes on the front. `carry.append`
followed by a reverse would also work and is arguably clearer.

**`break`, not `continue`.** Once one paragraph does not fit in the overlap
budget, stop. Continuing would skip it and pull in an earlier, smaller paragraph,
which produces an overlap with a hole in the middle. That is worse than no
overlap, because the chunk boundary is now in two places.

You still need: the split, the empty guard, resetting `current` and
`current_tokens` from the carry, and the final emit of whatever is left.

---

## Task F2: the filter dicts

The whole thing is two literals, so instead here is why the second one is
dangerous:

```python
"updated_at": {"$gte": to_epoch("2025-01-01T00:00:00Z")}
```

`to_epoch` returns `1735689600`. Write the string instead of the call and you get
`{"$gte": "2025-01-01T00:00:00Z"}`, which is syntactically valid, passes every
type check, raises nothing, and matches zero records. The notebook has a cell
that runs exactly that and reports `recall@5=0.000`.

Note also that `to_epoch` interprets the `Z` suffix as UTC. If you drop the `Z`,
Python treats the timestamp as local time and you get a different integer on a
machine in a different time zone. Compute it, do not hard-code the number you saw
in someone else's output.

Multiple keys at the same level are combined with an implicit AND. There is an
explicit `$and` operator, and you do not need it here.

---

## Task F3: `delete_document`

The listing, which is the part that changed in SDK v9:

```python
ids = sorted(item.id
             for page in index.list(prefix=f"{document_id}#", namespace=namespace)
             for item in page)
```

**Two levels of iteration.** `index.list()` returns an iterator of `ListResponse`
pages, and iterating a page yields `ListItem` objects. So the outer loop walks
pages and the inner loop walks items within a page. Pages default to 100 IDs.

**`item.id`, not `item`.** In SDK v8 and earlier this yielded id strings
directly. In v9 it yields objects with a single `id` field. The Module 02 slide
on structured IDs still shows the old form, which now silently builds a list of
objects and then fails inside `delete`. This is the most common v8 to v9 upgrade
break.

**The `#` in the prefix.** Without it, `prefix="man_thermostat_t4"` would also
match a hypothetical `man_thermostat_t400`. Including the separator anchors the
prefix at a document boundary.

**`sorted`.** Not required by the API. It makes the return value stable so tests
and diffs behave.

You still need: the namespace default, the guard against deleting an empty list,
the delete call itself, and the return.

---

## Task G1: the model swap

The two lines that carry the lesson:

```python
alt_embedder = get_embedder("alternate", dim=EMBED_DIM, background=background)
alt_index = build_index(best_records, alt_embedder)
```

**`best_records`, not new records.** The entire point is to change one variable.
Re-chunking here would confound the comparison and you would learn nothing about
either lever.

**Why the alternate is genuinely a different model.** It uses character n-gram
features in addition to words, which produces a different vocabulary, a different
learned geometry, and a different dimension. It is not the primary model resized.
A vector from one cannot be scored against an index built from the other, and
that is a property of the models rather than a limitation of this lab.

The comparison, and the part worth thinking about:

```python
best_label = max([lab for lab, _ in SWEEP],
                 key=lambda lab: results[lab][f"recall@{K}"])
chunking_delta = results[best_label][...] - results["B baseline fixed 1024/0"][...]
```

**Why the best sweep config and not 384/64.** The chunking lever is the whole
span from your starting point to the best configuration you found, because that
is the range genuinely available to you. Measuring it against an arbitrary
mid-range point would understate it and flatter the model swap. Be as careful
with the framing of a comparison as with the arithmetic.

You still need: the mismatch demonstration, the evaluate call, the model delta,
and the printing.

---

## Task H1: `incremental_refresh`

The forward pass:

```python
digest = content_hash(doc["text"])
if doc_id in existing:
    if existing[doc_id]["hash"] != digest:
        to_reembed.append(doc)
        chunk_ids_to_delete.extend(existing[doc_id]["chunk_ids"])
else:
    new_docs.append(doc)
```

**Why changed documents appear in both lists.** A changed document needs its new
content embedded **and** its old chunks removed. Skip the delete and you leave
stale chunks in the index alongside the fresh ones, both retrievable, with no way
for a query to tell them apart. That is worse than not refreshing at all: at
least a stale index is consistently stale.

**Why `content_hash` and not `hash`.** Python's built-in `hash()` is salted with
a per-process random seed, so `hash("abc")` returns a different value every time
the interpreter starts. A nightly job using it would find every document changed,
every night, forever, and you would pay to re-embed the entire corpus daily while
believing you had built an incremental pipeline. `content_hash` is SHA-256 and is
stable across processes and machines.

The backward pass, which is the one people forget:

```python
current = {d["document_id"] for d in documents}
for gone in sorted(set(existing) - current):
    chunk_ids_to_delete.extend(existing[gone]["chunk_ids"])
```

**Why a second pass in the other direction.** The forward loop only sees
documents that still exist. A document deleted upstream never appears in
`documents`, so nothing in that loop can possibly notice it. Set difference on
the IDs is the only way to find it.

**Deletes only, never a re-embed.** There is no content left to embed. Adding it
to `to_reembed` would raise a `KeyError` downstream when something tried to read
its text.

You still need: the three accumulators, and the tuple return in the documented
order.
