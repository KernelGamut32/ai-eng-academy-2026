# HINTS: Progressive Tier

**Week 5, Lab 02. Cordwell Support-Doc Retrieval.**

> **Pick one hint tier per task.** This file guides you toward the answer in
> three escalating steps. `HINTS_DETAILED.md` instead shows you the working core
> with line-by-line commentary and leaves you the assembly. Reading both wastes
> your time. Neither is cheating, and switching tiers between tasks is fine.

Read level 1. Try again. Only then read level 2. A capable engineer who is stuck
should be moving again by level 2.

---

## Task A1: `fraction_under(docs, size)`

**Level 1.** You need a count and a division. `count_tokens(text)` is already
imported and does the hard part. Guard the empty list so you never divide by
zero.

**Level 2.** Count how many documents satisfy the condition, then divide by the
total. A generator expression inside `sum()` counts booleans without building an
intermediate list. The condition is on `d["text"]`, not on `d`.

**Level 3.**

```python
if not docs:
    return 0.0
n_under = sum(1 for d in docs if count_tokens(...) <= size)
return n_under / len(docs)
```

---

## Task B1: `chunk_fixed(tokens, size, overlap)`

**Level 1.** This is a slice in a loop. The only question is what the loop step
is. Overlap means consecutive chunks share tokens, so the step must be smaller
than the chunk size.

**Level 2.** Step is `size - overlap`. Two edge cases to handle before the loop:
an empty input returns `[]`, and `overlap >= size` would make the step zero or
negative, which never advances. Clamp the step to a minimum of 1.

Python slicing is forgiving at the end. `tokens[8:12]` on a 10 element list
returns 2 elements rather than raising, which is why the final short chunk falls
out for free.

**Level 3.**

```python
if not tokens:
    return []
step = max(1, size - overlap)     # the guard
return [tokens[i:i + size] for i in range(0, len(tokens), step)]
```

---

## Task B2: `build_metadata(...)`

**Level 1.** This is one dictionary literal. The two things worth thinking about
are the timestamp type and the ID format, and both are called out on the slides.

**Level 2.** Timestamps must be `int`, produced by `to_epoch(iso_string)`. A
string date does not raise; it just silently never matches a `$gte`. The chunk
ID is `f"{document_id}#{chunk_index}"`, which is what makes prefix deletion
possible in Part F. Truncate the text with a slice, `chunk_text[:1200]`.

**Level 3.**

```python
return {
    "chunk_id": f"{doc['document_id']}#{chunk_index}",
    "document_id": doc["document_id"],
    "source": doc["source"],
    "doc_type": doc["doc_type"],
    "product_line": doc["product_line"],
    "created_at": to_epoch(doc["created_at"]),   # int, not str
    "updated_at": to_epoch(...),
    "is_active": doc["is_active"],
    "chunk_index": chunk_index,
    "total_chunks": total_chunks,
    "text": chunk_text[:1200],
}
```

---

## Task B3: `chunk_corpus(docs, strategy, **kwargs)`

**Level 1.** Outer loop over documents. Inner loop over the chunks of that
document. The `strategy` argument picks which chunker to call, and the table in
the notebook tells you which ones take tokens and which takes raw text.

**Level 2.** The token based strategies need `tokenize(doc["text"])` going in
and `" ".join(chunk)` coming out, because the chunker returns lists of tokens
and you want strings. The paragraph strategy takes and returns text directly, so
no conversion.

`total_chunks` is per document, so you cannot compute it until that document is
fully chunked. Build the list of pieces first, take its length, then loop with
`enumerate` to build records.

**Level 3.**

```python
records = []
for doc in docs:
    if strategy == "fixed":
        pieces = [" ".join(c) for c in chunk_fixed(tokenize(doc["text"]), **kwargs)]
    elif strategy == "sliding_v1":
        ...
    elif strategy == "paragraph":
        pieces = chunk_paragraphs(doc["text"], **kwargs)
    else:
        raise ValueError(...)

    total = len(pieces)              # only knowable now
    for i, piece in enumerate(pieces):
        meta = build_metadata(doc, i, total, piece)
        records.append({"id": meta["chunk_id"], "text": piece, "metadata": meta})
return records
```

---

## Task C1: `make_search(index, emb, flt)`

**Level 1.** The inner `search` function does three things: embed the query,
query the index, reshape the matches into dicts. The embedder is the one passed
in, not a new one.

**Level 2.** `emb.encode` takes a **list** and returns a 2D array, so encode
`[query_text]` and take element `[0]`. The index wants a plain list, so call
`.tolist()`.

Ask the index for `top_k * 4`, not `top_k`. Several chunks of the same document
will come back and get deduplicated to one document downstream, so asking for
exactly 5 chunks can leave you with 2 documents.

Every v9 data plane method is keyword only. `index.query(vector, 5)` is a
`TypeError`.

**Level 3.**

```python
def search(query_text, top_k=K):
    query_vector = emb.encode([query_text])[0]
    response = index.query(
        top_k=top_k * 4,               # over-fetch, dedup happens later
        vector=query_vector.tolist(),
        namespace=L.NAMESPACE,
        filter=flt,
        include_metadata=True,
    )
    return [{"chunk_id": m.id,
             "document_id": m.metadata["document_id"],
             "score": m.score,
             "text": m.metadata.get("text", "")}
            for m in response.matches]
return search
```

---

## Task C2: record the baseline

**Level 1.** One call to `evaluate`, one assignment into `results`.

**Level 2.** `evaluate(LABELED_QUERIES, baseline_search, k=K)`. The dictionary
key must match exactly, including spacing, because later cells look it up.

**Level 3.**

```python
results["B baseline fixed 1024/0"] = evaluate(LABELED_QUERIES, baseline_search, k=K)
```

---

## Task D1: `index_coverage(docs, records)`

**Level 1.** Compare two sets of document IDs: the ones in the corpus, and the
ones that actually appear in the records you are about to index. The difference
is what vanished.

**Level 2.** Build a set of `r["metadata"]["document_id"]` across all records.
Then a document is missing if its ID is not in that set. Sort the missing list
so the output is stable between runs.

**Level 3.**

```python
covered = {r["metadata"]["document_id"] for r in records}
missing = sorted(d["document_id"] for d in docs if d["document_id"] not in covered)
return {"documents_total": len(docs),
        "documents_indexed": len(covered),
        "documents_missing": len(missing),
        "missing_ids": missing}
```

---

## Task D2: `chunk_sliding_v2(tokens, size, stride)`

**Level 1.** Look hard at `range(0, len(tokens) - size, stride)` in the broken
version. Work out what that range is when `len(tokens)` is 300 and `size` is
512. The fix follows directly from what you find.

**Level 2.** `range(0, -212, 256)` is empty, so the loop body never runs and the
function returns `[]`. Two changes are needed. Short documents need an early
return producing exactly one chunk. The loop bound needs to be `len(tokens)`
rather than `len(tokens) - size`, plus a break once a window has reached the
end, so you do not emit a run of shrinking trailing chunks.

**Level 3.**

```python
if not tokens:
    return []
if len(tokens) <= size:
    return [tokens]          # the fix that matters

chunks = []
for i in range(0, len(tokens), stride):
    chunks.append(tokens[i:i + size])
    if i + size >= len(tokens):
        break                # tail captured, stop
return chunks
```

---

## Task E1: `chunk_paragraphs(text, size, overlap)`

**Level 1.** Accumulate paragraphs into a current chunk while tracking a running
token count. When the next paragraph would push you over `size`, close the
current chunk and start a new one. Never split a paragraph.

**Level 2.** The overlap step is the fiddly part. After emitting a chunk, seed
the next one with trailing paragraphs of the one you just emitted. Walk the
emitted list backwards, adding paragraphs to the front of the carry list while
the carry token total stays within `overlap`, and stop as soon as one would push
you over.

Guard the emit with "and the current chunk is not empty". Without that guard, a
single paragraph longer than `size` emits an empty chunk and then loops forever.

**Level 3.**

```python
paragraphs = split_paragraphs(text)
if not paragraphs:
    return []

chunks, current, current_tokens = [], [], 0
for para in paragraphs:
    n = count_tokens(para)
    if current_tokens and current_tokens + n > size:    # note the guard
        chunks.append("\n\n".join(current))
        carry, carry_tokens = [], 0
        for prev in reversed(current):
            # ... take from the end while it fits inside overlap ...
        current, current_tokens = list(carry), carry_tokens
    current.append(para)
    current_tokens += n

if current:
    chunks.append("\n\n".join(current))
return chunks
```

---

## Task F2: the filter dicts

**Level 1.** Pinecone filters are dicts mapping a field name to an operator
dict. The operators you need are `$eq` and `$gte`.

**Level 2.** `ACTIVE_ONLY` has one key. `ACTIVE_AND_RECENT` has two, and keys at
the same level are combined with an implicit AND. The date operand must be an
integer produced by `to_epoch`, not the ISO string.

**Level 3.**

```python
ACTIVE_ONLY = {"is_active": {"$eq": True}}
ACTIVE_AND_RECENT = {
    "is_active": {"$eq": True},
    "updated_at": {"$gte": to_epoch("2025-01-01T00:00:00Z")},
}
```

---

## Task F3: `delete_document(index, document_id, namespace)`

**Level 1.** Two calls. List the IDs sharing the document prefix, then delete
that list. The prefix is the document ID plus the separator you chose in Task
B2.

**Level 2.** `index.list(prefix=..., namespace=...)` returns an iterator of
**pages**, and iterating a page yields `ListItem` objects rather than strings.
You want `item.id`. That needs a nested comprehension: outer over pages, inner
over items.

Skip the delete when the list is empty. Deleting nothing is harmless but the
empty call is noise.

**Level 3.**

```python
namespace = L.NAMESPACE if namespace is None else namespace
ids = sorted(item.id
             for page in index.list(prefix=f"{document_id}#", namespace=namespace)
             for item in page)          # .id, not item
if ids:
    index.delete(ids=ids, namespace=namespace)
return ids
```

---

## Task G1: the model swap

**Level 1.** Four steps, all of which reuse functions you already wrote. Build
the alternate embedder, show the spaces differ, rebuild the same records with
it, and evaluate.

**Level 2.** `get_embedder("alternate", dim=EMBED_DIM, background=background)`.
Reuse `best_records`, because the point is to change **only** the model. Then
`build_index(best_records, alt_embedder)` and
`make_search(alt_index, alt_embedder)`.

For the deltas: the chunking lever is baseline to the best sweep configuration,
because that is the range genuinely available to you. The model lever is the
swap at fixed chunking.

**Level 3.**

```python
alt_embedder = get_embedder("alternate", dim=EMBED_DIM, background=background)
alt_index = build_index(best_records, alt_embedder)       # same records
results["G alternate model 384/64"] = evaluate(
    LABELED_QUERIES, make_search(alt_index, alt_embedder), k=K)

best_label = max([lab for lab, _ in SWEEP],
                 key=lambda lab: results[lab][f"recall@{K}"])
chunking_delta = results[best_label][...] - results["B baseline fixed 1024/0"][...]
model_delta = results["G alternate model 384/64"][...] - results["E paragraph 384/64"][...]
```

---

## Task H1: `incremental_refresh(documents, existing)`

**Level 1.** Three buckets, and the third is the one people forget. Changed,
new, and **removed**. Removed documents are in `existing` but not in
`documents`, so you only find them by comparing the other direction.

**Level 2.** Loop the current documents. If the ID is known, compare hashes and
on a mismatch add to `to_reembed` **and** extend the delete list with the old
chunk IDs. If the ID is unknown, it is new: embed it, nothing to delete.

Then a second pass in the other direction: `set(existing) - current` gives you
IDs that disappeared upstream. Those get deletes only, never a re-embed.

Use `content_hash`, not Python's `hash()`. The built-in is salted per process
and would report the entire corpus as changed on every run.

**Level 3.**

```python
to_reembed, chunk_ids_to_delete, new_docs = [], [], []
for doc in documents:
    digest = content_hash(doc["text"])
    doc_id = doc["document_id"]
    if doc_id in existing:
        if existing[doc_id]["hash"] != digest:
            to_reembed.append(doc)
            chunk_ids_to_delete.extend(existing[doc_id]["chunk_ids"])
    else:
        new_docs.append(doc)

current = {d["document_id"] for d in documents}
for gone in sorted(set(existing) - current):
    chunk_ids_to_delete.extend(existing[gone]["chunk_ids"])

return to_reembed, chunk_ids_to_delete, new_docs
```
