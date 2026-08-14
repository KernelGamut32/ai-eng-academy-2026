# Progressive Hints

Three levels per task. Read level 1 first and go back to work; escalate only when still stuck after a real attempt. Level 3 shows the key line or two, never the whole function. If you would rather read the working core with commentary than be led stepwise, use HINTS_DETAILED.md instead; pick one tier per task.

The final section, "When the corpus fights back," is keyed to symptoms, not tasks. Go there when something breaks in a way the task hints do not explain.

---

## Task 1: pick_device

**Level 1.** Torch exposes one availability probe per accelerator. Order matters, and the whole thing wants wrapping so a missing torch cannot crash a function whose contract is "never raise."

**Level 2.** Import torch inside the function, inside a try block. Check CUDA first, then MPS via `torch.backends`, and let every failure path fall through to returning "cpu". Use `hasattr` before touching `torch.backends.mps` so older builds cannot AttributeError.

**Level 3.** The MPS line is the one people miss:
```python
if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    return "mps"
```

## Task 2: read_text_tolerant

**Level 1.** Read bytes once, then try decodings in order of strictness. There is an encoding name that handles a UTF-8 BOM for free.

**Level 2.** `path.read_bytes()`, then loop over `("utf-8-sig", "cp1252")` attempting `raw.decode(encoding)`, catching UnicodeDecodeError and continuing. After the loop, the last resort decode with `errors="replace"`.

**Level 3.** The tolerant order in one expression:
```python
for encoding in ("utf-8-sig", "cp1252"):
    try:
        return raw.decode(encoding)
    except UnicodeDecodeError:
        continue
```

## Task 3: select_active_documents

**Level 1.** Group by doc_family, keep one winner per group, then re-emit winners in original corpus order. The dates are ISO strings, which sort correctly as plain strings.

**Level 2.** Build a dict from family to best document, comparing candidates on the tuple `(effective_date, version, doc_id)`. Then iterate the original list and keep each document only if it IS its family's winner.

**Level 3.** The comparison key that makes the choice deterministic:
```python
key = (doc.effective_date, doc.version, doc.doc_id)
```
and the order-preserving emit:
```python
return [d for d in documents if best[d.doc_family] is d]
```

## Task 4: chunk_document

**Level 1.** Three provided helpers, three document shapes. Sections from headings, paragraph packing within a section, and hard slicing for anything that is still too long after packing. Your job is composition plus one guarantee: nothing you return exceeds max_chars, ever.

**Level 2.** For each section from `split_into_sections`, run `_pack_paragraphs`. For each packed piece, check its length: small pieces become chunks directly; oversized pieces (a single paragraph can exceed max_chars all by itself) go through `_slice_with_overlap` and every slice becomes a chunk. Number chunks with one counter across the whole document.

**Level 3.** The guarantee lives in this branch:
```python
for piece in _pack_paragraphs(section, max_chars):
    if len(piece) <= max_chars:
        emit(piece)
    else:
        for sliced in _slice_with_overlap(piece, max_chars, overlap):
            emit(sliced)
```

## Task 5: dedupe_chunks

**Level 1.** One pass, one set of seen content hashes, first occurrence wins.

**Level 2.** For each chunk compute `content_hash(chunk.text)`. If the hash is in the seen set, increment a dropped counter; otherwise add it and keep the chunk. Return the kept list and the count.

**Level 3.**
```python
h = content_hash(chunk.text)
if h in seen:
    dropped += 1
    continue
seen.add(h)
```

## Task 6: upsert_chunks

**Level 1.** Three phases: validate alignment, build and size-check every record BEFORE the first upsert call, then batch. The metadata guard failing mid-batch would leave a half-written index; check everything first.

**Level 2.** Raise ValueError on length mismatch. Build all records up front: id from chunk_id, values from the vector, metadata from `self.build_metadata`. Size-check each with `self.metadata_size_bytes` against `config.METADATA_LIMIT_BYTES` and raise with the chunk_id and size in the message. Then slice the record list into batch_size pieces and call `self.index.upsert(vectors=batch)` per piece.

**Level 3.** The SDK shape that the fake index (and the real client) enforces:
```python
self.index.upsert(vectors=batch)   # keyword-only; positional raises TypeError
```

## Task 7: query_top_k

**Level 1.** One SDK call, then defend the rest of the codebase from SDK response objects: everything leaving this function is a plain dict.

**Level 2.** `self.index.query(vector=..., top_k=..., include_metadata=True)`. Matches may be attribute-style or dict-style; write one tiny accessor that tries `getattr` then falls back to `.get`, and use it for id, score, and metadata. Fill missing metadata fields with "" and seq with 0.

**Level 3.** The normalization accessor most teams write eventually:
```python
def field(m, name, default=None):
    value = getattr(m, name, None)
    return m.get(name, default) if value is None and isinstance(m, dict) else (value if value is not None else default)
```

## Task 8: offline_extractive_answer

**Level 1.** Split contexts into sentences, embed the question and all sentences in ONE embed call, score by dot product (vectors are unit length), then decide: abstain or emit the top sentences in reading order.

**Level 2.** Track sentences as (position, text, score). best_score is the max score. The abstention branch compares best_score to the threshold when the threshold is not None and returns `(config.ABSTAIN_TEXT, best_score)`. Otherwise take the top max_sentences by score, then sort those winners by position before joining, so the answer reads in document order.

**Level 3.** The reading-order restoration, which is the step everyone skips first try:
```python
top = sorted(scored, key=lambda s: s.score, reverse=True)[:max_sentences]
top.sort(key=lambda s: s.position)
```

## Task 9: retrieve and generate

**Level 1.** retrieve: embed, query, remember, return texts. generate: three backends, one of which never touches a model. The variant object carries every knob; generate should contain no literals.

**Level 2.** retrieve saves the full match dicts on `self.last_matches` before returning `[m["text"] for m in matches]`. generate branches on the backend: offline calls `offline_extractive_answer` with the variant's offline knobs; the live branch builds a system message from `self.variant.system_prompt` and a user message from `format_context_block(self.last_matches)` plus the question, then calls `chat_completion` with the variant's temperature and max_tokens.

**Level 3.** The live-path user message, in the solution's framing:
```python
user = f"QUESTION:\n{question}\n\nCONTEXT:\n{format_context_block(self.last_matches)}\n\nAnswer:"
```

## Task 10: the triad

**Level 1.** Three functions, three comparisons: question vs contexts (mean), response vs contexts (max), question vs response. The abstention rules are stated in each contract and they are asymmetric on purpose.

**Level 2.** All three follow the same skeleton: handle the degenerate and abstention cases first, embed everything in one call, dot products, clip with `_clip01`. context_relevance averages over contexts. groundedness takes the max over contexts, because one fully supporting source suffices. answer_relevance is a single similarity.

**Level 3.** The pair of abstention rules that create the deliberate tension:
```python
# groundedness: an abstention cannot hallucinate
if is_abstention(response): return 1.0
# answer_relevance: an abstention answers nothing
if is_abstention(response): return 0.0
```

## Task 11: pick_champion

**Level 1.** Two stages: gates eliminate, composite ranks. Count gate failures as you filter (the no-winner report needs them). Everything reads from dataframe columns named `metrics.<name>`.

**Level 2.** For each gate, build a boolean mask requiring the column to exist, be non-null, and meet the threshold; record how many rows each gate removed; AND the masks. If nothing survives, return the no_eligible_run dict. For survivors, compute the weighted sum across `criteria.weights` treating missing as 0.0, then sort by `(composite, groundedness, run_id)` descending on the first two.

**Level 3.** A missing-column-safe gate mask:
```python
col = f"metrics.{name}"
ok = df[col].notna() & (df[col] >= threshold) if col in df else pd.Series(False, index=df.index)
```

## Task 12: /search

**Level 1.** The endpoint is four lines of orchestration over things you already built: embed, query, map to the response model. FastAPI's Query declarations already validated the inputs.

**Level 2.** Pull embedder and store out of `state`, embed `q` (embed takes a list, take element 0), call `query_top_k`, build a SearchHit per match dict, wrap in SearchResponse.

**Level 3.**
```python
vector = state["embedder"].embed([q])[0]
matches = state["store"].query_top_k(vector, top_k=top_k)
```

---

## When the corpus fights back (symptom-keyed)

Real corpora misbehave. This one does it on purpose, four ways. Match your symptom below; each entry tells you which task's code owns the fix, without telling you the fix.

**Symptom: ingest dies with UnicodeDecodeError, mentioning a byte like 0x93.** One document was exported from an old Windows tool and is not UTF-8. This is exactly what Task 2's contract describes; your reader is currently less tolerant than its docstring promises. Find which file it is with `file corpus/*.md` or by catching the exception and printing the path; then make the reader honest.

**Symptom: your assistant answers a returns question with 90 days, and the eval disagrees.** The corpus contains two returns policies from different years in the same doc_family. If both were ingested, retrieval happily serves the stale one. Task 3 exists to prevent this. Check `select_active_documents` output: 30 documents, not 31, and the 2026 policy among them.

**Symptom: a ValueError about metadata size during ingest, or one document producing absurdly long chunks.** One operations document is a 41KB call-transcript digest with no headings at all and paragraphs longer than the slice limit. Heading-based chunking alone packs it into chunks that blow the metadata budget. Task 4's size guarantee must hold for this document specifically; the test `test_real_transcript_curveball_stays_within_limits` is the honest referee.

**Symptom: chunk counts look about 6 too high, or the same manual text appears twice in retrieval results.** A content migration left an archived byte-identical copy of one product manual under a different doc_id. Versioning (Task 3) does not catch it because the copy sits in its own doc_family; only content dedupe (Task 5) does. A follow-up worth discussing at standup once fixed: WHICH copy's chunks survived, and what that means for the doc_id your API reports as a source.

**Symptom: everything green but retrieval quality seems oddly literal.** Not a trap, just the hash embedder being a lexical instrument. Try `EMBEDDING_BACKEND=st` after M6 and compare; that comparison is a stretch goal for a reason.
