# Instruction-Tuning Schemas — Data Dictionary
### Week 2 · Lab 10 — Instruction-Tuning Schemas

> **All data is synthetic.** The notebook regenerates its Lab 09 inputs from
> `random.Random(42)` (domains `example.local`), then transforms them. Nothing maps to a
> real system.

---

## 1. Inputs (regenerated from Lab 09)

| File | Lines | Shape |
|---|---|---|
| `artifacts/jsonl/rag_chunks.jsonl` | **546** | `{doc_id, chunk_id, text, metadata{…, schema_version:"rag-chunk-v1"}}` |
| `artifacts/jsonl/corpus_sft.jsonl` | **120** | `{input, output, metadata{doc_id, type, lang}}` |

The RAG chunks are 300-char windows with 60-char overlap; their bodies share a boilerplate
intro, so **many chunks are near-identical** (only 274 of 546 texts are unique). That's the
raw material for the dedupe lesson in Part C.

---

## 2. The three SFT schemas

| Schema | Shape | Status |
|---|---|---|
| **Trio** | `{instruction, input, output, metadata}` | current, open-source SFT (Alpaca/FLAN) |
| **Prompt–Completion** | `{prompt, completion, metadata}` | **legacy** — baked-in template; rejected by chat FT endpoints |
| **Chat messages** | `{messages:[{role,content}…], metadata}` | **modern** — the hosted-FT target |

**Messages record** (the modern canonical):
```json
{"messages": [
  {"role": "system", "content": "You are a concise technical assistant."},
  {"role": "user", "content": "Instruction: ...\n\nContext: ..."},
  {"role": "assistant", "content": "..."}
], "metadata": {"doc_id": "...", "chunk_id": "...", "schema_version": "trio-from-rag-v1"}}
```
Current hosted fine-tuning requires **at least one `user` and one `assistant` message** per
line (enforced by the `ChatRow` validator in Part D).

---

## 3. Outputs & locked funnel (seed 42)

| Stage | File | Count |
|---|---|---|
| Trio from SFT (dedupe on `(instruction, output)`) | `instruct_trio.jsonl` | 120 → **61** |
| Legacy prompt–completion (rendered from Trio) | `instruct_prompt_completion.jsonl` | **61** |
| Trio from RAG — length band `[250,1200]` | *(intermediate)* | 546 → **403** |
| Trio from RAG — after text-dedupe | `instruct_trio_from_rag.jsonl` | 403 → **131** |
| Chat messages (from RAG Trio) | `instruct_chat_from_rag.jsonl` | **131** |
| Cleansed PC — decontaminate `Data Retention Policy` (9) | `instruct_prompt_completion_cleansed.jsonl` | 61 → **52** |
| Stats | `artifacts/stats/instruct_stats.json` | `{trio:61, pc:61, chat:131}` |

Validation (pydantic): `trio (61, 0)`, `pc (61, 0)`, `chat (131, 0)` — zero bad rows.

---

## 4. Hygiene knobs

| Knob | Value | Purpose |
|---|---|---|
| `min_out_chars` (Trio) | 20 | drop trivially short outputs |
| length band (RAG) | `250 ≤ len ≤ 1200` chars | skip thin/oversized chunks |
| dedupe key (RAG) | the chunk **text** (stable `set`, not `hash()`) | collapse boilerplate near-dups |
| `MAX_PROMPT_TOK / MAX_COMP_TOK` | 700 / 350 (word proxy) | length governance |
| `EVAL` | `{"Data Retention Policy"}` | eval-holdout decontamination trigger |

---

## 5. Currency fixes over the source lab (verified)

| Source | Problem | Resolution |
|---|---|---|
| `{prompt, completion}` labeled "OpenAI FT style" | it's the **legacy** format; chat models reject it | teach it as legacy; add **chat `messages`** as the modern target |
| chat template as `<system>…<user>…` tag string | not a real schema; not what APIs ingest | emit **role-based `messages`** lists |
| `key = hash(text) % 10**12` for dedupe | Python's `str` hash is **salted per process** → non-reproducible | dedupe on the **text** itself (stable `set`) |
| length band `200–1200` | tuned for 900-char chunks; keeps all 300-char chunks (no-op) | retuned to **`250–1200`** so the floor actually filters |
| word-count "token" filter | undercounts real tokens | kept for offline use; **flag** to use `tiktoken`/`transformers` in prod |
| decontamination trigger = shared template | would silently drop the *entire* set | scoped `EVAL` to a holdout slice (9 policy rows) — measurable |
| one-off pydantic sanity check | doesn't validate the dataset | `validate_file` runs over **every** line; `ChatRow` enforces user+assistant |
