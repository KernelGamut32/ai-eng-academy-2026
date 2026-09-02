# RAG Reranking Demo

Canonical two-stage retrieval: bi-encoder retrieve, cross-encoder rerank, generate. Built for the AI Engineering Academy; runs keyless and offline on the cohort's no-GPU Macs.

## Files

| File | Purpose |
|---|---|
| `DEMO_SCRIPT.md` | Instructor script: primer, pre-flight, step-by-step talk track, fallbacks, Q&A, slide example, verification ledger, currency flags. Start here. |
| `rerank_demo.py` | The implementation. Four commands: `demo`, `eval`, `showcase`, `replay`. |
| `data/cordwell_kb.json` | 40 synthetic Cordwell passages plus 10 gold queries with `why_hard` notes. |
| `requirements.txt` | Pins verified against PyPI on build day. |
| `cache/` | `demo` writes `last_run.json` here; `replay` reads it. Empty until first run. |

If not already done, access the OneDrive link at <https://gamuttechnologysvcs-my.sharepoint.com/:f:/p/asanders/IgD_SIVCz8YJQYh7BL3DUy4ZAVgU8-9SO8Lo3boIy-wwV8g?e=D4X53b>, navigate to the `models` folder, and download `all-MiniLM-L6-v2.zip` and `ms-marco-MiniLM-L6-v2.zip`. Unzip to a target folder on your system and use that folder path in the updates to `rerank_demo.py` (described below).

## Quick start

```bash
pip install -r requirements.txt
python rerank_demo.py showcase          # picks the query to present
python rerank_demo.py demo --qid Q1     # full walkthrough, records the fallback
python rerank_demo.py eval              # Recall@1, Recall@3, MRR before and after
python rerank_demo.py replay            # no models needed
```

Optional live generation: `LAB_BACKEND=ollama` or `LAB_BACKEND=lmstudio`, model tag in `LOCAL_MODEL` (default `gemma4`).

## Decisions

| Decision | Choice | Reason |
|---|---|---|
| Embedder | `all-MiniLM-L6-v2` | Ubiquitous, small, and imperfect enough that reranking visibly helps. Update line 41 in `rerank_demo.py` to point to local location of model. |
| Reranker | `ms-marco-MiniLM-L6-v2` | Reference cross-encoder in the Sentence Transformers docs; same parameter count as the embedder, so the demo isolates architecture, not size. Update line 41 in `rerank_demo.py` to point to local location of model. |
| Stage 1 pool | 10 | Large enough that gold is almost always present on a 40-passage corpus; small enough to rerank in well under a second on CPU |
| Final k | 3 | Makes the context-size argument concrete; adjustable in one constant |
| Hybrid BM25 | Not included | One concept per demo. Named in the discussion prompt as the production upgrade |
| Generation | Offline extractive stub by default | Keeps the demo keyless; both local backends available behind `LAB_BACKEND` on one OpenAI-compatible client path |
| Which query to present | Chosen by `showcase` at pre-flight | Model behavior could not be executed in the build sandbox; the instructor's machine decides |

## Known limits

- Ten gold queries: enough to show direction, not to quote a percentage. Say so.
- Real model outputs were not run at build time (Hugging Face Hub unreachable from the build sandbox). Every number in `DEMO_SCRIPT.md` is marked illustrative; the pre-flight run supplies the real ones.

---

## Demo: Reranking in RAG (two-stage retrieval)

**Companion files:** `rerank_demo.py`, `data/cordwell_kb.json`, `requirements.txt`, `README.md`.

**Runs on:** cohort Macs, CPU or MPS, no API key, no network after the one-time model download.

### The problem reranking solves

Dense retrieval compresses every passage into one vector once, ahead of time, and compresses the query into one vector at request time. Ranking is a dot product. That is why it is fast enough to search a million passages, and also why it is imprecise: the query and the passage never see each other. The passage vector was frozen before the question existed.

Consequence engineers recognize immediately: a bi-encoder is very good at "these two texts are about the same topic" and mediocre at "this passage answers this exact question." A passage about pressure washer PSI specs and a passage about returning outdoor power equipment without a receipt both sit near a query about returning a pressure washer without a receipt. Topic similarity is not answer relevance.

### What a cross-encoder does differently

A cross-encoder takes the query and one candidate passage together as a single input sequence, `[CLS] query [SEP] passage [SEP]`, runs one transformer forward pass, and emits one number: how relevant is this passage to this query. Because both texts are in the same attention window, every query token can attend to every passage token. It sees the interaction. That is the whole trick.

The cost is the same reason for the benefit: one forward pass per (query, passage) pair, at request time, with nothing precomputable. You cannot cross-encode a million passages per query. You can cross-encode ten.

### The canonical pattern

```
query
  |
  v
[stage 1] bi-encoder + vector index  ->  top-10 candidates   (fast, recall-oriented)
  |
  v
[stage 2] cross-encoder scores 10 pairs -> top-3 survivors    (slow per item, precision-oriented)
  |
  v
[stage 3] prompt = top-3 passages + question  ->  LLM
```

Engineering analogy: stage 1 is the cheap index scan that narrows to a candidate set; stage 2 is the expensive row-level filter you only run on the survivors. Nobody runs the expensive filter on the whole table.

### Why it also helps the generator

Two effects beyond "the right passage is now at rank 1":

1. **Context budget.** Sending 10 passages costs roughly 4 times the tokens of sending 3. Reranking lets you send fewer passages with more confidence. The demo prints the word count both ways.
2. **Position.** LLMs attend more reliably to material at the start and end of the context than to the middle. Reranking puts the best evidence first, where the model is most likely to use it.

### Models used, and why these

| Role | Model | Why it is the teaching choice |
|---|---|---|
| Bi-encoder | `sentence-transformers/all-MiniLM-L6-v2` | The most widely used small embedding model; 22M parameters, 384-dim, runs in well under a second on CPU. Deliberately not the strongest embedder available, which is what makes stage 2's contribution visible. |
| Cross-encoder | `cross-encoder/ms-marco-MiniLM-L6-v2` | The reference reranker in the Sentence Transformers documentation; trained on MS MARCO passage ranking, 22M parameters. Same size as the bi-encoder, so the demo isolates the architectural difference rather than a size difference. |
| Production alternative | `BAAI/bge-reranker-v2-m3` | Multilingual, materially stronger, about 570M parameters. Same `CrossEncoder` code path; set `RERANK_MODEL` to swap. Too slow for a live CPU demo; reasonable as an upgrade path. |
| Hosted alternative | Cohere Rerank, or the rerank endpoints in managed vector databases | Same two-stage idea behind an API. Not used here because the Academy stays keyless. |

Both demo models load through Sentence Transformers 6.0.1 (`SentenceTransformer` and `CrossEncoder`). Method names verified against the 6.0.1 source: `encode_document`, `encode_query`, `util.semantic_search`, `CrossEncoder.rank`.

### A note on scores

Cosine scores from stage 1 live in [-1, 1] and are comparable across queries. Cross-encoder scores are model-specific; the MS MARCO MiniLM cross-encoders emit raw logits, so a score of 6.1 versus 2.3 means "much more relevant" and nothing more. Compare reranker scores only within one query. If you want probabilities, `CrossEncoder(..., activation_fn=torch.nn.Sigmoid())` squashes them, but the ranking is unchanged.
