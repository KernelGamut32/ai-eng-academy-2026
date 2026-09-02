"""
Reranking in RAG: the canonical two-stage retrieval pattern.

    Stage 1  bi-encoder retrieve   fast, approximate, wide net      top-k = RETRIEVE_K
    Stage 2  cross-encoder rerank  slow, precise, narrow net        top-k = FINAL_K
    Stage 3  generate              only FINAL_K passages reach the prompt

Run modes
    python rerank_demo.py demo                 walk one query through all three stages
    python rerank_demo.py demo --query "..."   same, with your own question
    python rerank_demo.py eval                 Recall@1, Recall@3, MRR before and after reranking
    python rerank_demo.py showcase             pick the gold query where reranking moves the answer most
    python rerank_demo.py replay               print the last recorded run without loading any model

Environment
    LAB_BACKEND   offline (default) | lmstudio | ollama
    LOCAL_MODEL   model tag served by the local backend, default gemma4
    EMBED_MODEL   bi-encoder, default sentence-transformers/all-MiniLM-L6-v2
    RERANK_MODEL  cross-encoder, default cross-encoder/ms-marco-MiniLM-L6-v2
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")  # must be set before torch is imported
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# ---------------------------------------------------------------------------
# Config. Everything a student might reasonably change lives here.
# ---------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent
KB_PATH = HERE / "cordwell_kb.json"
CACHE_PATH = HERE / "cache" / "last_run.json"

EMBED_MODEL = str(Path(os.environ.get("W4L1_MODEL_DIR", "~/models/all-MiniLM-L6-v2")).expanduser().resolve())  # config variable, do not hard-code model paths
RERANK_MODEL = str(Path(os.getenv("RERANK_MODEL", "~/models/ms-marco-MiniLM-L6-v2")).expanduser().resolve())
# Heavier production-grade alternative, same code path:  RERANK_MODEL=BAAI/bge-reranker-v2-m3

RETRIEVE_K = 10   # stage 1 candidate pool
FINAL_K = 3       # stage 2 survivors that reach the prompt

LAB_BACKEND = os.getenv("LAB_BACKEND", "offline").lower()
LOCAL_MODEL = os.getenv("LOCAL_MODEL", "gemma4")
BACKEND_URLS = {
    "lmstudio": "http://localhost:1234/v1",
    "ollama": "http://localhost:11434/v1",
}


def pick_device() -> str:
    """CUDA if present, else Apple MPS, else CPU. Never hard-code a device string."""
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def load_kb() -> tuple[list[dict], list[dict]]:
    kb = json.loads(KB_PATH.read_text())
    return kb["passages"], kb["gold_queries"]


def passage_text(p: dict) -> str:
    # Title plus body is what gets embedded and what the reranker reads.
    return f"{p['title']}. {p['text']}"


# ---------------------------------------------------------------------------
# Stage 1: bi-encoder retrieval
# ---------------------------------------------------------------------------
def load_embedder(device: str):
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(EMBED_MODEL, device=device)


def build_index(embedder, passages: list[dict]):
    """Embed every passage once. In production this is the offline indexing job."""
    texts = [passage_text(p) for p in passages]
    return embedder.encode_document(texts, convert_to_tensor=True, normalize_embeddings=True)


def retrieve(embedder, index, query: str, k: int) -> list[dict]:
    """Cosine similarity between one query vector and every passage vector. O(n) dot products."""
    from sentence_transformers import util
    q = embedder.encode_query(query, convert_to_tensor=True, normalize_embeddings=True)
    hits = util.semantic_search(q, index, top_k=k)[0]
    return [{"idx": h["corpus_id"], "score": float(h["score"])} for h in hits]


# ---------------------------------------------------------------------------
# Stage 2: cross-encoder reranking
# ---------------------------------------------------------------------------
def load_reranker(device: str):
    from sentence_transformers import CrossEncoder
    return CrossEncoder(RERANK_MODEL, device=device)


def rerank(reranker, query: str, candidates: list[dict], passages: list[dict], k: int) -> list[dict]:
    """
    The cross-encoder reads (query, passage) as ONE sequence and outputs a relevance score.
    It sees the interaction between the two texts, which a bi-encoder cannot.
    Cost: one forward pass per candidate, so only run it on the small stage-1 pool.
    """
    texts = [passage_text(passages[c["idx"]]) for c in candidates]
    ranked = reranker.rank(query, texts, top_k=k)          # sorted, highest score first
    return [{"idx": candidates[r["corpus_id"]]["idx"], "score": float(r["score"])} for r in ranked]


# ---------------------------------------------------------------------------
# Stage 3: generation (offline path plus two local backends, one client code path)
# ---------------------------------------------------------------------------
PROMPT_TEMPLATE = """You are a Cordwell Home & Hardware store associate. Answer the customer's question
using ONLY the numbered passages below. If the passages do not contain the answer, say so.
Cite the passage number you used, like [2].

Passages:
{context}

Customer question: {question}
Answer:"""


def build_prompt(query: str, passages: list[dict], picks: list[dict]) -> str:
    context = "\n\n".join(
        f"[{i + 1}] ({passages[p['idx']]['id']}) {passage_text(passages[p['idx']])}"
        for i, p in enumerate(picks)
    )
    return PROMPT_TEMPLATE.format(context=context, question=query)


def generate(prompt: str, passages: list[dict], picks: list[dict]) -> str:
    if LAB_BACKEND == "offline":
        # Deterministic stand-in: quote the top reranked passage. Keeps the demo keyless and network-free.
        top = passages[picks[0]["idx"]]
        return f"(offline extractive answer) {top['text']} [1]"

    if LAB_BACKEND not in BACKEND_URLS:
        raise ValueError(f"LAB_BACKEND must be one of offline, lmstudio, ollama; got {LAB_BACKEND!r}")

    from openai import OpenAI  # both local servers speak the OpenAI chat API
    client = OpenAI(base_url=BACKEND_URLS[LAB_BACKEND], api_key="not-needed")
    try:
        resp = client.chat.completions.create(
            model=LOCAL_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=250,
        )
    except Exception as exc:  # explicit failure, never a silent fallback to another backend
        raise RuntimeError(
            f"{LAB_BACKEND} server not reachable at {BACKEND_URLS[LAB_BACKEND]} "
            f"or model {LOCAL_MODEL!r} not loaded. Start the server or set LAB_BACKEND=offline."
        ) from exc
    return resp.choices[0].message.content.strip()


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def rank_of(gold_ids: list[str], ranked: list[dict], passages: list[dict]) -> int | None:
    for pos, r in enumerate(ranked, start=1):
        if passages[r["idx"]]["id"] in gold_ids:
            return pos
    return None


def print_ranking(label: str, ranked: list[dict], passages: list[dict], gold: list[str], before=None) -> None:
    print(f"\n{label}")
    print(f"{'rank':>4}  {'score':>7}  {'id':<8} {'moved':<7} title")
    for pos, r in enumerate(ranked, start=1):
        p = passages[r["idx"]]
        flag = "  <-- GOLD" if p["id"] in gold else ""
        moved = ""
        if before is not None:
            prev = next((i for i, b in enumerate(before, start=1) if b["idx"] == r["idx"]), None)
            moved = f"{prev:>2} -> {pos:<2}" if prev else "new"
        print(f"{pos:>4}  {r['score']:>7.3f}  {p['id']:<8} {moved:<7} {p['title']}{flag}")


def context_size(picks: list[dict], passages: list[dict]) -> tuple[int, int]:
    words = sum(len(passage_text(passages[p["idx"]]).split()) for p in picks)
    return len(picks), words


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------
def load_models():
    device = pick_device()
    print(f"device: {device}")
    t0 = time.perf_counter()
    embedder = load_embedder(device)
    reranker = load_reranker(device)
    print(f"models loaded in {time.perf_counter() - t0:.1f}s  ({EMBED_MODEL}  +  {RERANK_MODEL})")
    return embedder, reranker


def run_demo(query: str, gold: list[str], passages: list[dict], embedder, reranker, index) -> dict:
    print("=" * 78)
    print(f"QUERY: {query}")
    print("=" * 78)

    t0 = time.perf_counter()
    stage1 = retrieve(embedder, index, query, RETRIEVE_K)
    t1 = time.perf_counter() - t0
    print_ranking(f"STAGE 1  bi-encoder top-{RETRIEVE_K}  (cosine, {t1 * 1000:.0f} ms)", stage1, passages, gold)

    t0 = time.perf_counter()
    stage2 = rerank(reranker, query, stage1, passages, FINAL_K)
    t2 = time.perf_counter() - t0
    print_ranking(f"STAGE 2  cross-encoder top-{FINAL_K}  (logit score, {t2 * 1000:.0f} ms)",
                  stage2, passages, gold, before=stage1)

    naive_picks = stage1[:FINAL_K]
    n1, w1 = context_size(stage1, passages)
    n3, w3 = context_size(stage2, passages)
    r_before = rank_of(gold, stage1, passages)
    r_after = rank_of(gold, stage2, passages)
    print(f"\nGold passage rank:   stage 1 = {r_before or 'miss'}   stage 2 = {r_after or 'miss'}")
    print(f"Context to the LLM:  send all {n1} candidates = {w1} words;  "
          f"send reranked top-{FINAL_K} = {w3} words  ({100 * (1 - w3 / w1):.0f}% smaller)")
    print(f"Naive top-{FINAL_K} without reranking would have contained gold: "
          f"{'yes' if rank_of(gold, naive_picks, passages) else 'NO'}")

    prompt = build_prompt(query, passages, stage2)
    print("\nSTAGE 3  prompt sent to the generator")
    print("-" * 78)
    print(prompt)
    print("-" * 78)
    answer = generate(prompt, passages, stage2)
    print(f"[{LAB_BACKEND}] {answer}\n")

    return {
        "query": query, "gold": gold, "backend": LAB_BACKEND,
        "stage1": [{"id": passages[r["idx"]]["id"], "title": passages[r["idx"]]["title"], "score": r["score"]} for r in stage1],
        "stage2": [{"id": passages[r["idx"]]["id"], "title": passages[r["idx"]]["title"], "score": r["score"]} for r in stage2],
        "gold_rank_before": r_before, "gold_rank_after": r_after,
        "ms_retrieve": round(t1 * 1000), "ms_rerank": round(t2 * 1000),
        "words_all_candidates": w1, "words_reranked": w3, "answer": answer,
    }


def run_eval(passages, gold_queries, embedder, reranker, index) -> list[dict]:
    rows = []
    for g in gold_queries:
        s1 = retrieve(embedder, index, g["query"], RETRIEVE_K)
        s2 = rerank(reranker, g["query"], s1, passages, RETRIEVE_K)  # rerank the full pool so MRR is comparable
        rows.append({"qid": g["qid"], "query": g["query"],
                     "before": rank_of(g["gold"], s1, passages), "after": rank_of(g["gold"], s2, passages)})

    def recall_at(k, key):
        return sum(1 for r in rows if r[key] and r[key] <= k) / len(rows)

    def mrr(key):
        return sum((1 / r[key]) if r[key] else 0 for r in rows) / len(rows)

    print(f"\n{'qid':<4} {'before':>6} {'after':>6}  query")
    for r in rows:
        arrow = "  ^" if (r["after"] or 99) < (r["before"] or 99) else ("  v" if (r["after"] or 99) > (r["before"] or 99) else "")
        print(f"{r['qid']:<4} {str(r['before'] or 'miss'):>6} {str(r['after'] or 'miss'):>6}  {r['query'][:60]}{arrow}")
    print(f"\n{'metric':<12} {'bi-encoder':>11} {'reranked':>10}")
    for name, k in (("Recall@1", 1), ("Recall@3", 3)):
        print(f"{name:<12} {recall_at(k, 'before'):>11.2f} {recall_at(k, 'after'):>10.2f}")
    print(f"{'MRR':<12} {mrr('before'):>11.3f} {mrr('after'):>10.3f}")
    print(f"\n(n = {len(rows)} queries, candidate pool = {RETRIEVE_K}. Small n: read direction, not decimals.)")
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["demo", "eval", "showcase", "replay"])
    ap.add_argument("--query", help="free-text question for the demo command")
    ap.add_argument("--qid", help="gold query id for the demo command, e.g. Q1")
    args = ap.parse_args()

    if LAB_BACKEND not in ("offline", *BACKEND_URLS):
        sys.exit(f"LAB_BACKEND must be one of offline, lmstudio, ollama; got {LAB_BACKEND!r}")
    passages, gold_queries = load_kb()

    if args.command == "replay":
        if not CACHE_PATH.exists():
            sys.exit("no cached run found; run `demo` once with the models available")
        cached = json.loads(CACHE_PATH.read_text())
        print(f"REPLAY of recorded run (backend was {cached['backend']})\nQUERY: {cached['query']}")
        for stage in ("stage1", "stage2"):
            print(f"\n{stage.upper()}")
            for pos, r in enumerate(cached[stage], start=1):
                flag = "  <-- GOLD" if r["id"] in cached["gold"] else ""
                print(f"{pos:>4}  {r['score']:>7.3f}  {r['id']:<8} {r['title']}{flag}")
        print(f"\nGold rank {cached['gold_rank_before']} -> {cached['gold_rank_after']}; "
              f"context {cached['words_all_candidates']} -> {cached['words_reranked']} words")
        print(f"\n{cached['answer']}")
        return

    embedder, reranker = load_models()
    t0 = time.perf_counter()
    index = build_index(embedder, passages)
    print(f"indexed {len(passages)} passages in {time.perf_counter() - t0:.2f}s  (shape {tuple(index.shape)})")

    if args.command == "eval":
        run_eval(passages, gold_queries, embedder, reranker, index)
        return

    if args.command == "showcase":
        rows = run_eval(passages, gold_queries, embedder, reranker, index)
        movers = [r for r in rows if r["before"] and r["after"] and r["after"] < r["before"]]
        if not movers:
            print("\nNo query improved under reranking on this machine. Use `demo --qid Q1` and explore the "
                  "cost and context-size argument instead.")
            return
        best = max(movers, key=lambda r: r["before"] - r["after"])
        print(f"\nSHOWCASE: {best['qid']}  (gold rank {best['before']} -> {best['after']})\n"
              f"  python rerank_demo.py demo --qid {best['qid']}")
        return

    # demo
    if args.query:
        query, gold = args.query, []
    else:
        qid = args.qid or "Q1"
        g = next((g for g in gold_queries if g["qid"] == qid), None)
        if g is None:
            sys.exit(f"unknown qid {qid}; valid: {', '.join(g['qid'] for g in gold_queries)}")
        query, gold = g["query"], g["gold"]

    result = run_demo(query, gold, passages, embedder, reranker, index)
    CACHE_PATH.parent.mkdir(exist_ok=True)
    CACHE_PATH.write_text(json.dumps(result, indent=2))
    print(f"recorded to {CACHE_PATH.relative_to(HERE)}  (fallback: python rerank_demo.py replay)")


if __name__ == "__main__":
    main()
