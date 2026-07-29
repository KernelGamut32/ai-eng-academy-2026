"""
lab_support.py
Week 5, Lab 02: Cordwell Support-Doc Retrieval

Pre-written plumbing, so that the lesson concept dominates your active time.

Provided here (do not rewrite these, read them):
    corpus loading and length statistics
    the labeled query set
    both embedding backends and the device selection routine
    the vector store backends (offline, Pinecone Local, Pinecone cloud)
    recall@k and MRR
    the evaluation driver
    plotting
    the soft check harness

You write, in the notebook:
    the chunking functions
    the metadata builder
    the index build call
    the query function
    the coverage audit
    the incremental refresh planner

Implementing MRR is not the lesson. The metric code is given so you spend your
time on the retrievals that feed it.

Verified against: pinecone 9.1.0, numpy 2.4.4, matplotlib 3.10.8, Python 3.12.3.
Runs unchanged on the cohort's Python 3.13.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import math
import os
import re
from typing import Any, Callable, Iterable, Iterator, Sequence
from pathlib import Path

import numpy as np

from cordwell_corpus import LABELED_QUERIES, build_documents

__all__ = [
    "LABELED_QUERIES",
    "load_corpus",
    "pick_device",
    "tokenize",
    "count_tokens",
    "split_paragraphs",
    "to_epoch",
    "content_hash",
    "get_embedder",
    "get_index",
    "recall_at_k",
    "reciprocal_rank",
    "evaluate",
    "length_report",
    "plot_length_distribution",
    "plot_metric_comparison",
    "check",
    "check_summary",
    "step",
    "reset_checks",
    "BACKEND",
    "EMBEDDER",
    "INDEX_NAME",
    "NAMESPACE",
]

# ---------------------------------------------------------------------------
# Configuration.
#
# Explicit selection. If you ask for a backend and it is not reachable, this
# module raises with a clear message rather than silently falling back to
# another one. A lab that quietly switches backends teaches you nothing about
# what your code is actually talking to.
# ---------------------------------------------------------------------------

BACKEND = os.getenv("LAB_BACKEND", "pinecone_local").strip().lower()
EMBEDDER = os.getenv("EMBED_BACKEND", "auto").strip().lower()

INDEX_NAME = os.getenv("CORDWELL_INDEX", "cordwell-support")
NAMESPACE = os.getenv("CORDWELL_NAMESPACE", "support")

PINECONE_LOCAL_HOST = os.getenv("PINECONE_LOCAL_HOST", "http://localhost:5080")

# Path to a sentence-transformers model directory that has already been
# downloaded and staged on this machine. Cohort machines do not download models
# at lab time, so this must point at a local directory, never at a hub name.
ST_MODEL_PATH = os.getenv("CORDWELL_ST_MODEL_PATH", "~/models/all-MiniLM-L6-v2")
ST_MODEL_PATH_B = os.getenv("CORDWELL_ST_MODEL_PATH_B", "~/models/all-mpnet-base-v2")

_VALID_BACKENDS = {"offline", "pinecone_local", "pinecone_cloud"}
if BACKEND not in _VALID_BACKENDS:
    raise ValueError(
        f"LAB_BACKEND={BACKEND!r} is not recognised. "
        f"Choose one of: {sorted(_VALID_BACKENDS)}"
    )


# ---------------------------------------------------------------------------
# Device selection. Never assume CUDA.
# ---------------------------------------------------------------------------


def pick_device() -> str:
    """
    Choose the best available torch device at runtime.

    Order is CUDA, then Apple MPS, then CPU. Cohort machines are Macs with no
    GPU access, so this resolves to mps or cpu there. Never hard-code "cuda".
    """
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
# Text utilities
# ---------------------------------------------------------------------------

_TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9'\-]*")


def tokenize(text: str) -> list[str]:
    """
    Split text into word tokens.

    This is a word tokenizer, not a byte pair encoder. A real BPE tokenizer
    produces roughly 1.3 times as many tokens on English prose, so a 384 "token"
    chunk here is about 500 real model tokens. The ratio is stable enough that
    chunk size decisions made on these counts transfer directly. Using a word
    tokenizer keeps chunk boundaries on word boundaries, which is what you want
    anyway, and removes a model download from the critical path.
    """
    return _TOKEN_RE.findall(text)


def count_tokens(text: str) -> int:
    """Number of word tokens in text."""
    return len(_TOKEN_RE.findall(text))


def split_paragraphs(text: str) -> list[str]:
    """
    Split a document into paragraph units on blank lines.

    These are the natural boundaries the module tells you to prefer. A markdown
    heading travels with the paragraph beneath it because the heading and its
    body are separated by a single newline, not a blank line.
    """
    return [p.strip() for p in text.split("\n\n") if p.strip()]


def to_epoch(iso: str) -> int:
    """
    Convert an ISO 8601 timestamp to an epoch integer.

    Pinecone numeric comparison operators need numbers, not ISO strings. A
    string date does not raise, it simply never matches, which is the worst
    possible failure mode because the query looks like it worked.
    """
    return int(dt.datetime.fromisoformat(iso.replace("Z", "+00:00")).timestamp())


def content_hash(content: str) -> str:
    """
    Stable content fingerprint for change detection.

    SHA-256, not Python's built-in hash(). The built-in is salted per process,
    so it gives a different answer every time the interpreter restarts and is
    useless for deciding whether a document changed since last night's run.
    """
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Corpus
# ---------------------------------------------------------------------------


def load_corpus() -> list[dict]:
    """
    Load the Cordwell support knowledge base.

    Returns a list of document dicts with keys: document_id, source, doc_type,
    product_line, created_at, updated_at, is_active, text.

    Fully synthetic and fully deterministic. Every call returns the same corpus.
    """
    return build_documents()


def length_report(docs: Sequence[dict]) -> dict[str, dict[str, float]]:
    """Token count statistics per doc_type. Used by Part A."""
    buckets: dict[str, list[int]] = {}
    for d in docs:
        buckets.setdefault(d["doc_type"], []).append(count_tokens(d["text"]))
    out: dict[str, dict[str, float]] = {}
    for k, v in sorted(buckets.items()):
        arr = np.array(sorted(v))
        out[k] = {
            "count": int(arr.size),
            "min": int(arr.min()),
            "p50": float(np.percentile(arr, 50)),
            "p90": float(np.percentile(arr, 90)),
            "max": int(arr.max()),
            "mean": float(arr.mean()),
        }
    return out


# ---------------------------------------------------------------------------
# Embedding backends
#
# Both expose the same two-method interface:
#     .dimension  -> int
#     .encode(list[str]) -> np.ndarray of shape (n, dimension), L2 normalized
#
# Critically, an embedder is FIT ONCE and then FROZEN. A real embedding model is
# trained elsewhere, months ago, by someone else. It does not change when you
# change your chunk size. Refitting the embedder per chunking configuration
# would confound the entire experiment: you would no longer be measuring
# chunking, you would be measuring chunking plus a different model.
# ---------------------------------------------------------------------------


class LsaEmbedder:
    """
    A latent semantic analysis embedder: TF-IDF followed by truncated SVD.

    This is a real embedding model in the technical sense. It maps text to a
    fixed length vector of floats, trained so that text about similar things
    lands nearby, and it learns that from word co-occurrence across a corpus. It
    is older and weaker than a transformer bi-encoder, and it is honest about
    being so.

    It matters that the dimension is FIXED and small relative to the vocabulary.
    That fixed capacity is exactly what makes a chunk covering four unrelated
    topics land at a compromise point between them instead of near all four.
    That is the dilution the module is about, and it is why this stand-in
    reproduces the behaviour rather than merely gesturing at it.

    Runs anywhere, downloads nothing, and is deterministic.
    """

    name = "lsa"

    def __init__(self, dim: int = 96, char_ngrams: bool = False) -> None:
        self.requested_dim = dim
        self.char_ngrams = char_ngrams
        self._fitted = False

    # -- feature extraction --------------------------------------------------
    def _features(self, text: str) -> list[str]:
        toks = [t.lower() for t in tokenize(text)]
        if not self.char_ngrams:
            return toks
        out = list(toks)
        for t in toks:
            padded = f"#{t}#"
            out.extend(padded[i : i + 4] for i in range(max(1, len(padded) - 3)))
        return out

    @staticmethod
    def _l2(matrix: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return matrix / norms

    def _bag(self, texts: Sequence[str]) -> np.ndarray:
        counts = np.zeros((len(texts), len(self.vocab)), dtype=np.float64)
        for row, text in enumerate(texts):
            for feat in self._features(text):
                col = self.vocab.get(feat)
                if col is not None:
                    counts[row, col] += 1.0
        weighted = np.log1p(counts) * self.idf
        return self._l2(weighted)

    # -- fit / encode --------------------------------------------------------
    def fit(self, background_texts: Sequence[str]) -> "LsaEmbedder":
        """
        Learn the embedding basis from a background corpus, then freeze it.

        The background corpus here is the raw documents split into paragraphs.
        It is deliberately NOT the chunks you are about to index, because the
        model must not move when your chunking moves.
        """
        docs_features = [self._features(t) for t in background_texts]
        doc_freq: dict[str, int] = {}
        for feats in docs_features:
            for feat in set(feats):
                doc_freq[feat] = doc_freq.get(feat, 0) + 1

        self.vocab = {feat: i for i, feat in enumerate(sorted(doc_freq))}
        n_docs = len(docs_features)
        self.idf = np.zeros(len(self.vocab), dtype=np.float64)
        for feat, col in self.vocab.items():
            self.idf[col] = math.log((n_docs + 1) / (doc_freq[feat] + 1)) + 1.0

        matrix = self._bag(background_texts)
        k = min(self.requested_dim, min(matrix.shape) - 1)
        _u, _s, vt = np.linalg.svd(matrix, full_matrices=False)
        self.components = vt[:k]
        self.dimension = k
        self._fitted = True
        return self

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("LsaEmbedder.encode called before fit")
        if isinstance(texts, str):
            raise TypeError("encode expects a sequence of strings, not a single string")
        projected = self._bag(list(texts)) @ self.components.T
        return self._l2(projected)


class SentenceTransformerEmbedder:
    """
    A transformer bi-encoder loaded from a LOCAL directory.

    Cohort machines do not download models at lab time. The model directory must
    already be staged on disk and its path passed in. Passing a hub name such as
    "sentence-transformers/all-MiniLM-L6-v2" will attempt a network fetch and
    fail, which is why this class refuses anything that is not an existing
    directory.
    """

    name = "sentence_transformers"

    def __init__(self, model_path: str, device: str | None = None) -> None:
        if not model_path:
            raise ValueError(
                "No model path set. Export CORDWELL_ST_MODEL_PATH to the local "
                "directory holding the staged model, for example "
                "./models/all-MiniLM-L6-v2. This lab never downloads a model."
            )
        if not os.path.isdir(model_path):
            raise FileNotFoundError(
                f"CORDWELL_ST_MODEL_PATH={model_path!r} is not a directory on this "
                "machine. Stage the model locally first. A hub name will not work: "
                "downloads are disabled for this lab."
            )
        # Hard-disable any hub access so a misconfiguration fails loudly and
        # immediately rather than hanging on a network timeout mid-lab.
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

        from sentence_transformers import SentenceTransformer

        self.device = device or pick_device()
        self.model = SentenceTransformer(model_path, device=self.device)
        self.dimension = int(self.model.get_sentence_embedding_dimension())

    def fit(self, background_texts: Sequence[str]) -> "SentenceTransformerEmbedder":
        """No-op. A pretrained bi-encoder is already fit. Present for interface parity."""
        return self

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        vectors = self.model.encode(
            list(texts),
            batch_size=32,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.asarray(vectors, dtype=np.float64)


def get_embedder(
    variant: str = "primary",
    dim: int = 96,
    background: Sequence[str] | None = None,
) -> LsaEmbedder | SentenceTransformerEmbedder:
    """
    Build and fit the embedder for the selected backend.

    variant is "primary" or "alternate". The alternate is a genuinely different
    embedding space with a different dimension, used in the Part G model swap.
    Vectors from the two cannot be compared. That is the point of Part G.

    background is the corpus used to fit an LSA basis. It is ignored by the
    transformer path, which is already trained.
    """
    if variant not in {"primary", "alternate"}:
        raise ValueError(f"variant must be 'primary' or 'alternate', got {variant!r}")

    resolved = EMBEDDER
    if resolved == "auto":
        resolved = "lsa" if BACKEND == "offline" else "sentence_transformers"

    if resolved == "sentence_transformers":
        path = Path(ST_MODEL_PATH if variant == "primary" else (ST_MODEL_PATH_B or ST_MODEL_PATH)).expanduser().resolve()
        return SentenceTransformerEmbedder(str(path))

    if resolved != "lsa":
        raise ValueError(
            f"EMBED_BACKEND={EMBEDDER!r} is not recognised. "
            "Choose auto, lsa, or sentence_transformers."
        )

    if background is None:
        docs = load_corpus()
        background = [p for d in docs for p in split_paragraphs(d["text"]) if count_tokens(p) >= 8]

    if variant == "primary":
        return LsaEmbedder(dim=dim, char_ngrams=False).fit(background)
    # The alternate uses character n-gram features as well as words, which
    # produces a different vocabulary, a different geometry, and a different
    # dimension. It is a different model, not the same model resized.
    return LsaEmbedder(dim=max(32, dim // 2), char_ngrams=True).fit(background)


# ---------------------------------------------------------------------------
# Vector store backends
#
# The offline backend mirrors the Pinecone v9 data plane surface exactly, so the
# code you write in the notebook is byte-identical across all three backends.
# Only the connection differs. That is the whole point of using an emulator or a
# faithful stand-in rather than a lookalike database with its own API.
# ---------------------------------------------------------------------------


class _Match:
    """Mirrors pinecone.ScoredVector for the fields this lab reads."""

    __slots__ = ("id", "score", "metadata", "values")

    def __init__(self, id: str, score: float, metadata: dict, values=None) -> None:
        self.id = id
        self.score = score
        self.metadata = metadata
        self.values = values

    def __repr__(self) -> str:
        return f"Match(id={self.id!r}, score={self.score:.4f})"


class _QueryResponse:
    """Mirrors pinecone.QueryResponse for the fields this lab reads."""

    __slots__ = ("matches", "namespace")

    def __init__(self, matches: list[_Match], namespace: str) -> None:
        self.matches = matches
        self.namespace = namespace


class _ListItem:
    """Mirrors pinecone.ListItem. Iterating a list page yields these, not strings."""

    __slots__ = ("id",)

    def __init__(self, id: str) -> None:
        self.id = id


class _ListPage:
    """Mirrors pinecone.ListResponse: iterable, yielding _ListItem."""

    def __init__(self, ids: list[str]) -> None:
        self.vectors = [_ListItem(i) for i in ids]

    def __iter__(self) -> Iterator[_ListItem]:
        return iter(self.vectors)

    def __len__(self) -> int:
        return len(self.vectors)


class _Stats:
    __slots__ = ("namespaces", "dimension", "total_vector_count", "metric")

    def __init__(self, namespaces, dimension, total, metric) -> None:
        self.namespaces = namespaces
        self.dimension = dimension
        self.total_vector_count = total
        self.metric = metric

    def __repr__(self) -> str:
        return (
            f"IndexStats(total_vector_count={self.total_vector_count}, "
            f"dimension={self.dimension}, metric={self.metric!r}, "
            f"namespaces={ {k: v['vector_count'] for k, v in self.namespaces.items()} })"
        )


def _matches_filter(metadata: dict, flt: dict | None) -> bool:
    """
    Evaluate a Pinecone metadata filter against one record.

    Supports the operators this lab uses: $eq, $ne, $gt, $gte, $lt, $lte, $in,
    $nin, $exists, plus $and and $or. A bare value is treated as $eq, which is
    how Pinecone treats it.

    Note what happens with a string on a numeric field: the comparison is simply
    False. No exception. That is precisely the silent failure the module warns
    about, and it is reproduced here rather than smoothed over.
    """
    if not flt:
        return True

    for key, condition in flt.items():
        if key == "$and":
            if not all(_matches_filter(metadata, sub) for sub in condition):
                return False
            continue
        if key == "$or":
            if not any(_matches_filter(metadata, sub) for sub in condition):
                return False
            continue

        actual = metadata.get(key)
        if not isinstance(condition, dict):
            condition = {"$eq": condition}

        for op, expected in condition.items():
            try:
                if op == "$eq":
                    ok = actual == expected
                elif op == "$ne":
                    ok = actual != expected
                elif op == "$gt":
                    ok = actual is not None and actual > expected
                elif op == "$gte":
                    ok = actual is not None and actual >= expected
                elif op == "$lt":
                    ok = actual is not None and actual < expected
                elif op == "$lte":
                    ok = actual is not None and actual <= expected
                elif op == "$in":
                    ok = actual in expected
                elif op == "$nin":
                    ok = actual not in expected
                elif op == "$exists":
                    ok = (key in metadata) == bool(expected)
                else:
                    raise ValueError(f"unsupported filter operator: {op}")
            except TypeError:
                # Comparing a string to an int, for example. Pinecone does not
                # raise here either. The record simply does not match.
                ok = False
            if not ok:
                return False
    return True


class OfflineIndex:
    """
    An in-memory vector index with the Pinecone v9 data plane surface.

    Exact brute force search rather than approximate nearest neighbour. At a few
    hundred vectors, exact search is instant and removes recall variance from
    the ANN index itself, so the numbers you measure are attributable to your
    chunking rather than to index internals.
    """

    def __init__(self, dimension: int, metric: str = "cosine") -> None:
        self.dimension = dimension
        self.metric = metric
        self._ns: dict[str, dict[str, tuple[np.ndarray, dict]]] = {}

    # -- writes --------------------------------------------------------------
    def upsert(self, *, vectors: Sequence[dict], namespace: str = "", **_: Any):
        store = self._ns.setdefault(namespace, {})
        for record in vectors:
            values = np.asarray(record["values"], dtype=np.float64)
            if values.shape[0] != self.dimension:
                raise ValueError(
                    f"Vector dimension {values.shape[0]} does not match index "
                    f"dimension {self.dimension}. This is the dimension coupling the "
                    "module warns about: the index dimension is fixed at creation "
                    "and every vector must match it exactly."
                )
            store[record["id"]] = (values, dict(record.get("metadata") or {}))
        return type("UpsertResponse", (), {"upserted_count": len(vectors)})()

    def delete(
        self,
        *,
        ids: Sequence[str] | None = None,
        delete_all: bool = False,
        filter: dict | None = None,
        namespace: str = "",
        **_: Any,
    ) -> None:
        store = self._ns.setdefault(namespace, {})
        if delete_all:
            store.clear()
            return
        if filter is not None:
            raise RuntimeError(
                "Delete by metadata filter is not supported on this backend. "
                "Pinecone cloud added it in late 2025, but Pinecone Local pins API "
                "version 2025-01, which predates it. Delete by ID prefix instead, "
                "which works everywhere and is what your ID scheme is for."
            )
        for vector_id in ids or []:
            store.pop(vector_id, None)

    # -- reads ---------------------------------------------------------------
    def query(
        self,
        *,
        top_k: int,
        vector: Sequence[float] | None = None,
        namespace: str = "",
        filter: dict | None = None,
        include_metadata: bool = False,
        include_values: bool = False,
        **_: Any,
    ) -> _QueryResponse:
        store = self._ns.get(namespace, {})
        if not store or vector is None:
            return _QueryResponse([], namespace)

        query_vector = np.asarray(vector, dtype=np.float64)
        if query_vector.shape[0] != self.dimension:
            raise ValueError(
                f"Query vector dimension {query_vector.shape[0]} does not match "
                f"index dimension {self.dimension}."
            )

        ids: list[str] = []
        rows: list[np.ndarray] = []
        metas: list[dict] = []
        for vector_id, (values, metadata) in store.items():
            if _matches_filter(metadata, filter):
                ids.append(vector_id)
                rows.append(values)
                metas.append(metadata)
        if not ids:
            return _QueryResponse([], namespace)

        matrix = np.vstack(rows)
        if self.metric == "cosine":
            denom = np.linalg.norm(matrix, axis=1) * np.linalg.norm(query_vector)
            denom[denom == 0] = 1.0
            scores = (matrix @ query_vector) / denom
        elif self.metric == "dotproduct":
            scores = matrix @ query_vector
        elif self.metric == "euclidean":
            scores = -np.linalg.norm(matrix - query_vector, axis=1)
        else:
            raise ValueError(f"unsupported metric: {self.metric}")

        order = np.argsort(-scores)[:top_k]
        matches = [
            _Match(
                ids[i],
                float(scores[i]),
                metas[i] if include_metadata else {},
                rows[i].tolist() if include_values else None,
            )
            for i in order
        ]
        return _QueryResponse(matches, namespace)

    def list(
        self,
        *,
        prefix: str | None = None,
        limit: int | None = None,
        namespace: str = "",
        **_: Any,
    ) -> Iterator[_ListPage]:
        """Paginated ID listing. Yields pages; iterating a page yields ListItem objects."""
        store = self._ns.get(namespace, {})
        matching = sorted(k for k in store if prefix is None or k.startswith(prefix))
        page_size = limit or 100
        for start in range(0, len(matching), page_size):
            yield _ListPage(matching[start : start + page_size])

    def describe_index_stats(self, **_: Any) -> _Stats:
        namespaces = {
            name: {"vector_count": len(store)} for name, store in self._ns.items()
        }
        total = sum(len(s) for s in self._ns.values())
        return _Stats(namespaces, self.dimension, total, self.metric)


def _connect_pinecone(dimension: int, metric: str, recreate: bool):
    """
    Connect to Pinecone and return a data plane index handle.

    This is the current SDK 9.x shape. It is materially different from the code
    printed on the Module 02 slides, which targets SDK 8 and earlier:

      pc.create_index    ->  pc.indexes.create
      pc.has_index       ->  pc.indexes.exists
      pc.describe_index  ->  pc.indexes.describe
      pc.delete_index    ->  pc.indexes.delete
      pc.Index(...)      ->  pc.index(...)
      GRPCClientConfig   ->  removed entirely in v9

    TLS is now selected by the URL scheme. normalize_host in the SDK preserves an
    http:// prefix and prepends https:// to anything else, so prefixing the
    emulator host with http:// is how you disable TLS. There is no secure=False
    flag any more.
    """
    from pinecone import Pinecone, ServerlessSpec

    if BACKEND == "pinecone_local":
        client = Pinecone(api_key="pclocal", host=PINECONE_LOCAL_HOST)
    else:
        api_key = os.getenv("PINECONE_API_KEY", "")
        if not api_key:
            raise RuntimeError(
                "LAB_BACKEND=pinecone_cloud but PINECONE_API_KEY is not set. "
                "Export it before launching Jupyter."
            )
        client = Pinecone(api_key=api_key)

    if recreate and client.indexes.exists(INDEX_NAME):
        client.indexes.delete(INDEX_NAME)

    if not client.indexes.exists(INDEX_NAME):
        client.indexes.create(
            name=INDEX_NAME,
            dimension=dimension,
            metric=metric,
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            # Pinecone Local ignores cloud and region but accepts them, so the
            # same call runs unchanged against both the emulator and production.
        )

    described = client.indexes.describe(INDEX_NAME)
    if BACKEND == "pinecone_local":
        # describe returns a bare host:port for the emulator. The http:// prefix
        # is what tells the SDK not to attempt a TLS handshake.
        host = described.host.replace("https://", "http://")
        return client.index(host=host)
    return client.index(name=INDEX_NAME)


def get_index(dimension: int, metric: str = "cosine", recreate: bool = True):
    """
    Return a data plane index handle for the selected backend.

    Every call you make after this point is identical across all three backends:
    upsert, query, list, delete, describe_index_stats.
    """
    if BACKEND == "offline":
        return OfflineIndex(dimension=dimension, metric=metric)
    return _connect_pinecone(dimension, metric, recreate)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def recall_at_k(retrieved_ids: Sequence[str], relevant_ids: set[str], k: int) -> float:
    """
    Fraction of the relevant documents that appear in the top k.

    Answers "did we find it at all". For retrieval augmented generation this is
    the metric that matters most, because if the right chunk is not in the top k
    then nothing downstream can recover it.
    """
    if not relevant_ids:
        return 0.0
    return len(set(retrieved_ids[:k]) & relevant_ids) / len(relevant_ids)


def reciprocal_rank(retrieved_ids: Sequence[str], relevant_ids: set[str]) -> float:
    """
    One over the rank of the first relevant result, or zero if none was found.

    Answers "was it near the top".
    """
    for rank, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in relevant_ids:
            return 1.0 / rank
    return 0.0


def evaluate(
    queries: Sequence[dict],
    search_fn: Callable[..., Sequence[dict]],
    k: int = 5,
    per_query: bool = False,
) -> dict:
    """
    Run the labeled query set through a search function and score it.

    search_fn(query_text, top_k=k) must return a list of dicts, each carrying at
    least a "document_id" key.

    Chunk hits are deduplicated up to their document before scoring, preserving
    rank order. Without that, three chunks from the same manual would count as
    three separate finds and inflate recall.
    """
    recalls: list[float] = []
    rrs: list[float] = []
    detail: dict[str, dict] = {}

    for query in queries:
        hits = search_fn(query["text"], top_k=k)

        seen: set[str] = set()
        doc_ids: list[str] = []
        for hit in hits:
            doc_id = hit["document_id"]
            if doc_id not in seen:
                seen.add(doc_id)
                doc_ids.append(doc_id)

        r = recall_at_k(doc_ids, query["relevant"], k)
        rr = reciprocal_rank(doc_ids, query["relevant"])
        recalls.append(r)
        rrs.append(rr)
        if per_query:
            detail[query["query_id"]] = {
                "text": query["text"],
                "relevant": sorted(query["relevant"]),
                "retrieved": doc_ids[:k],
                "recall": r,
                "rr": rr,
            }

    result = {
        f"recall@{k}": sum(recalls) / len(recalls),
        "mrr": sum(rrs) / len(rrs),
        "n_queries": len(queries),
    }
    if per_query:
        result["per_query"] = detail
    return result


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_length_distribution(docs: Sequence[dict]):
    """Histogram of document token counts, split by doc_type, on a log x axis."""
    import matplotlib.pyplot as plt

    by_type: dict[str, list[int]] = {}
    for d in docs:
        by_type.setdefault(d["doc_type"], []).append(count_tokens(d["text"]))

    fig, ax = plt.subplots(figsize=(9, 4.5))
    bins = np.logspace(1.5, 4.0, 40)
    for doc_type, lengths in sorted(by_type.items()):
        ax.hist(lengths, bins=bins, alpha=0.65, label=f"{doc_type} (n={len(lengths)})")
    ax.set_xscale("log")
    ax.set_xlabel("document length in word tokens (log scale)")
    ax.set_ylabel("documents")
    ax.set_title("Cordwell support corpus: length distribution by document type")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    return fig, ax


def plot_metric_comparison(results: dict[str, dict], k: int = 5):
    """Grouped bar chart of recall@k and MRR across named configurations."""
    import matplotlib.pyplot as plt

    labels = list(results)
    recalls = [results[name][f"recall@{k}"] for name in labels]
    mrrs = [results[name]["mrr"] for name in labels]

    x = np.arange(len(labels))
    width = 0.38
    fig, ax = plt.subplots(figsize=(max(8, 1.9 * len(labels)), 4.6))
    bars_r = ax.bar(x - width / 2, recalls, width, label=f"recall@{k}")
    bars_m = ax.bar(x + width / 2, mrrs, width, label="MRR")
    for group in (bars_r, bars_m):
        for bar in group:
            ax.annotate(
                f"{bar.get_height():.3f}",
                (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                textcoords="offset points",
                xytext=(0, 3),
                ha="center",
                fontsize=8,
            )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=18, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("score")
    ax.set_title("Retrieval quality by configuration")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    return fig, ax


# ---------------------------------------------------------------------------
# Soft check harness
#
# Checks report and move on. They never raise. A cold Run All on the student
# notebook produces a wall of FAIL and zero crashes, which means you can always
# see the whole board.
# ---------------------------------------------------------------------------

_CHECKS: list[tuple[str, str, str]] = []


class step:
    """
    Guard a provided driver cell so a cold Run All never throws a traceback.

    Used only around cells that call functions you have not written yet. It
    prints one line and moves on, so the check board at the bottom stays
    readable instead of being buried under stack traces.

    Usage:
        with step("Part D: build the sliding window index"):
            ...
    """

    def __init__(self, name: str) -> None:
        self.name = name

    def __enter__(self) -> "step":
        return self

    def __exit__(self, exc_type, exc, _tb) -> bool:
        if exc_type is None:
            return False
        if issubclass(exc_type, NotImplementedError):
            print(f"  [skipped] {self.name}: waiting on {exc}")
        elif issubclass(exc_type, (NameError, KeyError, TypeError, AttributeError)):
            print(f"  [skipped] {self.name}: {exc_type.__name__}: {exc}")
        else:
            print(f"  [skipped] {self.name}: {exc_type.__name__}: {exc}")
        return True  # suppress


def reset_checks() -> None:
    _CHECKS.clear()


def check(name: str, condition: Any, detail: str = "") -> bool:
    """
    Record one check. Never raises.

    A stub that raises NotImplementedError is reported as STUB rather than FAIL,
    so you can tell "not written yet" apart from "written and wrong".
    """
    try:
        passed = bool(condition() if callable(condition) else condition)
        status = "PASS" if passed else "FAIL"
    except NotImplementedError:
        status, passed = "STUB", False
        detail = detail or "not implemented yet"
    except Exception as exc:  # noqa: BLE001
        status, passed = "ERROR", False
        detail = f"{type(exc).__name__}: {exc}"

    _CHECKS.append((name, status, detail))
    marker = {"PASS": "PASS ", "FAIL": "FAIL ", "STUB": "STUB ", "ERROR": "ERR  "}[status]
    line = f"  {marker} {name}"
    if detail and status != "PASS":
        line += f"   [{detail}]"
    print(line)
    return passed


def check_summary() -> dict[str, int]:
    """Print and return the running tally."""
    tally = {"PASS": 0, "FAIL": 0, "STUB": 0, "ERROR": 0}
    for _name, status, _detail in _CHECKS:
        tally[status] += 1
    total = len(_CHECKS)
    print("=" * 62)
    print(
        f"  {tally['PASS']} passed, {tally['FAIL']} failed, "
        f"{tally['STUB']} not implemented, {tally['ERROR']} errored, "
        f"out of {total}"
    )
    print("=" * 62)
    if tally["FAIL"] or tally["ERROR"] or tally["STUB"]:
        print("  Still open:")
        for name, status, detail in _CHECKS:
            if status != "PASS":
                suffix = f"   [{detail}]" if detail else ""
                print(f"    {status:5s} {name}{suffix}")
    return tally
