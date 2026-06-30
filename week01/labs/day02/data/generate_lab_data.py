"""
generate_lab_data.py
=====================
Reproducible synthetic data generator for the Week 1 / Day 2 NumPy labs.

Run:
    python generate_lab_data.py            # writes all .npy files next to this script
    python generate_lab_data.py --check    # regenerate to /tmp and diff against committed files

Design notes
------------
* Everything uses the MODERN NumPy 2.x RNG API: numpy.random.default_rng(seed).
  No np.random.seed(), no np.random.rand() / randn(). Each array gets its OWN
  seeded generator so the files are independent and individually reproducible.
* All dtypes are explicit (NEP 50 discipline — no implicit promotion).
* The data is clearly fictional and AI-engineering flavoured (eval scores,
  request telemetry, token records) to match the Day 2 deck. No real systems
  or real customer data are referenced.

Target environment: Python 3.13, NumPy 2.x (tested on 2.4 / 2.5).
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


# --------------------------------------------------------------------------- #
# Lab 1 — The ndarray: shape, dtype, memory, save/load
# --------------------------------------------------------------------------- #
def make_lab1_image_batch() -> np.ndarray:
    """A small batch of 8-bit 'grayscale glyphs' (MNIST-shaped, but synthetic).

    Shape (64, 28, 28), dtype uint8. Used to reason about shape/ndim/size,
    nbytes, reshape to (64, 784), and C- vs F-contiguity.
    """
    rng = np.random.default_rng(seed=101)
    # Smooth-ish blobs so the images are not pure noise: low-res random field
    # upsampled by nearest-neighbour repeat, then jittered.
    coarse = rng.integers(0, 256, size=(64, 7, 7), dtype=np.uint8)
    big = np.repeat(np.repeat(coarse, 4, axis=1), 4, axis=2)  # (64, 28, 28)
    jitter = rng.integers(0, 24, size=big.shape, dtype=np.uint8)
    images = (big // 2 + jitter).astype(np.uint8)
    assert images.shape == (64, 28, 28) and images.dtype == np.uint8
    return images


def make_lab1_layer_weights() -> np.ndarray:
    """A synthetic dense-layer weight matrix.

    Shape (256, 128), dtype float64. Used to demonstrate the float64 -> float32
    memory halving and the casting= rules of astype().
    """
    rng = np.random.default_rng(seed=102)
    # Glorot-ish scaling just so the numbers look like real weights.
    fan_in, fan_out = 256, 128
    scale = np.sqrt(2.0 / (fan_in + fan_out))
    weights = (rng.standard_normal((fan_in, fan_out)) * scale).astype(np.float64)
    return weights


# --------------------------------------------------------------------------- #
# Lab 2 — Vectorization & broadcasting
# --------------------------------------------------------------------------- #
def make_lab2_request_features() -> np.ndarray:
    """Per-request telemetry for a fictional LLM gateway.

    Shape (500, 6), dtype float64. Columns are on DELIBERATELY different scales
    so z-score normalisation is clearly necessary. Column 0 contains a handful
    of slightly-negative values (sensor noise) to motivate np.where clipping.

    Columns:
        0 latency_ms        ~ N(120, 40), a few negative outliers injected
        1 input_tokens      integers in [50, 2000]
        2 output_tokens     integers in [10, 800]
        3 retrieved_docs    integers in [0, 20]
        4 prompt_cost_usd   small positive floats
        5 similarity_top1   U[0, 1)
    """
    rng = np.random.default_rng(seed=202)
    n = 500
    latency = rng.normal(120.0, 40.0, size=n)
    # Inject 8 negative "noise" readings to be clipped later.
    neg_idx = rng.choice(n, size=8, replace=False)
    latency[neg_idx] = rng.uniform(-15.0, -0.5, size=8)

    input_tokens = rng.integers(50, 2001, size=n).astype(np.float64)
    output_tokens = rng.integers(10, 801, size=n).astype(np.float64)
    retrieved_docs = rng.integers(0, 21, size=n).astype(np.float64)
    prompt_cost = (input_tokens * 3e-6 + output_tokens * 6e-6).astype(np.float64)
    similarity = rng.random(n)

    features = np.column_stack(
        [latency, input_tokens, output_tokens, retrieved_docs, prompt_cost, similarity]
    ).astype(np.float64)
    assert features.shape == (500, 6)
    return features


# --------------------------------------------------------------------------- #
# Lab 3 — Indexing, selection, views vs copies
# --------------------------------------------------------------------------- #
def make_lab3_eval_scores() -> np.ndarray:
    """Composite eval scores in [0, 1] for 240 model responses.

    Shape (240,), dtype float64. Used for slicing, boolean masks, compound
    conditions, and fancy indexing.
    """
    rng = np.random.default_rng(seed=303)
    # Beta-ish skew toward higher scores, but with a clear low tail.
    scores = rng.beta(5.0, 2.5, size=240).astype(np.float64)
    return scores


def make_lab3_shuffle_index() -> np.ndarray:
    """A precomputed permutation of 0..239.

    Shape (240,), dtype int64. Lets students perform a reproducible
    fancy-index shuffle WITHOUT needing the RNG API (that arrives in Lab 5).
    """
    rng = np.random.default_rng(seed=313)
    return rng.permutation(240).astype(np.int64)


def make_lab3_category_codes() -> np.ndarray:
    """Integer category codes (0..4) for each of the 240 responses.

    Shape (240,), dtype int8. Pairs with an inline vocabulary array to
    demonstrate lookup-table fancy indexing: CATEGORIES[codes].
    """
    rng = np.random.default_rng(seed=323)
    return rng.integers(0, 5, size=240, dtype=np.int8)


# --------------------------------------------------------------------------- #
# Lab 4 — Reductions, aggregations, sorting
# --------------------------------------------------------------------------- #
def make_lab4_metric_matrix() -> np.ndarray:
    """Per-response eval metrics with missing values.

    Shape (200, 4), dtype float64. Columns: relevance, faithfulness, fluency,
    coherence — all in [0, 1]. ~5% of cells are set to NaN to force the use of
    NaN-safe reductions.
    """
    rng = np.random.default_rng(seed=404)
    raw = rng.random((200, 4))
    nan_mask = rng.random((200, 4)) < 0.05
    raw[nan_mask] = np.nan
    return raw.astype(np.float64)


def make_lab4_records() -> np.ndarray:
    """A small structured ('record') array — like a table schema.

    Length 12. Fields: token_id (int32), score (float32), label (<U8).
    Used for the structured-array stretch goal and field-based sorting.
    """
    rng = np.random.default_rng(seed=414)
    dtype = np.dtype([("token_id", np.int32), ("score", np.float32), ("label", "U8")])
    labels = np.array(["positive", "neutral", "negative"])
    n = 12
    token_ids = rng.integers(100, 999, size=n).astype(np.int32)
    scores = rng.random(n).astype(np.float32)
    label_codes = rng.integers(0, 3, size=n)
    records = np.empty(n, dtype=dtype)
    records["token_id"] = token_ids
    records["score"] = scores
    records["label"] = labels[label_codes]
    return records


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
ARTIFACTS = {
    "lab1_image_batch.npy": make_lab1_image_batch,
    "lab1_layer_weights.npy": make_lab1_layer_weights,
    "lab2_request_features.npy": make_lab2_request_features,
    "lab3_eval_scores.npy": make_lab3_eval_scores,
    "lab3_shuffle_index.npy": make_lab3_shuffle_index,
    "lab3_category_codes.npy": make_lab3_category_codes,
    "lab4_metric_matrix.npy": make_lab4_metric_matrix,
    "lab4_records.npy": make_lab4_records,
}


def build(target_dir: str) -> None:
    os.makedirs(target_dir, exist_ok=True)
    for name, fn in ARTIFACTS.items():
        arr = fn()
        path = os.path.join(target_dir, name)
        np.save(path, arr, allow_pickle=False)
        print(f"  wrote {name:<28} shape={arr.shape!s:<14} dtype={arr.dtype}")


def check() -> int:
    """Regenerate to /tmp and confirm byte-for-byte stability."""
    import tempfile

    tmp = tempfile.mkdtemp()
    build(tmp)
    ok = True
    for name in ARTIFACTS:
        a = np.load(os.path.join(HERE, name), allow_pickle=False)
        b = np.load(os.path.join(tmp, name), allow_pickle=False)
        meta = a.dtype == b.dtype and a.shape == b.shape
        if a.dtype.names:  # structured array: equal_nan is not supported
            same = meta and np.array_equal(a, b)
        else:
            same = meta and np.array_equal(a, b, equal_nan=True)
        print(f"  {name:<28} {'OK' if same else 'MISMATCH'}")
        ok = ok and same
    return 0 if ok else 1


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--check", action="store_true", help="verify reproducibility")
    args = p.parse_args()
    print(f"NumPy {np.__version__}")
    if args.check:
        sys.exit(check())
    build(HERE)
    print("Done.")
