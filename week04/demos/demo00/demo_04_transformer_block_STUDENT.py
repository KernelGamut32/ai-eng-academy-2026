"""
Demo 04 (STUDENT): The transformer block. Assembling attention into the
repeating unit that gets stacked to make GPT-style models.

You already built raw attention in demo 03, so the attention head here is
provided. You will build what is new in this demo:
  the causal mask, the residual + norm rhythm, and the final vocab projection.

Fill each FILL marker as the instructor covers it. The file stops cleanly
at your first unfilled blank. Solution key: demo_04_transformer_block.py

Untrained weights mean the numbers are meaningless; watch the SHAPES and
the MASK. The shapes are the architecture.

Run: python demo_04_transformer_block_STUDENT.py
Requires numpy.
"""

import numpy as np

np.random.seed(0)
np.set_printoptions(precision=2, suppress=True)

def require(value, fill_num, hint):
    """Stops the script cleanly at the first unfilled blank."""
    if value is Ellipsis:
        print(f"\n>>> Stopped at FILL {fill_num}: {hint}")
        print(">>> Fill it in and rerun. Everything printed above is done.")
        raise SystemExit(0)
    return value

TOKENS = ["order", "status", "for", "invoice", "1042"]
T = len(TOKENS)        # sequence length
D_MODEL = 8            # residual stream width
N_HEAD = 2             # attention heads
D_HEAD = D_MODEL // N_HEAD

X = np.random.randn(T, D_MODEL)
print(f"Input: {T} tokens x {D_MODEL} dims -> X shape {X.shape}")

def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)

print()
print("=" * 70)
print("STEP 1: The causal mask, no peeking at the future")
print("=" * 70)

# FILL 1: build the causal mask.
# Contract: a (T, T) boolean array that is True STRICTLY ABOVE the main
#           diagonal (the future positions to block) and False elsewhere.
#           np.triu with k=1 gives the strict upper triangle; cast the
#           result with .astype(bool).
# Target output: the printed grid below shows X only to the right of the
#           diagonal, '.' on and below it.
mask = np.triu(np.ones((T, T)), k=1).astype(bool)  # FILL 1
mask = require(mask, 1, "boolean strict upper triangle, shape (T, T)")

print("mask", mask)

print("\nMasked positions (X = blocked, . = visible):\n")
print("          " + "".join(f"{t:>9s}" for t in TOKENS))
for i, t in enumerate(TOKENS):
    row = "".join(f"{'X' if mask[i, j] else '.':>9s}" for j in range(T))
    print(f"{t:>10s}{row}")
print("\nEach row is a query token; it may attend to itself and to the left.")

print()
print("=" * 70)
print("STEP 2: Multi-head causal attention")
print("=" * 70)

# Provided: one attention head, exactly your demo 03 code plus two lines
# that apply the mask (blocked cells get -inf, so softmax gives them 0.00).
def causal_attention_head(X, seed):
    rng = np.random.default_rng(seed)
    W_Q = rng.normal(size=(D_MODEL, D_HEAD)) * 0.3
    W_K = rng.normal(size=(D_MODEL, D_HEAD)) * 0.3
    W_V = rng.normal(size=(D_MODEL, D_HEAD)) * 0.3
    Q, K, V = X @ W_Q, X @ W_K, X @ W_V
    scores = Q @ K.T / np.sqrt(D_HEAD)
    scores[mask] = -np.inf          # blocked cells get zero weight after softmax
    w = softmax(scores)
    return w @ V, w

head_outs, head_weights = [], []
for h in range(N_HEAD):
    out, w = causal_attention_head(X, seed=h)
    head_outs.append(out)
    head_weights.append(w)
    print(f"  head {h}: output shape {out.shape}")

print("head_outs", head_outs)
print("head_weights", head_weights)

print(f"\nHead 0 attention weights (note the 0.00 upper triangle, the mask at work):\n")
print(head_weights[0])

# Provided: heads are concatenated, then mixed back to d_model.
concat = np.concatenate(head_outs, axis=-1)
W_O = np.random.randn(D_MODEL, D_MODEL) * 0.3
attn_out = concat @ W_O
print(f"\nconcat heads: {concat.shape} -> project with W_O -> {attn_out.shape}")
print("Why multiple heads: each can learn a different relation. Trained models")
print("grow heads for syntax, for coreference, for copying, and more.")

print()
print("=" * 70)
print("STEP 3: Residual + layer norm, then the MLP")
print("=" * 70)

# Provided: layer norm.
def layer_norm(x, eps=1e-5):
    mu = x.mean(-1, keepdims=True)
    var = x.var(-1, keepdims=True)
    return (x - mu) / np.sqrt(var + eps)

# FILL 2: apply the residual connection and normalize.
# Contract: ADD attn_out to the original input X (do not replace X), then
#           pass the sum through layer_norm. Shape stays (5, 8).
X1 = layer_norm(X + attn_out)  # FILL 2
X1 = require(X1, 2, "layer_norm of X plus attn_out")

print(f"\nAfter attention + residual + norm: {X1.shape}")

# MLP weights provided: expand 4x, nonlinearity, project back.
W1 = np.random.randn(D_MODEL, 4 * D_MODEL) * 0.3
W2 = np.random.randn(4 * D_MODEL, D_MODEL) * 0.3

# FILL 3: the position-wise MLP.
# Contract: multiply X1 by W1, apply ReLU (np.maximum of 0 and the
#           product), then multiply by W2. Same weights hit every token
#           independently. Shape returns to (5, 8).
mlp_out = np.maximum(0, X1 @ W1) @ W2  # FILL 3
mlp_out = require(mlp_out, 3, "ReLU(X1 @ W1) @ W2")

print("mlp_out", mlp_out)

# FILL 4: second residual + norm, same rhythm as FILL 2.
# Contract: layer_norm of X1 plus mlp_out.
X2 = layer_norm(X1 + mlp_out)  # FILL 4
X2 = require(X2, 4, "layer_norm of X1 plus mlp_out")

print("X2", X2)

print(f"MLP expands {D_MODEL} -> {4 * D_MODEL} -> {D_MODEL}; after residual + norm: {X2.shape}")

print("""
Rhythm of one block:
  gather context (attention) -> process it locally (MLP), with residuals
  and norms keeping the signal stable.

Engineering anchor: the residual stream is a shared bus. Each block reads
from the bus, computes a delta, and writes the delta back. Blocks
communicate only through this bus.
""")

print("=" * 70)
print("STEP 4: Stack blocks, project to vocabulary")
print("=" * 70)

# Provided: one full block as a function, so we can stack it.
def block(X, seed):
    rng = np.random.default_rng(seed)
    outs = [causal_attention_head(X, seed * 10 + h)[0] for h in range(N_HEAD)]
    W_O = rng.normal(size=(D_MODEL, D_MODEL)) * 0.3
    X = layer_norm(X + np.concatenate(outs, -1) @ W_O)
    W1 = rng.normal(size=(D_MODEL, 4 * D_MODEL)) * 0.3
    W2 = rng.normal(size=(4 * D_MODEL, D_MODEL)) * 0.3
    return layer_norm(X + np.maximum(0, X @ W1) @ W2)

H = X
for layer in range(3):
    H = block(H, seed=layer + 1)
    print(f"  after block {layer}: {H.shape}   (shape never changes, blocks stack cleanly)")

VOCAB_SIZE = 12
W_unembed = np.random.randn(D_MODEL, VOCAB_SIZE) * 0.3

# FILL 5: project the final hidden states to vocabulary scores.
# Contract: matrix-multiply H (5, 8) by W_unembed (8, 12) to get logits
#           of shape (5, 12): a score for every vocab token at every position.
logits = H @ W_unembed  # FILL 5
logits = require(logits, 5, "H matrix-multiplied by W_unembed")

print("logits", logits)

print(f"\nFinal projection to vocabulary: {H.shape} @ {W_unembed.shape} -> logits {logits.shape}")
print(f"Last row = scores over the vocab for the NEXT token after {TOKENS[-1]!r}:")
print(np.round(logits[-1], 2))

print("""
KEY TAKEAWAY
  A GPT-style model is: embeddings -> N identical blocks -> vocab scores.
  GPT-2 small is this exact recipe at d_model=768, 12 heads, 12 blocks.
  Untrained weights gave us noise scores; demo 06 trains real ones.
""")
