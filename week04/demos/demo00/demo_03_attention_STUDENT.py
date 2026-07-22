"""
Demo 03 (STUDENT): Attention. How a token gathers information from the
rest of the sentence. This is the heart of the transformer.

  Query  (Q): what this token is looking for
  Key    (K): what each token advertises about itself
  Value  (V): the payload each token hands over if selected

You will implement the four moves of attention:
  project -> score -> softmax -> weighted sum

Fill each FILL marker as the instructor covers it. The file stops cleanly
at your first unfilled blank. Solution key: demo_03_attention.py

Run: python demo_03_attention_STUDENT.py
Requires numpy.
"""

import numpy as np

np.set_printoptions(precision=2, suppress=True)

def require(value, fill_num, hint):
    """Stops the script cleanly at the first unfilled blank."""
    if value is Ellipsis:
        print(f"\n>>> Stopped at FILL {fill_num}: {hint}")
        print(">>> Fill it in and rerun. Everything printed above is done.")
        raise SystemExit(0)
    return value

# ---------------------------------------------------------------------------
# Provided: scenario sentence from a Cordwell return ticket. The question
# we want the model to resolve: what does 'it' refer to?
# ---------------------------------------------------------------------------
TOKENS = ["the", "drill", "arrived", "and", "it", "was", "defective"]

# Hand-set 4-dim vectors. Dimension meanings (for our toy only):
#   [is_object, is_pronoun, is_action, is_filler]
E = {
    "the":       [0.0, 0.0, 0.0, 1.0],
    "drill":     [1.0, 0.0, 0.0, 0.0],
    "arrived":   [0.0, 0.0, 1.0, 0.0],
    "and":       [0.0, 0.0, 0.0, 1.0],
    "it":        [0.0, 1.0, 0.0, 0.0],
    "was":       [0.0, 0.0, 0.5, 0.5],
    "defective": [0.2, 0.0, 1.0, 0.0],
}
X = np.array([E[t] for t in TOKENS], dtype=float)  # shape (7, 4)

print("=" * 70)
print("STEP 1: Q, K, V are three learned projections of the same input")
print("=" * 70)

# Provided: a projection matrix that makes pronouns SEARCH FOR objects.
# It maps the 'is_pronoun' direction onto the 'is_object' direction.
# In a trained model these matrices are learned; we hand-set them so the
# resulting pattern is readable.
W_Q = np.array([
    [1.0, 0.0, 0.0, 0.0],
    [4.0, 0.0, 0.0, 0.0],   # pronoun feature -> strong query for objects
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 0.2],
])
W_K = np.eye(4)      # keys: advertise your own features unchanged
W_V = np.eye(4)      # values: hand over your own features unchanged

# FILL 1, 2, 3: project the input into queries, keys, and values.
# Contract: matrix-multiply X (shape (7, 4)) by the matching projection.
#           All three results have shape (7, 4): one Q, K, V row per token.
Q = ...  # FILL 1
Q = require(Q, 1, "X matrix-multiplied by W_Q")
K = ...  # FILL 2
K = require(K, 2, "X matrix-multiplied by W_K")
V = ...  # FILL 3
V = require(V, 3, "X matrix-multiplied by W_V")

print(f"\nInput X: {X.shape}   Q: {Q.shape}   K: {K.shape}   V: {V.shape}")
print("\nIn a trained model, W_Q, W_K, W_V are learned. We hand-wrote W_Q so")
print("that a pronoun's query points in the same direction as object keys.")

print()
print("=" * 70)
print("STEP 2: Scores = how well each query matches each key")
print("=" * 70)

d_k = K.shape[1]

# FILL 4: score every query against every key.
# Contract: multiply Q by the transpose of K, then divide by the square
#           root of d_k (the scale keeps softmax gradients healthy).
#           Result shape (7, 7): row = query token, column = key token.
scores = ...  # FILL 4
scores = require(scores, 4, "Q times K-transpose, scaled by sqrt(d_k)")

print(f"\nscores = Q @ K.T / sqrt(d_k), shape {scores.shape} (query row, key column)\n")
header = "          " + "".join(f"{t:>10s}" for t in TOKENS)
print(header)
for i, t in enumerate(TOKENS):
    print(f"{t:>10s}" + "".join(f"{s:10.2f}" for s in scores[i]))

print()
print("=" * 70)
print("STEP 3: Softmax turns scores into weights that sum to 1")
print("=" * 70)

# Provided: numerically stable softmax.
def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)

# FILL 5: convert scores to attention weights.
# Contract: apply the provided softmax to `scores`. Each ROW of the result
#           sums to 1.0: it is one token's attention budget over the sentence.
weights = ...  # FILL 5
weights = require(weights, 5, "softmax applied to the scores matrix")

print("\nAttention weights (each row sums to 1.0):\n")
print(header)
for i, t in enumerate(TOKENS):
    print(f"{t:>10s}" + "".join(f"{w:10.2f}" for w in weights[i]))

it_row = weights[TOKENS.index("it")]
best = TOKENS[int(it_row.argmax())]
print(f"\nRead the 'it' row: its largest weight ({it_row.max():.2f}) is on {best!r}.")
print("The pronoun found its referent. This is what 'attends to' means.")

# Target output check: the 'it' row should read 0.53 on 'drill' and
# roughly 0.07 to 0.11 everywhere else. If not, revisit FILL 4 and 5.

# STOP AND DISCUSS:
# What would happen to this row if the sentence had two objects,
# 'the drill and the sander arrived and it was defective'?
# (Ambiguity shows up as split attention weight.)

print()
print("=" * 70)
print("STEP 4: Output = attention-weighted average of the values")
print("=" * 70)

# FILL 6: produce the attention output.
# Contract: matrix-multiply the weights (7, 7) by V (7, 4). Each output
#           row is that token's attention-weighted blend of all values.
out = ...  # FILL 6
out = require(out, 6, "weights matrix-multiplied by V")

print(f"\noutput = weights @ V, shape {out.shape}\n")
print(f"Vector for 'it' BEFORE attention: {X[TOKENS.index('it')]}")
print(f"Vector for 'it' AFTER  attention: {out[TOKENS.index('it')]}")
print("""
'it' now carries a large dose of the 'drill' vector. Downstream layers see
a pronoun enriched with its referent. Attention MOVES INFORMATION BETWEEN
POSITIONS; that is its entire job.

Engineering anchor: it is a soft, differentiable key-value store lookup.
Instead of returning one exact match, it returns a relevance-weighted
blend of every value, so gradients can flow and the lookup can be learned.

KEY TAKEAWAY
  attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V
  One line of math. Everything else in the architecture is arranged
  around letting this lookup run in parallel, in many heads, in stacks.
""")
