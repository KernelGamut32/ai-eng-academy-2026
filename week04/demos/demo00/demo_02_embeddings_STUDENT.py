"""
Demo 02 (STUDENT): Embeddings. Token IDs become vectors that carry meaning.

Fill each FILL marker as the instructor covers it. The file stops cleanly
at your first unfilled blank. Solution key: demo_02_embeddings.py

Run: python demo_02_embeddings_STUDENT.py
Requires numpy.
"""

import numpy as np

np.random.seed(42)  # reproducible run-to-run

def require(value, fill_num, hint):
    """Stops the script cleanly at the first unfilled blank."""
    if value is Ellipsis:
        print(f"\n>>> Stopped at FILL {fill_num}: {hint}")
        print(">>> Fill it in and rerun. Everything printed above is done.")
        raise SystemExit(0)
    return value

print("=" * 70)
print("STEP 1: An embedding table is just a lookup")
print("=" * 70)

VOCAB = ["drill", "driver", "sander", "invoice", "refund", "the", "was"]
D_MODEL = 4  # real models use 768 to 16384 dimensions; 4 fits on a slide

# Provided: at initialization the table is random noise. Training moves the rows.
embedding_table = np.round(np.random.randn(len(VOCAB), D_MODEL), 2)

print(f"\nTable shape: {embedding_table.shape}  (vocab_size x d_model)\n")
for i, tok in enumerate(VOCAB):
    print(f"  id {i}  {tok!r:10s} -> {embedding_table[i]}")

ids = [5, 0, 6]  # "the drill was"
print(f"\nLooking up IDs {ids} ('the drill was'):")

# FILL 1: look up the embeddings for `ids`.
# Contract: index the table with the list of IDs to get shape (3, 4),
#           one row per token, in order. Numpy indexes with a list directly.
looked_up = ...  # FILL 1
looked_up = require(looked_up, 1, "index embedding_table with the ids list")

print(looked_up)
print("\nThat is the entire operation: row lookup by integer index.")

print()
print("=" * 70)
print("STEP 2: After training, geometry means something")
print("=" * 70)

# Provided: hand-set vectors standing in for what training would learn.
# Tools point one way, billing concepts another.
trained = {
    "drill":   np.array([0.9, 0.8, 0.1, 0.0]),
    "driver":  np.array([0.8, 0.9, 0.2, 0.1]),
    "sander":  np.array([0.9, 0.7, 0.0, 0.2]),
    "invoice": np.array([0.1, 0.0, 0.9, 0.8]),
    "refund":  np.array([0.0, 0.2, 0.8, 0.9]),
}

def cosine(a, b):
    # FILL 2: cosine similarity between vectors a and b.
    # Contract: dot product of a and b, divided by the product of their
    #           L2 norms (np.linalg.norm). Returns a scalar near 1.0 for
    #           same-direction vectors, near 0.0 for unrelated ones.
    sim = ...  # FILL 2
    return require(sim, 2, "a dot b over the product of the two norms")

print("\nCosine similarity (1.0 = same direction, 0.0 = unrelated):\n")
pairs = [("drill", "driver"), ("drill", "sander"), ("drill", "invoice"),
         ("invoice", "refund"), ("sander", "refund")]
for a, b in pairs:
    print(f"  {a:8s} vs {b:8s}: {cosine(trained[a], trained[b]):.2f}")

# STOP AND DISCUSS:
# Which pairs scored high? Why would a model trained on support tickets
# push 'invoice' and 'refund' together?

print("""
Engineering anchor: an embedding space is a learned index where distance
approximates relatedness. A vector index is to semantic lookup what a
B-tree is to exact lookup.
""")

print("=" * 70)
print("STEP 3: Position matters, so we add it in")
print("=" * 70)

print("""
'dog bites man' and 'man bites dog' contain identical tokens. Attention
alone treats input as a set, so we inject order: each position gets its
own vector, added to the token vector before the first layer.
""")

pos_table = np.round(np.random.randn(3, D_MODEL) * 0.1, 2)
tok_vecs = embedding_table[ids]

# FILL 3: combine token vectors with position vectors.
# Contract: element-wise sum of tok_vecs and pos_table (both shape (3, 4)),
#           rounded to 2 decimals with np.round for readable printing.
combined = ...  # FILL 3
combined = require(combined, 3, "np.round of tok_vecs plus pos_table")

for p, tok_id in enumerate(ids):
    print(f"  pos {p} {VOCAB[tok_id]!r:8s} token{np.round(tok_vecs[p],2)} + pos{pos_table[p]}")
print(f"\nCombined input to the first transformer layer:\n{combined}")

print("""
KEY TAKEAWAY
  IDs -> embedding lookup -> vectors, plus position information.
  Every later computation happens on these vectors, and training is what
  arranges them so that geometry reflects meaning.
""")
