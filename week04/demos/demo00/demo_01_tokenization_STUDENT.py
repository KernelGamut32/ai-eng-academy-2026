"""
Demo 01 (STUDENT): Tokenization. Text becomes numbers.

Follow along and fill each FILL marker as the instructor covers it.
The file runs top to bottom at any point: it stops cleanly at your first
unfilled blank and tells you which one. Solution key: demo_01_tokenization.py

Run: python demo_01_tokenization_STUDENT.py
"""

def require(value, fill_num, hint):
    """Stops the script cleanly at the first unfilled blank."""
    if value is Ellipsis:
        print(f"\n>>> Stopped at FILL {fill_num}: {hint}")
        print(">>> Fill it in and rerun. Everything printed above is done.")
        raise SystemExit(0)
    return value

# ---------------------------------------------------------------------------
# Scenario: Cordwell Home & Hardware customer support messages.
# ---------------------------------------------------------------------------

SENTENCE = "the customer returned the cordless drill because it was defective"

print("=" * 70)
print("STEP 1: Word-level tokenization (the obvious first idea)")
print("=" * 70)

words = SENTENCE.split()
vocab = sorted(set(words))

print(vocab)

# FILL 1: build the word-to-ID mapping.
# Contract: a dict mapping each word in `vocab` to its index in `vocab`.
#           Example entry: 'because' -> 0 (it sorts first).
word_to_id = {w: i for i, w in enumerate(vocab)}
word_to_id = require(word_to_id, 1, "dict of word -> its index in vocab")

print(f"\nSentence: {SENTENCE!r}")
print(f"\nVocabulary ({len(vocab)} words):")
for w, i in word_to_id.items():
    print(f"  {i:2d} -> {w!r}")

# FILL 2: encode the sentence.
# Contract: a list of integer IDs, one per word in `words`, in order,
#           looked up from word_to_id.
# Target output for this sentence: [7, 2, 6, 7, 1, 4, 0, 5, 8, 3]
ids = [word_to_id[w] for w in words]
ids = require(ids, 2, "list of word_to_id[w] for each w in words")

print(f"\nToken IDs: {ids}")
print("\nThis is all the model receives. Not letters, not words. These integers.")

# STOP AND DISCUSS:
# Ask yourself: what breaks if a customer types a word we have never seen?

print()
print("=" * 70)
print("STEP 2: The unknown-word problem")
print("=" * 70)

new_message = "the sander was defective"
print(f"\nNew message: {new_message!r}")
for w in new_message.split():
    status = word_to_id.get(w, "<-- NOT IN VOCAB, word-level tokenizer fails here")
    print(f"  {w!r:14s} {status}")

print("""
Word-level vocabularies cannot cover every product name, typo, and SKU.
Real systems solve this with SUBWORD tokenization: frequent strings get
their own token, rare words get split into familiar pieces.
""")

print("=" * 70)
print("STEP 3: Subword intuition (hand-rolled, three merge rules)")
print("=" * 70)

# Provided: a real BPE tokenizer learns thousands of merge rules from
# data. We hand-write three rules to show the mechanism.
MERGES = ["def", "ective", "sand"]  # pretend these were learned from data

def subword_tokenize(word: str) -> list[str]:
    """Greedy longest-match against known pieces, else single characters."""
    pieces, i = [], 0
    while i < len(word):
        for m in sorted(MERGES, key=len, reverse=True):
            if word.startswith(m, i):
                pieces.append(m)
                i += len(m)
                break
        else:
            pieces.append(word[i])
            i += 1
    return pieces

for w in ["defective", "sander", "sanded"]:
    print(f"  {w!r:14s} -> {subword_tokenize(w)}")

print("""
'sander' and 'sanded' now share the piece 'sand'. The model can reuse what
it learned about sanding for both, and NO input is ever out of vocabulary:
worst case, a word falls back to single characters.

Engineering anchor: a subword vocab is a compression dictionary. Common
strings get short codes, everything else decomposes into known parts.

KEY TAKEAWAY
  text -> tokenizer -> list of integer IDs. That list is the model input.
  Production models use vocabularies of roughly 32k to 200k subword tokens.
""")
