"""
Demo 05 (STUDENT): The generation loop. How next-token scores become a
response.

A language model outputs ONE thing, a probability distribution over the
next token. A chatbot response is that operation run in a loop: predict,
pick, append, repeat. You will build the distribution, two picking
policies, and use the loop.

The "model" here is a word-level count table with the same contract a
transformer has: context in, next-token probabilities out. Demo 06 swaps
in a real transformer and the loop does not change.

Fill each FILL marker as the instructor covers it. The file stops cleanly
at your first unfilled blank. Solution key: demo_05_generation_loop.py

Run: python demo_05_generation_loop_STUDENT.py
Requires numpy.
"""

import numpy as np
from collections import defaultdict

rng = np.random.default_rng(7)

def require(value, fill_num, hint):
    """Stops the script cleanly at the first unfilled blank."""
    if value is Ellipsis:
        print(f"\n>>> Stopped at FILL {fill_num}: {hint}")
        print(">>> Fill it in and rerun. Everything printed above is done.")
        raise SystemExit(0)
    return value

# ---------------------------------------------------------------------------
# Provided: tiny synthetic corpus, Cordwell Home & Hardware support snippets.
# ---------------------------------------------------------------------------
CORPUS = """
the customer returned the drill because it was defective .
the customer returned the sander because it arrived damaged .
the agent issued a refund for the drill .
the agent issued a refund for the damaged sander .
the customer asked about the order status .
the agent checked the order status and confirmed the refund .
the drill was out of stock so the agent suggested the driver .
the customer accepted the driver and thanked the agent .
""".split()

print("=" * 70)
print("STEP 1: A minimal next-token model (count table)")
print("=" * 70)

# Provided: count how often each word follows each word.
counts = defaultdict(lambda: defaultdict(int))
for prev, nxt in zip(CORPUS, CORPUS[1:]):
    counts[prev][nxt] += 1

def next_token_probs(context: list[str]) -> tuple[list[str], np.ndarray]:
    """The model contract: context in, distribution over next token out.
    A transformer implements this same signature, just far better."""
    table = counts[context[-1]]
    toks = list(table)
    p = np.array([table[t] for t in toks], dtype=float)
    # FILL 1: turn raw counts into probabilities.
    # Contract: divide the count array by its own sum so the result is
    #           non-negative and sums to 1.0.
    probs = p / p.sum()  # FILL 1
    return toks, require(probs, 1, "counts divided by their sum")

toks, p = next_token_probs(["the"])
print("\nAfter the word 'the', the model's distribution is:\n")
for t, prob in sorted(zip(toks, p), key=lambda x: -x[1]):
    print(f"  {t:10s} {prob:.3f}  {'#' * int(prob * 40)}")

# STOP AND DISCUSS:
# This IS what a language model outputs. Not a sentence. This bar chart.
# Target output check: 'agent' should top the list at 0.263.

print()
print("=" * 70)
print("STEP 2: The loop: predict, pick, append, repeat")
print("=" * 70)

# Provided: the loop. Note the shape: predict -> pick -> append -> repeat.
def generate(prompt, n_tokens, pick_fn, label):
    context = prompt.split()
    for _ in range(n_tokens):
        toks, p = next_token_probs(context)     # predict
        # FILL 2: pick and append.
        # Contract: call pick_fn with toks and p to choose one token,
        #           store it in nxt. The append below adds it to context.
        nxt = pick_fn(toks, p)  # FILL 2
        context.append(require(nxt, 2, "the token pick_fn chooses from toks and p"))
        if context[-1] == ".":
            break
    print(f"  [{label:>18s}] {' '.join(context)}")

def greedy(toks, p):
    # FILL 3: greedy policy.
    # Contract: return the token whose probability is largest. np.argmax
    #           on p gives the winning index into toks.
    pick = toks[int(np.argmax(p))]  # FILL 3
    return require(pick, 3, "the token at the argmax index of p")

print("\nGreedy decoding (always take the argmax), run 3 times:\n")
for _ in range(3):
    generate("the customer", 12, greedy, "greedy")
print("\nSame output every time. Deterministic, and often repetitive or stuck.")

print()
print("=" * 70)
print("STEP 3: Sampling and temperature")
print("=" * 70)

def make_sampler(temperature):
    def sample(toks, p):
        # FILL 4: apply temperature to the distribution.
        # Contract: take the natural log of p (np.log), then divide by
        #           temperature. Low temperature stretches the gaps
        #           between logits (sharper), high temperature shrinks
        #           them (flatter). Store the result in logits.
        logits = np.log(p) / temperature  # FILL 4
        logits = require(logits, 4, "log of p, divided by temperature")
        # Provided: softmax back to probabilities, then sample.
        q = np.exp(logits - logits.max())
        q = q / q.sum()
        return toks[int(rng.choice(len(toks), p=q))]
    return sample

for temp in (0.7, 1.0, 1.5, 0.001):
    print(f"\nTemperature {temp}, run 3 times:")
    for _ in range(3):
        generate("the customer", 12, make_sampler(temp), f"sample T={temp}")

print("""
Low temperature sharpens the distribution toward the argmax (safe, samey).
High temperature flattens it (varied, riskier). Temperature is a knob on
the SELECTION step; the model's probabilities never changed.
""")

print("=" * 70)
print("STEP 4: Why context is everything")
print("=" * 70)

print("\nSame loop, different prompts:\n")
for prompt in ("the agent", "the drill", "the customer asked"):
    generate(prompt, 12, greedy, "greedy")
    generate(prompt, 12, make_sampler(1.0), "sample T=1.0")
    generate(prompt, 12, make_sampler(0.001), "sample T=0.01")

print("""
The output is a pure function of context plus the picking rule. "Prompt
engineering" is choosing context that steers this loop; there is no other
input channel.

Honest limitation of our toy: it sees only ONE previous word. It cannot
know that 'it' means the drill from five words back. THAT long-range
lookup is exactly what attention added in demo 03, and it is why the
transformer replaced count tables and RNNs.

KEY TAKEAWAY
  response = loop( model(context) -> distribution -> pick -> append )
  Greedy, temperature, top-k, top-p are policies for the 'pick' step.
  Demo 06 keeps this identical loop and swaps in a real transformer.
""")
