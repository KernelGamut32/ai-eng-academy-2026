"""
Demo 06 (STUDENT): Train a tiny transformer and watch generation go from
noise to language. The full arc in one script, every piece learned end to
end.

Most of the machinery is provided; you will write the lines that ARE the
lesson: the embedding composition, the generation loop (predict, pick,
append), and the training objective. Each one is a callback to an earlier
demo, named in its FILL comment.

Fill each FILL marker as the instructor covers it. The file stops cleanly
at your first unfilled blank. Solution key: demo_06_train_tiny_transformer.py

What to watch for live once everything is filled:
  1. BEFORE training: generation is random character soup.
  2. Loss falls as the model learns character statistics, then words,
     then transcript structure.
  3. AFTER training: the model produces CUSTOMER / AGENT dialogue with
     real words it was never explicitly programmed to produce.

Run: python demo_06_train_tiny_transformer_STUDENT.py
Runtime: about 1 to 3 minutes on a no-GPU Mac (MPS or CPU). Progress
prints every 50 steps, so a quiet pause of a few seconds is normal.
Requires: torch, numpy (see requirements.txt).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(1337)

def require(value, fill_num, hint):
    """Stops the script cleanly at the first unfilled blank."""
    if value is Ellipsis:
        print(f"\n>>> Stopped at FILL {fill_num}: {hint}")
        print(">>> Fill it in and rerun. Everything printed above is done.")
        raise SystemExit(0)
    return value

# ---------------------------------------------------------------------------
# Config. Small on purpose: fast on CPU, big enough to visibly learn.
# ---------------------------------------------------------------------------
BLOCK_SIZE = 64     # context window in characters
N_EMBD     = 64     # d_model
N_HEAD     = 4
N_LAYER    = 3
BATCH_SIZE = 32
TRAIN_STEPS = 600
LR = 3e-3

# Provided. Device: prefer CUDA, then Apple MPS, then CPU. Never hard-code cuda.
def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"

DEVICE = pick_device()
print(f"Device: {DEVICE}")

# ---------------------------------------------------------------------------
# Provided: synthetic training corpus, Cordwell support transcripts from
# templates. Fixed seed, generated at runtime, no external data files.
# ---------------------------------------------------------------------------
import random
random.seed(1337)

PRODUCTS = ["cordless drill", "orbital sander", "impact driver",
            "circular saw", "shop vacuum", "tile cutter"]
ISSUES = ["arrived damaged", "was defective", "stopped charging",
          "was missing parts", "never arrived"]
ACTIONS = ["issued a refund", "shipped a replacement",
           "escalated the ticket", "applied store credit"]

lines = []
for _ in range(400):
    p, i, a = random.choice(PRODUCTS), random.choice(ISSUES), random.choice(ACTIONS)
    lines.append(f"CUSTOMER: my {p} {i}. can you help?\n"
                 f"AGENT: sorry about the {p}. we have {a}.\n")
corpus = "".join(lines)
print(f"Corpus: {len(corpus):,} characters of synthetic support transcripts")

# ---------------------------------------------------------------------------
# Provided: tokenizer, character-level. Compare with subwords in demo 01.
# ---------------------------------------------------------------------------
chars = sorted(set(corpus))
VOCAB_SIZE = len(chars)
stoi = {c: i for i, c in enumerate(chars)}
itos = {i: c for c, i in stoi.items()}
encode = lambda s: [stoi[c] for c in s]
decode = lambda ids: "".join(itos[i] for i in ids)
print(f"Vocab: {VOCAB_SIZE} characters")

data = torch.tensor(encode(corpus), dtype=torch.long)

def get_batch():
    ix = torch.randint(len(data) - BLOCK_SIZE - 1, (BATCH_SIZE,))
    x = torch.stack([data[i:i + BLOCK_SIZE] for i in ix])
    y = torch.stack([data[i + 1:i + BLOCK_SIZE + 1] for i in ix])  # shifted by one
    return x.to(DEVICE), y.to(DEVICE)

# ---------------------------------------------------------------------------
# The model. Every piece maps to a demo you have already run:
#   embeddings + positions  -> demo 02
#   causal multi-head attn  -> demos 03 and 04
#   MLP, residuals, norms   -> demo 04
# ---------------------------------------------------------------------------
class Block(nn.Module):
    """Provided: exactly the block you built in demo 04, in torch."""
    def __init__(self):
        super().__init__()
        self.ln1 = nn.LayerNorm(N_EMBD)
        self.attn = nn.MultiheadAttention(N_EMBD, N_HEAD, batch_first=True)
        self.ln2 = nn.LayerNorm(N_EMBD)
        self.mlp = nn.Sequential(
            nn.Linear(N_EMBD, 4 * N_EMBD), nn.GELU(),
            nn.Linear(4 * N_EMBD, N_EMBD),
        )

    def forward(self, x):
        T = x.shape[1]
        causal = torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), 1)
        h = self.ln1(x)
        a, _ = self.attn(h, h, h, attn_mask=causal)  # the mask from demo 04
        x = x + a                                    # residual: read bus, write delta
        x = x + self.mlp(self.ln2(x))
        return x

class TinyGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok_emb = nn.Embedding(VOCAB_SIZE, N_EMBD)   # demo 02, step 1
        self.pos_emb = nn.Embedding(BLOCK_SIZE, N_EMBD)   # demo 02, step 3
        self.blocks = nn.Sequential(*[Block() for _ in range(N_LAYER)])
        self.ln_f = nn.LayerNorm(N_EMBD)
        self.head = nn.Linear(N_EMBD, VOCAB_SIZE)         # demo 04, step 4

    def forward(self, idx):
        T = idx.shape[1]
        pos = torch.arange(T, device=idx.device)
        # FILL 1: build the block input. (Callback: demo 02, step 3.)
        # Contract: token embeddings of idx PLUS position embeddings of
        #           pos. Both come from the tables defined in __init__;
        #           call them like functions and add the results.
        x = ...  # FILL 1
        x = require(x, 1, "tok_emb of idx plus pos_emb of pos")
        x = self.blocks(x)
        return self.head(self.ln_f(x))                    # logits (B, T, vocab)

model = TinyGPT().to(DEVICE)
n_params = sum(p.numel() for p in model.parameters())
print(f"Model: {N_LAYER} blocks, {N_HEAD} heads, d_model {N_EMBD}, {n_params:,} params\n")

# ---------------------------------------------------------------------------
# Generation: byte-for-byte the loop from demo 05, count table swapped for
# a transformer. You write pick and append again, for real this time.
# ---------------------------------------------------------------------------
@torch.no_grad()
def generate(prompt: str, n_tokens: int = 120, temperature: float = 0.8) -> str:
    model.eval()
    idx = torch.tensor([encode(prompt)], dtype=torch.long, device=DEVICE)
    for _ in range(n_tokens):
        logits = model(idx[:, -BLOCK_SIZE:])       # predict (last BLOCK_SIZE tokens)
        logits = logits[:, -1, :] / temperature    # last position, temperature applied
        # FILL 2: distribution. (Callback: demo 05, step 1.)
        # Contract: softmax over the logits (F.softmax, last dim) so each
        #           row is a probability distribution over the vocab.
        probs = ...  # FILL 2
        probs = require(probs, 2, "softmax of logits over the last dim")
        # FILL 3: pick. (Callback: demo 05, step 3.)
        # Contract: sample ONE token id from probs. torch.multinomial with
        #           num_samples 1 draws from a probability row.
        nxt = ...  # FILL 3
        nxt = require(nxt, 3, "one sample drawn from probs")
        # FILL 4: append. (Callback: demo 05, step 2.)
        # Contract: concatenate nxt onto idx along dim 1 (the sequence
        #           dim) with torch.cat, so the loop feeds its own output
        #           back in as context.
        idx = ...  # FILL 4
        idx = require(idx, 4, "idx with nxt concatenated on the sequence dim")
    model.train()
    return decode(idx[0].tolist())

PROMPT = "CUSTOMER: my "

print("=" * 70)
print("BEFORE TRAINING (random weights)")
print("=" * 70)
print(generate(PROMPT))
print("\nRandom character soup. The architecture is all there; the weights")
print("know nothing yet. Everything that follows is learned from data.\n")

# STOP AND PREDICT: what will the loss curve do, and what will the model
# learn FIRST? (Spaces and common letters come before words, words before
# dialogue structure.)

print("=" * 70)
print(f"TRAINING ({TRAIN_STEPS} steps)")
print("=" * 70)
opt = torch.optim.AdamW(model.parameters(), lr=LR)
start_loss = None
for step in range(TRAIN_STEPS + 1):
    xb, yb = get_batch()
    logits = model(xb)
    # FILL 5: the entire training objective. (Callback: 'predict the next
    # token' is all there is.)
    # Contract: cross-entropy (F.cross_entropy) between the model's
    #           logits and the shifted targets yb. Flatten first so every
    #           position is one prediction: logits become
    #           (B*T, VOCAB_SIZE) via .view(-1, VOCAB_SIZE), yb becomes
    #           (B*T,) via .view(-1).
    loss = ...  # FILL 5
    loss = require(loss, 5, "cross-entropy of flattened logits vs flattened yb")
    opt.zero_grad()
    loss.backward()
    opt.step()
    if start_loss is None:
        start_loss = loss.item()
    if step % 50 == 0:
        print(f"  step {step:4d}  loss {loss.item():.3f}")

print(f"\nLoss note: random guessing over {VOCAB_SIZE} chars is ln({VOCAB_SIZE}) = "
      f"{math.log(VOCAB_SIZE):.2f}. We started near that and ended far below it.")

print()
print("=" * 70)
print("AFTER TRAINING")
print("=" * 70)
print(generate(PROMPT))

print()
print("=" * 70)
print("SAME MODEL, TEMPERATURE KNOB (demo 05, step 3, now for real)")
print("=" * 70)
for temp in (0.5, 1.3):
    print(f"\n--- temperature {temp} ---")
    print(generate(PROMPT, n_tokens=90, temperature=temp))

print("""
KEY TAKEAWAY
  Nothing in this script told the model what a word is, what products
  exist, or how a support dialogue flows. One objective (predict the next
  character) plus the transformer architecture recovered all of it from
  raw text. Production LLMs are this same recipe with subword tokens,
  thousands of times more parameters, and vastly more data, followed by
  instruction tuning on top.
""")
