# Week 5, Lab 02: Cordwell Support-Doc Retrieval

Cordwell Home and Hardware has a support knowledge base: short FAQ entries,
medium troubleshooting guides, and long installation manuals. Support engineers
keep saying the same thing. The right document exists. It does not come back.

In this lab you build the index, measure how bad it is, find a bug that no
exception will ever show you, and improve retrieval with evidence.

**Duration:** about 3 hours for Parts A through F, 4 hours with both stretch
parts.

**Where this stops.** The lab ends at "the right chunks came back". Turning
chunks into a grounded answer is the next module. No text generation happens
here, which is why this lab has no LM Studio or Ollama backend. The analogous
choice, two independent embedding paths, is offered instead.

---

## File map

| File | What it is |
|---|---|
| `lab02_student.ipynb` | Your working notebook. Cold Run All produces zero crashes and 3 of 65 checks passing |
| `lab_support.py` | Pre-written plumbing. Read it, do not edit it |
| `cordwell_corpus.py` | The synthetic corpus and the labeled query set |
| `HINTS.md` | Progressive tier: three escalating levels per task |
| `HINTS_DETAILED.md` | Detailed tier: the working core with line-by-line commentary |
| `requirements.txt` | Pinned, verified dependencies |
| `docker-compose.yaml` | Pinecone Local emulator |
| `Instructor_Demo_Script.md` | Instructor only |
| `Instructor_Code_Walkthrough.md` | Instructor only |
| `Module02_Slide_Corrections.md` | Instructor only |
| `lab02_solution.ipynb` | Instructor only, handed out after the lab |

**Pick one hint tier per task. Reading both wastes your time.** The progressive
tier guides you to the answer. The detailed tier shows you the working core with
commentary and withholds the assembly. Neither is cheating and neither is
better; they suit different moods. The solution notebook is the only fully
assembled artifact and it comes out after the lab.

---

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter lab
```

Python 3.13 on cohort machines. Verified on 3.12.3, runs unchanged on 3.13.

All data is synthetic and generated inside the lab from fixed literals and fixed
seeds. Every product, part number, policy, and fault code is invented. There are
no cross-lab dependencies and nothing to download.

---

## Backend selection

Two independent environment variables. Set them **before** launching Jupyter.
Explicit selection raises a clear error when the thing you asked for is not
reachable, rather than silently switching to something else.

### `LAB_BACKEND` controls where the vectors live

| Value | What it is | Needs |
|---|---|---|
| `offline` (default) | In-memory numpy index mirroring the Pinecone v9 API exactly | Nothing |
| `pinecone_local` | The real Pinecone emulator in Docker | Docker |
| `pinecone_cloud` | Pinecone serverless | API key, network |

The API is identical across all three for everything the lab builds: create,
upsert, query, filter, list, delete. Only the connection differs. That is the
entire reason for using an emulator instead of a lookalike database.

### `EMBED_BACKEND` controls the embedding model

| Value | What it is | Needs |
|---|---|---|
| `auto` (default) | `lsa` when offline, `sentence_transformers` otherwise | |
| `lsa` | TF-IDF plus truncated SVD, in numpy | Nothing |
| `sentence_transformers` | A transformer bi-encoder from a **local directory** | A staged model |

```bash
# The default. Works on any machine, no Docker, no downloads.
export LAB_BACKEND=offline

# The real Pinecone API, still with the zero-dependency embedder.
docker compose up -d
export LAB_BACKEND=pinecone_local
export EMBED_BACKEND=lsa

# The real Pinecone API and a real transformer bi-encoder.
export LAB_BACKEND=pinecone_local
export EMBED_BACKEND=sentence_transformers
export CORDWELL_ST_MODEL_PATH=./models/all-MiniLM-L6-v2
```

### About the LSA embedder

The default embedder is latent semantic analysis: TF-IDF followed by truncated
SVD. That is a real embedding model in the technical sense. It maps text to a
fixed length vector of floats, it learns from word co-occurrence across a
corpus, and similar text lands nearby. It is older and weaker than a transformer
bi-encoder, and the lab says so rather than pretending otherwise.

It was chosen deliberately, not as a convenience. The property that makes chunk
size matter is **fixed capacity**: one vector of `d` numbers regardless of how
much text goes in, so a chunk spanning four unrelated topics is forced to a
compromise point between them. Low-dimensional LSA has that property. Raw
TF-IDF does not, which is why a bag-of-words stand-in would have shown the
opposite effect and taught the wrong lesson.

### About the sentence-transformers path

**This lab never downloads a model.** `CORDWELL_ST_MODEL_PATH` must point at a
directory already on disk. Passing a hub name raises immediately with an
explanatory message rather than hanging on a network call mid-lab.

To stage a model once, on a machine that does have access:

```bash
pip install "huggingface_hub[cli]"
hf download sentence-transformers/all-MiniLM-L6-v2 \
    --local-dir ./models/all-MiniLM-L6-v2
```

Then distribute that directory with the lab materials. About 90 MB.

For Part G, stage a second model at a different dimension and set
`CORDWELL_ST_MODEL_PATH_B`. `all-mpnet-base-v2` at 768 dimensions against
`all-MiniLM-L6-v2` at 384 makes the "spaces are not interchangeable" point
loudly, because the dimension mismatch raises.

---

## Preflight for the Docker path

Run this once on a cohort machine before class. It takes two minutes.

```bash
docker compose pull
docker compose up -d
sleep 20

python3 - <<'PY'
from pinecone import Pinecone, ServerlessSpec
pc = Pinecone(api_key="pclocal", host="http://localhost:5080")
print("control plane reachable:", [i.name for i in pc.indexes.list()])

if not pc.indexes.exists("preflight"):
    pc.indexes.create(name="preflight", dimension=4, metric="cosine",
                      spec=ServerlessSpec(cloud="aws", region="us-east-1"))

host = pc.indexes.describe("preflight").host
if not host.startswith(("http://", "https://")):
    host = f"http://{host}"
idx = pc.index(host=host)

idx.upsert(vectors=[{"id": "a#0", "values": [1, 0, 0, 0], "metadata": {"n": 1}}],
           namespace="support")
r = idx.query(top_k=1, vector=[1, 0, 0, 0], namespace="support",
              include_metadata=True)
print("data plane reachable:", r.matches)
print("list by prefix:", [i.id for p in idx.list(prefix="a#", namespace="support") for i in p])
pc.indexes.delete("preflight")
print("PREFLIGHT OK")
PY
```

If that prints `PREFLIGHT OK`, the Docker path is good and the whole lab will
run on `pinecone_local`.

---

## Currency flags

**`⚠️ CURRENCY FLAG 1. The Module 02 slides do not run on the current SDK.**
`pinecone==9.1.0` is a ground-up rewrite. `GRPCClientConfig` was removed, the
control plane methods moved to `pc.indexes.*`, `pc.Index()` became a deprecated
shim, and `index.list()` now yields objects rather than strings. Drop-in
replacements for every affected slide are in `Module02_Slide_Corrections.md`.
Read that before teaching.

**`⚠️ CURRENCY FLAG 2. API version skew, unverified.** The SDK sends
`X-Pinecone-Api-Version: 2025-10`. Pinecone Local implements `2025-01`.
Pinecone's own documentation shows curl examples against Local sending
`2025-10`, which suggests the emulator tolerates it, but this could not be
verified during the build because the build environment had no Docker. **Run the
preflight above.** If it fails on a version header, pin `pinecone==8.1.2`, which
still sends `2025-10` but retains `GRPCClientConfig`, or `pinecone==6.0.2`,
which sends `2025-01` exactly and matches the slides as originally written.

**`⚠️ CURRENCY FLAG 3. Delete by metadata filter is now supported on
serverless.** The slide saying it is not is outdated; Pinecone shipped it around
October 2025. It is still absent from Pinecone Local because of the API version
pin. The lab reproduces that split deliberately and the ID design lesson is
unaffected. See correction 5.

---

## Measured results

Produced on the `offline` backend with `EMBED_DIM=96`. Your figures will differ
on `sentence_transformers`, and that is data rather than a mistake.

| Part | Configuration | Chunks | Documents indexed | recall@5 | MRR |
|---|---|---|---|---|---|
| B | fixed 1024, no overlap | 218 | 202 of 202 | 0.700 | 0.629 |
| D | sliding 512/256, slide version | 65 | **6 of 202** | 0.528 | **0.733** |
| D | sliding 512/256, corrected | 267 | 202 of 202 | 0.733 | 0.679 |
| E | paragraph 512/64 | 240 | 202 of 202 | 0.717 | 0.661 |
| E | paragraph 384/64 | 257 | 202 of 202 | 0.750 | 0.661 |
| E | paragraph 256/32 | 291 | 202 of 202 | 0.783 | 0.600 |
| E | paragraph 192/24 | 318 | 202 of 202 | 0.817 | 0.653 |
| F | 384/64 plus `is_active` filter | 257 | 202 of 202 | 0.750 | 0.681 |
| G | 384/64, alternate model | 257 | 202 of 202 | 0.683 | 0.527 |

Corpus: 202 documents, 35 FAQ entries at 52 to 88 tokens, 161 troubleshooting
articles at 76 to 314 tokens, 6 installation manuals at 2,874 to 3,583 tokens.
Three documents are superseded revisions marked `is_active: False`. Labeled
query set: 30 queries with human-assigned relevant document IDs.

**The headline.** The slide version of the sliding window chunker silently drops
196 of 202 documents, recall falls to 0.528, and **MRR rises to 0.733**. A team
watching a ranking dashboard would have shipped it.

---

## Check counts

| Notebook | Result |
|---|---|
| `lab02_student.ipynb` cold Run All | 3 of 65 passing, **0 hard crashes** |
| `lab02_solution.ipynb` | **65 of 65 passing**, 0 crashes |

The three that pass cold are the setup checks. Everything else is yours.

Both notebooks are generated from a single shared specification, so the student
stubs and the solution bodies cannot drift apart.
