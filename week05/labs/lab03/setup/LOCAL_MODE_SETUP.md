# Local Mode Setup

Local mode runs the lab against the real artifacts: the Pinecone Local
vector index from Module 02 and SmolLM2-360M-Instruct with the Cordwell
LoRA adapter from Lab 01. The offline backend is the default and needs
none of this; set up local mode only if you want live model behavior.

Local model outputs vary slightly between runs even at low temperature.
The notebook checks are calibrated against the deterministic offline
backend; in local mode treat check thresholds as guidance, not gates.

## 1. Pinecone Local (Docker)

```bash
From the location where **pinecone-local-latest.tar** has been stored, run `docker load -i pinecone-local-latest.tar` to load the image to local cache
```

```bash
docker run --rm -d \
  --name pinecone-local \
  -e PORT=5080 \
  -e PINECONE_HOST=localhost \
  -p 5080-5090:5080-5090 \
  ghcr.io/pinecone-io/pinecone-local:latest
```

Notes:

* **cordwell_corpus.jsonl** and **eval_queries.json** are expected to be in a folder called **data**
* Pinecone Local does not persist records across container restarts.
  After every `docker start pinecone-local`, rebuild the index (run from the same venv):

```bash
python setup/bootstrap_index.py
```

The bootstrap embeds the 14 corpus chunks with all-MiniLM-L6-v2 and
upserts them with `text` and `source` metadata, which is the shape the
notebook's retriever expects.

## 2. The Lab 01 adapter

Copy your trained adapter directory from Lab 01 into this lab folder as
`adapters/cordwell` (the directory that contains
`adapter_config.json` and the adapter weights). Override the location
with the `ADAPTER_PATH` environment variable if you keep it elsewhere.

The notebook loads it through the transformers-native path
(`model.load_adapter(...)`), which is what makes
`model.enable_adapters()` and `model.disable_adapters()` valid. If you
experiment on your own with `PeftModel.from_pretrained` instead, those
two calls will raise `ValueError("No adapter loaded")`; with that
wrapper you toggle via `model.base_model.disable_adapter_layers()`.

## 3. Judge server (optional)

By default the judge is the in-process base model (the adapter is
temporarily disabled while judging, so the fine-tune is not grading its
own homework). To use a separate local server instead:

LM Studio: load your model, start the server (port 1234), then

```bash
export JUDGE_BACKEND=lmstudio
export JUDGE_MODEL=openai/gpt-oss-20b    # confirm the exact tag your machine serves
```

Ollama: `ollama serve` (port 11434), then

```bash
export JUDGE_BACKEND=ollama
export JUDGE_MODEL=gemma4    # confirm with: ollama list
```

Both servers expose an OpenAI compatible endpoint, so the notebook uses
one client code path for both. Ollama's OpenAI compatibility is
documented as experimental; give it a one-question smoke test before
class.
