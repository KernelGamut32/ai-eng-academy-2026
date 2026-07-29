"""Build (or rebuild) the Cordwell index in Pinecone Local.

Pinecone Local does NOT persist records across container restarts, so run
this script after every `docker start`. It is idempotent: an existing
index is deleted and rebuilt.

Usage:
    python setup/bootstrap_index.py

Environment (all optional):
    PINECONE_HOST   default http://localhost:5080
    INDEX_NAME      default cordwell-support
    EMBED_MODEL     default sentence-transformers/all-MiniLM-L6-v2
"""
import json
import os
import sys
import time
from pathlib import Path

from pinecone import Pinecone, ServerlessSpec

PINECONE_HOST = os.environ.get("PINECONE_HOST", "http://localhost:5080")
INDEX_NAME = os.environ.get("INDEX_NAME", "cordwell-support")
EMBED_MODEL = str(Path(os.environ.get("EMBED_MODEL",
                             "~/models/all-MiniLM-L6-v2")).expanduser().resolve())
CORPUS = Path(__file__).resolve().parent.parent / "data" / "cordwell_corpus.jsonl"
DIM = 384  # all-MiniLM-L6-v2 output dimension


def main() -> int:
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
    except ImportError:
        print("langchain-huggingface is not installed. "
              "pip install -r requirements.txt first.")
        return 1

    pc = Pinecone(api_key="pclocal", host=PINECONE_HOST)

    existing = [ix.name for ix in pc.list_indexes()]
    if INDEX_NAME in existing:
        print(f"Deleting existing index {INDEX_NAME!r}...")
        pc.delete_index(INDEX_NAME)

    print(f"Creating index {INDEX_NAME!r} (dim {DIM}, cosine)...")
    pc.create_index(
        name=INDEX_NAME,
        dimension=DIM,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )

    index_host = pc.describe_index(INDEX_NAME).host
    index = pc.Index(host=f"http://{index_host}")

    print(f"Embedding corpus with {EMBED_MODEL}...")
    embedder = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

    records = [json.loads(line) for line in open(CORPUS)]
    texts = [r["text"] for r in records]
    vectors = embedder.embed_documents(texts)

    print(f"Upserting {len(records)} chunks...")
    index.upsert(vectors=[
        {
            "id": r["id"],
            "values": v,
            "metadata": {"text": r["text"], "source": r["source"]},
        }
        for r, v in zip(records, vectors)
    ])

    # Pinecone Local is fast, but give the upsert a beat before verifying.
    time.sleep(1)
    stats = index.describe_index_stats()
    count = stats.get("total_vector_count", 0)
    print(f"Index ready: {count} vectors in {INDEX_NAME!r} at {index_host}")
    if count != len(records):
        print("WARNING: vector count does not match corpus size. "
              "Re-run this script.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
