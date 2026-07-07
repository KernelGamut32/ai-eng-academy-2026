# corpus_api.py — standalone LLM Corpus API (modern FastAPI lifespan).
# Run from this folder:   uvicorn corpus_api:app --port 8009
# Then: http://127.0.0.1:8009/health   ·   http://127.0.0.1:8009/v1/corpus?page=1&page_size=3
from contextlib import asynccontextmanager
from pathlib import Path
import pandas as pd
from fastapi import FastAPI, HTTPException

# Resolve the CSV relative to this file so CWD doesn't matter.
CORPUS_CSV = str((Path(__file__).parent / "data" / "corpus_llm.csv").resolve())


# Lifespan handler: runs startup code before `yield` and shutdown code after.
# FastAPI calls this once when the app boots (not per request), so it's the
# right place to load data we want to reuse across all requests.
@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- Startup: load the corpus once and keep it in memory ---------------
    # Read the CSV a single time at boot instead of re-reading it on every
    # request (reading per request would be slow and wasteful).
    df = pd.read_csv(CORPUS_CSV)

    # Replace missing values (NaN) with empty strings in text columns.
    # pandas represents blank cells as NaN, which serializes to `NaN` in JSON
    # (invalid JSON for many clients); "" keeps the API responses clean.
    for c in ["body_text", "title", "section", "tags", "source_url"]:
        df[c] = df[c].fillna("")

    # Stash the DataFrame on `app.state` — the app-scoped store FastAPI gives
    # us for shared resources. Handlers read it via `app.state.df`. Preferred
    # over a module-level global: it's tied to this app instance and testable.
    app.state.df = df

    # Hand control to FastAPI to serve requests. Everything before `yield` is
    # startup; anything after it (none here) would run on shutdown/cleanup.
    yield


app = FastAPI(title="LLM Corpus API", version="2.0.0", lifespan=lifespan)


@app.get("/health")
async def health():
    return {"ok": True}


@app.get("/v1/corpus")
async def list_docs(page: int = 1, page_size: int = 50,
                    q: str | None = None, language: str | None = None, conf: str | None = None):
    if page < 1 or not (1 <= page_size <= 200):
        raise HTTPException(400, "bad paging params")
    df = app.state.df
    if q:                                 # literal substring match (regex=False)
        m = (df["body_text"].str.contains(q, case=False, na=False, regex=False)
             | df["title"].str.contains(q, case=False, na=False, regex=False))
        df = df[m]
    if language:
        df = df[df["language"] == language]
    if conf:
        df = df[df["confidentiality"] == conf]
    start = (page - 1) * page_size
    return df.iloc[start:start + page_size].to_dict(orient="records")


@app.get("/v1/corpus/{doc_id}")
async def get_doc(doc_id: str):
    row = app.state.df.loc[app.state.df["doc_id"] == doc_id]
    if row.empty:
        raise HTTPException(404, "not found")
    return row.iloc[0].to_dict()
