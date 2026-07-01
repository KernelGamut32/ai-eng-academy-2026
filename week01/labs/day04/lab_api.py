"""
lab_api.py  —  Local practice API for Week 1 / Day 4 (Data Acquisition)
=======================================================================
A small, fully self-contained FastAPI application that stands in for a real
production REST API. It runs on localhost so students can practice every
`requests` habit from the deck — pagination, retries/backoff, rate-limit
(Retry-After) handling, bearer-token auth, POST with a JSON body, and nested
JSON — **without ever calling an external service**.

The data is synthetic and clearly fictional (an AI-assistant platform's request
log). Nothing here talks to a real model or a real vendor.

--------------------------------------------------------------------------
RUNNING IT
--------------------------------------------------------------------------
Option A — from a terminal (simplest mental model):
    pip install fastapi uvicorn
    uvicorn lab_api:app --host 127.0.0.1 --port 8000
    # leave it running; open the notebook in another terminal/tab

Option B — from inside the notebook (self-contained; used by the labs):
    the notebooks import `start_server()` from this file and launch it on a
    background thread, so the API and your client code live in one notebook.

--------------------------------------------------------------------------
AUTH
--------------------------------------------------------------------------
Every /v1/* endpoint requires a bearer token:
    Authorization: Bearer <LAB_API_KEY>
The expected token is read from the env var LAB_API_KEY (default "local-dev-key").
Missing/incorrect -> 401. /health and /admin/reset are open.

--------------------------------------------------------------------------
ENDPOINTS  (all /v1/* need the bearer token)
--------------------------------------------------------------------------
GET  /health
        -> {"status": "ok"}                              (no auth)

GET  /v1/events?page=<int>&per_page=<int>
        Page/per_page pagination over 250 synthetic events.
        -> {"data": [event, ...], "page", "per_page", "total", "has_more"}
        Last page is smaller than per_page (signals the end).

GET  /v1/events/cursor?limit=<int>&cursor=<str|null>
        Cursor/next-token pagination over the same 250 events.
        -> {"data": [event, ...], "next_cursor": <str|null>}
        next_cursor is null on the final page.

GET  /v1/articles/nested?page=<int>&per_page=<int>
        Records with a nested `author` object and a `tags` list, for
        pd.json_normalize / .explode practice.
        -> {"data": [article, ...], "page", "per_page", "total"}

GET  /v1/unreliable?key=<str>&fail_times=<int>
        Fails with 503 the first `fail_times` requests (per key), then 200.
        Use it to see Retry + exponential backoff actually recover.
        -> 503 (until threshold) then {"data": [...], "attempts": <int>}

GET  /v1/rate-limited?key=<str>
        Returns 429 with a `Retry-After: 1` header on the first request
        (per key), then 200. For Retry-After handling.
        -> 429 (first) then {"data": [...], "attempts": <int>}

POST /v1/summarize     body: {"text": <str>, "model": <str?>}
        A fake, deterministic "summary" (first sentence + a token estimate).
        Mimics an LLM completion endpoint without calling one.
        -> {"id", "model", "summary", "input_tokens", "output_tokens"}

POST /admin/reset
        Clears the per-key counters used by /v1/unreliable and
        /v1/rate-limited so the retry demos are repeatable.       (no auth)

Every event object looks like:
    {
      "event_id": int, "user_id": int, "created_at": "YYYY-MM-DDTHH:MM:SS",
      "model": str, "category": str|null, "region": str,
      "score": float, "input_tokens": int, "output_tokens": int,
      "latency_ms": float, "response_length": int
    }
"""
from __future__ import annotations

import os
from datetime import datetime, timedelta

import numpy as np
from fastapi import Depends, FastAPI, Header, HTTPException, Response
from pydantic import BaseModel

# --------------------------------------------------------------------------- #
# Synthetic dataset (deterministic, generated once at import)
# --------------------------------------------------------------------------- #
MODELS = ["atlas-mini", "atlas-pro", "nova-4", "orion-8b"]
CATEGORIES = ["factual", "creative", "code", "summary", "refusal"]
REGIONS = ["AMER", "EU", "APAC", "LATAM"]
N_EVENTS = 250


def _build_events() -> list[dict]:
    rng = np.random.default_rng(seed=4001)
    base = datetime(2024, 1, 1, 8, 0, 0)
    events: list[dict] = []
    for i in range(N_EVENTS):
        created = base + timedelta(hours=int(rng.integers(0, 24 * 120)))
        score = float(np.clip(rng.normal(74, 14), 0, 100).round(2))
        in_tok = int(rng.integers(20, 4000))
        out_tok = int(rng.integers(5, 1500))
        # category is null ~10% of the time (data-quality lessons; exceeds the 5% flag)
        category = None if rng.random() < 0.10 else CATEGORIES[int(rng.integers(0, len(CATEGORIES)))]
        events.append(
            {
                "event_id": 1000 + i,
                "user_id": int(rng.integers(10_000, 10_200)),
                "created_at": created.isoformat(timespec="seconds"),
                "model": MODELS[int(rng.integers(0, len(MODELS)))],
                "category": category,
                "region": REGIONS[int(rng.integers(0, len(REGIONS)))],
                "score": score,
                "input_tokens": in_tok,
                "output_tokens": out_tok,
                "latency_ms": float(np.clip(rng.normal(600, 220), 20, None).round(1)),
                "response_length": int(out_tok * rng.integers(3, 6)),
            }
        )
    # Inject two exact duplicate rows (for df.duplicated() lessons in Lab 2)
    events.append(dict(events[10]))
    events.append(dict(events[20]))
    return events


EVENTS = _build_events()  # length 252 (250 + 2 duplicates)


def _build_articles() -> list[dict]:
    """Nested records: author object + tags list (for json_normalize)."""
    rng = np.random.default_rng(seed=4002)
    authors = [("Alice", 42), ("Bob", 17), ("Carol", 88), ("Dan", 5)]
    all_tags = ["nlp", "python", "ml", "rag", "eval", "prompt"]
    out = []
    for i in range(40):
        name, aid = authors[int(rng.integers(0, len(authors)))]
        k = int(rng.integers(1, 4))
        tags = list(np.random.default_rng(seed=i).choice(all_tags, size=k, replace=False))
        out.append(
            {
                "id": 1 + i,
                "title": f"Post {i:02d}",
                "author": {"name": name, "id": aid},
                "tags": tags,
                "word_count": int(rng.integers(200, 3000)),
            }
        )
    return out


ARTICLES = _build_articles()

# --------------------------------------------------------------------------- #
# App + auth
# --------------------------------------------------------------------------- #
app = FastAPI(title="AI Academy Day 4 Practice API", version="1.0")

_COUNTERS: dict[str, int] = {}  # per-key hit counters for the flaky endpoints


def _expected_key() -> str:
    return os.environ.get("LAB_API_KEY", "local-dev-key")


def require_auth(authorization: str | None = Header(default=None)) -> None:
    """Reject requests without a correct 'Authorization: Bearer <key>' header."""
    expected = f"Bearer {_expected_key()}"
    if authorization != expected:
        raise HTTPException(status_code=401, detail="Missing or invalid bearer token")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/admin/reset")
def reset() -> dict:
    _COUNTERS.clear()
    return {"reset": True}


@app.get("/v1/events")
def get_events(page: int = 1, per_page: int = 100, _=Depends(require_auth)) -> dict:
    if page < 1 or per_page < 1:
        raise HTTPException(status_code=400, detail="page and per_page must be >= 1")
    total = len(EVENTS)
    start = (page - 1) * per_page
    end = min(start + per_page, total)
    data = EVENTS[start:end] if start < total else []
    return {
        "data": data,
        "page": page,
        "per_page": per_page,
        "total": total,
        "has_more": end < total,
    }


@app.get("/v1/events/cursor")
def get_events_cursor(limit: int = 100, cursor: str | None = None, _=Depends(require_auth)) -> dict:
    if limit < 1:
        raise HTTPException(status_code=400, detail="limit must be >= 1")
    start = 0 if cursor is None else int(cursor)
    end = min(start + limit, len(EVENTS))
    data = EVENTS[start:end]
    next_cursor = str(end) if end < len(EVENTS) else None
    return {"data": data, "next_cursor": next_cursor}


@app.get("/v1/articles/nested")
def get_articles(page: int = 1, per_page: int = 20, _=Depends(require_auth)) -> dict:
    total = len(ARTICLES)
    start = (page - 1) * per_page
    end = min(start + per_page, total)
    return {"data": ARTICLES[start:end], "page": page, "per_page": per_page, "total": total}


@app.get("/v1/unreliable")
def unreliable(key: str = "default", fail_times: int = 2, _=Depends(require_auth)) -> dict:
    n = _COUNTERS.get(key, 0) + 1
    _COUNTERS[key] = n
    if n <= fail_times:
        raise HTTPException(status_code=503, detail=f"Transient error (attempt {n})")
    return {"data": EVENTS[:5], "attempts": n}


@app.get("/v1/rate-limited")
def rate_limited(response: Response, key: str = "rl", _=Depends(require_auth)):
    n = _COUNTERS.get(key, 0) + 1
    _COUNTERS[key] = n
    if n == 1:
        return Response(
            status_code=429,
            headers={"Retry-After": "1"},
            content='{"detail": "Rate limited"}',
            media_type="application/json",
        )
    return {"data": EVENTS[:5], "attempts": n}


class SummarizeRequest(BaseModel):
    text: str
    model: str = "atlas-pro"


@app.post("/v1/summarize")
def summarize(req: SummarizeRequest, _=Depends(require_auth)) -> dict:
    text = req.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="text must not be empty")
    first_sentence = text.split(".")[0].strip()
    summary = (first_sentence[:120] + "...") if len(first_sentence) > 120 else first_sentence
    input_tokens = max(1, len(text.split()))
    output_tokens = max(1, len(summary.split()))
    import hashlib

    digest = int(hashlib.sha256(text.encode()).hexdigest(), 16) % 100000
    return {
        "id": f"sum_{digest:05d}",
        "model": req.model,
        "summary": summary,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
    }


# --------------------------------------------------------------------------- #
# Background-thread runner (used by the notebooks)
# --------------------------------------------------------------------------- #
def start_server(host: str = "127.0.0.1", port: int = 8000, log_level: str = "warning"):
    """Start uvicorn on a daemon thread and return (server, thread).

    Signal handlers are disabled because we are not on the main thread. The
    caller should poll GET /health until it returns 200 before making requests.
    """
    import threading
    import uvicorn

    class _ThreadedServer(uvicorn.Server):
        def install_signal_handlers(self):  # no-op off the main thread
            pass

    config = uvicorn.Config(app, host=host, port=port, log_level=log_level)
    server = _ThreadedServer(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    return server, thread


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")
