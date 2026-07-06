"""
lab_api.py  —  Local practice API for Week 2 / Lab 01 (Data Engineering)
=======================================================================
A small, self-contained FastAPI app that serves the synthetic Cordwell Home &
Hardware SQLite database (`cordwell.db`) as a REST/JSON API on localhost. It lets
students practice real `requests` habits — query params, pagination, error
handling, and exponential backoff that respects `429` and `5xx` — **without any
external network calls.** The API and the SQL half of the lab read the *same*
database, so it's one dataset exposed two ways.

--------------------------------------------------------------------------
RUNNING IT
--------------------------------------------------------------------------
Option A — from inside the notebook (what the lab does):
    from lab_api import start_server
    server, _ = start_server(port=8000)      # background thread; poll /health

Option B — from a terminal:
    uvicorn lab_api:app --host 127.0.0.1 --port 8000

The database path comes from the env var CORDWELL_DB (default "cordwell.db" in
the current directory). Build it first with `python build_cordwell_db.py`.

--------------------------------------------------------------------------
ENDPOINTS
--------------------------------------------------------------------------
GET  /health
        -> {"status": "ok", "orders": <int>}                         (no db rows leaked)

GET  /v1/orders?limit=<int>&cursor=<int>&region=<str>&channel=<str>
        Keyset (cursor) pagination over the orders table, ordered by order_id.
        Optional exact-match filters on store_region and channel.
        -> {"data": [order, ...], "next_cursor": <int|null>, "count": <int>}
        Pass the returned next_cursor back as `cursor` for the next page; it is
        null on the final page. `limit` is capped at 1000.

GET  /v1/unreliable?key=<str>&fail_times=<int>
        Returns 503 for the first `fail_times` requests (per key), then 200.
        Use it to watch exponential backoff actually recover.
        -> 503 (until threshold) then {"data": [...], "attempts": <int>}

GET  /v1/rate-limited?key=<str>
        Returns 429 with a `Retry-After: 2` header on the first request (per
        key), then 200. For Retry-After handling.
        -> 429 (first) then {"data": [...], "attempts": <int>}

POST /admin/reset
        Clears the per-key counters used by the two endpoints above so the
        retry demos are repeatable.

Each `order` object mirrors the orders table:
    {order_id, customer_id, store_id, store_region, channel, order_date, order_ts}
"""
from __future__ import annotations

import os
import sqlite3

from fastapi import FastAPI, HTTPException, Query, Response

DB_PATH = os.environ.get("CORDWELL_DB", "cordwell.db")

app = FastAPI(title="Cordwell Week 2 Practice API", version="1.0")

_COUNTERS: dict[str, int] = {}  # per-key hit counters for the flaky endpoints

_ORDER_COLUMNS = [
    "order_id", "customer_id", "store_id", "store_region",
    "channel", "order_date", "order_ts",
]


def _connect() -> sqlite3.Connection:
    """Open a fresh connection (sync endpoints run in a threadpool)."""
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


@app.get("/health")
def health() -> dict:
    try:
        conn = _connect()
        n = conn.execute("SELECT COUNT(*) FROM orders").fetchone()[0]
        conn.close()
        return {"status": "ok", "orders": int(n)}
    except sqlite3.Error as exc:
        raise HTTPException(status_code=503, detail=f"database not ready: {exc}")


@app.post("/admin/reset")
def reset() -> dict:
    _COUNTERS.clear()
    return {"reset": True}


@app.get("/v1/orders")
def get_orders(
    limit: int = Query(100, ge=1, le=1000),
    cursor: int = Query(0, ge=0),
    region: str | None = None,
    channel: str | None = None,
) -> dict:
    """Keyset pagination over orders, ordered by order_id, with optional filters."""
    where = ["order_id > ?"]
    params: list = [cursor]
    if region is not None:
        where.append("store_region = ?")
        params.append(region)
    if channel is not None:
        where.append("channel = ?")
        params.append(channel)
    sql = (
        f"SELECT {', '.join(_ORDER_COLUMNS)} FROM orders "
        f"WHERE {' AND '.join(where)} ORDER BY order_id LIMIT ?"
    )
    params.append(limit)

    conn = _connect()
    try:
        rows = [dict(r) for r in conn.execute(sql, params).fetchall()]
    finally:
        conn.close()

    # next_cursor is the last order_id IFF a full page came back (more may remain)
    next_cursor = rows[-1]["order_id"] if len(rows) == limit else None
    return {"data": rows, "next_cursor": next_cursor, "count": len(rows)}


@app.get("/v1/unreliable")
def unreliable(key: str = "default", fail_times: int = 2) -> dict:
    n = _COUNTERS.get(key, 0) + 1
    _COUNTERS[key] = n
    if n <= fail_times:
        raise HTTPException(status_code=503, detail=f"Transient error (attempt {n})")
    return {"data": [{"ok": True}], "attempts": n}


@app.get("/v1/rate-limited")
def rate_limited(key: str = "rl"):
    n = _COUNTERS.get(key, 0) + 1
    _COUNTERS[key] = n
    if n == 1:
        return Response(
            status_code=429,
            headers={"Retry-After": "2"},
            content='{"detail": "Rate limited"}',
            media_type="application/json",
        )
    return {"data": [{"ok": True}], "attempts": n}


# --------------------------------------------------------------------------- #
# Background-thread runner (used by the notebook)
# --------------------------------------------------------------------------- #
def start_server(host: str = "127.0.0.1", port: int = 8000, log_level: str = "warning"):
    """Start uvicorn on a daemon thread and return (server, thread).

    Signal handlers are disabled because we are not on the main thread. Poll
    GET /health until it returns 200 before making requests.
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
