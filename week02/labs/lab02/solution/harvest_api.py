"""
harvest_api.py  —  Local practice API for Week 2 / Lab 02 (Resilient Harvester)
==============================================================================
Extends Lab 01's Cordwell API with the two things a *production* harvester must
survive: **Bearer-token auth** and **rate limiting** (real `429` + `Retry-After`).
It serves the same synthetic Cordwell Home & Hardware SQLite database on localhost
— no external calls, no second server (the Lab-03 draft's separate Datasette + proxy
are folded into this one app).

--------------------------------------------------------------------------
RUNNING IT
--------------------------------------------------------------------------
    from harvest_api import start_server
    server, _ = start_server(port=8000)      # background thread; poll /health

The DB path comes from env var CORDWELL_DB (default "cordwell.db"). The expected
bearer token comes from HARVEST_TOKEN (default "cordwell-dev-token").

--------------------------------------------------------------------------
AUTH
--------------------------------------------------------------------------
Every /v1/* endpoint requires:   Authorization: Bearer <HARVEST_TOKEN>
  * missing / malformed header -> 401
  * wrong token                -> 403
/health and /admin/reset are open (no token).

--------------------------------------------------------------------------
ENDPOINTS
--------------------------------------------------------------------------
GET  /health                                                    (no auth)
        -> {"status": "ok", "orders": <int>}

GET  /v1/orders?limit=&cursor=&region=&channel=&since=          (auth)
        Keyset (cursor) pagination over orders, ordered by order_id, with optional
        exact-match filters on region/channel and an incremental filter
        `since` (order_date >= since, ISO 'YYYY-MM-DD').
        -> {"data": [order, ...], "next_cursor": <int|null>, "count": <int>}

GET  /v1/unreliable?key=&fail_times=                            (auth)
        503 for the first `fail_times` requests per key, then 200. For backoff.

GET  /v1/rate-limited?key=                                      (auth)
        429 + `Retry-After: 2` on the first request per key, then 200.

POST /admin/reset                                               (no auth)
        Clears the per-key counters used by the two endpoints above.

Rate limiting: a real fixed-window limiter runs on every /v1/* request
(HARVEST_RATE_LIMIT requests per 60s per token, default 10000 so a normal harvest
never trips it). The deterministic /v1/rate-limited endpoint is what the lab uses
to *exercise* 429 handling without making the whole harvest flaky.
"""
from __future__ import annotations

import os
import sqlite3
import time

from fastapi import Depends, FastAPI, Header, HTTPException, Query, Response

DB_PATH = os.environ.get("CORDWELL_DB", "cordwell.db")
RATE_LIMIT = int(os.environ.get("HARVEST_RATE_LIMIT", "10000"))  # per 60s per token

app = FastAPI(title="Cordwell Week 2 Harvest API", version="1.0")

_COUNTERS: dict[str, int] = {}          # per-key counters for the flaky endpoints
_RATE: dict[tuple[str, int], int] = {}  # (token, minute-window) -> request count

_ORDER_COLUMNS = [
    "order_id", "customer_id", "store_id", "store_region",
    "channel", "order_date", "order_ts",
]


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def _expected_token() -> str:
    return os.environ.get("HARVEST_TOKEN", "cordwell-dev-token")


def require_auth(authorization: str | None = Header(default=None)) -> str:
    """Enforce 'Authorization: Bearer <token>' and a per-token rate limit."""
    if authorization is None or not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="Missing or malformed bearer token")
    token = authorization.split(" ", 1)[1]
    if token != _expected_token():
        raise HTTPException(status_code=403, detail="Invalid token")
    # fixed-window rate limit
    window = int(time.time()) // 60
    key = (token, window)
    _RATE[key] = _RATE.get(key, 0) + 1
    if _RATE[key] > RATE_LIMIT:
        reset = (window + 1) * 60 - int(time.time())
        raise HTTPException(status_code=429, detail="Rate limit exceeded",
                            headers={"Retry-After": str(max(reset, 1))})
    return token


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
    since: str | None = None,
    _token: str = Depends(require_auth),
) -> dict:
    where = ["order_id > ?"]
    params: list = [cursor]
    if region is not None:
        where.append("store_region = ?"); params.append(region)
    if channel is not None:
        where.append("channel = ?"); params.append(channel)
    if since is not None:
        where.append("order_date >= ?"); params.append(since)
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

    next_cursor = rows[-1]["order_id"] if len(rows) == limit else None
    return {"data": rows, "next_cursor": next_cursor, "count": len(rows)}


@app.get("/v1/unreliable")
def unreliable(key: str = "default", fail_times: int = 2, _token: str = Depends(require_auth)) -> dict:
    n = _COUNTERS.get(key, 0) + 1
    _COUNTERS[key] = n
    if n <= fail_times:
        raise HTTPException(status_code=503, detail=f"Transient error (attempt {n})")
    return {"data": [{"ok": True}], "attempts": n}


@app.get("/v1/rate-limited")
def rate_limited(key: str = "rl", _token: str = Depends(require_auth)):
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


def start_server(host: str = "127.0.0.1", port: int = 8000, log_level: str = "warning"):
    """Start uvicorn on a daemon thread and return (server, thread)."""
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
