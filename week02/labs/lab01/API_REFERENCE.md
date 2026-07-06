# Week 2 · Lab 01 — Local API Reference (`lab_api.py`)

A small **FastAPI** app that serves `cordwell.db` on `http://127.0.0.1:8000`. It runs
on localhost — **no external calls**. The API and the SQL half of the lab read the same
database.

## Running it
```python
from lab_api import start_server
server, _ = start_server(port=8000)   # background thread; poll GET /health
```
or standalone: `uvicorn lab_api:app --host 127.0.0.1 --port 8000`.
The DB path comes from env var `CORDWELL_DB` (default `cordwell.db`); build it first
with `python build_cordwell_db.py`.

## Endpoints

### `GET /health`
`-> {"status": "ok", "orders": <int>}`. Liveness probe; poll after starting.

### `GET /v1/orders`  — keyset (cursor) pagination + filters
Query params:
- `limit` (int, default 100, **max 1000**) — page size.
- `cursor` (int, default 0) — return orders with `order_id > cursor`.
- `region` (str, optional) — exact match on `store_region`.
- `channel` (str, optional) — exact match on `channel`.

Returns:
```json
{"data": [ {order}, ... ], "next_cursor": <int|null>, "count": <int>}
```
`next_cursor` is the last `order_id` of the page (pass it back as `cursor`) and is
`null` on the final page. Each `order` has: `order_id, customer_id, store_id,
store_region, channel, order_date, order_ts`.

### `GET /v1/unreliable`  — transient failures (backoff practice)
Query params: `key` (str, default `"default"`), `fail_times` (int, default 2).
Returns `503` for the first `fail_times` requests **per key**, then
`{"data": [...], "attempts": <int>}`. Call `POST /admin/reset` to replay.

### `GET /v1/rate-limited`  — 429 + Retry-After
Query param: `key` (str, default `"rl"`). First request per key returns `429` with
header `Retry-After: 2`; the next returns `{"data": [...], "attempts": <int>}`.

### `POST /admin/reset`
Clears the per-key counters for the two endpoints above. `-> {"reset": true}`.

## Which part uses what
| Part | Uses |
|---|---|
| A (API extraction) | `/health`, `/v1/orders`, `/v1/unreliable`, `/v1/rate-limited`, `/admin/reset` |
| B, C (SQL) | reads `cordwell.db` directly with pandas — no API |
