# Week 2 · Lab 02 — Local API Reference (`harvest_api.py`)

A **FastAPI** app serving `cordwell.db` on `http://127.0.0.1:8000`, with **Bearer-token
auth** and a **rate limiter** (real `429` + `Retry-After`). Localhost only — no external
calls. Folds the Lab-03 draft's separate Datasette + auth/rate-limit proxy into one app.

## Running it
```python
from harvest_api import start_server
server, _ = start_server(port=8000)   # background thread; poll GET /health
```
Env vars: `CORDWELL_DB` (default `cordwell.db`), `HARVEST_TOKEN` (default
`cordwell-dev-token`), `HARVEST_RATE_LIMIT` (default 10000/min/token).

## Auth
Every `/v1/*` endpoint requires `Authorization: Bearer <HARVEST_TOKEN>`:
- missing / malformed header → **401**
- wrong token → **403**

`/health` and `/admin/reset` are open.

## Endpoints

### `GET /health`  *(no auth)*
`-> {"status": "ok", "orders": <int>}`.

### `GET /v1/orders`  *(auth)* — keyset pagination + filters
Params: `limit` (1–1000, default 100), `cursor` (default 0, returns `order_id > cursor`),
`region`, `channel`, `since` (`order_date >= since`, ISO `YYYY-MM-DD`).
```json
{"data": [ {order}, ... ], "next_cursor": <int|null>, "count": <int>}
```
`next_cursor` = last `order_id` of a full page (pass back as `cursor`); `null` on the last
page. Order fields: `order_id, customer_id, store_id, store_region, channel, order_date,
order_ts`.

### `GET /v1/unreliable`  *(auth)* — 503 then 200
Params: `key` (default `"default"`), `fail_times` (default 2). Returns `503` for the first
`fail_times` requests **per key**, then `{"data":[...],"attempts":<int>}`. Reset with
`POST /admin/reset`.

### `GET /v1/rate-limited`  *(auth)* — 429 + Retry-After
Param: `key` (default `"rl"`). First request per key → `429` + `Retry-After: 2`; next → 200.

### `POST /admin/reset`  *(no auth)*
Clears the per-key counters for the two flaky endpoints. `-> {"reset": true}`.

## Which part uses what
| Part | Uses |
|---|---|
| A (harvester) | `/health`, `/v1/orders` (+ `since`), `/v1/unreliable`, `/v1/rate-limited`, `/admin/reset` |
| B (SQL) | reads `cordwell.db` directly with pandas |
| C (stretch) | `/v1/orders`, `/v1/unreliable` (retry budget) |
