# Day 4 — Local Practice API Reference (`lab_api.py`)

**Week 1 · Day 4 · AI Engineering Academy** · Gamut Technology Services

You cannot call external APIs from this environment — and you don't need to.
`lab_api.py` is a small **FastAPI** app that runs on **localhost** and behaves like a
real production REST API: bearer-token auth, pagination (two styles), transient
failures, rate limits, and a POST endpoint. It's used by **Lab 1** and **Lab 2**.
(Lab 3 needs no server — it mocks HTTP with the `responses` library.)

The data is synthetic and clearly fictional: an AI-assistant platform's request log
(252 events) plus 40 nested "article" records. Nothing talks to a real model or vendor.

---

## Running the API

### Option A — inside the notebook (what the labs do)
The lab notebooks import a helper and launch the server on a background thread, so the
API and your client code live in one notebook:

```python
from lab_api import start_server
server, _thread = start_server(port=8000)   # returns immediately
# ...poll GET /health until it returns 200, then make requests...
server.should_exit = True                    # stop it at the end
```

You don't have to write this — each lab's setup cell does it for you (Lab 1 uses port
8001, Lab 2 uses port 8002).

### Option B — from a terminal (standalone)
```bash
pip install fastapi uvicorn
uvicorn lab_api:app --host 127.0.0.1 --port 8000
# leave it running; open the notebook in another tab
```
Then point the notebook's `BASE_URL` at `http://127.0.0.1:8000`.

---

## Authentication

Every `/v1/*` endpoint requires a bearer token:

```
Authorization: Bearer <LAB_API_KEY>
```

The expected token comes from the environment variable **`LAB_API_KEY`** (default
`local-dev-key`). Missing or wrong → **401 Unauthorized**. `/health` and `/admin/reset`
are open (no token). The labs read the key from `os.environ["API_KEY"]` — never
hardcoded inline — to model the secrets-from-environment habit from the deck.

---

## Endpoints

### `GET /health`  *(no auth)*
Liveness check. Poll this after starting the server.
- **Returns:** `{"status": "ok"}`

### `GET /v1/events`  *(page/per_page pagination)*
The main dataset: 252 synthetic events (250 unique + 2 deliberate duplicates).
- **Query params:** `page` (int, default 1), `per_page` (int, default 100)
- **Returns:**
  ```json
  {"data": [ {event}, ... ], "page": 1, "per_page": 100, "total": 252, "has_more": true}
  ```
- The **last page is smaller** than `per_page` (at `per_page=100`: pages of 100, 100, 52).
  Stop paginating when a page has fewer than `per_page` items.
- **Errors:** `400` if `page`/`per_page` < 1; `401` without a token.

### `GET /v1/events/cursor`  *(cursor pagination)*
Same 252 events, exposed via a next-token cursor.
- **Query params:** `limit` (int, default 100), `cursor` (str, optional — omit on the first call)
- **Returns:** `{"data": [ {event}, ... ], "next_cursor": "<str>" | null}`
- Pass the returned `next_cursor` back as the `cursor` param on the next call. Stop when
  `next_cursor` is `null`.

### `GET /v1/articles/nested`  *(nested JSON)*
40 records with a nested `author` object and a `tags` list — for `pd.json_normalize`
and `.explode`.
- **Query params:** `page` (int, default 1), `per_page` (int, default 20)
- **Returns:** `{"data": [ {article}, ... ], "page", "per_page", "total": 40}`
- Each article: `{"id": int, "title": str, "author": {"name": str, "id": int}, "tags": [str, ...], "word_count": int}`

### `GET /v1/unreliable`  *(transient failures → retry practice)*
Fails with `503` the first `fail_times` requests **per key**, then succeeds. Use it to
watch `Retry` + exponential backoff recover.
- **Query params:** `key` (str, default `"default"`), `fail_times` (int, default 2)
- **Returns:** `503` until the threshold, then `{"data": [ {event}, ...5 ], "attempts": <int>}`
- Call `POST /admin/reset` first to make the demo repeatable.

### `GET /v1/rate-limited`  *(429 + Retry-After)*
Returns `429 Too Many Requests` with a **`Retry-After: 1`** header on the first request
**per key**, then succeeds.
- **Query params:** `key` (str, default `"rl"`)
- **Returns:** `429` (first) then `{"data": [...], "attempts": <int>}`
- Call `POST /admin/reset` first to reset the counter.

### `POST /v1/summarize`  *(POST with a JSON body)*
Mimics an LLM completion endpoint deterministically (no real model).
- **Body (JSON):** `{"text": "<str, required>", "model": "<str, default 'atlas-pro'>"}`
- **Returns:**
  ```json
  {"id": "sum_XXXXX", "model": "atlas-pro", "summary": "<first sentence>",
   "input_tokens": <int>, "output_tokens": <int>}
  ```
- Send with `requests.post(url, json=payload, ...)` — use `json=`, not `data=`.
- **Errors:** `400` if `text` is empty; `401` without a token.

### `POST /admin/reset`  *(no auth)*
Clears the per-key counters used by `/v1/unreliable` and `/v1/rate-limited` so the
retry demos are repeatable.
- **Returns:** `{"reset": true}`

---

## The `event` object

```json
{
  "event_id": 1000,
  "user_id": 10042,
  "created_at": "2024-04-25T02:00:00",
  "model": "atlas-pro",
  "category": "code",          // null ~8% of the time (data-quality lessons)
  "region": "AMER",
  "score": 81.35,              // eval score 0-100
  "input_tokens": 512,
  "output_tokens": 240,
  "latency_ms": 612.4,
  "response_length": 960       // ~ output_tokens * (3-5); correlates with output_tokens
}
```

**Data facts useful for the labs**
- 252 rows total = 250 unique + **2 exact duplicate rows** (for `df.duplicated()`).
- `category` is null in **20 rows (~7.9%)** — above a 5% flag threshold.
- `response_length` is derived from `output_tokens`, so those two correlate strongly.

---

## Which lab uses what

| Lab | Uses the server? | Endpoints |
|---|---|---|
| **Lab 1 — HTTP with requests** | Yes (port 8001) | `/health`, `/v1/events`, `/v1/events/cursor`, `/v1/unreliable`, `/v1/rate-limited`, `/v1/summarize`, `/v1/articles/nested`, `/admin/reset` |
| **Lab 2 — Jupyter EDA** | Yes (port 8002) | `/health`, `/v1/events`, `/v1/articles/nested` |
| **Lab 3 — Modules & tests** | No — uses `responses` mocking | (none; fully offline) |
