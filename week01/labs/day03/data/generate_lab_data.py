"""
generate_lab_data.py
=====================
Reproducible synthetic data generator for the Week 1 / Day 3 pandas labs.

Run:
    python generate_lab_data.py            # writes all files next to this script
    python generate_lab_data.py --check    # regenerate to /tmp and diff row counts / schemas

Design notes
------------
* One coherent, clearly-fictional world: a made-up AI-assistant platform.
  `users` subscribe to a plan; `events` are individual model interactions with
  eval scores and token counts. No real company, product, or customer data.
* Values are drawn with the MODERN NumPy RNG (numpy.random.default_rng(seed));
  every table gets its own seed so it is independently reproducible.
* Written with pandas 3.x. CSVs are the "raw source" inputs the labs ingest;
  the labs themselves produce Parquet. Two JSON files feed the read_json /
  json_normalize exercises.

Files produced
--------------
  users.csv               2000 users; plan has a few nulls (isna lessons)
  events.csv              8000 model-interaction events (FK user_id -> users)
  events_sample.json      first 200 events, orient="records" (read_json)
  sessions_nested.json    nested payload for json_normalize
  monthly_tokens_long.csv small long-form table for pivot / melt

Target environment: Python 3.13, pandas 3.x, pyarrow.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))

PLANS = ["free", "pro", "enterprise"]
REGIONS = ["AMER", "EU", "APAC", "LATAM"]  # "NA" avoided: it is a CSV null sentinel
# Clearly-fictional model names (no real vendors/trademarks).
MODELS = ["atlas-mini", "atlas-pro", "nova-4", "orion-8b"]

N_USERS = 2000
N_EVENTS = 8000
USER_ID_START = 10_000


# --------------------------------------------------------------------------- #
# users
# --------------------------------------------------------------------------- #
def make_users() -> pd.DataFrame:
    rng = np.random.default_rng(seed=3001)
    user_id = np.arange(USER_ID_START, USER_ID_START + N_USERS, dtype=np.int32)

    # signup dates across 2023-2024
    start = np.datetime64("2023-01-01")
    day_offsets = rng.integers(0, 730, size=N_USERS)
    signup_date = start + day_offsets.astype("timedelta64[D]")

    plan = rng.choice(PLANS, size=N_USERS, p=[0.61, 0.28, 0.11])
    region = rng.choice(REGIONS, size=N_USERS, p=[0.45, 0.30, 0.18, 0.07])

    # seats scale with plan
    seats = np.ones(N_USERS, dtype=np.int32)
    seats[plan == "pro"] = rng.integers(1, 6, size=(plan == "pro").sum())
    seats[plan == "enterprise"] = rng.integers(5, 51, size=(plan == "enterprise").sum())

    df = pd.DataFrame(
        {
            "user_id": user_id,
            "signup_date": signup_date,
            "plan": plan,
            "region": region,
            "seats": seats,
        }
    )

    # Inject ~1% nulls into plan (for isna lessons). Use object so NaN survives CSV.
    null_idx = rng.choice(N_USERS, size=int(0.01 * N_USERS), replace=False)
    df["plan"] = df["plan"].astype(object)
    df.loc[null_idx, "plan"] = np.nan
    return df


# --------------------------------------------------------------------------- #
# events
# --------------------------------------------------------------------------- #
def make_events(users: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(seed=3002)
    event_id = np.arange(1, N_EVENTS + 1, dtype=np.int32)

    # Each event belongs to a random user (many-to-one -> merge validate="1:m").
    u_pos = rng.integers(0, N_USERS, size=N_EVENTS)
    user_id = users["user_id"].to_numpy()[u_pos]
    signup = users["signup_date"].to_numpy()[u_pos]

    # event_date is signup + [0, 400] days (so days_active >= 0)
    gap_days = rng.integers(0, 400, size=N_EVENTS).astype("timedelta64[D]")
    event_date = signup + gap_days

    model = rng.choice(MODELS, size=N_EVENTS, p=[0.40, 0.25, 0.20, 0.15])

    # eval score 0-100, skewed high; clip to [0, 100]
    score = np.clip(rng.normal(74, 14, size=N_EVENTS), 0, 100).astype(np.float64)

    input_tokens = rng.integers(20, 4000, size=N_EVENTS, dtype=np.int32)
    output_tokens = rng.integers(5, 1500, size=N_EVENTS, dtype=np.int32)
    latency_ms = np.clip(rng.normal(600, 220, size=N_EVENTS), 20, None).round(1)

    df = pd.DataFrame(
        {
            "event_id": event_id,
            "user_id": user_id.astype(np.int32),
            "event_date": event_date,
            "model": model,
            "score": score.round(2),
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "latency_ms": latency_ms,
        }
    )
    return df


# --------------------------------------------------------------------------- #
# JSON payloads
# --------------------------------------------------------------------------- #
def make_events_sample_json(events: pd.DataFrame) -> str:
    """First 200 events as an orient='records' JSON string."""
    sample = events.head(200).copy()
    sample["event_date"] = sample["event_date"].astype("datetime64[us]").astype(str)
    return sample.to_json(orient="records")


def make_sessions_nested(events: pd.DataFrame) -> dict:
    """A nested payload: sessions, each with meta + a list of event dicts.

    Shaped for pandas.json_normalize(record_path=..., meta=...).
    """
    rng = np.random.default_rng(seed=3003)
    sub = events.head(60).copy()
    sub["event_date"] = sub["event_date"].astype("datetime64[us]").astype(str)

    sessions = []
    for s in range(6):
        rows = sub.iloc[s * 10 : (s + 1) * 10]
        sessions.append(
            {
                "session_id": f"sess_{1000 + s}",
                "user_id": int(rows.iloc[0]["user_id"]),
                "events": [
                    {
                        "event_id": int(r.event_id),
                        "model": r.model,
                        "score": float(r.score),
                    }
                    for r in rows.itertuples()
                ],
            }
        )
    return {"generated": "synthetic", "data": sessions}


# --------------------------------------------------------------------------- #
# long-form monthly usage (for pivot / melt)
# --------------------------------------------------------------------------- #
def make_monthly_tokens_long(events: pd.DataFrame) -> pd.DataFrame:
    """Small tidy/long table: (user_id, month, tokens) for 12 users x 6 months."""
    rng = np.random.default_rng(seed=3004)
    users = np.arange(USER_ID_START, USER_ID_START + 12, dtype=np.int32)
    months = ["2024-01", "2024-02", "2024-03", "2024-04", "2024-05", "2024-06"]
    rows = []
    for u in users:
        for m in months:
            rows.append((int(u), m, int(rng.integers(500, 50_000))))
    return pd.DataFrame(rows, columns=["user_id", "month", "tokens"])


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def build(target_dir: str) -> None:
    os.makedirs(target_dir, exist_ok=True)

    users = make_users()
    events = make_events(users)

    users_path = os.path.join(target_dir, "users.csv")
    users.to_csv(users_path, index=False)
    print(f"  wrote users.csv              rows={len(users):<6} cols={users.shape[1]}")

    events_path = os.path.join(target_dir, "events.csv")
    events.to_csv(events_path, index=False)
    print(f"  wrote events.csv             rows={len(events):<6} cols={events.shape[1]}")

    sample_json = make_events_sample_json(events)
    with open(os.path.join(target_dir, "events_sample.json"), "w") as f:
        f.write(sample_json)
    print("  wrote events_sample.json     (200 records, orient='records')")

    nested = make_sessions_nested(events)
    with open(os.path.join(target_dir, "sessions_nested.json"), "w") as f:
        json.dump(nested, f, indent=2)
    print("  wrote sessions_nested.json   (6 sessions, nested events)")

    long_df = make_monthly_tokens_long(events)
    long_df.to_csv(os.path.join(target_dir, "monthly_tokens_long.csv"), index=False)
    print(f"  wrote monthly_tokens_long.csv rows={len(long_df)} (12 users x 6 months)")


def check() -> int:
    import tempfile

    tmp = tempfile.mkdtemp()
    build(tmp)
    ok = True
    for name in [
        "users.csv",
        "events.csv",
        "events_sample.json",
        "sessions_nested.json",
        "monthly_tokens_long.csv",
    ]:
        a = os.path.join(HERE, name)
        b = os.path.join(tmp, name)
        same = os.path.exists(a) and open(a, "rb").read() == open(b, "rb").read()
        print(f"  {name:<28} {'OK' if same else 'MISMATCH/ABSENT'}")
        ok = ok and same
    return 0 if ok else 1


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--check", action="store_true", help="verify reproducibility")
    args = p.parse_args()
    print(f"pandas {pd.__version__}, NumPy {np.__version__}")
    if args.check:
        sys.exit(check())
    build(HERE)
    print("Done.")
