"""
lab_support.py
Week 5, Lab 01: LoRA Catalog Normalization (Cordwell Home and Hardware)

Pre-written plumbing so the lesson concept dominates your active time.
Provided here, per the lab briefing slide:
  data generation and loading, the JSON schema validator, the metric
  computation, the evaluation loop, plotting, and the backend selector.

You write: the LoraConfig, the SFTConfig, the trainer.train() call,
and the data quality audit in Part B.

Backends (TRAIN_BACKEND environment variable):
  peft_mps  (default)  real fine-tune with peft + trl on MPS or CPU
  mlx                  Apple-native path, see APPENDIX_MLX.md (not wired here)
  offline              tiny built-in stand-in model plus deterministic
                       scripted generation, no downloads, no live model

Verified against: torch 2.13.0, transformers 5.14.1, peft 0.19.1,
trl 1.9.1, datasets 5.0.0. Python 3.12, runs unchanged on 3.13.
"""

import os

# Set MPS fallback and offline flags BEFORE importing torch or datasets.
# Some PyTorch ops still lack Metal kernels; the fallback runs them on CPU
# per-op instead of crashing. Harmless on machines without MPS.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("WANDB_DISABLED", "true")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

TRAIN_BACKEND = os.environ.get("TRAIN_BACKEND", "peft_mps").strip().lower()
_VALID_BACKENDS = ("peft_mps", "mlx", "offline")

if TRAIN_BACKEND == "offline":
    # Offline env vars must be set before the datasets import.
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import hashlib
import json
import random
import re
from pathlib import Path

# Model ID lives in config, never hard-coded at call sites.
BASE_MODEL_DIR = Path(os.environ.get("W5L1_MODEL_DIR", "~/models/SmolLM2-360M-Instruct")).expanduser().resolve()  # config variable, do not hard-code model paths
BASE_MODEL = str(BASE_MODEL_DIR)  # config variable, do not hard-code model paths

SYSTEM_PROMPT = "Extract product attributes as strict JSON."

DATA_DIR = os.environ.get("LAB_DATA_DIR", "./data")
TRAIN_FILE = os.path.join(DATA_DIR, "cordwell_sft.jsonl")
CLEAN_FILE = os.path.join(DATA_DIR, "cordwell_sft_clean.jsonl")
EVAL_FILE = os.path.join(DATA_DIR, "cordwell_eval.jsonl")

SEED = 42
N_TRAIN = 450
N_EVAL = 50

REQUIRED_FIELDS = ("sku_mfg", "category", "material", "size_nominal_in",
                   "pack_qty", "attributes")

TAXONOMY = (
    "plumbing.valves.ball",
    "plumbing.valves.gate",
    "plumbing.fittings.elbow",
    "fasteners.screws.deck",
    "fasteners.bolts.hex",
    "electrical.wire.thhn",
    "paint.interior.eggshell",
    "tools.hand.hammer",
)


# ---------------------------------------------------------------------------
# Device selection. Never assume CUDA. The cohort Macs have no GPU.
# ---------------------------------------------------------------------------

def pick_device() -> str:
    """Auto-select device best-effort: CUDA, then Apple MPS, then CPU."""
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


DEVICE = pick_device()


def backend_banner() -> None:
    """Print the active backend and device, and fail loudly on a bad value."""
    if TRAIN_BACKEND not in _VALID_BACKENDS:
        raise ValueError(
            f"TRAIN_BACKEND={TRAIN_BACKEND!r} is not one of {_VALID_BACKENDS}. "
            "Set the environment variable before launching Jupyter. "
            "Explicit selection, no silent fallback."
        )
    if TRAIN_BACKEND == "mlx":
        raise NotImplementedError(
            "The mlx path is documented in APPENDIX_MLX.md and runs from the "
            "command line, not from this notebook. Use TRAIN_BACKEND=peft_mps "
            "or TRAIN_BACKEND=offline here."
        )
    print(f"TRAIN_BACKEND = {TRAIN_BACKEND}")
    print(f"DEVICE        = {DEVICE}")
    print(f"BASE_MODEL    = {BASE_MODEL}"
          + ("  (ignored in offline mode)" if TRAIN_BACKEND == "offline" else ""))


# ---------------------------------------------------------------------------
# Synthetic data generation: Cordwell supplier blurbs and gold labels.
# Fully deterministic from SEED. Regenerated inside the lab, no external
# dependencies, no cross-lab data.
# ---------------------------------------------------------------------------

_SIZES_IN = [0.25, 0.375, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0]

_SIZE_TEXT = {0.25: "1/4in", 0.375: "3/8in", 0.5: "1/2in", 0.75: "3/4in",
              1.0: "1in", 1.25: "1-1/4in", 1.5: "1-1/2in", 2.0: "2in"}

_PRODUCTS = [
    # (category, materials, sized, attribute builder keys)
    ("plumbing.valves.ball", ["brass", "pvc", "stainless"], True,
     {"port": ["full", "standard"], "pressure_rating": ["600 WOG", "150 WSP"],
      "end_type": ["FNPT", "sweat", "push-fit"]}),
    ("plumbing.valves.gate", ["brass", "bronze"], True,
     {"stem": ["rising", "non-rising"], "pressure_rating": ["200 WOG", "125 WSP"]}),
    ("plumbing.fittings.elbow", ["copper", "pvc", "black-iron"], True,
     {"angle": ["90", "45"], "end_type": ["sweat", "FNPT", "slip"]}),
    ("fasteners.screws.deck", ["coated-steel", "stainless"], True,
     {"drive": ["T25", "square"], "thread": ["coarse", "fine"]}),
    ("fasteners.bolts.hex", ["zinc-steel", "stainless"], True,
     {"grade": ["5", "8"], "thread": ["coarse", "fine"]}),
    ("electrical.wire.thhn", ["copper"], False,
     {"gauge": ["12 AWG", "14 AWG", "10 AWG"], "jacket": ["THHN", "THWN-2"]}),
    ("paint.interior.eggshell", ["acrylic-latex"], False,
     {"finish": ["eggshell"], "coverage": ["400 sqft", "350 sqft"]}),
    ("tools.hand.hammer", ["steel", "fiberglass"], False,
     {"head_oz": ["16", "20", "22"], "handle": ["fiberglass", "hickory", "steel"]}),
]

_NOUNS = {
    "plumbing.valves.ball": "ball valve",
    "plumbing.valves.gate": "gate valve",
    "plumbing.fittings.elbow": "elbow fitting",
    "fasteners.screws.deck": "deck screws",
    "fasteners.bolts.hex": "hex bolts",
    "electrical.wire.thhn": "building wire",
    "paint.interior.eggshell": "interior paint",
    "tools.hand.hammer": "claw hammer",
}

_SKU_PREFIX = {
    "plumbing.valves.ball": "BV", "plumbing.valves.gate": "GV",
    "plumbing.fittings.elbow": "EL", "fasteners.screws.deck": "DS",
    "fasteners.bolts.hex": "HB", "electrical.wire.thhn": "WR",
    "paint.interior.eggshell": "PT", "tools.hand.hammer": "HM",
}


def _make_gold(rng: random.Random, idx: int) -> dict:
    category, materials, sized, attr_space = _PRODUCTS[idx % len(_PRODUCTS)]
    material = rng.choice(materials)
    size = rng.choice(_SIZES_IN) if sized else None
    pack = rng.choice([1, 6, 10, 12, 24, 50, 100])
    attrs = {k: rng.choice(v) for k, v in attr_space.items()}
    sku = f"{_SKU_PREFIX[category]}-{rng.randint(1000, 9899)}"
    return {"sku_mfg": sku, "category": category, "material": material,
            "size_nominal_in": size, "pack_qty": pack, "attributes": attrs}


def _make_blurb(rng: random.Random, gold: dict) -> str:
    """Render a supplier blurb in one of several supplier house styles."""
    noun = _NOUNS[gold["category"]]
    size_txt = _SIZE_TEXT.get(gold["size_nominal_in"], "")
    mat = gold["material"].replace("-", " ")
    attrs = gold["attributes"]
    attr_bits = [f"{v}" if k in ("gauge", "pressure_rating") else f"{v} {k.replace('_', ' ')}"
                 for k, v in attrs.items()]
    style = rng.randrange(4)
    if style == 0:
        parts = ["HEAVY DUTY" if rng.random() < 0.5 else "PRO GRADE",
                 size_txt, mat.upper(), noun + ",",
                 ", ".join(attr_bits) + ",",
                 f"sold {gold['pack_qty']}/case.",
                 f"MFG#{gold['sku_mfg']}."]
        return " ".join(p for p in parts if p)
    if style == 1:
        parts = [size_txt, mat, noun, "|",
                 " | ".join(attr_bits), "|",
                 f"pk {gold['pack_qty']}", "|",
                 f"mfg {gold['sku_mfg']}"]
        return " ".join(p for p in parts if p)
    if style == 2:
        lead = f"{mat.title()} {noun}"
        if size_txt:
            lead += f", {size_txt} nominal"
        return (f"{lead}. Features {', '.join(attr_bits)}. "
                f"Case quantity {gold['pack_qty']}. "
                f"Manufacturer part {gold['sku_mfg']}.")
    parts = [f"{gold['sku_mfg']}:", size_txt, mat, noun + ";",
             "; ".join(attr_bits) + ";",
             f"qty {gold['pack_qty']}"]
    return " ".join(p for p in parts if p)


def _record(rid: str, gold: dict, blurb: str) -> dict:
    return {"rid": rid, "messages": [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": blurb},
        {"role": "assistant", "content": json.dumps(gold)}]}


def build_datasets(force: bool = False) -> dict:
    """Generate the training file (with planted defects) and the clean eval file.

    Deterministic from SEED. Returns a small summary dict.
    The three planted defect types are the subject of Part B; the specific
    record ids live in the instructor solution, not here.
    """
    os.makedirs(DATA_DIR, exist_ok=True)
    if (not force and os.path.exists(TRAIN_FILE) and os.path.exists(EVAL_FILE)):
        n_train = sum(1 for _ in open(TRAIN_FILE))
        n_eval = sum(1 for _ in open(EVAL_FILE))
        return {"train": n_train, "eval": n_eval, "regenerated": False}

    rng = random.Random(SEED)
    train_records = []
    for i in range(N_TRAIN):
        gold = _make_gold(rng, rng.randrange(len(_PRODUCTS)))
        blurb = _make_blurb(rng, gold)
        train_records.append(_record(f"T{i:04d}", gold, blurb))

    # Plant three defect types at deterministic positions.
    defect_rng = random.Random(SEED + 1)
    all_idx = list(range(N_TRAIN))
    defect_rng.shuffle(all_idx)
    idx_invalid = sorted(all_idx[0:8])     # type 1: assistant not valid JSON
    idx_drift = sorted(all_idx[8:18])      # type 2: sku_mfg key renamed mfg_sku
    idx_unit = []                          # type 3: inches replaced by millimeters
    for i in all_idx[18:]:
        gold = json.loads(train_records[i]["messages"][2]["content"])
        if gold["size_nominal_in"] is not None:
            idx_unit.append(i)
        if len(idx_unit) == 6:
            break
    idx_unit = sorted(idx_unit)

    for i in idx_invalid:
        content = train_records[i]["messages"][2]["content"]
        # Python-repr style: single quotes. json.loads rejects it.
        train_records[i]["messages"][2]["content"] = content.replace('"', "'")
    for i in idx_drift:
        gold = json.loads(train_records[i]["messages"][2]["content"])
        gold = {("mfg_sku" if k == "sku_mfg" else k): v for k, v in gold.items()}
        train_records[i]["messages"][2]["content"] = json.dumps(gold)
    for i in idx_unit:
        gold = json.loads(train_records[i]["messages"][2]["content"])
        gold["size_nominal_in"] = round(gold["size_nominal_in"] * 25.4, 2)
        train_records[i]["messages"][2]["content"] = json.dumps(gold)

    with open(TRAIN_FILE, "w") as f:
        for r in train_records:
            f.write(json.dumps(r) + "\n")

    eval_records = []
    for i in range(N_EVAL):
        gold = _make_gold(rng, rng.randrange(len(_PRODUCTS)))
        blurb = _make_blurb(rng, gold)
        eval_records.append(_record(f"E{i:04d}", gold, blurb))
    with open(EVAL_FILE, "w") as f:
        for r in eval_records:
            f.write(json.dumps(r) + "\n")

    return {"train": N_TRAIN, "eval": N_EVAL, "regenerated": True}


def load_jsonl(path: str) -> list:
    """Load a jsonl file into a list of dicts."""
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def gold_of(record: dict) -> dict:
    """Parse the gold assistant JSON from an eval record."""
    return json.loads(record["messages"][2]["content"])


def blurb_of(record: dict) -> str:
    """The supplier blurb (user message) from a record."""
    return record["messages"][1]["content"]


# ---------------------------------------------------------------------------
# Schema validator and helpers. Pre-written: parsing plumbing is not the
# lesson. Part B uses these to audit the training data.
# ---------------------------------------------------------------------------

def validate_output(text: str) -> tuple:
    """Strict schema check on a raw model output string.

    Returns (ok, problems). ok is True only when the WHOLE string parses as
    JSON and conforms to the target schema. No fence-stripping, no partial
    credit: the downstream pipeline consumes this string as-is.
    """
    problems = []
    try:
        obj = json.loads(text)
    except (json.JSONDecodeError, TypeError) as exc:
        return False, [f"not valid JSON: {exc}"]
    if not isinstance(obj, dict):
        return False, ["top level is not an object"]
    for field in REQUIRED_FIELDS:
        if field not in obj:
            problems.append(f"missing field: {field}")
    extras = [k for k in obj if k not in REQUIRED_FIELDS]
    for k in extras:
        problems.append(f"unexpected field: {k}")
    if problems:
        return False, problems
    if not isinstance(obj["sku_mfg"], str) or not obj["sku_mfg"]:
        problems.append("sku_mfg must be a non-empty string")
    if obj["category"] not in TAXONOMY:
        problems.append(f"category not in taxonomy: {obj['category']!r}")
    if not isinstance(obj["material"], str) or not obj["material"]:
        problems.append("material must be a non-empty string")
    if obj["size_nominal_in"] is not None and not isinstance(obj["size_nominal_in"], (int, float)):
        problems.append("size_nominal_in must be a number or null")
    if not isinstance(obj["pack_qty"], int) or isinstance(obj["pack_qty"], bool):
        problems.append("pack_qty must be an integer")
    if not isinstance(obj["attributes"], dict):
        problems.append("attributes must be an object")
    return (len(problems) == 0), problems


_FRACTIONS = {"1/4": 0.25, "3/8": 0.375, "1/2": 0.5, "3/4": 0.75,
              "1": 1.0, "1-1/4": 1.25, "1-1/2": 1.5, "2": 2.0}
_SIZE_RE = re.compile(r"\b(1-1/4|1-1/2|1/4|3/8|1/2|3/4|1|2)in\b")


def extract_size_from_blurb(blurb: str):
    """Best-effort nominal size in inches parsed from the blurb text.

    Returns a float or None. Pre-written so Part B is about the data quality
    concept, not about regex. Use it to cross-check a labeled size against
    what the blurb actually says.
    """
    m = _SIZE_RE.search(blurb)
    if not m:
        return None
    return _FRACTIONS[m.group(1)]


# ---------------------------------------------------------------------------
# Metrics. All three are deterministic for this task, which is why the task
# was chosen. Defined over ALL eval records: an output that fails to parse
# scores zero on every field it failed to deliver.
# ---------------------------------------------------------------------------

def evaluate_outputs(outputs: list, eval_records: list) -> dict:
    """Score raw output strings against gold eval records.

    Returns a dict:
      schema_valid_rate   fraction of outputs passing validate_output
      category_accuracy   fraction with exactly correct category string
      per_field           dict field -> exact-match fraction
      failures            list of (rid, problems) for schema-invalid outputs
    """
    assert len(outputs) == len(eval_records), "one output per eval record"
    n = len(eval_records)
    valid = 0
    cat_ok = 0
    field_hits = {f: 0 for f in REQUIRED_FIELDS}
    failures = []
    for out, rec in zip(outputs, eval_records):
        gold = gold_of(rec)
        ok, problems = validate_output(out)
        if ok:
            valid += 1
            obj = json.loads(out)
            if obj["category"] == gold["category"]:
                cat_ok += 1
            for f in REQUIRED_FIELDS:
                if obj.get(f) == gold[f]:
                    field_hits[f] += 1
        else:
            failures.append((rec["rid"], problems))
    return {
        "n": n,
        "schema_valid_rate": valid / n,
        "category_accuracy": cat_ok / n,
        "per_field": {f: field_hits[f] / n for f in REQUIRED_FIELDS},
        "failures": failures,
    }


def print_results(name: str, results: dict) -> None:
    """Readable one-block summary of an evaluate_outputs result."""
    print(f"=== {name} (n={results['n']}) ===")
    print(f"schema_valid_rate: {results['schema_valid_rate']:.0%}")
    print(f"category_accuracy: {results['category_accuracy']:.0%}")
    print("per-field exact match:")
    for f, v in results["per_field"].items():
        print(f"  {f:<16} {v:.0%}")
    if results["failures"]:
        rid, problems = results["failures"][0]
        print(f"example failure ({rid}): {problems[0]}")


# ---------------------------------------------------------------------------
# Capability regression set (stretch Part G). Ten generic prompts a healthy
# instruct model answers in plain prose. Keyword scoring keeps it
# deterministic and dependency-free.
# ---------------------------------------------------------------------------

CAPABILITY_PROMPTS = [
    ("What is a unit test?", ["test"]),
    ("Name one benefit of code review.", ["review", "bug", "quality", "catch", "knowledge"]),
    ("What does an HTTP 404 status mean?", ["not found", "404", "resource", "exist"]),
    ("Explain what a cache does, in one sentence.", ["cache", "store", "fast", "reuse"]),
    ("What is the capital of France?", ["paris"]),
    ("What does SQL stand for?", ["structured query language", "sql"]),
    ("Give one reason to pin dependency versions.", ["reproduc", "same", "consistent", "break", "version"]),
    ("What is version control used for?", ["track", "history", "change", "collaborat"]),
    ("What does an API do?", ["interface", "communicat", "request", "program"]),
    ("Why write documentation?", ["understand", "explain", "future", "onboard", "reference", "maintain"]),
]


def score_capability(outputs: list) -> dict:
    """Fraction of generic prompts answered with a plausible prose answer.

    A response scores 1 when it contains any expected keyword AND does not
    look like the catalog JSON schema leaking into general conversation.
    Deterministic and crude by design: it is a smoke alarm, not a judge.
    """
    assert len(outputs) == len(CAPABILITY_PROMPTS)
    hits = []
    for out, (_prompt, keywords) in zip(outputs, CAPABILITY_PROMPTS):
        low = out.lower()
        keyword_ok = any(k in low for k in keywords)
        format_bleed = '"sku_mfg"' in out or '"pack_qty"' in out
        hits.append(1 if (keyword_ok and not format_bleed) else 0)
    return {"n": len(hits), "capability_score": sum(hits) / len(hits),
            "per_prompt": hits}


# ---------------------------------------------------------------------------
# Offline scripted generation. Deterministic per record and phase via
# SHA-256 (never the built-in hash(): it is salted per process and not
# reproducible). Mimics the real failure modes of a small instruct model
# so the metrics and plots teach the same lesson without a live model.
# ---------------------------------------------------------------------------

def _bucket(key: str) -> int:
    return int(hashlib.sha256(key.encode()).hexdigest(), 16) % 1000


def _noisy_gold(gold: dict, key: str) -> dict:
    """Copy of gold with one deterministic wrong attribute value."""
    obj = json.loads(json.dumps(gold))
    if obj["attributes"]:
        keys = sorted(obj["attributes"])
        k = keys[_bucket(key + "attr") % len(keys)]
        obj["attributes"][k] = "unknown"
    return obj


def _scripted_output(record: dict, phase: str) -> str:
    gold = gold_of(record)
    b = _bucket(f"{record['rid']}|{phase}")
    if phase == "baseline":
        if b < 240:
            return json.dumps(gold)
        if b < 390:
            return json.dumps(_noisy_gold(gold, record["rid"]))
        if b < 500:
            return ("Here are the extracted attributes for this product:\n"
                    + json.dumps(gold, indent=2))
        if b < 640:
            return "```json\n" + json.dumps(gold, indent=2) + "\n```"
        if b < 780:
            return json.dumps(gold).replace('"', "'")
        if b < 880:
            drifted = {("mfg_sku" if k == "sku_mfg" else "size" if k == "size_nominal_in" else k): v
                       for k, v in gold.items()}
            return json.dumps(drifted)
        text = json.dumps(gold)
        return text[: int(len(text) * 0.6)]
    if phase == "tuned_r16":
        if b < 830:
            return json.dumps(gold)
        if b < 900:
            # Valid JSON, wrong taxonomy leaf: schema-valid is not correct.
            obj = json.loads(json.dumps(gold))
            siblings = [c for c in TAXONOMY if c != obj["category"]]
            obj["category"] = siblings[_bucket(record["rid"] + "cat") % len(siblings)]
            return json.dumps(obj)
        if b < 960:
            return json.dumps(_noisy_gold(gold, record["rid"]))
        drifted = {("mfg_sku" if k == "sku_mfg" else k): v for k, v in gold.items()}
        return json.dumps(drifted)
    if phase == "tuned_r4":
        if b < 700:
            return json.dumps(gold)
        if b < 790:
            obj = json.loads(json.dumps(gold))
            siblings = [c for c in TAXONOMY if c != obj["category"]]
            obj["category"] = siblings[_bucket(record["rid"] + "cat4") % len(siblings)]
            return json.dumps(obj)
        if b < 860:
            return json.dumps(_noisy_gold(gold, record["rid"]))
        if b < 930:
            drifted = {("mfg_sku" if k == "sku_mfg" else k): v for k, v in gold.items()}
            return json.dumps(drifted)
        return "```json\n" + json.dumps(gold) + "\n```"
    raise ValueError(f"unknown phase: {phase}")


def _scripted_capability(phase: str) -> list:
    """Scripted answers to CAPABILITY_PROMPTS for baseline and tuned models."""
    answers = [
        "A unit test checks one small piece of code in isolation.",
        "Code review catches bugs before they ship and spreads knowledge.",
        "HTTP 404 means the requested resource was not found on the server.",
        "A cache stores results so repeated requests are served fast.",
        "The capital of France is Paris.",
        "SQL stands for Structured Query Language.",
        "Pinned versions make builds reproducible across machines.",
        "Version control tracks changes and lets teams collaborate on history.",
        "An API is an interface that lets programs communicate by request.",
        "Documentation helps future readers understand and maintain the code.",
    ]
    if phase == "baseline":
        return list(answers)
    out = list(answers)
    # Exactly one prompt degrades: format bleed, where the narrow fine-tune
    # leaks its schema into a general question. Part G surfaces this.
    i = _bucket(f"capdegrade|{phase}") % len(out)
    out[i] = ('{"sku_mfg": null, "category": null, "material": null, '
              '"size_nominal_in": null, "pack_qty": 1, "attributes": {}}')
    return out


# ---------------------------------------------------------------------------
# Offline stand-in model: a tiny Llama-architecture model plus tokenizer,
# built in-process with the SAME module names as the real base model
# (q_proj, v_proj, and friends). Your LoraConfig, SFTConfig, and
# trainer.train() run for real against it. Only the generated text is
# scripted in offline mode; the training API path is genuine.
# ---------------------------------------------------------------------------

_STANDIN_DIR = os.path.join(DATA_DIR, "offline_standin")


def build_offline_standin():
    """Return (model, tokenizer) for a tiny local Llama-style model."""
    import torch
    from transformers import (AutoModelForCausalLM, AutoTokenizer,
                              LlamaConfig, LlamaForCausalLM,
                              PreTrainedTokenizerFast)
    if not os.path.exists(os.path.join(_STANDIN_DIR, "config.json")):
        from tokenizers import Tokenizer, models, trainers, pre_tokenizers
        corpus = []
        if os.path.exists(TRAIN_FILE):
            for rec in load_jsonl(TRAIN_FILE)[:120]:
                corpus.append(blurb_of(rec))
                corpus.append(rec["messages"][2]["content"])
        corpus.append(SYSTEM_PROMPT)
        tok = Tokenizer(models.BPE(unk_token="<unk>"))
        tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        trainer = trainers.BpeTrainer(
            vocab_size=1000,
            special_tokens=["<unk>", "<pad>", "<|im_start|>", "<|im_end|>"])
        tok.train_from_iterator(corpus, trainer)
        hf_tok = PreTrainedTokenizerFast(
            tokenizer_object=tok, unk_token="<unk>", pad_token="<pad>",
            eos_token="<|im_end|>")
        hf_tok.chat_template = (
            "{% for message in messages %}"
            "{{ '<|im_start|>' + message['role'] + '\n' + message['content'] "
            "+ '<|im_end|>' + '\n' }}{% endfor %}"
            "{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}"
            "{% endif %}")
        cfg = LlamaConfig(
            vocab_size=len(hf_tok), hidden_size=64, intermediate_size=128,
            num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
            max_position_embeddings=512)
        torch.manual_seed(SEED)
        model = LlamaForCausalLM(cfg)
        os.makedirs(_STANDIN_DIR, exist_ok=True)
        model.save_pretrained(_STANDIN_DIR)
        hf_tok.save_pretrained(_STANDIN_DIR)
    import torch
    tokenizer = AutoTokenizer.from_pretrained(_STANDIN_DIR)
    model = AutoModelForCausalLM.from_pretrained(_STANDIN_DIR, dtype=torch.float32)
    return model, tokenizer


def load_base_model():
    """Backend-aware base model loader. Returns (model, tokenizer).

    peft_mps: downloads BASE_MODEL (first run only), places it on DEVICE.
              No device_map='auto' on MPS: unified memory cannot offload
              layers the way CUDA can.
    offline:  tiny built-in stand-in, no network.
    """
    backend_banner_ok = TRAIN_BACKEND in _VALID_BACKENDS
    if not backend_banner_ok:
        backend_banner()  # raises with the clear error
    if TRAIN_BACKEND == "offline":
        return build_offline_standin()
    if TRAIN_BACKEND == "mlx":
        backend_banner()  # raises NotImplementedError with pointer
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    dtype = torch.bfloat16 if DEVICE == "mps" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, dtype=dtype)
    model = model.to(DEVICE)
    return model, tokenizer


# ---------------------------------------------------------------------------
# Evaluation loop. Pre-written: generation plumbing is not the lesson.
# ---------------------------------------------------------------------------

def generate_for_eval(model, tokenizer, eval_records: list, phase: str,
                      max_new_tokens: int = 160) -> list:
    """One raw output string per eval record.

    phase is one of: 'baseline', 'tuned_r16', 'tuned_r4'. In offline mode
    the model is ignored and outputs come from the deterministic script.
    In live mode the phase string only labels progress output; what matters
    is which model object you pass in.
    """
    if TRAIN_BACKEND == "offline":
        return [_scripted_output(rec, phase) for rec in eval_records]
    import torch
    outputs = []
    model.eval()
    for i, rec in enumerate(eval_records):
        msgs = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": blurb_of(rec)}]
        # transformers 5: apply_chat_template returns a BatchEncoding by
        # default (return_dict=True). Unpack it into generate.
        inputs = tokenizer.apply_chat_template(
            msgs, tokenize=True, add_generation_prompt=True,
            return_tensors="pt").to(model.device)
        with torch.no_grad():
            gen = model.generate(**inputs, max_new_tokens=max_new_tokens,
                                 do_sample=False,
                                 pad_token_id=tokenizer.pad_token_id
                                 if tokenizer.pad_token_id is not None
                                 else tokenizer.eos_token_id)
        new_tokens = gen[0, inputs["input_ids"].shape[1]:]
        outputs.append(tokenizer.decode(new_tokens, skip_special_tokens=True).strip())
        if (i + 1) % 10 == 0:
            print(f"  [{phase}] generated {i + 1}/{len(eval_records)}")
    return outputs


def generate_capability(model, tokenizer, phase: str,
                        max_new_tokens: int = 60) -> list:
    """Answers to CAPABILITY_PROMPTS. Scripted in offline mode."""
    if TRAIN_BACKEND == "offline":
        return _scripted_capability(phase)
    import torch
    outputs = []
    model.eval()
    for prompt, _kw in CAPABILITY_PROMPTS:
        msgs = [{"role": "user", "content": prompt}]
        inputs = tokenizer.apply_chat_template(
            msgs, tokenize=True, add_generation_prompt=True,
            return_tensors="pt").to(model.device)
        with torch.no_grad():
            gen = model.generate(**inputs, max_new_tokens=max_new_tokens,
                                 do_sample=False,
                                 pad_token_id=tokenizer.pad_token_id
                                 if tokenizer.pad_token_id is not None
                                 else tokenizer.eos_token_id)
        new_tokens = gen[0, inputs["input_ids"].shape[1]:]
        outputs.append(tokenizer.decode(new_tokens, skip_special_tokens=True).strip())
    return outputs


# ---------------------------------------------------------------------------
# Plotting. Pre-written.
# ---------------------------------------------------------------------------

def plot_before_after(results_by_name: dict) -> None:
    """Bar chart of schema-valid rate and category accuracy per run."""
    import matplotlib
    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt
    names = list(results_by_name)
    valid = [results_by_name[n]["schema_valid_rate"] for n in names]
    cat = [results_by_name[n]["category_accuracy"] for n in names]
    x = range(len(names))
    width = 0.38
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar([i - width / 2 for i in x], valid, width, label="schema-valid rate")
    ax.bar([i + width / 2 for i in x], cat, width, label="category accuracy")
    ax.set_xticks(list(x))
    ax.set_xticklabels(names)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("fraction of 50 eval records")
    ax.set_title("Cordwell catalog normalization: before and after")
    for i, v in enumerate(valid):
        ax.text(i - width / 2, v + 0.02, f"{v:.0%}", ha="center", fontsize=9)
    for i, v in enumerate(cat):
        ax.text(i + width / 2, v + 0.02, f"{v:.0%}", ha="center", fontsize=9)
    ax.legend()
    fig.tight_layout()
    plt.show()


def plot_field_breakdown(results_by_name: dict) -> None:
    """Grouped bars of per-field exact match for each run."""
    import matplotlib.pyplot as plt
    names = list(results_by_name)
    fields = list(REQUIRED_FIELDS)
    fig, ax = plt.subplots(figsize=(8, 4))
    width = 0.8 / len(names)
    for j, name in enumerate(names):
        vals = [results_by_name[name]["per_field"][f] for f in fields]
        xs = [i + j * width for i in range(len(fields))]
        ax.bar(xs, vals, width, label=name)
    ax.set_xticks([i + width * (len(names) - 1) / 2 for i in range(len(fields))])
    ax.set_xticklabels(fields, rotation=20, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("exact match fraction")
    ax.set_title("Per-field exact match")
    ax.legend()
    fig.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Soft check harness. check() records PASS or FAIL and never raises, so a
# cold Run All on the stubbed notebook produces zero hard crashes.
# ---------------------------------------------------------------------------

def need(value):
    """Guard for checks that depend on an earlier part.

    Raises NotImplementedError when the value is still None (its part has
    not been implemented yet), so the check reports NOT IMPLEMENTED
    instead of a confusing FAIL. Returns the value otherwise.
    """
    if value is None:
        raise NotImplementedError
    return value


_CHECKS = []


def check(name: str, fn) -> bool:
    """Run fn(); PASS when it returns truthy. Catches every exception.

    A NotImplementedError (an untouched stub) reports as NOT IMPLEMENTED
    rather than FAIL, so you can tell 'not started' from 'wrong'.
    """
    try:
        result = bool(fn())
        status = "PASS" if result else "FAIL"
    except NotImplementedError:
        result, status = False, "NOT IMPLEMENTED"
    except Exception as exc:
        result, status = False, f"FAIL ({type(exc).__name__}: {exc})"
    _CHECKS.append((name, status))
    print(f"[{status}] {name}")
    return result


def checkpoint_summary() -> None:
    """Print the tally of all checks run so far in this kernel session."""
    passed = sum(1 for _, s in _CHECKS if s == "PASS")
    print(f"\n{passed}/{len(_CHECKS)} checks passing")
    for name, status in _CHECKS:
        if status != "PASS":
            print(f"  [{status}] {name}")


def reset_checks() -> None:
    _CHECKS.clear()
