"""Support module for Week 5 Lab 04: Experiment Tracking with Weights & Biases.

Everything in this file is pre-written plumbing. It is the Cordwell RAG
pipeline from Modules 01 through 03 compressed into one deterministic,
dependency-light module so that this lab can focus on tracking. Nothing
here is the lesson. The lesson is what you wrap around it.

The pipeline is real: it chunks a corpus, embeds chunks, retrieves by
cosine similarity, generates an answer from the retrieved context, and
scores the result with the same metric family as Module 03 (context
recall, context precision, faithfulness, answer relevancy, abstention).
It is deterministic so that every student sees the same numbers and the
checks can be exact.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import socket
from pathlib import Path

# ---------------------------------------------------------------------------
# Backend selection. Call configure_wandb_backend() BEFORE importing wandb.
# ---------------------------------------------------------------------------

VALID_BACKENDS = ("offline", "local", "cloud")


def configure_wandb_backend() -> str:
    """Read WANDB_LAB_BACKEND and set the wandb environment variables.

    Must run before `import wandb`, because wandb reads these variables
    at import and init time. Returns the backend name.

    offline (default): WANDB_MODE=offline. No server, no account. Runs
        are written to ./wandb/offline-run-* and can be synced later.
    local: WANDB_BASE_URL=http://localhost:8080. Requires the Docker
        W&B server from the setup guide and a one-time `wandb login
        --host=http://localhost:8080`.
    cloud: the standard hosted endpoint. Requires a free account and
        `wandb login`.
    """
    backend = os.environ.get("WANDB_LAB_BACKEND", "local").strip().lower()
    if backend not in VALID_BACKENDS:
        raise ValueError(
            f"WANDB_LAB_BACKEND={backend!r} is not one of {VALID_BACKENDS}"
        )
    if backend == "offline":
        os.environ["WANDB_MODE"] = "offline"
        os.environ.pop("WANDB_BASE_URL", None)
    elif backend == "local":
        os.environ.pop("WANDB_MODE", None)
        os.environ["WANDB_BASE_URL"] = "http://localhost:8080"
    else:  # cloud
        os.environ.pop("WANDB_MODE", None)
        os.environ.pop("WANDB_BASE_URL", None)
    os.environ.setdefault("WANDB_SILENT", "true")
    return backend


def server_reachable(host: str = "localhost", port: int = 8080, timeout: float = 2.0) -> bool:
    """True if something is listening on the local W&B server port."""
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


# ---------------------------------------------------------------------------
# Corpus: 16 Cordwell Home & Hardware support documents.
# ---------------------------------------------------------------------------

CORPUS = [
    {
        "doc_id": "drillmaster-battery",
        "title": "DrillMaster 20V Battery Care",
        "text": (
            "The DrillMaster 20V uses a lithium ion battery pack rated at 4.0 "
            "amp hours. Charge the battery fully before first use, which takes "
            "about 90 minutes on the rapid charger. Store the battery indoors "
            "between 40 and 80 degrees Fahrenheit. For long term storage keep "
            "the charge level near 50 percent and recharge every three months. "
            "Never store the battery attached to the tool. A battery that will "
            "not hold a charge after 500 cycles has reached the end of its "
            "rated service life and qualifies for the recycling program at any "
            "Cordwell store."
        ),
    },
    {
        "doc_id": "drillmaster-chuck",
        "title": "DrillMaster 20V Chuck and Bit Changes",
        "text": (
            "The DrillMaster 20V has a keyless half inch chuck. To change a "
            "bit, hold the collar and rotate the sleeve counterclockwise to "
            "open the jaws. Insert the bit fully, then rotate the sleeve "
            "clockwise until it clicks twice. The two clicks confirm the "
            "ratcheting lock is engaged. If a bit slips under load, open the "
            "jaws completely, clear any debris from the teeth with a dry "
            "brush, and retighten. Do not use pliers on the sleeve; the "
            "housing is composite and will crack."
        ),
    },
    {
        "doc_id": "trimpro-blade",
        "title": "TrimPro Hedge Trimmer Blade Replacement",
        "text": (
            "Replacement blades for the TrimPro 24 inch hedge trimmer are "
            "sold as part number TP-24-BLD. Disconnect the battery before any "
            "blade work. Remove the six torx T25 screws on the underside of "
            "the bar, lift the guard, and slide the old blade set out toward "
            "the tip. Coat the new blade set with light machine oil before "
            "installation. Torque the screws to 35 inch pounds in a cross "
            "pattern. Blades should be replaced when cutting performance "
            "drops or after roughly 80 hours of use."
        ),
    },
    {
        "doc_id": "aquaflow-cartridge",
        "title": "AquaFlow Kitchen Faucet Cartridge Service",
        "text": (
            "A dripping AquaFlow kitchen faucet almost always needs a new "
            "ceramic cartridge, part number AF-C40. Shut off both supply "
            "valves under the sink first. Pry off the handle cap, remove the "
            "set screw with a 3 millimeter hex key, and lift the handle off. "
            "Unscrew the retaining nut by hand and pull the cartridge "
            "straight up. Match the tabs on the new cartridge to the slots in "
            "the valve body. The cartridge is covered for five years under "
            "the AquaFlow limited warranty."
        ),
    },
    {
        "doc_id": "paintperfect-cleaning",
        "title": "PaintPerfect Sprayer Cleaning",
        "text": (
            "Clean the PaintPerfect airless sprayer immediately after each "
            "use. For latex paint flush with warm water; for oil based paint "
            "flush with mineral spirits. Run the flush liquid through the "
            "pump until it comes out clear, usually two to three quarts. "
            "Remove the spray tip and soak it in the matching solvent for "
            "ten minutes, then clear the orifice with the supplied pick. "
            "Never use a wire brush on the tip. Store the pump with a "
            "tablespoon of pump preserver in the fluid section."
        ),
    },
    {
        "doc_id": "warranty-policy",
        "title": "Cordwell Tool Warranty Policy",
        "text": (
            "Cordwell brand power tools carry a three year limited warranty "
            "from the date of purchase. Batteries and chargers carry a two "
            "year warranty. The warranty covers defects in materials and "
            "workmanship. It does not cover normal wear items such as blades, "
            "brushes, and spray tips, and it does not cover damage from "
            "misuse or unauthorized repair. A receipt or order number is "
            "required for all warranty claims. Claims are started at the "
            "service desk of any store or through the online portal."
        ),
    },
    {
        "doc_id": "returns-policy",
        "title": "Cordwell Returns and Exchanges",
        "text": (
            "Unused items in original packaging may be returned within 90 "
            "days with a receipt for a full refund to the original payment "
            "method. Opened power tools may be returned within 30 days if "
            "all parts and accessories are included. Gas powered equipment "
            "that has been fueled cannot be returned and is serviced under "
            "warranty instead. Custom cut lumber, mixed paint, and special "
            "orders are final sale. Without a receipt, returns are issued as "
            "store credit at the lowest recent selling price."
        ),
    },
    {
        "doc_id": "deckshield-coverage",
        "title": "DeckShield Stain Coverage and Drying",
        "text": (
            "One gallon of DeckShield semi transparent stain covers about "
            "250 square feet on smooth wood and about 150 square feet on "
            "rough sawn wood. Apply with a pad applicator in thin coats. The "
            "surface is dry to the touch in two hours and ready for light "
            "foot traffic in 24 hours. Wait 48 hours before replacing "
            "furniture. Do not apply DeckShield if rain is expected within "
            "12 hours or if the surface temperature is below 50 degrees "
            "Fahrenheit."
        ),
    },
    {
        "doc_id": "gripfast-anchors",
        "title": "GripFast Drywall Anchor Load Ratings",
        "text": (
            "GripFast self drilling drywall anchors are rated by size. The "
            "small blue anchor holds up to 30 pounds, the medium gray anchor "
            "holds up to 50 pounds, and the large black anchor holds up to "
            "75 pounds in half inch drywall. Ratings assume a static load "
            "pulling straight down. For shelving, divide the rating by two "
            "to allow for lever forces. GripFast anchors are not rated for "
            "ceiling mounts; use a toggle bolt into a joist for any "
            "overhead load."
        ),
    },
    {
        "doc_id": "flowmax-filter",
        "title": "FlowMax Shop Vacuum Filter Guide",
        "text": (
            "The FlowMax 12 gallon shop vacuum ships with a standard "
            "cartridge filter for dry debris. For drywall dust or fine "
            "sawdust install the high efficiency filter, part number "
            "FM-HE12, or the vacuum will exhaust fine dust back into the "
            "room. For wet pickup remove the cartridge filter entirely and "
            "install the foam sleeve. Tap the cartridge filter clean after "
            "each use and replace it when the pleats stay gray after "
            "tapping."
        ),
    },
    {
        "doc_id": "levelline-laser",
        "title": "LevelLine Laser Level Calibration Check",
        "text": (
            "Check LevelLine laser calibration monthly. Place the unit 15 "
            "feet from a wall, mark the beam center, rotate the unit 180 "
            "degrees, and mark again. If the two marks differ by more than "
            "an eighth of an inch, the unit needs factory recalibration. "
            "The self leveling pendulum must be unlocked during use and "
            "locked for transport. A flashing beam means the unit is out of "
            "its four degree self leveling range; reposition the base."
        ),
    },
    {
        "doc_id": "mulchmaster-line",
        "title": "MulchMaster String Trimmer Line Loading",
        "text": (
            "The MulchMaster 40V string trimmer uses 0.080 inch twisted "
            "line. Cut two lengths of ten feet each. Insert each length "
            "into an eyelet on the spool head and wind both in the "
            "direction of the arrow, keeping the lines parallel and snug. "
            "Leave six inches free at each eyelet. Bump the head lightly on "
            "the ground during use to feed more line. If the line welds "
            "together from heat, unwind and rewind it loosely."
        ),
    },
    {
        "doc_id": "seedstart-schedule",
        "title": "SeedStart Fertilizer Application Schedule",
        "text": (
            "Apply SeedStart lawn fertilizer four times per season. The "
            "first application goes down when soil reaches 55 degrees "
            "Fahrenheit in spring. The second follows eight weeks later. "
            "The third goes down in early fall, and the final winterizer "
            "application goes down after the last mow. Use the drop "
            "spreader setting printed on the bag. Water within 24 hours of "
            "each application unless rain does the work for you. Do not "
            "apply to a stressed or dormant lawn in summer heat."
        ),
    },
    {
        "doc_id": "pipeseal-tape",
        "title": "PipeSeal Thread Tape Usage",
        "text": (
            "Wrap PipeSeal thread tape clockwise when viewed from the pipe "
            "end, so tightening the fitting does not unwrap it. Use three "
            "wraps for plastic fittings and five wraps for metal fittings. "
            "PipeSeal white tape is rated for water lines only. Use the "
            "yellow gas rated tape for any gas connection. Thread tape is "
            "not a substitute for pipe dope on tapered iron fittings larger "
            "than one inch; use both on those joints."
        ),
    },
    {
        "doc_id": "safeglow-detector",
        "title": "SafeGlow Smoke Detector Placement",
        "text": (
            "Install SafeGlow smoke detectors on every level of the home "
            "and inside every bedroom. Mount ceiling units at least four "
            "inches from any wall. Mount wall units between four and twelve "
            "inches below the ceiling. Keep detectors at least ten feet "
            "from cooking appliances to limit nuisance alarms. Replace the "
            "battery every year and the entire detector every ten years. "
            "Test each unit monthly with the test button."
        ),
    },
    {
        "doc_id": "toolbench-assembly",
        "title": "ToolBench Pro Workbench Assembly Notes",
        "text": (
            "The ToolBench Pro ships in two cartons. Assembly requires two "
            "people and about 45 minutes. Attach the legs to the frame with "
            "the sixteen M8 bolts finger tight, square the frame on a flat "
            "floor, then torque all bolts to 18 foot pounds. Install the "
            "bamboo top with the twelve wood screws from underneath, never "
            "from above. The bench is rated for 1500 pounds distributed "
            "load. Level the feet with the adjustable pads before loading "
            "any weight."
        ),
    },
    {
        "doc_id": "catalog-drillmaster",
        "title": "Catalog: DrillMaster 20V Drill Driver",
        "text": (
            "The DrillMaster 20V drill driver delivers pro grade torque in a "
            "compact body. Includes battery, rapid charger, belt clip, and "
            "carry bag. Brushless motor, two speed gearbox, LED work light. "
            "The battery platform is shared across the full Cordwell 20V "
            "line. A favorite for cabinet installs and deck builds."
        ),
    },
    {
        "doc_id": "catalog-trimpro",
        "title": "Catalog: TrimPro 24 Inch Hedge Trimmer",
        "text": (
            "Shape hedges fast with the TrimPro 24 inch hedge trimmer. Dual "
            "action blades cut cleaner with less vibration. Lightweight "
            "housing, wrap around front handle, and tool free debris sweep. "
            "Replacement blades and accessories available at every Cordwell "
            "store and online."
        ),
    },
    {
        "doc_id": "catalog-gripfast",
        "title": "Catalog: GripFast Anchor Assortment Pack",
        "text": (
            "The GripFast anchor assortment pack covers every drywall job. "
            "Self drilling anchors in three sizes with matching screws, "
            "organized in a reusable case. No pilot hole, no wall damage. "
            "Hang shelving, mirrors, and frames with confidence using "
            "GripFast drywall anchors."
        ),
    },
    {
        "doc_id": "catalog-flowmax",
        "title": "Catalog: FlowMax 12 Gallon Shop Vacuum",
        "text": (
            "The FlowMax 12 gallon wet dry shop vacuum handles sawdust, "
            "spills, and jobsite cleanup. Large rear wheels, onboard "
            "accessory storage, and a blower port. Filters and bags for "
            "every pickup type are sold separately. The quiet motor keeps "
            "shop conversations possible."
        ),
    },
    {
        "doc_id": "catalog-mulchmaster",
        "title": "Catalog: MulchMaster 40V String Trimmer",
        "text": (
            "Trim and edge with the MulchMaster 40V string trimmer. "
            "Telescoping shaft, pivoting head, and a bump feed line system. "
            "Shares the 40V battery platform with Cordwell mowers and "
            "blowers. Spools of trimmer line in every size are stocked in "
            "the garden aisle."
        ),
    },
    {
        "doc_id": "catalog-safeglow",
        "title": "Catalog: SafeGlow Smoke Detector Two Pack",
        "text": (
            "Protect the whole home with the SafeGlow smoke detector two "
            "pack. Photoelectric sensing, ten year sealed battery option, "
            "and a low profile design. Simple twist mount installation on "
            "wall or ceiling. Pair with SafeGlow carbon monoxide detectors "
            "for complete coverage."
        ),
    },
    {
        "doc_id": "catalog-pipeseal",
        "title": "Catalog: PipeSeal Tape and Fitting Prep",
        "text": (
            "PipeSeal thread tape seals threaded fittings on water and gas "
            "lines. Stock up on white tape, yellow gas rated tape, and pipe "
            "dope in one trip. Every plumbing repair starts with clean "
            "threads and the right tape. Find PipeSeal products in the "
            "plumbing aisle."
        ),
    },
    {
        "doc_id": "catalog-deckshield",
        "title": "Catalog: DeckShield Stain Family",
        "text": (
            "DeckShield stains protect and beautify outdoor wood. Choose "
            "clear sealer, semi transparent stain, or solid color stain in "
            "forty tones. UV blockers and mildew resistance built in. One "
            "gallon and five gallon sizes. Bring a board sample for a "
            "custom color match at the paint desk."
        ),
    },
]

# ---------------------------------------------------------------------------
# Evaluation query set: 12 queries, 10 answerable, 2 not answerable.
# ---------------------------------------------------------------------------

EVAL_QUERIES = [
    {
        "query_id": "q01",
        "question": "How long does the DrillMaster 20V battery take to charge on the rapid charger?",
        "relevant_doc": "drillmaster-battery",
        "answer_keywords": ["90 minutes", "rapid charger"],
        "answerable": True,
    },
    {
        "query_id": "q02",
        "question": "What part number do I need for replacement TrimPro hedge trimmer blades?",
        "relevant_doc": "trimpro-blade",
        "answer_keywords": ["TP-24-BLD"],
        "answerable": True,
    },
    {
        "query_id": "q03",
        "question": "How many years is the warranty on Cordwell power tool batteries and chargers?",
        "relevant_doc": "warranty-policy",
        "answer_keywords": ["two year", "batteries and chargers"],
        "answerable": True,
    },
    {
        "query_id": "q04",
        "question": "How many days do I have to return an opened power tool?",
        "relevant_doc": "returns-policy",
        "answer_keywords": ["30 days", "opened power tools"],
        "answerable": True,
    },
    {
        "query_id": "q05",
        "question": "How many square feet does a gallon of DeckShield cover on rough sawn wood?",
        "relevant_doc": "deckshield-coverage",
        "answer_keywords": ["150 square feet", "rough sawn"],
        "answerable": True,
    },
    {
        "query_id": "q06",
        "question": "How much weight can the large black GripFast drywall anchor hold?",
        "relevant_doc": "gripfast-anchors",
        "answer_keywords": ["75 pounds", "large black anchor"],
        "answerable": True,
    },
    {
        "query_id": "q07",
        "question": "Which FlowMax filter should I use for drywall dust?",
        "relevant_doc": "flowmax-filter",
        "answer_keywords": ["FM-HE12", "high efficiency filter"],
        "answerable": True,
    },
    {
        "query_id": "q08",
        "question": "What size line does the MulchMaster 40V string trimmer use?",
        "relevant_doc": "mulchmaster-line",
        "answer_keywords": ["0.080 inch", "twisted line"],
        "answerable": True,
    },
    {
        "query_id": "q09",
        "question": "How many wraps of PipeSeal tape should I use on metal fittings?",
        "relevant_doc": "pipeseal-tape",
        "answer_keywords": ["five wraps", "metal fittings"],
        "answerable": True,
    },
    {
        "query_id": "q10",
        "question": "How far from cooking appliances should SafeGlow smoke detectors be installed?",
        "relevant_doc": "safeglow-detector",
        "answer_keywords": ["ten feet", "cooking appliances"],
        "answerable": True,
    },
    {
        "query_id": "q11",
        "question": "Does Cordwell price match online retailers on power tools?",
        "relevant_doc": None,
        "answer_keywords": [],
        "answerable": False,
    },
    {
        "query_id": "q12",
        "question": "What is the horsepower of the Cordwell RidgeRunner riding mower?",
        "relevant_doc": None,
        "answer_keywords": [],
        "answerable": False,
    },
]

EVAL_SET_VERSION = "cordwell-eval:v2"

# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------


def chunk_docs(chunk_size: int, chunk_overlap: int = 48) -> list[dict]:
    """Split every corpus document into overlapping character chunks.

    Splits at whitespace boundaries so words are never cut in half.
    Returns a list of {"chunk_id", "doc_id", "text"} dicts.
    """
    chunks = []
    for doc in CORPUS:
        text = doc["text"]
        start = 0
        idx = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            if end < len(text):
                space = text.rfind(" ", start, end)
                if space > start:
                    end = space
            piece = text[start:end].strip()
            if piece:
                chunks.append(
                    {
                        "chunk_id": f"{doc['doc_id']}#{idx}",
                        "doc_id": doc["doc_id"],
                        "text": piece,
                    }
                )
                idx += 1
            if end >= len(text):
                break
            start = max(end - chunk_overlap, start + 1)
    return chunks


# ---------------------------------------------------------------------------
# Deterministic embedder: hashed word and bigram features, cosine ready.
# ---------------------------------------------------------------------------

EMBED_DIM = 512
EMBEDDING_MODEL_NAME = "cordwell-hash-512 (offline stand-in)"

_word_re = re.compile(r"[a-z0-9.]+")


def _tokens(text: str) -> list[str]:
    return _word_re.findall(text.lower())


def _slot(feature: str) -> int:
    digest = hashlib.md5(feature.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big") % EMBED_DIM


def embed(text: str) -> list[float]:
    """Map text to a fixed vector using hashed unigrams and bigrams.

    Term counts are dampened with 1 + log(count) so a word repeated ten
    times does not dominate the vector. Deterministic across machines
    because it uses md5, not Python's salted built-in hash.
    """
    vec = [0.0] * EMBED_DIM
    toks = _tokens(text)
    counts: dict[str, float] = {}
    for tok in toks:
        key = "u:" + tok
        counts[key] = counts.get(key, 0.0) + 1.0
    for a, b in zip(toks, toks[1:]):
        key = "b:" + a + " " + b
        counts[key] = counts.get(key, 0.0) + 0.5
    for feature, count in counts.items():
        vec[_slot(feature)] += 1.0 + math.log(count)
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


def _cosine(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def build_index(chunks: list[dict]) -> list[dict]:
    """Attach an embedding to every chunk. This list is the 'index'."""
    return [{**c, "vector": embed(c["text"])} for c in chunks]


def retrieve(index: list[dict], question: str, top_k: int) -> list[dict]:
    """Return the top_k chunks by cosine similarity to the question."""
    qv = embed(question)
    scored = [
        {**c, "score": _cosine(qv, c["vector"])} for c in index
    ]
    scored.sort(key=lambda c: (-c["score"], c["chunk_id"]))
    return scored[:top_k]


# ---------------------------------------------------------------------------
# Deterministic generator: the Module 01 adapter, simulated.
# ---------------------------------------------------------------------------

GENERIC_FILLER = (
    "As a general rule, most Cordwell products include a one year "
    "satisfaction guarantee on top of the stated terms."
)


def _find_answer_sentence(contexts: list[dict], keywords: list[str]) -> str | None:
    """Return the first context sentence containing all answer keywords."""
    for chunk in contexts:
        for sentence in re.split(r"(?<=[.!?])\s+", chunk["text"]):
            low = sentence.lower()
            if keywords and all(k.lower() in low for k in keywords):
                return sentence.strip()
    return None


def generate_answer(question: str, contexts: list[dict], adapter_active: bool, keywords: list[str]) -> dict:
    """Deterministic stand-in for the Cordwell model.

    Adapter on: strict JSON discipline. Answers only from context and
    abstains with NOT_IN_DOCS when the context does not contain the
    answer.

    Adapter off: the base model. Always answers, pads with a generic
    ungrounded sentence, and never abstains, even when it should.
    """
    grounded = _find_answer_sentence(contexts, keywords)
    sources = sorted({c["doc_id"] for c in contexts})
    if adapter_active:
        if grounded is None:
            return {"answer": "NOT_IN_DOCS", "sources": [], "abstained": True}
        return {"answer": grounded, "sources": sources, "abstained": False}
    # Base model behavior
    if grounded is not None:
        answer = grounded + " " + GENERIC_FILLER
    else:
        first = contexts[0]["text"] if contexts else ""
        lead = re.split(r"(?<=[.!?])\s+", first)[0] if first else ""
        answer = (lead + " " + GENERIC_FILLER).strip()
    return {"answer": answer, "sources": sources, "abstained": False}


# ---------------------------------------------------------------------------
# Deterministic judges: the Module 03 metric family.
# ---------------------------------------------------------------------------

_STOP = set(
    "the a an of on in for to and or with at by is are be it its this that "
    "as from under after before when if any all use used using do does not "
    "you your".split()
)


def _content_tokens(text: str) -> set[str]:
    return {t for t in _tokens(text) if t not in _STOP and len(t) > 2}


def score_faithfulness(answer: str, contexts: list[dict]) -> float:
    """Claim decomposition scoring, as in Module 03.

    Split the answer into sentences (claims). A claim is supported when
    at least 70 percent of its content tokens appear in the retrieved
    context. Faithfulness is supported claims divided by total claims.
    """
    context_tokens = set()
    for c in contexts:
        context_tokens |= _content_tokens(c["text"])
    claims = [s for s in re.split(r"(?<=[.!?])\s+", answer) if s.strip()]
    if not claims:
        return 0.0
    supported = 0
    for claim in claims:
        toks = _content_tokens(claim)
        if not toks:
            supported += 1
            continue
        overlap = len(toks & context_tokens) / len(toks)
        if overlap >= 0.7:
            supported += 1
    return supported / len(claims)


def score_relevancy(answer: str, question: str, keywords: list[str]) -> float:
    """Answer relevancy: does the answer address what was asked?

    Combines keyword coverage (did the expected facts appear?) with
    topical overlap between answer and question.
    """
    low = answer.lower()
    if keywords:
        kw_cov = sum(1 for k in keywords if k.lower() in low) / len(keywords)
    else:
        kw_cov = 0.0
    q_toks = _content_tokens(question)
    a_toks = _content_tokens(answer)
    topical = len(q_toks & a_toks) / len(q_toks) if q_toks else 0.0
    return round(0.7 * kw_cov + 0.3 * topical, 4)


# ---------------------------------------------------------------------------
# The full evaluation pass.
# ---------------------------------------------------------------------------


def run_rag_eval(adapter_active: bool, top_k: int, chunk_size: int, chunk_overlap: int = 48) -> dict:
    """Run the complete Cordwell RAG evaluation once.

    Returns {"aggregate": {...}, "per_query": [...]}. Deterministic:
    the same arguments always produce the same numbers.
    """
    chunks = chunk_docs(chunk_size, chunk_overlap)
    index = build_index(chunks)
    per_query = []
    for q in EVAL_QUERIES:
        contexts = retrieve(index, q["question"], top_k)
        result = generate_answer(q["question"], contexts, adapter_active, q["answer_keywords"])
        retrieved_docs = [c["doc_id"] for c in contexts]
        if q["answerable"]:
            hit_chunks = [
                c for c in contexts
                if c["doc_id"] == q["relevant_doc"]
                and all(k.lower() in c["text"].lower() for k in q["answer_keywords"])
            ]
            context_recall = 1.0 if hit_chunks else 0.0
            relevant_retrieved = sum(1 for d in retrieved_docs if d == q["relevant_doc"])
            context_precision = relevant_retrieved / len(retrieved_docs) if retrieved_docs else 0.0
        else:
            context_recall = None
            context_precision = None
        if result["abstained"]:
            faithfulness = None
            relevancy = None
        else:
            faithfulness = round(score_faithfulness(result["answer"], contexts), 4)
            relevancy = score_relevancy(result["answer"], q["question"], q["answer_keywords"])
        per_query.append(
            {
                "query_id": q["query_id"],
                "question": q["question"],
                "answerable": q["answerable"],
                "answer": result["answer"],
                "sources": result["sources"],
                "abstained": result["abstained"],
                "context_recall": context_recall,
                "context_precision": context_precision,
                "faithfulness": faithfulness,
                "answer_relevancy": relevancy,
            }
        )

    def _mean(vals):
        vals = [v for v in vals if v is not None]
        return round(sum(vals) / len(vals), 4) if vals else 0.0

    answerable = [r for r in per_query if r["answerable"]]
    unanswerable = [r for r in per_query if not r["answerable"]]
    aggregate = {
        "context_recall": _mean([r["context_recall"] for r in answerable]),
        "context_precision": _mean([r["context_precision"] for r in answerable]),
        "faithfulness": _mean([r["faithfulness"] for r in per_query]),
        "answer_relevancy": _mean([r["answer_relevancy"] for r in per_query]),
        "abstention_rate": round(sum(1 for r in per_query if r["abstained"]) / len(per_query), 4),
        "correct_abstention": round(
            sum(1 for r in unanswerable if r["abstained"]) / len(unanswerable), 4
        ) if unanswerable else 0.0,
    }
    return {"aggregate": aggregate, "per_query": per_query}


# ---------------------------------------------------------------------------
# Artifact staging helpers: files students will log as artifacts.
# ---------------------------------------------------------------------------


def write_adapter_dir(base_dir: str = "artifacts_staging") -> str:
    """Write a small stand-in adapter directory and return its path.

    In Module 01 this directory came out of SFTTrainer. Here it is a
    faithful miniature: an adapter_config.json and a weights file, so
    that artifact logging exercises the real API on real files.
    """
    path = Path(base_dir) / "cordwell_adapter"
    path.mkdir(parents=True, exist_ok=True)
    (path / "adapter_config.json").write_text(
        json.dumps(
            {
                "peft_type": "LORA",
                "r": 16,
                "lora_alpha": 32,
                "target_modules": ["q_proj", "v_proj"],
                "base_model_name_or_path": "HuggingFaceTB/SmolLM2-360M-Instruct",
            },
            indent=2,
        )
    )
    (path / "adapter_model.bin").write_bytes(
        hashlib.sha256(b"cordwell-adapter-weights-v1").digest() * 64
    )
    return str(path)


def write_eval_set_file(base_dir: str = "artifacts_staging") -> str:
    """Write the eval query set to a JSONL file and return its path."""
    path = Path(base_dir)
    path.mkdir(parents=True, exist_ok=True)
    fp = path / "cordwell_eval_v2.jsonl"
    with fp.open("w") as f:
        for q in EVAL_QUERIES:
            f.write(json.dumps(q) + "\n")
    return str(fp)
