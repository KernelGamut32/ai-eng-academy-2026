# Week 5, Lab 01: LoRA Catalog Normalization

Cordwell Home and Hardware ingests product data from 300 suppliers, each sending free-text blurbs in a different house style. Downstream systems need one strict JSON attribute record per product. In this lab you baseline a frozen small model on that task, audit the training data for planted quality defects, configure and run a LoRA fine-tune with peft and trl, and re-measure on the same held-out records to prove the improvement with real numbers.

Duration: about 3 hours core, 4 hours with both stretch goals.

## File map

| File | What it is |
|---|---|
| `lab01_student.ipynb` | Your working notebook. Stubs raise NotImplementedError; a cold Run All produces zero crashes and 3 of 22 checks passing |
| `lab_support.py` | Pre-written plumbing: data generation, schema validator, metrics, evaluation loop, backend selector, plotting. Read it, do not edit it |
| `HINTS.md` | Progressive tier: three escalating levels per task |
| `HINTS_DETAILED.md` | Detailed tier: working core with line-by-line commentary |
| `requirements.txt` | Pinned, verified dependency set |
| `demo_script.md` | Instructor walkthrough to the hand-off point |
| `APPENDIX_MLX.md` | The Apple-native mlx-lm alternative path (reference only) |
| `lab01_solution.ipynb` | Instructor only, withheld until after the lab |
| `instructor_code_walkthrough.md` | Instructor only |

Pick one hint tier per task. Reading both wastes time. The solution notebook remains the only fully assembled artifact and is handed out after the lab.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Python 3.13 (verified on 3.12.3, runs unchanged on 3.13). All data is synthetic and generated inside the lab from a fixed seed. There are no cross-lab data dependencies, no external services, and nothing to run in Docker for this lab: training happens in-process, and the only network access is the one-time base model download in live mode.

## Backend selection

Set `TRAIN_BACKEND` in the shell **before** launching Jupyter. Explicit selection raises a clear error rather than silently switching.

| Value | Path | When to use it |
|---|---|---|
| `peft_mps` (default) | Real fine-tune with peft and trl on Apple MPS, CPU fallback per-op | The normal path. This is the transferable API you will use on cluster hardware |
| `offline` | Tiny built-in stand-in model plus deterministic scripted outputs | No network, no downloads, always completes every part in minutes. Also the live-demo fallback |
| `mlx` | Apple-native command line path | Reference only, see `APPENDIX_MLX.md`. Not wired into the notebook |

```bash
TRAIN_BACKEND=offline jupyter lab    # example: offline mode
```

In offline mode your LoraConfig, SFTConfig, and training call all execute for real against a tiny local model; only the generated evaluation text is scripted. The training API path is genuine either way.

Live mode notes for cohort Macs: no CUDA exists on these machines and none is assumed. The device is auto-selected at runtime (CUDA, then MPS, then CPU). `device_map="auto"` is never used on MPS. The base model is roughly 0.7 GB on first download and is cached afterward. Expect training to take about 10 to 25 minutes and each 50-record evaluation about 2 to 5 minutes; confirm on your own machine during the dry run.

## Base model

The model ID lives in a config variable, never hard-coded:

```bash
BASE_MODEL=HuggingFaceTB/SmolLM2-360M-Instruct   # the default
```

Override with the environment variable if the cohort standardizes on a different small instruct model. Confirm the ID resolves before class.

## Verified environment

All API calls in this lab were executed against: torch 2.13.0, transformers 5.14.1, peft 0.19.1, trl 1.9.1, accelerate 1.14.0, datasets 5.0.0. Three currency notes baked into the materials, each verified by running the code:

- `SFTConfig` takes `max_length` and `eval_strategy`. The older `max_seq_length` and `evaluation_strategy` names are gone; tutorials showing them are stale.
- `from_pretrained` takes `dtype`. The older `torch_dtype` still works but warns and is deprecated.
- `apply_chat_template` now returns a `BatchEncoding` by default; unpack it into `generate` with two stars.

⚠️ CURRENCY FLAG: metric numbers quoted for live mode (baseline roughly 20 to 45 percent schema-valid, tuned above 90) are estimates for SmolLM2-360M-Instruct and were not run against the live model at build time; offline-mode numbers (34, 98, 90 percent) are exact and deterministic. Confirm live numbers on cohort hardware during the dry run and adjust the expected ranges in the notebook text if needed.
