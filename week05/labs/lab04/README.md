# Week 5 Lab 04: Instrument the Cordwell Pipeline
## Experiment tracking with Weights & Biases

Students wrap the week's Cordwell RAG evaluation in W&B tracking: a
config that passes the re-run test, stage-prefixed metrics, versioned
artifacts with lineage, a three-run controlled experiment, a comparison
that attributes each metric change to one variable, per-query failure
analysis with a wandb.Table, and a stretch grid sweep. Core lab about
3.5 hours plus a 30 minute stretch. Maps to Module 04 slide deck
objectives 1 through 5; the Part E recommendation writeup is the
deliverable.

## Files

| File | Purpose |
|---|---|
| `Lab04_WandB_Tracking_STUDENT.ipynb` | Stubbed notebook students work in. Cold Run All: 3 PASS, 29 TODO, zero crashes |
| `Lab04_WandB_Tracking_SOLUTION.ipynb` | Fully executed instructor solution, 32 of 32 checks passing. Withhold until after the lab |
| `lab_support.py` | Pre-written plumbing: the deterministic Cordwell RAG pipeline, backend selector, artifact staging. Not the lesson |
| `demo_script.md` | Instructor I-do walkthrough to the hand-off point, with timing, expected output, and recovery paths |
| `walkthrough.md` | Line-by-line plain-language explanation of every solution function, for instructor prep |
| `HINTS.md` | Progressive tier: three escalating levels per task |
| `HINTS_DETAILED.md` | Detailed tier: working core with commentary, assembly withheld. Both hint files go to students |
| `setup/LOCAL_SERVER_SETUP.md` | Docker W&B server setup for the local backend |
| `requirements.txt` | Pinned dependencies with currency flags |

## Quick start

```bash
pip install -r requirements.txt
jupyter lab   # offline backend, no other setup
```

Optional backends, chosen by environment variable before Jupyter starts:

| `WANDB_LAB_BACKEND` | Needs |
|---|---|
| `offline` (default) | Nothing; checks calibrated here |
| `local` | Docker W&B server on port 8080, see setup guide |
| `cloud` | Free wandb.ai account and `wandb login` |

Every line of student code is identical across backends.

## Locked numbers (offline backend, deterministic)

Part D at chunk size 384: adapter-off-topk5 faithfulness 0.5 and recall
1.0; adapter-on-topk5 faithfulness 1.0 and recall 1.0; adapter-on-topk1
faithfulness 1.0 and recall 0.8. Part F worst three: q12, q11, q06.
Part G frontier pick: chunk_size 256, top_k 3. Full derivations in
`walkthrough.md` section 9.

## Verification ledger

Execution-confirmed in the build sandbox (Python 3.12.3, wandb 0.28.1):
solution notebook end to end via nbclient with 32 of 32 checks passing;
student notebook cold Run All with zero hard errors at 3 PASS and 29
TODO; completability confirmed by construction, since both notebooks are
generated from one source where stubs are replaced by the verified
solution bodies; offline behavior of `wandb.init`, `run.log`,
`run.summary`, `wandb.Table`, `log_artifact`, and artifact manifests;
the exact offline failure modes of `run.use_artifact` (TypeError) and
`wandb.sweep` and `wandb.Api` (UsageError, no API key or server); the
`wandb server start` CLI exists and is the documented Docker path.

Literature-sourced, confirm before class on the live paths: the local
server first-boot license flow and UI screens (Docker unavailable in the
build sandbox); cloud backend login flow; exact behavior of any wandb
version newer than 0.28.1. The instructor pre-flight in `demo_script.md`
covers all three.

## Currency flags

- wandb pinned at 0.28.1; re-verify the 32 checks on any newer version.
- `wandb server start` help text confirms it is the supported local
  Docker path and notes production self-hosting uses the Kubernetes
  operator; unchanged as of build time, cheap to reconfirm.
- The Week 6 boundary from the slides holds: MLflow and the tracking
  landscape are Monday; nothing in this lab depends on OpenAI APIs, so
  the October 2026 model shutdown does not touch it.
