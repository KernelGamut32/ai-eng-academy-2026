# Appendix: The Apple-Native Path (mlx-lm)

Reference material for `TRAIN_BACKEND=mlx`. This path is deliberately not wired into the notebook: it runs from the command line, its API surface differs from the peft and trl path the course teaches, and the peft path is the one that transfers to cluster hardware. Use this appendix when a machine is too slow for the MPS path or when you want to see the Apple-native ecosystem.

⚠️ CURRENCY FLAG: unlike everything in the notebook and lab_support.py, the commands below were NOT executed at build time. mlx runs only on Apple Silicon and the build environment was Linux. The command shapes reflect mlx-lm documentation as of the build date. Confirm against `python -m mlx_lm lora --help` on an actual cohort Mac before teaching from this page, and expect flag names to move: the mlx ecosystem iterates quickly.

## What mlx changes and what it does not

The concepts are identical: frozen base, low-rank adapters, the same r and scale ideas, the same data. What changes is the runtime (Apple's MLX array framework instead of PyTorch) and the interface (a CLI instead of a Trainer object). mlx also quantizes natively on Apple Silicon, which is the closest a Mac gets to the QLoRA story from the concept slides.

## Sketch of the equivalent run

Data: mlx-lm's LoRA trainer consumes jsonl with a messages column, the same shape as `cordwell_sft_clean.jsonl`. It expects `train.jsonl` and `valid.jsonl` in a data directory, so split the clean file first (a ten-line Python script or a re-export from the notebook's `splits` object).

```bash
pip install mlx-lm

python -m mlx_lm lora \
  --model HuggingFaceTB/SmolLM2-360M-Instruct \
  --train \
  --data ./data/mlx \
  --num-layers 8 \
  --batch-size 4 \
  --iters 300 \
  --learning-rate 2e-4 \
  --adapter-path ./adapters/mlx_catalog
```

Evaluation: generate with the adapter attached and pipe the outputs back through the notebook's scoring, which is pure Python and backend-agnostic:

```bash
python -m mlx_lm generate \
  --model HuggingFaceTB/SmolLM2-360M-Instruct \
  --adapter-path ./adapters/mlx_catalog \
  --prompt "..."
```

Then score the collected strings with `ls.evaluate_outputs(outputs, eval_records)` exactly as in Part A. The metric layer not caring where the text came from is the same design point the offline backend exploits.

## Teaching note

If a learner completes the lab on this path, the deliverable that matters is unchanged: baseline metrics, tuned metrics, same fifty records, same validator. Grade the numbers and the reasoning, not the toolchain.
