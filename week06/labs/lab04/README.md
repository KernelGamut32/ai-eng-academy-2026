# Week 6 Lab 2.1: Evaluating Text Generation with BLEU

Cordwell Home and Hardware scenario. Students compare two simulated text-rewriting systems with corpus-level and sentence-level BLEU, decompose scores into n-gram precisions and the brevity penalty, catch a system gaming the metric via verbatim passthrough, and rebuild clipped precision and the brevity penalty by hand.

Duration: about 120 minutes (25 instructor-led, 85 student work, 10 debrief).

## Files

| File | Audience | Purpose |
|---|---|---|
| `demo_script.md` | Instructor | I-do walkthrough from cold start to the Section 3 hand-off, timing ledger, recovery notes, debrief beats |
| `week06_lab21_bleu_student.ipynb` | Students | Stubbed notebook with 8 TODO tasks, worked target outputs, and a soft check harness (`run_checks()`) |
| `week06_lab21_bleu_solution.ipynb` | Instructor only | Fully executed solution, 8 of 8 checks passing, plus executed stretch goal solutions |
| `walkthrough.md` | Instructor | Line-by-line walkthrough of all solution and supporting code in plain terms, verified numbers ledger, corrections log, currency flags, flagged decisions |
| `HINTS.md` | Students | Progressive tier: three escalating levels per task |
| `HINTS_DETAILED.md` | Students | Detailed tier: working core of each task with line-by-line commentary, assembly withheld |
| `requirements.txt` | Both | Pinned dependencies, verified at build time |

Distribute to students: the student notebook, both hint files, and `requirements.txt`. Withhold the solution notebook until after the lab.

## Environment

Fully offline, CPU only. No GPU, no network calls, no API keys, no Docker, no local LLM server. The standing LM Studio and Ollama backend convention applies to labs that call an LLM; no cell here performs inference, so no backend switch is included by design.

Setup is one install:

```bash
pip install -r requirements.txt
```

The corpus is generated in-notebook from a dedicated seeded RNG, so every student produces a byte-identical corpus and the check harness verifies exact values.

## Verification status

- Solution notebook: executed clean, 0 errors, PASSED 8 of 8, on the pinned stack above.
- Student notebook: cold Run All produces 0 hard crashes and PASSED 0 of 8 with all tasks reporting TODO.
- Completability: the solution notebook is the student notebook with stubs replaced by solutions; identical shared cells, full pass count confirmed.
- Every quantitative claim in the notebooks, demo script, and walkthrough was executed against `sacrebleu 2.6.0`, `pandas 3.0.2`, `numpy 2.4.4`, `matplotlib 3.10.8` before being written down. See the verified numbers ledger in `walkthrough.md`.

## Headline verified numbers

Corpus BLEU: System A 67.51, System B 19.55. System B's 4-gram precision (59.4) exceeds System A's (55.9); the gap is almost entirely the brevity penalty (0.293 versus 1.000). B outscores A on exactly 33 rows, and all 33 are verbatim passthrough copies of the reference. Stretch: System C scores 100.0 unigram precision and 0.10 BLEU; chrF ranks C above B while BLEU ranks B above C.

## Instructor notes

- Read `demo_script.md` before class; the debrief beats depend on the numbers above landing in order.
- The Task 2 check intentionally defends the lab's central surprise (B's higher 4-gram precision) against students "fixing" it.
- Decisions that warrant your judgment (passthrough rate, delta thresholds, the tease framing) are tabled at the end of `walkthrough.md`.
