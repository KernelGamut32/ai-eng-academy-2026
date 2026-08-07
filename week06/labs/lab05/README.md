# Week 6 Lab 2.2: Evaluating Summarization with ROUGE

Cordwell Home and Hardware scenario. Two hour hands-on lab for Week 6 Day 2, following the Module 02 NLP evaluation material. Students implement two extractive summarizers, score them with ROUGE-1, ROUGE-2, and ROUGE-L, discover that corpus averages hide a bimodal failure mode, and measure exactly which ROUGE component catches hallucination and redundancy.

## Files

| File | Audience | Purpose |
|---|---|---|
| `demo_script.md` | Instructor | I-do walkthrough from setup to hand-off, with talk track, the live argument order trap demo, floor notes, and timing ledger |
| `lab2_2_rouge_student.ipynb` | Students | Stubbed notebook with 8 checked tasks and 2 optional stretches. Opens with 0 of 8 passing and zero tracebacks on a cold Run All |
| `lab2_2_rouge_solution.ipynb` | Instructor only | Fully implemented and executed, 8 of 8 passing, including both stretch solutions. Withhold until after the lab |
| `walkthrough.md` | Instructor | Line by line explanation of every piece of code, background primer, verified numbers ledger, decisions table, currency flags |
| `HINTS.md` | Students | Progressive tier: three escalating levels per task |
| `HINTS_DETAILED.md` | Students | Detailed tier: working core with line commentary, assembly withheld |
| `README.md` | Both | This file |
| `requirements.txt` | Both | Pinned dependencies |

Both hint files go to students at lab start; they self select one tier per task. The solution notebook is the only fully assembled artifact and stays instructor only.

## Environment

- Python 3.13 (cohort standard; the notebook also runs on 3.12)
- Install: `pip install -r requirements.txt`
- Core lab: fully offline and deterministic. No API keys, no network access, no Docker services, no model downloads. There is nothing to stand up before class.
- Stretch 2 only: needs `torch` and `sentence-transformers` plus one time network access for a 90 MB model download. Marked optional in the notebook and INSTRUCTOR VERIFY in the solution.

Verification state: all numbers in the notebooks, checks, demo script, and walkthrough were produced by executing the solution against the pinned versions in requirements.txt, and the full pipeline was run twice with byte identical output. The student notebook was executed cold end to end: zero tracebacks, scoreboard 0 of 8. The solution notebook executes to 8 of 8.

## Timing (120 minute slot)

| Block | Minutes |
|---|---|
| Instructor demo (see demo_script.md) | 22 |
| Tasks 1 to 3: splitter and both summarizers | 25 |
| Tasks 4 and 5: ROUGE function and corpus scoring | 25 |
| Task 6 plus checkpoint discussion | 25 |
| Tasks 7 and 8 plus reflection | 25 |
| Stretch 1 and 2 | fast finishers |

## Instructor notes

- The two teaching peaks are the Task 6 checkpoint (the layout slice reverses the room's read of the corpus means) and the Task 8 prediction moment (recall rises under hallucination). The demo script's floor notes carry the exact numbers and suggested lines for both.
- The Task 4 check detects swapped scorer arguments by pattern and names the bug in its failure message. The demo script includes a live demonstration of the swap; do not skip it, it is the lab's most production relevant thirty seconds.
- Decisions made during the build that warrant your review (scenario rename, corpus redesign, stretch split, and more) are in the decisions table at the end of `walkthrough.md`.
- Currency flags, including the rouge-score argument order verification and the offline splitter tradeoff, are at the end of `walkthrough.md`.
