# Week 6 Demo: BERTScore vs BLEU and ROUGE

Instructor demo notebook for Module 02, Segment 5, slides 36 to 38. Walk it live, then hand the executed notebook to students for review. Cordwell Home and Hardware scenario, continuing the minimal pair (slide 31) and the Cordwell Disagreement (slide 30) so the BERTScore results land against numbers the room has already seen.

## Files

- `bertscore_demo.ipynb`: the demo. Six parts, ~20 to 25 minutes.
- `requirements.txt`: pinned dependencies.
- `README.md`: this file.

## Build-time verification status

All BLEU, chrF++, and ROUGE cells were executed at build time against the pinned libraries; their outputs are populated in the notebook and match the Module 02 verified numbers ledger exactly (minimal pair: BLEU 59.46 both candidates; Cordwell Disagreement: BLEU 69.97 vs 6.89, ROUGE-1 F 83.87 vs 34.29).

**Four cells are marked [INSTRUCTOR-VERIFY] and ship with empty outputs**: the two `score(...)` cells in Parts 2 and 3, the BERTScore cell in Part 4, and the `plot_example` cell in Part 5. The build sandbox blocks HuggingFace model downloads, so these could not be executed at build time. The pre-flight run below populates them with real output before class. Do not quote any BERTScore number aloud that you have not seen your own machine produce.

## Pre-flight checklist (run the day before)

1. `pip install -r requirements.txt` in the class environment.
2. Open the notebook and Run All on a networked machine. The first `score(...)` call downloads `roberta-large` (~1.4 GB); allow time for it. Later runs hit the local cache.
3. Confirm all four instructor-verify cells now show real output and that the qualitative shapes match the expectation notes in the markdown cell under each one:
   - Part 2: synonym clearly above antonym.
   - Part 3: raw scores compressed high, rescaled scores spread out, unrelated sentence near or below zero after rescaling.
   - Part 4: System B rises sharply relative to its n-gram scores; note where System A lands and rehearse the read-out either way.
   - Part 5: the heatmap renders; the `big` row lights up on the `large` column.
4. Time the Part 2 scoring cell on the actual class machine. If model load drags on the no-GPU Mac, switch `BERTSCORE_MODEL` to `"distilbert-base-uncased"` (~260 MB, rescale baseline bundled, verified present in the package) and re-run pre-flight with that model.
5. Save the notebook with outputs populated. This saved copy is also the in-class fallback.

## Fallback path

If the classroom network fails, the model cache from pre-flight makes the score cells run fully offline. If the machine itself fails, walk the saved pre-flight notebook's outputs on screen; every number shown is one you executed yourself.

## Timing ledger (~22 min core)

- Part 1, n-gram recap with prediction hook: 4 min
- Part 2, BERTScore on the minimal pair: 5 min (includes model-load dead air; narrate the mechanics from slide 36 while it loads)
- Part 3, rescaling: 4 min
- Part 4, Cordwell Disagreement revisited: 6 min (the central caveat lives here; do not rush it)
- Part 5, alignment heatmap: optional 2 min
- Part 6, side-by-side table and hand-off to Segment 6: 3 min

## Currency flags

- ⚠️ `bert-score` 0.3.13 (May 2023) predates `transformers` 5.x. The cohort stack pins `transformers` 5.14.1, and the sandbox install resolved that pairing without conflict, but the actual scoring path could not be exercised at build time. The pre-flight run is the verification. If `score(...)` raises on the cohort stack, the failure is a `transformers` 5.x compatibility break in `bert-score`; the fix is an isolated venv with an older `transformers`, and that finding should go back into the Module 02 materials.
- ⚠️ Version-string quirk, also taught in the notebook: `bert_score.__version__` reports `0.3.12` for the 0.3.13 distribution. The notebook reports versions via `importlib.metadata` for this reason.

## Student hand-off

After class, distribute the executed notebook as-is. It is a walkthrough for review, not a lab: no stubs, no hidden solution. The Part 4 markdown carries the caveat students most need to retain (paraphrase fixed, factuality not), and Part 6 is the reference table for the rest of the week.
