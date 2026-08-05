"""Generate the Cordwell Home and Hardware synthetic review test set.

Reproduces the Module 01 slide-deck numbers exactly:
  N = 2,000 reviews, 67 safety escalations (base rate 3.35%)
  At threshold 0.50: TP=53, FN=14, FP=107, TN=1826
  Accuracy 0.9395, Precision 0.3312, Recall 0.7910, F1 0.4670
  MCC 0.4880, Balanced accuracy 0.8678
  ROC-AUC 0.9621, Average precision 0.6040 (score-based, used by later demos)

All data is synthetic and clearly fictional. Review text is decorative
flavor for human reading during error analysis; the y_score column is the
output of a simulated classifier and is the source of truth for metrics.

Internal seeds are pinned below. They were tuned so the generated score
distribution lands on the deck's published numbers to four decimals.
Do not change them if you need the deck and the notebook to agree.
"""

import numpy as np
import pandas as pd

# ---------------------------------------------------------------- constants
N_POS, N_NEG = 67, 1933            # 67 / 2000 = 3.35% base rate
TP, FN, FP, TN = 53, 14, 107, 1826  # target matrix at threshold 0.50

SCORE_SEED = 1267                   # seed for the tuned score draw
TEXT_SEED = 42                      # seed for review text templating
BETA_PARAMS = (2.3262, 1.4692,      # positives scoring >= 0.50
               1.8844, 1.0171,      # positives scoring <  0.50 (the misses)
               1.4505, 2.6211,      # negatives scoring >= 0.50 (false alarms)
               0.8850, 4.4081)      # negatives scoring <  0.50


def build_scores() -> tuple[np.ndarray, np.ndarray]:
    """Draw scores from tuned Beta distributions, then apply one rank swap.

    The single adjacent-rank swap (positions 29 and 30 in descending score
    order) nudges average precision onto the deck's published 0.6040 without
    moving any score across the 0.50 threshold, so the confusion matrix is
    untouched.
    """
    a_ph, b_ph, a_pl, b_pl, a_nh, b_nh, a_nl, b_nl = BETA_PARAMS
    rng = np.random.default_rng(SCORE_SEED)
    pos_hi = 0.5 + 0.5 * rng.beta(a_ph, b_ph, TP)   # 53 positives above cut
    pos_lo = 0.5 * rng.beta(a_pl, b_pl, FN)         # 14 positives below cut
    neg_hi = 0.5 + 0.5 * rng.beta(a_nh, b_nh, FP)   # 107 negatives above cut
    neg_lo = 0.5 * rng.beta(a_nl, b_nl, TN)         # 1826 negatives below cut

    y = np.concatenate([np.ones(N_POS, int), np.zeros(N_NEG, int)])
    s = np.concatenate([pos_hi, pos_lo, neg_hi, neg_lo])

    order = np.argsort(-s)
    a, b = order[29], order[30]
    s[a], s[b] = s[b], s[a]
    return y, s


# ------------------------------------------------------------- review text
PRODUCTS = [
    "cordless drill", "space heater", "step ladder", "extension cord",
    "table saw", "smoke detector", "paint sprayer", "garden hose",
    "shop vacuum", "circular saw", "LED work light", "power strip",
    "chainsaw", "pressure washer", "leaf blower", "tile cutter",
    "wet tile saw", "angle grinder", "heat gun", "utility knife",
]

SAFETY_OBVIOUS = [
    "The {p} sparked and scorched the outlet within minutes of plugging it in.",
    "My {p} caught fire during normal use. There were flames and smoke.",
    "The blade guard on the {p} snapped off mid cut and I was injured.",
    "This {p} shocked me twice while it was switched off. Serious hazard.",
    "The {p} overheated until the housing melted and burned my hand.",
    "The ladder rung on this {p} collapsed under normal weight. I fell.",
    "The cord insulation on the {p} split open exposing live wire.",
]

SAFETY_SUBTLE = [
    "After an hour the {p} gave off a strange chemical smell and got very warm, then it quit.",
    "The {p} makes a loud pop when starting and the lights in the room dim.",
    "Noticed the plug on the {p} is discolored and soft after a week of light use.",
    "The {p} vibrates so hard the fasteners back out. One piece flew off near my face.",
    "The battery pack on the {p} swelled up after the second charge.",
    "The {p} keeps tripping the breaker and the switch feels hot to the touch.",
    "There is a faint burning odor whenever the {p} runs longer than ten minutes.",
]

ROUTINE = [
    "Great {p}, the battery lasts all weekend and it feels solid.",
    "The {p} arrived two days late but works exactly as described.",
    "Decent {p} for the price. The case is flimsy but the tool is fine.",
    "I returned the {p} because the color did not match the listing.",
    "The {p} is louder than my old one but gets the job done.",
    "Assembly instructions for the {p} were confusing. Product is okay.",
    "Bought this {p} as a gift. My brother in law loves it.",
    "The {p} is fine, though the packaging was crushed in shipping.",
    "Five stars. This {p} replaced one I had for fifteen years.",
    "The {p} is underpowered for hardwood but great for softwood.",
]

ROUTINE_SPICY = [
    "This {p} is a hot mess. The finish looks burned out and cheap. Works fine though.",
    "The instructions warn about fire safety on every page. The {p} itself is fine.",
    "My old {p} finally died so I collapsed and bought this one. No regrets.",
    "The reviews scared me but my {p} has zero issues after a month.",
    "Shockingly good value. This {p} outperforms brands twice the price.",
]


def build_text(y: np.ndarray, s: np.ndarray) -> list[str]:
    rng = np.random.default_rng(TEXT_SEED)
    texts = []
    for label, score in zip(y, s):
        product = PRODUCTS[rng.integers(len(PRODUCTS))]
        if label == 1 and score >= 0.5:
            pool = SAFETY_OBVIOUS
        elif label == 1:
            pool = SAFETY_SUBTLE          # the misses read as subtle reports
        elif score >= 0.5:
            pool = ROUTINE_SPICY          # false alarms read as spicy routine
        else:
            pool = ROUTINE
        texts.append(pool[rng.integers(len(pool))].format(p=product))
    return texts


def main() -> None:
    y, s = build_scores()
    texts = build_text(y, s)

    df = pd.DataFrame({
        "review_id": [f"CW-{i:05d}" for i in range(1, len(y) + 1)],
        "review_text": texts,
        "y_true": y,
        "y_score": s,
    })

    # Shuffle rows so the file does not leak label order, then re-id.
    df = df.sample(frac=1.0, random_state=TEXT_SEED).reset_index(drop=True)
    df["review_id"] = [f"CW-{i:05d}" for i in range(1, len(df) + 1)]

    df.to_csv("cordwell_test_set.csv", index=False)
    print(f"Wrote cordwell_test_set.csv with {len(df)} rows, "
          f"{int(df.y_true.sum())} positives "
          f"({df.y_true.mean():.4f} base rate)")


if __name__ == "__main__":
    main()
