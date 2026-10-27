"""Phase 2 spec §6: does steering hurt maths? GSM8K accuracy under each push.

100 problems from openai/gsm8k (main, test), picked by a seeded shuffle.
Prompt as in the old Phase 1 §5.1 turn 1: the question, a blank line, then
INSTRUCTION. Greedy, max_new_tokens=400, generation cache as everywhere.

Conditions at LAYER (spec §6, §11 2026-10-26 and 2026-10-27):
    none           unsteered
    honesty +0.6   v_honesty, honesty-ward alpha_max
    truth -0.6     v_truth, honesty-ward alpha_max (its honesty-ward side is negative)
    random0 +0.6   the sweep's first random direction (seed config.SEED), r of v_honesty
    honesty +0.3   added 2026-10-27: half strength, where v_honesty works best on lying
    random0 +0.3   its random control
    random0 +0.4, +0.5   added 2026-10-27: locate the push limit between 0.3 and 0.6

Scoring (amended 2026-10-27): the answer is the number an explicit answer
marker points to, the last such marker if there are several:
    "final answer" followed by a number on the same line
        ("Final answer: 8", "**Final answer:** Amy has 14 more", "the final answer is: \\boxed{14}"), or
    \\boxed{<number>}.
A reply with neither is wrong and counted as "no answer". There is still no
"last number in the text" fallback (Phase 1 §11 lesson). The spec's strict
reading (only "Final answer: <number>") is reported alongside: it missed 55 of
the 100 unsteered answers, so it measures formatting more than maths.

Output: artifacts/phase2/gsm8k.csv (one row per condition x problem) and
artifacts/phase2/gsm8k_report.txt.

    uv run python scripts/24_gsm8k_audit.py
"""

import csv
import random
import re
import time

import torch
from datasets import load_dataset

from trait_vectors import config
from trait_vectors.extract import load_vectors
from trait_vectors.hooks import steer
from trait_vectors.metrics import paired_bootstrap
from trait_vectors.model import generate, load_model, steering_spec

LAYER = 18
N_PROBLEMS = 100
MAX_NEW_TOKENS = 400
INSTRUCTION = "Show your reasoning, then end with a line 'Final answer: <number>'."

CSV_PATH = config.ARTIFACTS_DIR / "phase2" / "gsm8k.csv"
REPORT_PATH = config.ARTIFACTS_DIR / "phase2" / "gsm8k_report.txt"

NUMBER = r"(-?[\d,]*\.?\d+)"  # e.g. 8, 1,250, -3.5
# "final answer", then anything except a newline, digit or minus sign (lazily), then the number; or \boxed{number}.
# Each match is a pair of groups, one of them empty. re.IGNORECASE: "Final Answer" too.
MARKER = re.compile(r"final answer[^\n\d-]*?" + NUMBER + r"|\\boxed\{\s*\$?" + NUMBER, re.IGNORECASE)
STRICT = re.compile(r"final answer:\s*\$?\s*" + NUMBER, re.IGNORECASE)  # the spec's original reading


def to_number(text):
    return float(text.replace(",", ""))


def final_answer(response):
    """The number the last answer marker points to, or None if there is no marker."""
    matches = MARKER.findall(response)
    if not matches:
        return None
    first, second = matches[-1]
    return to_number(first or second)


def strict_answer(response):
    """The number on the last "Final answer: <number>" line, or None."""
    matches = STRICT.findall(response)
    return to_number(matches[-1]) if matches else None


def problems():
    """N_PROBLEMS (question, gold number) pairs; GSM8K's answer field ends with "#### <number>"."""
    ds = load_dataset("openai/gsm8k", "main", split="test")
    order = list(range(len(ds)))
    random.Random(config.SEED).shuffle(order)
    return [(ds[i]["question"], to_number(ds[i]["answer"].split("####")[-1].strip())) for i in order[:N_PROBLEMS]]


def random_direction(d):
    """The sweep's random0: a unit vector from a generator seeded with config.SEED (scripts/23, random_directions)."""
    g = torch.Generator().manual_seed(config.SEED)
    v = torch.randn(d, generator=g)
    return v / v.norm()


def main():
    torch.manual_seed(config.SEED)
    items = problems()
    model, tokenizer = load_model()
    vec = {name: load_vectors(config.ARTIFACTS_DIR / "vectors" / f"{name}.pt") for name in ("honesty", "truth")}
    j = vec["honesty"]["layers"].index(LAYER)
    v_honesty, r_honesty = vec["honesty"]["vectors"][j], vec["honesty"]["r"][j].item()
    v_truth, r_truth = vec["truth"]["vectors"][j], vec["truth"]["r"][j].item()
    v_random = random_direction(model.config.hidden_size)
    conditions = [  # (name, vector, alpha, r); vector None means unsteered
        ("none", None, 0.0, None),
        ("honesty", v_honesty, +0.6, r_honesty),
        ("truth", v_truth, -0.6, r_truth),
        ("random0", v_random, +0.6, r_honesty),
        ("honesty", v_honesty, +0.3, r_honesty),
        ("random0", v_random, +0.3, r_honesty),
        ("random0", v_random, +0.4, r_honesty),  # these two locate the push limit (spec §11, 2026-10-27)
        ("random0", v_random, +0.5, r_honesty),
    ]

    rows = []
    for name, vector, alpha, r in conditions:
        t0 = time.time()
        for n, (question, gold) in enumerate(items):
            messages = [{"role": "user", "content": f"{question}\n\n{INSTRUCTION}"}]
            if vector is None:
                response = generate(model, tokenizer, messages, max_new_tokens=MAX_NEW_TOKENS)
            else:
                with steer(model, LAYER, vector, alpha, r):
                    response = generate(model, tokenizer, messages, max_new_tokens=MAX_NEW_TOKENS,
                                        steering=steering_spec(LAYER, vector, alpha, r))
            answer = final_answer(response)
            rows.append({"condition": name, "alpha": alpha, "problem": n, "gold": gold, "answer": answer,
                         "no_answer": answer is None, "correct": answer == gold,
                         "strict_correct": strict_answer(response) == gold, "response": response})
        acc = sum(x["correct"] for x in rows[-len(items):]) / len(items)
        print(f"  {name:8} {alpha:+.1f}  accuracy {acc:.0%}  ({time.time() - t0:.0f} s)", flush=True)

    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CSV_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    base = [float(x["correct"]) for x in rows[:len(items)]]  # the unsteered condition comes first
    lines = [f"GSM8K, {len(items)} test problems, layer {LAYER} (spec §6; scoring amended 2026-10-27)",
             f"{'condition':9} {'alpha':>6}  {'accuracy':>8}  {'strict':>6}  {'no answer':>9}  "
             f"{'minus unsteered, 95% CI':>24}"]
    for k in range(len(conditions)):
        cell = rows[k * len(items):(k + 1) * len(items)]
        correct = [float(x["correct"]) for x in cell]
        mean, lo, hi = paired_bootstrap(correct, base)
        diff = "" if k == 0 else f"{mean:+.0%} [{lo:+.0%}, {hi:+.0%}]"
        lines.append(f"{cell[0]['condition']:9} {cell[0]['alpha']:>+6.1f}  {sum(correct) / len(cell):>8.0%}  "
                     f"{sum(x['strict_correct'] for x in cell) / len(cell):>6.0%}  "
                     f"{sum(x['no_answer'] for x in cell):>9}  {diff:>24}")
    print("\n" + "\n".join(lines))
    REPORT_PATH.write_text("\n".join(lines) + "\n")
    print(f"\nwrote {CSV_PATH}\nwrote {REPORT_PATH}")


if __name__ == "__main__":
    main()
