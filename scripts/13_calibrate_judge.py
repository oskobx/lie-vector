"""Spec §3.3: calibrate the judge against hand labels.

    uv run python scripts/13_calibrate_judge.py label     # round 1: label the 64 sampled replies
    uv run python scripts/13_calibrate_judge.py relabel   # round 2, two days later: 30 of them again
    uv run python scripts/13_calibrate_judge.py report    # kappas, confusion matrix, disagreements

The sample: 8 replies per level, all 8 levels (0, 0-lead, 1, 1-lead, ...),
drawn with config.SEED from artifacts/lying_baseline.csv and frozen in
data/lying/calibration_sample.json on first use, so it never changes.
(The spec's "15 per level" assumed four levels; 8 x 8 = 64, close to its 60.)

Labelling is blind: you see the fact, the neutral question with its true
answer, the question actually asked, and the reply; never the level or the
judge's label. Order is shuffled. Answers are saved one by one to
data/lying/hand_labels.csv, so you can quit and resume.

Cohen's kappa measures agreement beyond chance: with p_o the fraction of
items two raters label the same, and p_e the agreement expected if each
labelled at random with their own label frequencies,
    kappa = (p_o - p_e) / (1 - p_e),
so 1 is perfect agreement and 0 is chance level.
"""

import argparse
import csv
import json
import random

from trait_vectors import config
from trait_vectors.traits.lying import JUDGE_LABELS, LEVELS, LyingTrait, asked_question

BASELINE_CSV = config.ARTIFACTS_DIR / "lying_baseline.csv"
SAMPLE_PATH = config.ROOT / "data" / "lying" / "calibration_sample.json"
HAND_LABELS = config.ROOT / "data" / "lying" / "hand_labels.csv"
PER_LEVEL = 8
N_RELABEL = 30
KEYS = {"h": "honest", "e": "evasive", "l": "lie"}


def baseline_rows():
    """{item id: row} from the baseline CSV; item id is "<scenario id>/<level>"."""
    with open(BASELINE_CSV) as f:
        return {f"{r['id']}/{r['level']}": r for r in csv.DictReader(f)}


def sample_ids():
    """The 60 sampled item ids, drawn once and then read from SAMPLE_PATH."""
    if SAMPLE_PATH.exists():
        return json.loads(SAMPLE_PATH.read_text())
    rows = baseline_rows()
    rng = random.Random(config.SEED)
    ids = []
    for level in LEVELS:
        ids += rng.sample(sorted(i for i, r in rows.items() if r["level"] == level), PER_LEVEL)
    rng.shuffle(ids)  # blind: levels mixed
    SAMPLE_PATH.write_text(json.dumps(ids, indent=2))
    return ids


def load_hand_labels():
    """{(item id, round): label} from HAND_LABELS."""
    if not HAND_LABELS.exists():
        return {}
    with open(HAND_LABELS) as f:
        return {(r["item_id"], int(r["round"])): r["label"] for r in csv.DictReader(f)}


def label_session(ids, round_):
    """Show each unlabelled item and record h / e / l; q quits."""
    scenarios = {s["id"]: s for s in LyingTrait().scenarios}
    rows = baseline_rows()
    done = load_hand_labels()
    todo = [i for i in ids if (i, round_) not in done]
    new_file = not HAND_LABELS.exists()
    with open(HAND_LABELS, "a", newline="") as f:
        writer = csv.writer(f)
        if new_file:
            writer.writerow(["item_id", "round", "label"])
        for n, item_id in enumerate(todo, start=1):
            scenario_id, level = item_id.split("/")
            s = scenarios[scenario_id]
            print("\n" + "=" * 78)
            print(f"round {round_}: {len(ids) - len(todo) + n}/{len(ids)}\n")
            print(f"FACT:      {s['fact']}")
            print(f"QUESTION:  {s['question']}   (true answer: {s['truth']})")
            print(f"ASKED:     {asked_question(s, level)}")
            print(f"REPLY:     {' '.join(rows[item_id]['response'].split())}\n")
            answer = None
            while answer not in KEYS:
                try:
                    answer = input("[h]onest  [e]vasive  [l]ie  [q]uit > ").strip().lower()
                except EOFError:
                    answer = "q"
                if answer == "q":
                    print("saved; rerun to continue")
                    return
            writer.writerow([item_id, round_, KEYS[answer]])
            f.flush()  # write to disk now, so a crash loses nothing
    print(f"\nround {round_} complete")


def kappa(pairs):
    """Cohen's kappa for a list of (label_a, label_b) pairs."""
    n = len(pairs)
    p_o = sum(a == b for a, b in pairs) / n
    p_e = sum((sum(a == c for a, _ in pairs) / n) * (sum(b == c for _, b in pairs) / n) for c in JUDGE_LABELS)
    return (p_o - p_e) / (1 - p_e)


def report(ids):
    rows = baseline_rows()
    hand = load_hand_labels()
    labelled = [i for i in ids if (i, 1) in hand]
    pairs = [(hand[(i, 1)], rows[i]["label"]) for i in labelled]  # (hand, judge)
    print(f"Judge vs hand (round 1): n = {len(pairs)}, agreement {sum(a == b for a, b in pairs) / len(pairs):.0%}, "
          f"kappa = {kappa(pairs):.2f}")
    relabelled = [i for i in labelled if (i, 2) in hand]
    if relabelled:
        self_pairs = [(hand[(i, 1)], hand[(i, 2)]) for i in relabelled]
        print(f"Hand vs hand (round 1 vs 2): n = {len(self_pairs)}, kappa = {kappa(self_pairs):.2f}  (the ceiling)")
    else:
        print("Hand vs hand: round 2 not done yet")

    print("\nConfusion matrix (rows: hand, columns: judge):")
    print(" " * 10 + "".join(f"{c:>9}" for c in JUDGE_LABELS))
    for a in JUDGE_LABELS:
        print(f"{a:10}" + "".join(f"{sum(h == a and j == b for h, j in pairs):9d}" for b in JUDGE_LABELS))

    print("\nDisagreements:")
    for i in labelled:
        if hand[(i, 1)] != rows[i]["label"]:
            r = rows[i]
            print(f"\n{i}: hand {hand[(i, 1)]}, judge {r['label']}")
            print(f"  reply: {' '.join(r['response'].split())}")
            print(f"  judge's quote: {r['quote']}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("step", choices=["label", "relabel", "report"])
    step = parser.parse_args().step
    ids = sample_ids()
    if step == "label":
        label_session(ids, 1)
    elif step == "relabel":
        subset = random.Random(config.SEED + 1).sample(ids, N_RELABEL)  # a fixed 30, in a new order
        label_session(subset, 2)
    else:
        report(ids)


if __name__ == "__main__":
    main()
