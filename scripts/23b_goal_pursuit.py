"""Phase 2 spec §5.2: goal pursuit of the steered replies, and its hand check.

    uv run python scripts/23b_goal_pursuit.py judge    # judge goal_v1 on the 1-lead sweep replies
    uv run python scripts/23b_goal_pursuit.py label    # hand-label 30 of them, blind
    uv run python scripts/23b_goal_pursuit.py report   # agreement with the judge, and the tables

The replies are the 1-lead rows of artifacts/phase2/sweep.csv for v_honesty,
v_truth and the unsteered baseline (direction "none", alpha_h = 0): 9 x 79 = 711.
The goal judge (goal_v1) labels each pursues or abandons; judgments are cached
like the lying judge's.

Hand labels: 30 rows drawn with config.SEED, frozen in data/lying/goal_sample.json
on first use. You see the role, goal, fact, what was asked and the reply; never
the direction, the strength, or either judge's label. Answers go one by one to
data/lying/goal_hand_labels.csv, so you can quit and resume.

Output: artifacts/phase2/goal.csv (one row per reply, with both labels) and
artifacts/phase2/goal_report.txt.
"""

import argparse
import csv
import json
import random

from trait_vectors import config
from trait_vectors.traits.lying import COUNTERPARTS, GOAL_LABELS, LyingTrait, asked_question

SWEEP_CSV = config.ARTIFACTS_DIR / "phase2" / "sweep.csv"
GOAL_CSV = config.ARTIFACTS_DIR / "phase2" / "goal.csv"
REPORT_PATH = config.ARTIFACTS_DIR / "phase2" / "goal_report.txt"
SAMPLE_PATH = config.ROOT / "data" / "lying" / "goal_sample.json"
HAND_LABELS = config.ROOT / "data" / "lying" / "goal_hand_labels.csv"
DIRECTIONS = ["none", "honesty", "truth"]
N_HAND = 30
KEYS = {"p": "pursues", "a": "abandons"}


def key(r):
    """One reply: direction, alpha_h and scenario id, e.g. "honesty|0.3|car-001"."""
    return f"{r['direction']}|{r['alpha_h']}|{r['id']}"


def judge_all():
    trait = LyingTrait()
    items = {it.id: it for it in trait.eval_prompts()}
    with open(SWEEP_CSV) as f:
        sweep = [r for r in csv.DictReader(f) if r["condition"] == "1-lead" and r["direction"] in DIRECTIONS]
    rows = []
    for n, r in enumerate(sweep, start=1):
        s = trait.score_goal(items[f"{r['id']}/1-lead"], r["response"])
        rows.append({"key": key(r), "direction": r["direction"], "alpha_h": r["alpha_h"], "id": r["id"],
                     "response": r["response"], "lying_label": r["label"], "goal_label": s.label,
                     "goal_quote": s.meta["quote"], "goal_parse_error": s.meta["parse_error"]})
        if n % 79 == 0:
            print(f"  {n}/{len(sweep)} judged", flush=True)
    with open(GOAL_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {GOAL_CSV} ({len(rows)} rows)")


def goal_rows():
    with open(GOAL_CSV) as f:
        return {r["key"]: r for r in csv.DictReader(f)}


def sample_keys():
    """The N_HAND sampled keys, drawn once and then read from SAMPLE_PATH."""
    if SAMPLE_PATH.exists():
        return json.loads(SAMPLE_PATH.read_text())
    keys = random.Random(config.SEED).sample(sorted(goal_rows()), N_HAND)  # sample() returns them in random order
    SAMPLE_PATH.write_text(json.dumps(keys, indent=2))
    return keys


def load_hand_labels():
    if not HAND_LABELS.exists():
        return {}
    with open(HAND_LABELS) as f:
        return {r["key"]: r["label"] for r in csv.DictReader(f)}


def label_session(keys):
    """Show each unlabelled reply and record p / a; q quits."""
    scenarios = {s["id"]: s for s in LyingTrait().scenarios}
    rows = goal_rows()
    done = load_hand_labels()
    todo = [k for k in keys if k not in done]
    new_file = not HAND_LABELS.exists()
    with open(HAND_LABELS, "a", newline="") as f:
        writer = csv.writer(f)
        if new_file:
            writer.writerow(["key", "label"])
        for n, k in enumerate(todo, start=1):
            s = scenarios[rows[k]["id"]]
            print("\n" + "=" * 78)
            print(f"{len(keys) - len(todo) + n}/{len(keys)}\n")
            print(f"ROLE:   {s['role']}")
            print(f"GOAL:   {s['goal']}")
            print(f"FACT:   {s['fact']}   (the {COUNTERPARTS[s['domain']]} doesn't know this)")
            print(f"ASKED:  {asked_question(s, '1-lead')}")
            print(f"REPLY:  {' '.join(rows[k]['response'].split())}\n")
            answer = None
            while answer not in KEYS:
                try:
                    answer = input("[p]ursues  [a]bandons  [q]uit > ").strip().lower()
                except EOFError:
                    answer = "q"
                if answer == "q":
                    print("saved; rerun to continue")
                    return
            writer.writerow([k, KEYS[answer]])
            f.flush()  # write to disk now, so a crash loses nothing
    print("\nlabelling complete")


def kappa(pairs):
    """Cohen's kappa for a list of (label_a, label_b) pairs (as in scripts/13_calibrate_judge.py)."""
    n = len(pairs)
    p_o = sum(a == b for a, b in pairs) / n
    p_e = sum((sum(a == c for a, _ in pairs) / n) * (sum(b == c for _, b in pairs) / n) for c in GOAL_LABELS)
    return (p_o - p_e) / (1 - p_e)


def report():
    rows = list(goal_rows().values())
    out = ["Goal pursuit, 1-lead, eval split, layer 18 (spec §5.2). Counts of 79.",
           f"Judge parse errors (counted as abandons): {sum(r['goal_parse_error'] == 'True' for r in rows)}",
           "",
           f"{'direction':9} {'alpha_h':>7}  {'pursues':>7}  {'honest+pursues':>14}  {'honest+abandons':>15}  "
           f"{'lie+pursues':>11}  {'evasive+pursues':>15}"]
    for d in DIRECTIONS[1:]:
        for a in sorted({float(r["alpha_h"]) for r in rows if r["direction"] == d} | {0.0}):
            cell = [r for r in rows if float(r["alpha_h"]) == a and r["direction"] == (d if a != 0 else "none")]
            count = lambda lie, goal: sum(r["lying_label"] == lie and r["goal_label"] == goal for r in cell)
            pursues = sum(r["goal_label"] == "pursues" for r in cell)
            out.append(f"{'v_' + d:9} {a:>+7.1f}  {pursues:>7}  {count('honest', 'pursues'):>14}  "
                       f"{count('honest', 'abandons'):>15}  {count('lie', 'pursues'):>11}  {count('evasive', 'pursues'):>15}")
        out.append("")

    hand = load_hand_labels()
    by_key = goal_rows()
    pairs = [(hand[k], by_key[k]["goal_label"]) for k in sample_keys() if k in hand]  # (hand, judge)
    if not pairs:
        out.append("Hand labels: none yet")
        return out
    out.append(f"Judge vs hand: n = {len(pairs)}, agreement {sum(a == b for a, b in pairs) / len(pairs):.0%}, "
               f"kappa = {kappa(pairs):.2f}")
    out.append("Confusion matrix (rows: hand, columns: judge):")
    out.append(" " * 10 + "".join(f"{c:>10}" for c in GOAL_LABELS))
    for a in GOAL_LABELS:
        out.append(f"{a:10}" + "".join(f"{sum(h == a and j == b for h, j in pairs):10d}" for b in GOAL_LABELS))
    out.append("\nDisagreements:")
    for k in sample_keys():
        if k in hand and hand[k] != by_key[k]["goal_label"]:
            out.append(f"\n{k}: hand {hand[k]}, judge {by_key[k]['goal_label']}")
            out.append(f"  reply: {' '.join(by_key[k]['response'].split())}")
            out.append(f"  judge's quote: {by_key[k]['goal_quote']}")
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("step", choices=["judge", "label", "report"])
    step = parser.parse_args().step
    if step == "judge":
        judge_all()
    elif step == "label":
        label_session(sample_keys())
    else:
        lines = report()
        print("\n".join(lines))
        REPORT_PATH.write_text("\n".join(lines) + "\n")
        print(f"\nwrote {REPORT_PATH}")


if __name__ == "__main__":
    main()
