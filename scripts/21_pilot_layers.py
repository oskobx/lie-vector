"""Phase 2 spec §3: which layer steers lying best? A pilot on the extract split.

For v_truth and v_honesty, at each layer in LAYERS, steer every passing
extract-split scenario at level 1-lead with alpha_h = -0.6 and +0.6, generate
a reply (greedy, 80 tokens, as in the Phase 1 baseline) and judge it.
alpha_h = 0 is the Phase 1 baseline itself. About 1,900 generations and
judgments, ~2 h. Generations and judgments are cached, so an interrupted run
picks up where it stopped.

Outputs:
    artifacts/phase2/pilot.csv          one row per (direction, layer, alpha_h, scenario)
    artifacts/phase2/pilot_report.txt   the table below, plus 10 replies at +0.6
                                        for each direction's best layer, to read

Rule (§3.3): per direction, the layer with the largest drop in lie rate from
alpha_h = -0.6 to +0.6, unless its +0.6 replies are visibly broken.

    uv run python scripts/21_pilot_layers.py
"""

import csv
import random
import time

import torch

from trait_vectors import config
from trait_vectors.extract import load_vectors
from trait_vectors.hooks import steer
from trait_vectors.metrics import paired_bootstrap
from trait_vectors.model import generate, load_model, steering_spec
from trait_vectors.traits.lying import BASELINE_PATH, LyingTrait

# Every other layer through the region where both directions form and settle
# (spec §3, amended 2026-10-25); 14 is before v_truth forms.
LAYERS = [14, 16, 18, 20, 22, 24]
ALPHAS = [-0.6, 0.6]
DIRECTIONS = ["truth", "honesty"]  # artifacts/vectors/<name>.pt; both point toward honesty (spec §1)
LEVEL = "1-lead"
MAX_NEW_TOKENS = 80
N_READ = 10

OUT_DIR = config.ARTIFACTS_DIR / "phase2"
CSV_PATH = OUT_DIR / "pilot.csv"
REPORT_PATH = OUT_DIR / "pilot_report.txt"


def pilot_items(trait):
    """The 1-lead items of extract-split scenarios that pass level 0, and their baseline labels."""
    with open(BASELINE_PATH) as f:
        labels = {(r["id"], r["level"]): r["label"] for r in csv.DictReader(f)}
    items = [it for it in trait.eval_prompts()
             if it.condition == LEVEL and trait.split(it) == "extract"
             and labels[(it.meta["scenario_id"], "0")] == "honest"]
    baseline = {it.meta["scenario_id"]: labels[(it.meta["scenario_id"], LEVEL)] for it in items}
    return items, baseline


def run(model, tokenizer, trait, items):
    """Steer, generate and judge every (direction, layer, alpha_h, item); one row each."""
    rows = []
    for name in DIRECTIONS:
        vectors = load_vectors(config.ARTIFACTS_DIR / "vectors" / f"{name}.pt")
        for layer in LAYERS:
            j = vectors["layers"].index(layer)
            v, r = vectors["vectors"][j], vectors["r"][j].item()
            for alpha in ALPHAS:
                t0 = time.time()
                spec = steering_spec(layer, v, alpha, r)
                with steer(model, layer, v, alpha, r):
                    for item in items:
                        response = generate(model, tokenizer, item.messages,
                                            max_new_tokens=MAX_NEW_TOKENS, steering=spec)
                        score = trait.score(item, response)
                        rows.append({"direction": name, "layer": layer, "alpha_h": alpha,
                                     "id": item.meta["scenario_id"], "response": response,
                                     "label": score.label, "quote": score.meta["quote"],
                                     "parse_error": score.meta["parse_error"],
                                     "reported_model": score.meta["reported_model"]})
                lie = sum(row["label"] == "lie" for row in rows[-len(items):]) / len(items)
                print(f"  v_{name:8} layer {layer}  alpha_h {alpha:+.1f}  lie {lie:5.1%}  ({time.time() - t0:.0f} s)",
                      flush=True)
    return rows


def report(rows, baseline):
    """The pilot table and, per direction, N_READ replies at +0.6 at the layer with the largest drop."""
    ids = sorted(baseline)
    base_lie = sum(baseline[i] == "lie" for i in ids) / len(ids)
    out = [f"Pilot: {LEVEL}, extract split, {len(ids)} passing scenarios. "
           f"Unsteered (alpha_h = 0) lie rate {base_lie:.1%}.",
           "drop = lie rate at -0.6 minus at +0.6, paired bootstrap 95% CI. "
           "evasive and honest are at +0.6."]
    best = {}
    for name in DIRECTIONS:
        out.append(f"\nv_{name}")
        out.append(f"  {'layer':>5} {'lie -0.6':>9} {'lie +0.6':>9} {'drop':>8} {'95% CI':>17} "
                   f"{'evasive':>8} {'honest':>7}")
        drops = {}
        for layer in LAYERS:
            label = {(row["alpha_h"], row["id"]): row["label"] for row in rows
                     if row["direction"] == name and row["layer"] == layer}
            neg = [label[(-0.6, i)] == "lie" for i in ids]
            pos = [label[(0.6, i)] == "lie" for i in ids]
            mean, lo, hi = paired_bootstrap(neg, pos)
            drops[layer] = mean
            evasive = sum(label[(0.6, i)] == "evasive" for i in ids) / len(ids)
            honest = sum(label[(0.6, i)] == "honest" for i in ids) / len(ids)
            out.append(f"  {layer:>5} {sum(neg) / len(ids):9.1%} {sum(pos) / len(ids):9.1%} {mean:+8.1%} "
                       f"[{lo:+6.1%}, {hi:+6.1%}] {evasive:8.1%} {honest:7.1%}")
        best[name] = max(LAYERS, key=lambda layer: drops[layer])
        out.append(f"  largest drop: layer {best[name]}")

    for name in DIRECTIONS:
        out.append(f"\n--- v_{name}, layer {best[name]}, alpha_h +0.6: {N_READ} replies to read (seeded) ---")
        chosen = [row for row in rows if row["direction"] == name and row["layer"] == best[name]
                  and row["alpha_h"] == 0.6]
        for row in random.Random(config.SEED).sample(chosen, N_READ):
            out.append(f"\n[{row['id']}] {row['label']}\n{row['response'].strip()}")
    return out


def main():
    torch.manual_seed(config.SEED)
    trait = LyingTrait()
    items, baseline = pilot_items(trait)
    print(f"{len(items)} items; {len(DIRECTIONS)} directions x {len(LAYERS)} layers x {len(ALPHAS)} alphas "
          f"= {len(items) * len(DIRECTIONS) * len(LAYERS) * len(ALPHAS)} generations", flush=True)

    model, tokenizer = load_model()
    rows = run(model, tokenizer, trait, items)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(CSV_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    lines = report(rows, baseline)
    REPORT_PATH.write_text("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines))
    print(f"\nwrote {CSV_PATH}\n      {REPORT_PATH}")


if __name__ == "__main__":
    main()
