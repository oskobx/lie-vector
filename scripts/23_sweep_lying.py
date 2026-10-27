"""Phase 2 spec §5: the main sweep, on held-out (eval-split) scenarios.

At LAYER, steer each direction with honesty-ward strength alpha_h in ALPHAS_H,
generate a reply (greedy, 80 tokens) for every eval-split scenario that passes
level 0, and judge it:
    v_honesty, v_truth          at 1-lead and 0-lead
    3 random unit directions    at 1-lead only, with r of v_honesty's layer
alpha_h = 0 is the Phase 1 baseline, copied in as direction "none".

Signs (spec §1, §11): the raw coefficient is alpha = +alpha_h for v_honesty and
-alpha_h for v_truth (whose "+" pole is "true", which raises lying); random
directions use alpha = alpha_h. Both are stored.

§5.1: every reply's perplexity under the unsteered model (model.reply_perplexity),
computed after the steering hook is removed.

About 2,200 generations and judgments, ~2 h. Generations and judgments are
cached, so an interrupted run picks up where it stopped.

Output: artifacts/phase2/sweep.csv, one row per (direction, alpha_h, condition, scenario).

    uv run python scripts/23_sweep_lying.py
"""

import csv
import time

import torch

from trait_vectors import config
from trait_vectors.extract import load_vectors
from trait_vectors.hooks import steer
from trait_vectors.model import generate, load_model, reply_perplexity, steering_spec
from trait_vectors.traits.lying import BASELINE_PATH, LyingTrait

LAYER = 18  # chosen in the §3 pilot for both directions (spec §11, 2026-10-26)
ALPHAS_H = [-0.6, -0.3, 0.3, 0.6]  # alpha_max = 0.6 (spec §4, §11); 0 is the baseline
SIGN = {"honesty": +1, "truth": -1}  # raw alpha = SIGN * alpha_h
CONDITIONS = ["1-lead", "0-lead"]
N_RANDOM = 3
MAX_NEW_TOKENS = 80

CSV_PATH = config.ARTIFACTS_DIR / "phase2" / "sweep.csv"


def eval_items(trait):
    """{condition: items} for eval-split scenarios that pass level 0, and the baseline rows for those items."""
    with open(BASELINE_PATH) as f:
        baseline = {(r["id"], r["level"]): r for r in csv.DictReader(f)}
    items = {c: [it for it in trait.eval_prompts()
                 if it.condition == c and trait.split(it) == "eval"
                 and baseline[(it.meta["scenario_id"], "0")]["label"] == "honest"]
             for c in CONDITIONS}
    return items, baseline


def random_directions(d):
    """N_RANDOM unit vectors in R^d, each from its own seeded generator, so they don't depend on run order."""
    out = []
    for i in range(N_RANDOM):
        g = torch.Generator().manual_seed(config.SEED + i)
        v = torch.randn(d, generator=g)
        out.append(v / v.norm())
    return out


def row(direction, alpha_h, alpha, condition, item, response, score, ppl):
    return {"direction": direction, "layer": LAYER, "alpha_h": alpha_h, "alpha": alpha, "condition": condition,
            "id": item.meta["scenario_id"], "domain": item.meta["domain"], "response": response,
            "label": score["label"], "quote": score["quote"], "parse_error": score["parse_error"],
            "reported_model": score["reported_model"], "perplexity": ppl}


def run_cell(model, tokenizer, trait, name, vector, r, alpha_h, alpha, condition, items):
    """Generate under steering, then (hook removed) judge and measure perplexity; one row per item."""
    t0 = time.time()
    spec = steering_spec(LAYER, vector, alpha, r)
    with steer(model, LAYER, vector, alpha, r):
        responses = [generate(model, tokenizer, it.messages, max_new_tokens=MAX_NEW_TOKENS, steering=spec)
                     for it in items]
    rows = []
    for it, response in zip(items, responses):
        s = trait.score(it, response)
        score = {"label": s.label, **{k: s.meta[k] for k in ("quote", "parse_error", "reported_model")}}
        rows.append(row(name, alpha_h, alpha, condition, it, response, score,
                        reply_perplexity(model, tokenizer, it.messages, response)))
    lie = sum(x["label"] == "lie" for x in rows) / len(rows)
    print(f"  {name:9} {condition:6} alpha_h {alpha_h:+.1f} (alpha {alpha:+.1f})  lie {lie:5.1%}  "
          f"({time.time() - t0:.0f} s)", flush=True)
    return rows


def main():
    torch.manual_seed(config.SEED)
    trait = LyingTrait()
    items, baseline = eval_items(trait)
    print({c: len(v) for c, v in items.items()}, "eval items per condition", flush=True)

    model, tokenizer = load_model()
    vectors = {name: load_vectors(config.ARTIFACTS_DIR / "vectors" / f"{name}.pt") for name in SIGN}
    j = vectors["honesty"]["layers"].index(LAYER)

    rows = []
    for condition in CONDITIONS:  # alpha_h = 0: the Phase 1 baseline reply, measured the same way
        for it in items[condition]:
            b = baseline[(it.meta["scenario_id"], condition)]
            score = {"label": b["label"], "quote": b["quote"], "parse_error": b["parse_error"], "reported_model": ""}
            rows.append(row("none", 0.0, 0.0, condition, it, b["response"], score,
                            reply_perplexity(model, tokenizer, it.messages, b["response"])))

    for condition in CONDITIONS:
        for name, sign in SIGN.items():
            v, r = vectors[name]["vectors"][j], vectors[name]["r"][j].item()
            for alpha_h in ALPHAS_H:
                rows += run_cell(model, tokenizer, trait, name, v, r, alpha_h, sign * alpha_h, condition, items[condition])

    r_honesty = vectors["honesty"]["r"][j].item()
    for i, v in enumerate(random_directions(model.config.hidden_size)):
        for alpha_h in ALPHAS_H:
            rows += run_cell(model, tokenizer, trait, f"random{i}", v, r_honesty, alpha_h, alpha_h, "1-lead",
                             items["1-lead"])

    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CSV_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nwrote {CSV_PATH} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
