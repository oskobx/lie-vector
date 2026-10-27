"""Find where steered text degenerates (Phase 0 definition-of-done item 4; Phase 2 spec §4).

Pushes alpha past the sweep's range at one layer, on a few prompts, and prints
each reply so it can be read, with the trait's score (a number for sentiment,
the judge's label for lying) and two crude fluency numbers: length in
characters and the fraction of distinct words (repetition loops drive it
toward zero).

    uv run python scripts/04_degeneration_probe.py      # Phase 0: sentiment, layer 18
    uv run python scripts/04_degeneration_probe.py --vector vectors/honesty.pt --layer 18 \\
        --prompts lying --alphas=-1.5,-1.0,-0.6,0,0.6,1.0,1.5          # Phase 2 §4

--vector is a path inside artifacts/. --prompts lying takes N_PROMPTS 1-lead
items from extract-split scenarios that pass level 0, sampled with config.SEED.
"""

import argparse
import csv
import random

import torch

from trait_vectors import config
from trait_vectors.extract import load_vectors
from trait_vectors.hooks import steer
from trait_vectors.model import generate, load_model, steering_spec
from trait_vectors.traits import lying, sentiment

LAYER_FRACTION = 0.65  # Phase 0's best layer (18 for L = 28), the default when --layer is not given
PHASE0_ALPHAS = [0.6, 1.0, 1.5, 2.0, 3.0, -1.0, -1.5, -2.0, -3.0]
N_PROMPTS = {"sentiment": 3, "lying": 5}
MAX_NEW_TOKENS = {"sentiment": config.MAX_NEW_TOKENS, "lying": 80}  # lying: as in the Phase 1 baseline


def distinct_word_fraction(text):
    words = text.lower().split()
    return len(set(words)) / len(words) if words else 0.0


def prompts_for(name):
    """(trait, items) for --prompts."""
    if name == "sentiment":
        trait = sentiment.SentimentTrait()
        return trait, trait.eval_prompts()[:N_PROMPTS[name]]
    trait = lying.LyingTrait()
    with open(lying.BASELINE_PATH) as f:
        level0 = {r["id"]: r["label"] for r in csv.DictReader(f) if r["level"] == "0"}
    items = [it for it in trait.eval_prompts()
             if it.condition == "1-lead" and trait.split(it) == "extract" and level0[it.meta["scenario_id"]] == "honest"]
    return trait, random.Random(config.SEED).sample(items, N_PROMPTS[name])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vector", default="sentiment.pt", help="path inside artifacts/")
    parser.add_argument("--layer", type=int, help="default: round(LAYER_FRACTION * L)")
    parser.add_argument("--prompts", choices=["sentiment", "lying"], default="sentiment")
    parser.add_argument("--alphas", help="comma-separated; default: Phase 0's list")
    args = parser.parse_args()

    torch.manual_seed(config.SEED)
    model, tokenizer = load_model()
    trait, items = prompts_for(args.prompts)
    alphas = [float(a) for a in args.alphas.split(",")] if args.alphas else PHASE0_ALPHAS

    data = load_vectors(config.ARTIFACTS_DIR / args.vector)
    assert data["model_id"] == config.MODEL_ID
    layer = args.layer if args.layer is not None else round(LAYER_FRACTION * model.config.num_hidden_layers)
    idx = data["layers"].index(layer)
    vector, r = data["vectors"][idx], data["r"][idx].item()
    print(f"{args.vector}, layer {layer}, r = {r:.1f}\n", flush=True)

    for alpha in alphas:
        print(f"===== alpha {alpha:+.1f} =====")
        spec = steering_spec(layer, vector, alpha, r)
        with steer(model, layer, vector, alpha, r):
            for item in items:
                response = generate(model, tokenizer, item.messages,
                                    max_new_tokens=MAX_NEW_TOKENS[args.prompts], steering=spec)
                score = trait.score(item, response)
                shown = f"{score.value:.3f}" if score.value is not None else score.label
                print(f"[{shown}  len {len(response):3d}  distinct {distinct_word_fraction(response):.2f}] "
                      f"{item.id}: {item.messages[-1]['content']}")
                print(f"    {response}\n", flush=True)


if __name__ == "__main__":
    main()
