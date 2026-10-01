"""Find where steered text degenerates (definition-of-done item 4).

The main sweep stayed fluent up to |alpha| = 0.6. This pushes further at the
best layer, on a few prompts, and prints each reply so it can be read, with
two crude fluency numbers: length in characters and the fraction of distinct
words (repetition loops drive it toward zero).
"""

import torch

from trait_vectors import config
from trait_vectors.extract import load_vectors
from trait_vectors.hooks import steer
from trait_vectors.model import generate, load_model
from trait_vectors.traits import sentiment

LAYER_FRACTION = 0.65  # the best layer of the main sweep (18 for L = 28)
ALPHAS = [0.6, 1.0, 1.5, 2.0, 3.0, -1.0, -1.5, -2.0, -3.0]
N_PROMPTS = 3


def distinct_word_fraction(text):
    words = text.lower().split()
    return len(set(words)) / len(words) if words else 0.0


def main():
    torch.manual_seed(config.SEED)
    model, tokenizer = load_model()
    trait = sentiment.SentimentTrait()
    prompts = trait.eval_prompts()[:N_PROMPTS]

    data = load_vectors(config.ARTIFACTS_DIR / "sentiment.pt")
    assert data["model_id"] == config.MODEL_ID
    layer = round(LAYER_FRACTION * model.config.num_hidden_layers)
    idx = data["layers"].index(layer)
    vector, r = data["vectors"][idx], data["r"][idx].item()
    print(f"layer {layer}, r = {r:.1f}\n", flush=True)

    for alpha in ALPHAS:
        print(f"===== alpha {alpha:+.1f} =====")
        with steer(model, layer, vector, alpha, r):
            for prompt in prompts:
                response = generate(model, tokenizer, prompt)
                score = trait.score(prompt, response)
                print(f"[score {score:.3f}  len {len(response):3d}  distinct {distinct_word_fraction(response):.2f}] "
                      f"{prompt}")
                print(f"    {response}\n", flush=True)


if __name__ == "__main__":
    main()
