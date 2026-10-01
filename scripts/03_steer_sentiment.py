"""Step 8: sanity sweep. Steer with the sentiment vectors, score replies, plot.

Grid: layers at LAYER_FRACTIONS of L x alpha in ALPHAS, over the trait's
neutral eval prompts. Then, at the best layer, the same alpha sweep with three
random unit directions as a control. Outputs go to artifacts/:
    sentiment_sweep.csv   one row per (layer, alpha, direction, prompt)
    sentiment_sweep.png   mean score vs alpha, one line per layer, controls overlaid
and a table of responses for one prompt across alpha is printed.
"""

import csv
import time

import matplotlib

matplotlib.use("Agg")  # draw to a file, no window
import matplotlib.pyplot as plt
import torch

from trait_vectors import config
from trait_vectors.extract import load_vectors
from trait_vectors.hooks import steer
from trait_vectors.model import generate, load_model
from trait_vectors.traits import sentiment

ALPHAS = [-0.6, -0.3, -0.15, 0.0, 0.15, 0.3, 0.6]
N_RANDOM = 3

# Which layers to sweep, as fractions of L. The extraction table showed the
# sentiment direction only becomes clear from about 0.6 L onward (|v|/r jumps
# from ~0.02 to ~0.1), so the grid is concentrated in the second half. The
# first sweep at {L/4, L/2, 3L/4} found nothing at L/4.
LAYER_FRACTIONS = [0.50, 0.65, 0.75, 0.85]

# Fixed categorical colours, one per layer in order; controls are grey and dashed.
LAYER_COLOURS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]


def sweep(model, tokenizer, trait, layer, vector, r, prompts, direction):
    """One row per (alpha, prompt): steer, generate, score."""
    rows = []
    for alpha in ALPHAS:
        t0 = time.time()
        with steer(model, layer, vector, alpha, r):
            for prompt in prompts:
                response = generate(model, tokenizer, prompt)
                rows.append({
                    "layer": layer,
                    "alpha": alpha,
                    "direction": direction,
                    "prompt": prompt,
                    "response": response,
                    "score": trait.score(prompt, response),
                })
        mean = sum(row["score"] for row in rows[-len(prompts):]) / len(prompts)
        print(f"  layer {layer:2d}  {direction:9}  alpha {alpha:+.2f}  mean score {mean:.3f}  ({time.time() - t0:.0f} s)",
              flush=True)  # flush so progress shows even when stdout is a pipe
    return rows


def mean_by_alpha(rows, layer, direction):
    """{alpha: mean score} for one (layer, direction)."""
    out = {}
    for alpha in ALPHAS:
        scores = [row["score"] for row in rows
                  if row["layer"] == layer and row["direction"] == direction and row["alpha"] == alpha]
        out[alpha] = sum(scores) / len(scores)
    return out


def main():
    torch.manual_seed(config.SEED)
    model, tokenizer = load_model()
    trait = sentiment.SentimentTrait()
    prompts = trait.eval_prompts()

    data = load_vectors(config.ARTIFACTS_DIR / "sentiment.pt")
    assert data["model_id"] == config.MODEL_ID, "vectors were extracted from a different model"
    L, d = model.config.num_hidden_layers, model.config.hidden_size
    layers = [round(f * L) for f in LAYER_FRACTIONS]
    print(f"layers {layers}, {len(prompts)} prompts, alphas {ALPHAS}", flush=True)

    # --- main sweep: the sentiment direction at each layer in the grid ---
    rows = []
    for layer in layers:
        idx = data["layers"].index(layer)
        rows += sweep(model, tokenizer, trait, layer, data["vectors"][idx], data["r"][idx].item(),
                      prompts, "sentiment")

    # best layer = largest spread of mean score between the extreme alphas
    spreads = {}
    for layer in layers:
        m = mean_by_alpha(rows, layer, "sentiment")
        spreads[layer] = m[ALPHAS[-1]] - m[ALPHAS[0]]
    best = max(spreads, key=spreads.get)
    print(f"\nspread (mean at alpha={ALPHAS[-1]} minus mean at alpha={ALPHAS[0]}) per layer:")
    for layer in layers:
        print(f"  layer {layer:2d}: {spreads[layer]:+.3f}")
    print(f"best layer: {best}\n")

    # --- control: random unit directions at the best layer, same r ---
    idx = data["layers"].index(best)
    r_best = data["r"][idx].item()
    for k in range(N_RANDOM):
        gen = torch.Generator().manual_seed(config.SEED + 1 + k)
        random_vector = torch.randn(d, generator=gen)
        rows += sweep(model, tokenizer, trait, best, random_vector, r_best, prompts, f"random{k}")

    # --- CSV ---
    csv_path = config.ARTIFACTS_DIR / "sentiment_sweep.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {csv_path}")

    # --- plot: mean score vs alpha ---
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for layer, colour in zip(layers, LAYER_COLOURS):
        m = mean_by_alpha(rows, layer, "sentiment")
        ax.plot(ALPHAS, [m[a] for a in ALPHAS], marker="o", color=colour, linewidth=2, label=f"layer {layer}")
    for k in range(N_RANDOM):
        m = mean_by_alpha(rows, best, f"random{k}")
        ax.plot(ALPHAS, [m[a] for a in ALPHAS], marker="o", color="#888888", linewidth=1.5,
                linestyle="--", label=f"random, layer {best}" if k == 0 else None)
    ax.set_xlabel("alpha")
    ax.set_ylabel("mean P(positive) over eval prompts")
    ax.set_ylim(0, 1)
    ax.set_title(f"Sentiment steering, {config.MODEL_ID}")
    ax.grid(True, color="#dddddd", linewidth=0.5)
    ax.legend()
    fig.tight_layout()
    png_path = config.ARTIFACTS_DIR / "sentiment_sweep.png"
    fig.savefig(png_path, dpi=150)
    print(f"wrote {png_path}")

    # --- responses for one prompt across alpha, at the best layer ---
    prompt = prompts[0]
    print(f"\nresponses at layer {best} for: {prompt!r}\n")
    for row in rows:
        if row["layer"] == best and row["direction"] == "sentiment" and row["prompt"] == prompt:
            print(f"alpha {row['alpha']:+.2f}  score {row['score']:.3f}")
            print(f"  {row['response']}\n")


if __name__ == "__main__":
    main()
