"""Diagnostic (after §5.2): is v_honesty partly a "drop the role" direction?

A "role" direction from the same scenarios with and without the character:

    d_role = mean over scenarios of ( h at level 1 - h at level 0 )

Level 1 has the role, goal and secret fact and asks the neutral question;
level 0 gives the fact with no role and asks the same question. Scenarios: extract
split, level-0 passing, honest at level 1, so honesty does not vary.

Then cos(d_role, v) per layer for v = v_honesty and v_truth. v_honesty =
honest - lie, so if it partly means "leave the role", cos(d_role, v_honesty) < 0.

Also cos^2, the share of ||v||^2 along d_role. No chance reference: a
sign-flip reference (mean of s_i (h1_i - h0_i), s_i = +-1) does not work here,
because the differences mostly point the same way, so flipping signs just gives
+-d_role again and the "chance" cosines are as large as the real one. For scale,
two random unit vectors in R^d have cosine about +-1/sqrt(d).

Caveat: level 1 and level 0 also differ in wording (system prompt, "Answer yes
or no" vs "Please answer directly"), so d_role is "role prompt", not only
"being in character".

Output: artifacts/phase2/role_direction_report.txt

    uv run python scripts/22b_role_direction.py
"""

import csv

import torch

from trait_vectors import config
from trait_vectors.extract import last_token_activations, load_vectors
from trait_vectors.model import load_model
from trait_vectors.traits.base import split_by_id
from trait_vectors.traits.lying import BASELINE_PATH, LyingTrait, messages_for

LAYERS = [14, 16, 18, 20, 22, 24]

REPORT_PATH = config.ARTIFACTS_DIR / "phase2" / "role_direction_report.txt"


def cos(a, b):
    return (a @ b / (a.norm() * b.norm())).item()


def main():
    torch.manual_seed(config.SEED)
    trait = LyingTrait()
    with open(BASELINE_PATH) as f:
        labels = {(r["id"], r["level"]): r["label"] for r in csv.DictReader(f)}
    chosen = [s for s in trait.scenarios
              if split_by_id(s["id"]) == "extract" and labels[(s["id"], "0")] == "honest"
              and labels[(s["id"], "1")] == "honest"]

    model, tokenizer = load_model()
    vectors = {name: load_vectors(config.ARTIFACTS_DIR / "vectors" / f"{name}.pt") for name in ("honesty", "truth")}

    h1 = last_token_activations(model, tokenizer, [messages_for(s, "1") for s in chosen], LAYERS)  # (n, layers, d)
    h0 = last_token_activations(model, tokenizer, [messages_for(s, "0") for s in chosen], LAYERS)
    d_role = (h1 - h0).mean(dim=0)  # (layers, d): mean of the per-scenario differences

    lines = [f"{len(chosen)} scenarios (extract split, honest at levels 0 and 1)",
             f"  {'layer':>5}  {'direction':>9}  {'cos(d_role, v)':>14}  {'cos^2':>6}"]
    for j, layer in enumerate(LAYERS):
        for name in ("honesty", "truth"):
            v = vectors[name]["vectors"][vectors[name]["layers"].index(layer)]
            c = cos(d_role[j], v)
            lines.append(f"  {layer:>5}  {'v_' + name:>9}  {c:>+14.3f}  {c * c:>6.3f}")
    lines.append(f"\nfor scale: random unit vectors in R^{d_role.shape[1]} have cosine about +-{d_role.shape[1] ** -0.5:.3f}")

    print("\n".join(lines))
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines) + "\n")
    print(f"\nwrote {REPORT_PATH}")


if __name__ == "__main__":
    main()
