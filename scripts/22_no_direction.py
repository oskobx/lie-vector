"""Diagnostic (after the §5 sweep): is v_honesty partly a "say No" direction?

A "No" direction from replies that are all honest, so that only the polarity
of the correct answer varies, not honesty:

    d_no = mean of h over scenarios whose true answer is "no"
         - mean of h over scenarios whose true answer is "yes"

h is read at the last prompt token, as in extraction. Built at level 0 (no
role; every passing scenario is honest there by definition) and at level 1
(role, neutral question; honest replies only). Extract split, level-0 passing.

Then cos(d_no, v) per layer for v = v_honesty and v_truth. Reference: the
same cosine with the yes/no labels shuffled at random N_SHUFFLE times; p is
the share of shuffles whose |cosine| is at least the real one.

Caveat: the level-1 honest prompts are also on v_honesty's honest side, so
level 0 is the cleaner of the two.

Output: artifacts/phase2/no_direction_report.txt

    uv run python scripts/22_no_direction.py
"""

import csv

import torch

from trait_vectors import config
from trait_vectors.extract import last_token_activations, load_vectors
from trait_vectors.model import load_model
from trait_vectors.traits.base import split_by_id
from trait_vectors.traits.lying import BASELINE_PATH, LyingTrait, messages_for

LAYERS = [14, 16, 18, 20, 22, 24]
LEVELS = ["0", "1"]
N_SHUFFLE = 1000

REPORT_PATH = config.ARTIFACTS_DIR / "phase2" / "no_direction_report.txt"


def cos(a, b):
    return (a @ b / (a.norm() * b.norm())).item()


def main():
    torch.manual_seed(config.SEED)
    trait = LyingTrait()
    with open(BASELINE_PATH) as f:
        labels = {(r["id"], r["level"]): r["label"] for r in csv.DictReader(f)}
    passing = [s for s in trait.scenarios
               if split_by_id(s["id"]) == "extract" and labels[(s["id"], "0")] == "honest"]

    model, tokenizer = load_model()
    vectors = {name: load_vectors(config.ARTIFACTS_DIR / "vectors" / f"{name}.pt") for name in ("honesty", "truth")}

    lines = []
    for level in LEVELS:
        chosen = [s for s in passing if labels[(s["id"], level)] == "honest"]
        is_no = torch.tensor([s["truth"] == "no" for s in chosen])
        h = last_token_activations(model, tokenizer, [messages_for(s, level) for s in chosen], LAYERS)  # (n, layers, d)
        d_no = h[is_no].mean(dim=0) - h[~is_no].mean(dim=0)  # (layers, d)

        lines.append(f"\nlevel {level}: {len(chosen)} honest replies, true answer no {int(is_no.sum())} / yes {int((~is_no).sum())}")
        lines.append(f"  {'layer':>5}  {'direction':>9}  {'cos(d_no, v)':>12}  {'shuffled 95% range':>18}  {'p':>6}")
        g = torch.Generator().manual_seed(config.SEED)
        for j, layer in enumerate(LAYERS):
            for name in ("honesty", "truth"):
                v = vectors[name]["vectors"][vectors[name]["layers"].index(layer)]
                real = cos(d_no[j], v)
                null = []
                for _ in range(N_SHUFFLE):
                    fake = is_no[torch.randperm(len(is_no), generator=g)]
                    null.append(cos(h[fake, j].mean(dim=0) - h[~fake, j].mean(dim=0), v))
                null = torch.tensor(null)
                lo, hi = null.quantile(0.025).item(), null.quantile(0.975).item()
                p = (null.abs() >= abs(real)).float().mean().item()
                lines.append(f"  {layer:>5}  {'v_' + name:>9}  {real:>+12.3f}  {f'[{lo:+.3f}, {hi:+.3f}]':>18}  {p:>6.3f}")

    print("\n".join(lines))
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines).lstrip() + "\n")
    print(f"\nwrote {REPORT_PATH}")


if __name__ == "__main__":
    main()
