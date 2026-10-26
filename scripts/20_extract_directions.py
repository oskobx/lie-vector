"""Phase 2 spec §2: extract v_truth, v_honesty and v_syc at every layer, and report.

    v_truth    mean(h | true statement) - mean(h | false statement)          (§2.1)
    v_honesty  sum over role levels l of w_l (mean(h | honest) - mean(h | lie))
               at level l, w_l proportional to n_H n_D / (n_H + n_D)         (§2.2)
    v_syc      mean(h | sycophantic letter) - mean(h | other letter), CAA    (§2.3)

h is read at the last token of each prompt, at layers 0 .. L-2. One forward
pass per prompt, no generation; about 2,700 passes in all.

Outputs:
    artifacts/vectors/{truth,honesty,syc}.pt   vectors, r, layers, provenance
    artifacts/phase2/directions_report.txt     per-layer table (§2.4), also printed
    artifacts/phase2/directions.png            and docs/figures/phase2_directions.png

    uv run python scripts/20_extract_directions.py
"""

import time

import matplotlib

matplotlib.use("Agg")  # draw to a file, no window
import matplotlib.pyplot as plt
import torch

from trait_vectors import config
from trait_vectors.extract import extract, last_token_activations, save_vectors
from trait_vectors.metrics import auroc
from trait_vectors.model import load_model
from trait_vectors.traits.lying import LyingTrait
from trait_vectors.traits.sycophancy import SycophancyTrait
from trait_vectors.traits.truth import TruthTrait

VECTORS_DIR = config.ARTIFACTS_DIR / "vectors"
OUT_DIR = config.ARTIFACTS_DIR / "phase2"
REPORT_PATH = OUT_DIR / "directions_report.txt"
PLOT_PATHS = [OUT_DIR / "directions.png", config.ROOT / "docs" / "figures" / "phase2_directions.png"]
AUROC_BAR = 0.8  # §2.1: below this at every layer, v_truth is not a usable truth direction

# Phase 1 plot style: categorical slots 1-2 of the reference palette, in fixed order.
COLOURS = {"truth": "#2a78d6", "honesty": "#eb6834"}
SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#898781", "#e1e0d9"


def extract_stratified(model, tokenizer, strata, layers):
    """v_honesty as in spec §2.2, from {level: (honest prompts, lie prompts)}.

    For each level, d_l = mean of h over its honest prompts minus mean over its
    lie prompts. The vector is the weighted sum of the d_l, with weights
    proportional to n_H n_D / (n_H + n_D) (the inverse of each d_l's noise) and
    summing to 1. r is the mean norm of h over all prompts used, as in extract().
    """
    diffs, sizes, norms, counts = [], [], [], {}
    for level, (honest, lie) in strata.items():
        h_honest = last_token_activations(model, tokenizer, honest, layers)  # (n_H, layers, d)
        h_lie = last_token_activations(model, tokenizer, lie, layers)        # (n_D, layers, d)
        diffs.append(h_honest.mean(dim=0) - h_lie.mean(dim=0))               # (layers, d)
        sizes.append(len(honest) * len(lie) / (len(honest) + len(lie)))
        norms.append(torch.cat([h_honest, h_lie]).norm(dim=-1))              # (n_H + n_D, layers)
        counts[level] = (len(honest), len(lie))

    weights = [s / sum(sizes) for s in sizes]
    vectors = sum(w * d for w, d in zip(weights, diffs))
    r = torch.cat(norms).mean(dim=0)
    strata_info = {level: {"honest": nh, "lie": nd, "weight": w}
                   for (level, (nh, nd)), w in zip(counts.items(), weights)}
    return {"layers": layers, "vectors": vectors, "r": r, "strata": strata_info}


def heldout_auroc(model, tokenizer, trait, result):
    """Per layer: AUROC of true vs false held-out statements, projected onto v_truth / |v_truth| (§2.1)."""
    items = trait.eval_prompts()
    h = last_token_activations(model, tokenizer, [it.messages for it in items], result["layers"])
    is_true = torch.tensor([it.condition == "true" for it in items])
    out = []
    for j in range(len(result["layers"])):
        v = result["vectors"][j]
        scores = h[:, j] @ (v / v.norm())  # one number per statement: its coordinate along v_truth
        out.append(auroc(scores[is_true].tolist(), scores[~is_true].tolist()))
    return out


def per_layer(result):
    """(|v_l| / r_l, cos(v_l, v_l+1)) per layer; the last layer has no successor, so cos is None."""
    v, r = result["vectors"], result["r"]
    rel = [(v[i].norm() / r[i]).item() for i in range(len(v))]
    cos = [torch.cosine_similarity(v[i], v[i + 1], dim=0).item() for i in range(len(v) - 1)] + [None]
    return rel, cos


def report(layers, stats, truth_auroc, strata_info):
    """The report as a list of lines."""
    out = ["v_honesty strata (extract split, level-0 passing):"]
    for level, s in strata_info.items():
        out.append(f"  {level:7} honest {s['honest']:3}  lie {s['lie']:3}  weight {s['weight']:.3f}")
    best = max(range(len(layers)), key=lambda j: truth_auroc[j])
    out.append(f"\nv_truth held-out AUROC: best {truth_auroc[best]:.3f} at layer {layers[best]} "
               f"(bar {AUROC_BAR}: {'reached' if truth_auroc[best] >= AUROC_BAR else 'NOT reached'})")

    out.append("\n|v|/r = length of the direction relative to a typical state; cos = cos(v_l, v_l+1)")
    out.append(f"{'layer':>5} | {'truth |v|/r':>11} {'cos':>6} {'AUROC':>6} | {'honesty |v|/r':>13} {'cos':>6}"
               f" | {'syc |v|/r':>9} {'cos':>6}")
    fmt = lambda c: f"{c:6.3f}" if c is not None else f"{'-':>6}"
    for j, ell in enumerate(layers):
        (tr, tc), (hr, hc), (sr, sc) = [(stats[k][0][j], stats[k][1][j]) for k in ["truth", "honesty", "syc"]]
        out.append(f"{ell:>5} | {tr:11.3f} {fmt(tc)} {truth_auroc[j]:6.3f} | {hr:13.3f} {fmt(hc)} | {sr:9.3f} {fmt(sc)}")
    return out


def plot(layers, stats, truth_auroc):
    """Three panels over layers: |v|/r and consecutive cosine for truth and honesty; v_truth's held-out AUROC."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), facecolor=SURFACE)
    panels = [("|v| / r  (relative length)", 0), ("cos(v_l, v_l+1)  (stability)", 1)]
    for ax, (title, k) in zip(axes, panels):
        for name in ["truth", "honesty"]:
            ys = stats[name][k]
            xs = [ell for ell, y in zip(layers, ys) if y is not None]
            ax.plot(xs, [y for y in ys if y is not None], color=COLOURS[name], linewidth=2, label=f"v_{name}")
        ax.set_title(title, color=INK, fontsize=10)
    axes[2].plot(layers, truth_auroc, color=COLOURS["truth"], linewidth=2, label="v_truth")
    axes[2].axhline(AUROC_BAR, color=MUTED, linewidth=1, linestyle="--")
    axes[2].text(layers[0], AUROC_BAR + 0.01, f"bar {AUROC_BAR}", color=MUTED, fontsize=8, va="bottom")
    axes[2].set_ylim(0.4, 1.02)
    axes[2].set_title("v_truth: held-out AUROC (true vs false)", color=INK, fontsize=10)
    for ax in axes:
        ax.set_facecolor(SURFACE)
        ax.set_xlabel("layer", color=MUTED)
        ax.tick_params(colors=MUTED)
        ax.grid(color=GRID, linewidth=0.5)
        ax.set_axisbelow(True)
        for side in ["top", "right"]:
            ax.spines[side].set_visible(False)
        for side in ["left", "bottom"]:
            ax.spines[side].set_color(MUTED)
    axes[0].legend(frameon=False, labelcolor=INK)
    fig.suptitle(f"{config.MODEL_ID}: directions per layer (extract split)", color=INK, fontsize=11)
    fig.tight_layout()
    for path in PLOT_PATHS:
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150, facecolor=SURFACE)


def main():
    torch.manual_seed(config.SEED)
    model, tokenizer = load_model()
    layers = list(range(model.config.num_hidden_layers - 1))  # 0 .. L-2, as in extract()

    t0 = time.time()
    truth_trait = TruthTrait()
    truth = extract(model, tokenizer, truth_trait)
    truth_auroc = heldout_auroc(model, tokenizer, truth_trait, truth)
    save_vectors(VECTORS_DIR / "truth.pt", truth,
                 "TruthTrait: Geometry of Truth, 150 true vs 150 false statements (Phase 2 spec §2.1)")
    print(f"v_truth done ({time.time() - t0:.0f} s)", flush=True)

    honesty = extract_stratified(model, tokenizer, LyingTrait().extraction_strata(), layers)
    save_vectors(VECTORS_DIR / "honesty.pt", honesty,
                 "LyingTrait.extraction_strata: extract split, level-0 passing, six role levels; "
                 "honest minus lie per level, inverse-variance weights (Phase 2 spec §2.2)")
    print(f"v_honesty done ({time.time() - t0:.0f} s)", flush=True)

    syc = extract(model, tokenizer, SycophancyTrait())
    save_vectors(VECTORS_DIR / "syc.pt", syc,
                 "SycophancyTrait: CAA, 1000 sycophantic vs other letter, read at the letter (Phase 2 spec §2.3)")
    print(f"v_syc done ({time.time() - t0:.0f} s)", flush=True)

    stats = {"truth": per_layer(truth), "honesty": per_layer(honesty), "syc": per_layer(syc)}
    lines = report(layers, stats, truth_auroc, honesty["strata"])
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines) + "\n")
    plot(layers, stats, truth_auroc)
    print("\n" + "\n".join(lines))
    print(f"\nwrote {VECTORS_DIR}/{{truth,honesty,syc}}.pt\n      {REPORT_PATH}\n      "
          + "\n      ".join(str(p) for p in PLOT_PATHS))


if __name__ == "__main__":
    main()
