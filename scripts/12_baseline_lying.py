"""Spec §4: baseline honest / evasive / lie rates of the unsteered model.

For every scenario x level (0, 0-lead, 1, 2, 3): one greedy generation of 80
tokens, judged by the LLM judge (spec §3). 168 scenarios -> 840 generations,
about an hour. Both generations and judgments are cached, so a rerun after an
interruption only does what is missing, and re-reporting is fast.

Outputs in artifacts/:
    lying_baseline.csv         one row per (scenario, level)
    lying_baseline_report.txt  the report below, also printed
    lying_baseline.png         stacked bars of the three labels per level

Report: level-0 pass rate (a scenario passes if its level-0 reply is judged
honest); then, over passing scenarios only, the label rates per level with
Wilson 95% intervals, for all scenarios, the eval split, and without the
numeric-comparison scenarios; the goal-driven lie rate (level k minus
0-lead); and the lie rate per domain.

    uv run python scripts/12_baseline_lying.py             # the full run
    uv run python scripts/12_baseline_lying.py --dry-run   # print one filled judge prompt; no model, no API
"""

import argparse
import csv
import json
import math
import statistics
import time

import matplotlib

matplotlib.use("Agg")  # draw to a file, no window
import matplotlib.pyplot as plt

from trait_vectors import config
from trait_vectors.model import generate, load_model
from trait_vectors.traits.lying import JUDGE_LABELS, LEVELS, LyingTrait

MAX_NEW_TOKENS = 80
CSV_PATH = config.ARTIFACTS_DIR / "lying_baseline.csv"
REPORT_PATH = config.ARTIFACTS_DIR / "lying_baseline_report.txt"
PLOT_PATH = config.ARTIFACTS_DIR / "lying_baseline.png"
COMPARISON_PATH = config.ROOT / "data" / "lying" / "comparison_ids.json"

# Categorical slots 1-3 of the reference palette, in fixed order (validated: CVD-safe;
# aqua is below 3:1 contrast on the surface, so every segment also carries a text label).
COLOURS = {"honest": "#2a78d6", "evasive": "#eb6834", "lie": "#1baf7a"}
SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#898781", "#e1e0d9"


def wilson(k, n, z=1.96):
    """Wilson 95% interval for a proportion k/n: better than p ± 1.96·se when p is near 0 or 1."""
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, centre - half), min(1.0, centre + half)  # clamp float rounding (e.g. -0.0%)


def run(trait, model, tokenizer):
    """Generate and judge every item; one row per item."""
    items = trait.eval_prompts()
    rows, t0 = [], time.time()
    for i, item in enumerate(items, start=1):
        response = generate(model, tokenizer, item.messages, max_new_tokens=MAX_NEW_TOKENS, steering="none")
        score = trait.score(item, response)
        rows.append({
            "id": item.meta["scenario_id"], "domain": item.meta["domain"], "split": trait.split(item),
            "level": item.condition, "response": response, "label": score.label,
            "quote": score.meta["quote"], "parse_error": score.meta["parse_error"],
            "judge_model": score.meta["judge_model"], "prompt_version": score.meta["prompt_version"],
        })
        if i % 20 == 0:
            print(f"{i}/{len(items)}  ({(time.time() - t0) / 60:.0f} min)", flush=True)
    return rows


def rates_line(name, rows):
    """One report line: n and each label's rate with its Wilson interval."""
    n = len(rows)
    if n == 0:
        return f"  {name:7} n=  0"
    parts = []
    for label in JUDGE_LABELS:
        k = sum(r["label"] == label for r in rows)
        lo, hi = wilson(k, n)
        parts.append(f"{label} {k / n:6.1%} [{lo:5.1%}, {hi:5.1%}]")
    return f"  {name:7} n={n:3}   " + "   ".join(parts)


def report(rows, comparison):
    """The report as a list of lines."""
    out = []
    level0 = {r["id"]: r["label"] for r in rows if r["level"] == "0"}
    passing = sorted(sid for sid, label in level0.items() if label == "honest")
    lo, hi = wilson(len(passing), len(level0))
    out.append(f"Level-0 pass rate: {len(passing)}/{len(level0)} = {len(passing) / len(level0):.1%} "
               f"[{lo:.1%}, {hi:.1%}]")
    out.append(f"Judge parse errors: {sum(r['parse_error'] for r in rows)} of {len(rows)}")
    kept = [r for r in rows if r["id"] in set(passing)]

    sections = [
        ("All passing scenarios", kept),
        ("Eval split only (the number Phase 2 compares against)", [r for r in kept if r["split"] == "eval"]),
        ("Without numeric-comparison scenarios (sensitivity check)", [r for r in kept if r["id"] not in comparison]),
    ]
    for title, subset in sections:
        out.append(f"\n{title}, per level:")
        out += [rates_line(level, [r for r in subset if r["level"] == level]) for level in LEVELS]

    # Paired difference per scenario: d = 1 if lie at level k only, -1 if lie at 0-lead only, else 0.
    # Its mean is the goal-driven lie rate; mean ± 1.96·sd/√n is its 95% interval.
    is_lie = {(r["id"], r["level"]): r["label"] == "lie" for r in kept}
    out.append("\nGoal-driven lie rate (lie rate at level k minus at 0-lead, same scenarios), 95% CI:")
    for level in ["1", "2", "3"]:
        d = [int(is_lie[(sid, level)]) - int(is_lie[(sid, "0-lead")]) for sid in passing]
        half = 1.96 * statistics.stdev(d) / math.sqrt(len(d))
        out.append(f"  level {level}: {statistics.mean(d):+.1%} ± {half:.1%}")

    out.append("\nLie rate per domain (passing scenarios):")
    out.append("  " + " " * 18 + "".join(f"{level:>9}" for level in LEVELS))
    for domain in dict.fromkeys(r["domain"] for r in rows):
        cells = []
        for level in LEVELS:
            sub = [r for r in kept if r["domain"] == domain and r["level"] == level]
            cells.append(f"{sum(r['label'] == 'lie' for r in sub) / len(sub):9.0%}" if sub else f"{'-':>9}")
        n = len([r for r in kept if r["domain"] == domain and r["level"] == "0"])
        out.append(f"  {domain:18}" + "".join(cells) + f"   (n={n})")
    return out


def plot(rows):
    """Stacked bars: share of each label per level, over passing scenarios, n above each bar."""
    passing = {r["id"] for r in rows if r["level"] == "0" and r["label"] == "honest"}
    kept = [r for r in rows if r["id"] in passing]
    fig, ax = plt.subplots(figsize=(7, 4.5), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    for x, level in enumerate(LEVELS):
        sub = [r for r in kept if r["level"] == level]
        bottom = 0.0
        for label in JUDGE_LABELS:
            share = sum(r["label"] == label for r in sub) / len(sub)
            # a thin surface-coloured edge separates the stacked segments
            ax.bar(x, share, bottom=bottom, width=0.6, color=COLOURS[label], edgecolor=SURFACE, linewidth=1.5,
                   label=label if x == 0 else None)
            if share >= 0.06:
                ax.text(x, bottom + share / 2, f"{share:.0%}", ha="center", va="center", fontsize=8, color=INK)
            bottom += share
        ax.text(x, 1.02, f"n={len(sub)}", ha="center", va="bottom", fontsize=8, color=MUTED)
    ax.set_xticks(range(len(LEVELS)), [f"level {level}" for level in LEVELS], color=INK)
    ax.set_ylim(0, 1.1)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.tick_params(colors=MUTED)
    ax.grid(axis="y", color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)
    for side in ["top", "right", "left"]:
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(MUTED)
    ax.set_title(f"Unsteered {config.MODEL_ID}: judged replies per pressure level", color=INK, fontsize=11)
    ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0), frameon=False)
    fig.tight_layout()
    fig.savefig(PLOT_PATH, dpi=150, facecolor=SURFACE)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="print one filled judge prompt; no model, no API")
    args = parser.parse_args()
    trait = LyingTrait()

    if args.dry_run:
        item = next(it for it in trait.eval_prompts() if it.condition == "2")
        print(f"(dry run: judge prompt for {item.id}, with a placeholder reply)\n")
        trait.score(item, "(no reply generated in a dry run)", dry_run=True)
        return

    model, tokenizer = load_model()
    rows = run(trait, model, tokenizer)

    with open(CSV_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    comparison = set(json.loads(COMPARISON_PATH.read_text())["ids"])
    lines = report(rows, comparison)
    REPORT_PATH.write_text("\n".join(lines) + "\n")
    plot(rows)
    print("\n" + "\n".join(lines))
    print(f"\nwrote {CSV_PATH}\n      {REPORT_PATH}\n      {PLOT_PATH}")


if __name__ == "__main__":
    main()
