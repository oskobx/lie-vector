"""Phase 3 (docs/phase3_spec.md): detection. From the model's state at the last
prompt token, before it writes anything, can we tell whether it is about to lie?

    uv run python scripts/30_detection.py states   # one forward pass per prompt, ~2 min
    uv run python scripts/30_detection.py report   # fit on extract, freeze, score eval, Gate 3; CPU, minutes

states (spec §2): every scenario that passes level 0, both splits, at level 0
and the six role levels (158 x 7 = 1,106 prompts). One forward pass each, no
generation. Saved per prompt: h_18 at the last prompt token, the next-token
logits of the two answer words, and what the report needs about the prompt
(scenario, level, split, Phase 1 label, true answer, negated leading question,
first word of the Phase 1 reply). Three checks, printed:
    1. the honest / lie prompts are exactly LyingTrait.extraction_strata(split=...)
    2. logit(Yes) > logit(No) wherever the greedy Phase 1 reply starts "Yes",
       and the reverse for "No" (same prompt, same model: must hold always)
    3. v_honesty rebuilt from these states equals artifacts/vectors/honesty.pt

report (spec §3-6, as amended in §9): first fit(), which sees the extract
rows only: each direction's sign, the probe (lambda by 5-fold cross-validation
over extract scenarios, out-of-fold scores), d_fact from level 0, the shortcut
detector (same cross-validation), and the 100 random directions. All of it is
saved to artifacts/phase3/detectors.pt and read back from there; only then is
the eval split scored. Gate 3 (spec §5), the §6 numbers, and the figure.

Outputs: artifacts/phase3/states.pt, detectors.pt, report.txt;
docs/figures/phase3_detection.png.
"""

import argparse
import csv
import re
import shutil
import time

import matplotlib

matplotlib.use("Agg")  # draw to a file, no window
import matplotlib.pyplot as plt
import numpy as np
import torch

from trait_vectors import config
from trait_vectors.extract import load_vectors, logistic_regression
from trait_vectors.hooks import read_activations
from trait_vectors.metrics import auroc, stratified_auroc
from trait_vectors.model import build_inputs, load_model
from trait_vectors.traits.lying import ANSWER_WORDS, BASELINE_PATH, ROLE_LEVELS, LyingTrait, is_negated

LAYER = 18
LEVELS = ["0"] + ROLE_LEVELS  # level 0 for the fact direction (spec §3), the role levels for detection

DIRECTIONS = ["honesty", "honesty_neutral", "truth"]   # artifacts/vectors/<name>.pt
DETECTORS = [f"v_{name}" for name in DIRECTIONS] + ["probe"]  # the four Gate 3 is applied to
# shortcut: as specified in §3 (pooled; broken as a control, see spec §9). shortcut_level: per level, §9 2026-10-28.
CONTROLS = ["shortcut_level", "shortcut"]
LAMBDAS = [0.001, 0.01, 0.1, 1, 10]
N_FOLDS = 5
N_RANDOM = 100
N_DRAWS = 2000
GATE_AUROC = 0.70

VECTORS_DIR = config.ARTIFACTS_DIR / "vectors"
OUT_DIR = config.ARTIFACTS_DIR / "phase3"
STATES_PATH = OUT_DIR / "states.pt"
DETECTORS_PATH = OUT_DIR / "detectors.pt"
REPORT_PATH = OUT_DIR / "report.txt"
FIG_PATH = config.ROOT / "docs" / "figures" / "phase3_detection.png"

# Plot style of Phases 1-2 (scripts/25_report_phase2.py).
SURFACE, INK, MUTED, GRID, BAND = "#fcfcfb", "#0b0b0b", "#898781", "#e1e0d9", "#ecebe6"
DETECTOR_COLOUR = "#2a78d6"


# ---------- states ----------

def first_word(response):
    """The answer word the reply starts with ("Yes" / "No"), or "" for any other start ("Not", "I", ...)."""
    for word in ANSWER_WORDS:
        if re.match(rf"{word}\b", response.strip()):
            return word
    return ""


def check_labels(trait, items, labels):
    """Check 1: per split and role level, the honest and lie prompts are those of extraction_strata."""
    for split in ("extract", "eval"):
        strata = trait.extraction_strata(split=split)
        assert list(strata) == ROLE_LEVELS, f"{split}: a level was skipped"
        for level, (honest, lie) in strata.items():
            for side, expected in (("honest", honest), ("lie", lie)):
                mine = [it.messages for it, label in zip(items, labels)
                        if trait.split(it) == split and it.condition == level and label == side]
                assert mine == expected, f"{split} {level} {side}: differs from extraction_strata"


def rebuilt_honesty(states, splits, levels, labels):
    """Check 3: v_honesty as in scripts/20_extract_directions.py, from the extract-split states.

    Per level, mean(honest) - mean(lie); summed with weights n_H n_D / (n_H + n_D), normalised to sum 1.
    """
    diffs, sizes = [], []
    for level in ROLE_LEVELS:
        at = [i for i in range(len(levels)) if splits[i] == "extract" and levels[i] == level]
        honest = [i for i in at if labels[i] == "honest"]
        lie = [i for i in at if labels[i] == "lie"]
        diffs.append(states[honest].mean(dim=0) - states[lie].mean(dim=0))
        sizes.append(len(honest) * len(lie) / (len(honest) + len(lie)))
    return sum(s / sum(sizes) * d for s, d in zip(sizes, diffs))


def states():
    torch.manual_seed(config.SEED)
    trait = LyingTrait()
    with open(BASELINE_PATH) as f:
        baseline = {(r["id"], r["level"]): r for r in csv.DictReader(f)}
    items = [it for it in trait.eval_prompts()
             if it.condition in LEVELS and baseline[(it.meta["scenario_id"], "0")]["label"] == "honest"]
    rows = [baseline[(it.meta["scenario_id"], it.condition)] for it in items]
    labels = [r["label"] for r in rows]
    check_labels(trait, items, labels)
    print(f"{len(items)} prompts; check 1 passed (labels = extraction_strata on both splits)", flush=True)

    model, tokenizer = load_model()
    token_ids = []
    for word in ANSWER_WORDS:
        ids = tokenizer.encode(word, add_special_tokens=False)
        assert len(ids) == 1, f"answer word {word!r} is {len(ids)} tokens"
        token_ids.append(ids[0])

    h, logits = [], []
    t0 = time.time()
    for k, it in enumerate(items):
        inputs = build_inputs(tokenizer, it.messages).to(model.device)
        with read_activations(model, [LAYER]) as acts:
            with torch.inference_mode():
                out = model(**inputs)
        h.append(acts[LAYER][0, -1])                          # (d,): batch 0, last prompt position
        logits.append(out.logits[0, -1, token_ids].float().cpu())  # (2,): next-token logits of Yes, No
        if (k + 1) % 100 == 0:
            print(f"  {k + 1}/{len(items)}  ({time.time() - t0:.0f} s)", flush=True)

    payload = {
        "states": torch.stack(h),        # (n, d) float32
        "logits": torch.stack(logits),   # (n, 2) float32, columns in the order of answer_words
        "answer_words": list(ANSWER_WORDS), "token_ids": token_ids,
        "scenario_id": [it.meta["scenario_id"] for it in items],
        "level": [it.condition for it in items],
        "split": [trait.split(it) for it in items],
        "label": labels,
        "truth": [it.meta["truth"] for it in items],
        "negated": [is_negated(it.meta["leading"]) for it in items],
        "first_word": [first_word(r["response"]) for r in rows],
        "layer": LAYER, "model_id": config.MODEL_ID, "dtype": str(config.DTYPE), "seed": config.SEED,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(payload, STATES_PATH)
    print(f"wrote {STATES_PATH} ({time.time() - t0:.0f} s)\n")

    for split in ("extract", "eval"):
        for level in LEVELS:
            at = [lab for lab, s, lev in zip(labels, payload["split"], payload["level"]) if s == split and lev == level]
            print(f"  {split:7} {level:6}  honest {at.count('honest'):3}  evasive {at.count('evasive'):3}  "
                  f"lie {at.count('lie'):3}")

    diff = payload["logits"][:, 0] - payload["logits"][:, 1]  # logit(Yes) - logit(No)
    yes = [i for i, w in enumerate(payload["first_word"]) if w == "Yes"]
    no = [i for i, w in enumerate(payload["first_word"]) if w == "No"]
    bad = [i for i in yes if diff[i] <= 0] + [i for i in no if diff[i] >= 0]
    print(f"\ncheck 2: {len(yes)} replies start Yes, {len(no)} No; logit sign disagrees with the reply in {len(bad)}")
    for i in bad[:10]:
        print(f"  {payload['scenario_id'][i]}/{payload['level'][i]}  first word {payload['first_word'][i]}  "
              f"logit(Yes) - logit(No) = {diff[i]:+.2f}")

    saved = load_vectors(VECTORS_DIR / "honesty.pt")
    v_saved = saved["vectors"][saved["layers"].index(LAYER)]
    v_new = rebuilt_honesty(payload["states"], payload["split"], payload["level"], labels)
    cos = (v_new @ v_saved / (v_new.norm() * v_saved.norm())).item()
    print(f"check 3: v_honesty rebuilt vs saved: cosine {cos:.6f}, "
          f"|difference| / |saved| = {((v_new - v_saved).norm() / v_saved.norm()).item():.2e}")


# ---------- report: fitting (extract split only) ----------

def unit(v):
    return v / np.linalg.norm(v)


def direction(name):
    """The unit vector of a Phase 2 direction at LAYER."""
    vec = load_vectors(VECTORS_DIR / f"{name}.pt")
    return unit(vec["vectors"][vec["layers"].index(LAYER)].numpy())


def random_directions(d):
    """N_RANDOM unit vectors, seed config.SEED + i, as in scripts/23_sweep_lying.py (random0-2 are the first three)."""
    out = []
    for i in range(N_RANDOM):
        g = torch.Generator().manual_seed(config.SEED + i)
        out.append(unit(torch.randn(d, generator=g).numpy()))
    return np.stack(out)  # (N_RANDOM, d)


def fit_logistic(X, y, level, lam, centre):
    """logistic_regression on X (centred per level if centre=True). Returns {"w", "b", "mu", "lam"}."""
    mu = {lev: X[level == lev].mean(axis=0) for lev in np.unique(level)} if centre else None
    Xc = X - np.stack([mu[lev] for lev in level]) if centre else X
    w, b = logistic_regression(torch.from_numpy(Xc), torch.from_numpy(y.astype(float)), lam)
    return {"w": w.numpy(), "b": b, "mu": mu, "lam": lam}


def logistic_score(model, X, level):
    """w . (x - mu_level) + b, the fitted log-odds of "lie". Per level this is w . x plus a constant,
    so its stratified AUROC is that of w . h (spec §3)."""
    if model["mu"] is not None:
        X = X - np.stack([model["mu"][lev] for lev in level])
    return X @ model["w"] + model["b"]


def fit_per_level(X, y, level, lam):
    """One logistic regression per level, no centring (the per-level shortcut detector, spec §9, 2026-10-28)."""
    return {"per_level": {lev: fit_logistic(X[level == lev], y[level == lev], level[level == lev], lam, centre=False)
                          for lev in np.unique(level)}, "lam": lam}


def per_level_score(model, X, level):
    """Each prompt scored by its own level's logistic regression."""
    out = np.empty(len(X))
    for lev, m in model["per_level"].items():
        here = level == lev
        out[here] = logistic_score(m, X[here], level[here])
    return out


def choose_lambda(X, y, level, sid, fit_fn, score_fn):
    """lambda by 5-fold cross-validation over scenarios (spec §3, §9.1). Returns (lambda, {lambda: CV AUROC}).

    fit_fn(X, y, level, lam) fits a model, score_fn(model, X, level) scores with it.

    Each scenario goes into one fold with all its levels. Every prompt is scored by
    the model fitted without its fold, and one stratified AUROC is taken over all
    of these out-of-fold scores. A tie goes to the larger lambda.
    """
    ids = sorted(set(sid))
    perm = np.random.default_rng(config.SEED).permutation(len(ids))
    fold_of = {ids[j]: k % N_FOLDS for k, j in enumerate(perm)}
    fold = np.array([fold_of[s] for s in sid])
    cv = {}
    for lam in LAMBDAS:
        oof = np.zeros(len(y))
        for k in range(N_FOLDS):
            test = fold == k
            model = fit_fn(X[~test], y[~test], level[~test], lam)
            oof[test] = score_fn(model, X[test], level[test])
        cv[lam] = stratified_auroc(oof, y, level)
    best = max(sorted(LAMBDAS, reverse=True), key=lambda lam: cv[lam])  # max keeps the first, i.e. largest, of ties
    return best, cv


def features(H, word, d_fact):
    """The shortcut detector's two features (spec §3): word-reader and fact-reader, shape (n, 2)."""
    return np.stack([word, H @ d_fact], axis=1)


def raw_scores(fitted, H, word, level):
    """Unsigned scores of the four detectors and the shortcut detector, plus the two features alone."""
    out = {f"v_{name}": H @ fitted["v"][name] for name in DIRECTIONS}
    out["probe"] = logistic_score(fitted["probe"], H, level)
    F = features(H, word, fitted["d_fact"])
    out["shortcut"] = logistic_score(fitted["shortcut"], F, level)
    out["shortcut_level"] = per_level_score(fitted["shortcut_level"], F, level)
    out["word"] = word                   # logit(Yes) - logit(No): + = about to say Yes
    out["fact"] = H @ fitted["d_fact"]    # + = "the true answer is no"
    return out


def fit(d):
    """Everything that is fitted (spec §2), from `d`, which holds the extract rows only."""
    rows = (d["level"] != "0") & (d["label"] != "evasive")  # the role-level honest and lie prompts
    X, y, level, sid = d["H"][rows], d["label"][rows] == "lie", d["level"][rows], d["sid"][rows]
    zero = d["level"] == "0"
    fitted = {"v": {name: direction(name) for name in DIRECTIONS},
              "d_fact": unit(d["H"][zero & (d["truth"] == "no")].mean(axis=0)
                             - d["H"][zero & (d["truth"] == "yes")].mean(axis=0))}
    centred = lambda X, y, level, lam: fit_logistic(X, y, level, lam, centre=True)
    pooled = lambda X, y, level, lam: fit_logistic(X, y, level, lam, centre=False)
    lam, cv = choose_lambda(X, y, level, sid, centred, logistic_score)
    fitted["probe"] = centred(X, y, level, lam)
    F = features(X, d["word"][rows], fitted["d_fact"])
    lam_s, cv_s = choose_lambda(F, y, level, sid, pooled, logistic_score)  # as specified in §3
    fitted["shortcut"] = pooled(F, y, level, lam_s)
    lam_l, cv_l = choose_lambda(F, y, level, sid, fit_per_level, per_level_score)  # §9, 2026-10-28
    fitted["shortcut_level"] = fit_per_level(F, y, level, lam_l)
    fitted["cv"] = {"probe": cv, "shortcut": cv_s, "shortcut_level": cv_l}

    # Signs (spec §3): the orientation with AUROC >= 0.5 on extract. In-sample for every detector.
    raw = raw_scores(fitted, X, d["word"][rows], level)
    fitted["extract_auroc"] = {name: stratified_auroc(raw[name], y, level) for name in DETECTORS + CONTROLS}
    fitted["sign"] = {name: 1 if a >= 0.5 else -1 for name, a in fitted["extract_auroc"].items()}
    fitted["random"] = random_directions(X.shape[1])
    fitted["fact_level0_extract_auroc"] = auroc(d["H"][zero & (d["truth"] == "no")] @ fitted["d_fact"],
                                                d["H"][zero & (d["truth"] == "yes")] @ fitted["d_fact"])
    return fitted


# ---------- report: scoring (eval split) ----------

def load_split(p, split):
    """The rows of states.pt in one split, as numpy arrays."""
    keep = np.array(p["split"]) == split
    return {"H": p["states"].numpy()[keep],
            "word": (p["logits"][:, 0] - p["logits"][:, 1]).numpy()[keep],  # logit(Yes) - logit(No)
            "level": np.array(p["level"])[keep], "label": np.array(p["label"])[keep],
            "sid": np.array(p["scenario_id"])[keep], "truth": np.array(p["truth"])[keep],
            "negated": np.array(p["negated"])[keep], "first_word": np.array(p["first_word"])[keep]}


def bootstrap(scores, y, level, sid, views):
    """{view: {name: N_DRAWS AUROCs}}: paired bootstrap over scenarios (spec §4).

    Each draw resamples the scenarios with replacement, each with all its levels,
    and recomputes every detector's stratified AUROC on the same resample, so
    differences between detectors are paired. `views` are row masks (subsets).
    """
    ids = sorted(set(sid))
    rows_of = [np.flatnonzero(sid == s) for s in ids]
    rng = np.random.default_rng(config.SEED)
    out = {v: {name: np.empty(N_DRAWS) for name in scores} for v in views}
    for j in range(N_DRAWS):
        idx = np.concatenate([rows_of[k] for k in rng.integers(0, len(ids), len(ids))])
        for v, mask in views.items():
            sub = idx[mask[idx]]
            for name, s in scores.items():
                out[v][name][j] = stratified_auroc(s[sub], y[sub], level[sub])
    return out


def interval(x):
    lo, hi = np.percentile(x, [2.5, 97.5])
    return float(lo), float(hi)


def plot(A, ci, rand, n_scenarios):
    names = DETECTORS + CONTROLS
    shown = {"v_honesty": "v_honesty", "v_honesty_neutral": "v_honesty_neutral", "v_truth": "v_truth",
             "probe": "probe", "shortcut_level": "shortcut detector,\nper level (control)",
             "shortcut": "shortcut detector as\nspecified (broken, §9)"}
    fig, ax = plt.subplots(figsize=(7.5, 4.0), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    lo5, hi95 = np.percentile(rand, [5, 95])
    ax.axvspan(lo5, hi95, color=BAND, linewidth=0)
    ax.text((lo5 + hi95) / 2, len(names) - 0.45, "100 random directions\n5-95%", ha="center", va="bottom",
            fontsize=7, color=MUTED)
    ax.axvline(0.5, color=MUTED, linewidth=0.8)
    ax.axvline(GATE_AUROC, color=MUTED, linewidth=0.8, linestyle="--")
    ax.text(GATE_AUROC, -0.75, " gate 0.70", ha="left", va="center", fontsize=7, color=MUTED)
    for row, name in enumerate(reversed(names)):
        colour = MUTED if name in CONTROLS else DETECTOR_COLOUR
        lo, hi = ci[name]
        ax.plot([lo, hi], [row, row], color=colour, linewidth=2, solid_capstyle="round")
        ax.plot(A[name], row, "o", color=colour, markersize=8, markeredgecolor=SURFACE, markeredgewidth=2)
        ax.text(hi + 0.01, row, f"{A[name]:.2f}", va="center", fontsize=8, color=INK)
    ax.set_yticks(range(len(names)), [shown[n] for n in reversed(names)], color=INK, fontsize=9)
    ax.set_ylim(-1, len(names) + 0.3)
    ax.set_xlim(0.2, 1.0)
    ax.set_xlabel(f"stratified AUROC, lie vs honest (eval split, {n_scenarios} scenarios, 95% interval)",
                  color=MUTED, fontsize=8)
    ax.tick_params(axis="x", colors=MUTED, labelsize=8)
    ax.grid(axis="x", color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)
    for side in ["top", "right", "left"]:
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(MUTED)
    ax.tick_params(axis="y", length=0)
    ax.set_title("Does the layer-18 state before the reply predict a lie?", color=INK, fontsize=10, loc="left")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "phase3_detection.png", dpi=150, facecolor=SURFACE)
    FIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(OUT_DIR / "phase3_detection.png", FIG_PATH)
    plt.close(fig)


def gate(A, ci, boot, control, p95, out):
    """Gate 3 (spec §5) for the four detectors against one shortcut detector; returns the names that pass."""
    out.append(f"  {'detector':18} {'AUROC':>6} {'95% interval':>15}   {'minus ' + control:>26}   "
               f"1  2  3   reads lying?")
    passing = []
    for name in DETECTORS:
        dlo, dhi = interval(boot["all"][name] - boot["all"][control])  # paired: same draws
        c1 = A[name] >= GATE_AUROC and ci[name][0] > 0.5
        c2 = dlo > 0
        c3 = A[name] > p95
        if c1 and c2 and c3:
            passing.append(name)
        mark = lambda c: "✓" if c else "✗"
        out.append(f"  {name:18} {A[name]:6.3f} [{ci[name][0]:.3f}, {ci[name][1]:.3f}]   "
                   f"{A[name] - A[control]:+.3f} [{dlo:+.3f}, {dhi:+.3f}]     {mark(c1)}  {mark(c2)}  {mark(c3)}   "
                   f"{'YES' if name in passing else 'no'}")
    if not passing:
        reading = "none passes: no linear lie signal before the reply beyond the planned word and the fact"
    elif passing == ["probe"]:
        reading = "only the probe passes: lying is linearly readable, but not along a difference-of-means direction"
    else:
        reading = "a Phase 2 direction passes: it reads lying though it does not steer it"
        if "v_truth" in passing:
            reading += "; v_truth passes: evidence for picture A"
    out.append(f"  => passing: {', '.join(passing) or 'none'}. Reading (spec §5 table): {reading}")
    return passing


def report():
    p = torch.load(STATES_PATH)
    torch.save(fit(load_split(p, "extract")), DETECTORS_PATH)
    # From here on only the frozen file is used. weights_only=False: our own file, holding numpy arrays.
    fitted = torch.load(DETECTORS_PATH, weights_only=False)

    d = load_split(p, "eval")
    role = d["level"] != "0"
    raw = raw_scores(fitted, d["H"][role], d["word"][role], d["level"][role])
    s = {name: fitted["sign"][name] * raw[name] for name in DETECTORS + CONTROLS}
    s.update(word=raw["word"], fact=raw["fact"])
    label, level, sid = d["label"][role], d["level"][role], d["sid"][role]
    rand_all = d["H"][role] @ fitted["random"].T  # (n, N_RANDOM)

    rows = label != "evasive"  # honest and lie: the prompts AUROCs are computed on
    y, lev, ids = label[rows] == "lie", level[rows], sid[rows]
    sc = {name: x[rows] for name, x in s.items()}
    A = {name: stratified_auroc(x, y, lev) for name, x in sc.items()}
    rand = np.array([stratified_auroc(rand_all[rows, i], y, lev) for i in range(N_RANDOM)])
    rand = np.maximum(rand, 1 - rand)
    p95 = float(np.percentile(rand, 95))

    first, lead = d["first_word"][role][rows], np.char.endswith(lev, "-lead")
    views = {"all": np.ones(len(y), bool), "yes": first == "Yes"}
    boot = bootstrap({n: sc[n] for n in DETECTORS + CONTROLS}, y, lev, ids, views)
    ci = {n: interval(boot["all"][n]) for n in DETECTORS + CONTROLS}

    out = [f"Phase 3 detection, layer {LAYER}, eval split: {len(set(ids))} scenarios, "
           f"{(~y).sum()} honest, {y.sum()} lies (evasive dropped); intervals: paired bootstrap over scenarios, "
           f"{N_DRAWS} draws, 95%.", ""]
    out.append("== Fitted on the extract split, frozen in detectors.pt before eval was scored ==")
    for name in DETECTORS + CONTROLS:
        out.append(f"  {name:18} sign {fitted['sign'][name]:+d}   extract AUROC (in-sample, signed) "
                   f"{max(fitted['extract_auroc'][name], 1 - fitted['extract_auroc'][name]):.3f}")
    for name in ["probe"] + CONTROLS:
        cv = fitted["cv"][name]
        out.append(f"  {name} lambda by 5-fold CV (out-of-fold stratified AUROC): "
                   + "  ".join(f"{lam:g}: {a:.3f}" for lam, a in cv.items()) + f"  -> {fitted[name]['lam']:g}")
    w = fitted["shortcut"]["w"]
    out.append(f"  shortcut (pooled) weights: word {w[0]:+.3f} per logit, fact {w[1]:+.3f} per unit projection")
    out.append("  shortcut_level weights (word per logit, fact per unit projection): " + "  ".join(
        f"{L} {m['w'][0]:+.2f}/{m['w'][1]:+.2f}" for L, m in fitted["shortcut_level"]["per_level"].items()))
    zero = d["level"] == "0"
    fact_eval0 = auroc(d["H"][zero & (d["truth"] == "no")] @ fitted["d_fact"],
                       d["H"][zero & (d["truth"] == "yes")] @ fitted["d_fact"])
    out.append(f"  d_fact reads the fact at level 0 (true answer no vs yes): AUROC extract "
               f"{fitted['fact_level0_extract_auroc']:.3f} (in-sample), eval {fact_eval0:.3f}")

    out.append("\n== Gate 3 (spec §5) ==")
    out.append(f"  random directions, max(A, 1 - A): median {np.median(rand):.3f}, 5-95% "
               f"[{np.percentile(rand, 5):.3f}, {np.percentile(rand, 95):.3f}], max {rand.max():.3f}")
    for name in CONTROLS:
        out.append(f"  {name}: {A[name]:.3f} [{ci[name][0]:.3f}, {ci[name][1]:.3f}]")
    out.append("  criteria: 1 AUROC >= 0.70 and interval above 0.5; 2 paired difference to the shortcut detector, "
               f"interval > 0; 3 above the random 95th percentile ({p95:.3f})")
    out.append("\n  As specified (§3 pooled shortcut detector; criterion 2 vacuous: control broken, spec §9):")
    gate(A, ci, boot, "shortcut", p95, out)
    out.append("\n  Amended, post hoc (per-level shortcut detector, spec §9 2026-10-28); the Phase 3 reading rests on this:")
    gate(A, ci, boot, "shortcut_level", p95, out)

    names = DETECTORS + CONTROLS + ["word", "fact"]
    out.append("\n== §6 AUROC per level (lie vs honest within the level; word / fact = the two features alone) ==")
    out.append(f"  {'level':7} {'n_H':>4} {'n_L':>4}  " + " ".join(f"{n[:12]:>12}" for n in names))
    for L in ROLE_LEVELS:
        m = lev == L
        out.append(f"  {L:7} {(m & ~y).sum():4} {(m & y).sum():4}  "
                   + " ".join(f"{auroc(sc[n][m & y], sc[n][m & ~y]):12.3f}" for n in names))
    out.append(f"  {'strat.':7} {(~y).sum():4} {y.sum():4}  " + " ".join(f"{A[n]:12.3f}" for n in names))

    out.append("\n== §6 by first word of the reply ==")
    yes = views["yes"]
    used = [L for L in ROLE_LEVELS if min((yes & (lev == L) & y).sum(), (yes & (lev == L) & ~y).sum()) >= 5]
    out.append(f"  starts 'Yes' ({(yes & ~y).sum()} honest, {(yes & y).sum()} lies), stratified over levels "
               f"with >= 5 per side: {', '.join(used)}")
    for n in DETECTORS + CONTROLS:
        lo, hi = interval(boot["yes"][n])
        out.append(f"    {n:18} {stratified_auroc(sc[n][yes], y[yes], lev[yes]):.3f} [{lo:.3f}, {hi:.3f}]")
    for n in ("word", "fact"):
        out.append(f"    {n:18} {stratified_auroc(sc[n][yes], y[yes], lev[yes]):.3f}")
    no = first == "No"
    out.append(f"  starts 'No' ({(no & ~y).sum()} honest, {(no & y).sum()} lies): pooled over levels, "
               "not stratified; too few to judge")
    out.append("    " + "  ".join(f"{n} {auroc(sc[n][no & y], sc[n][no & ~y]):.3f}" for n in names))

    out.append("\n== §6 negated leading questions (keyword flag; -lead prompts of flagged scenarios dropped) ==")
    keep = ~(d["negated"][role][rows] & lead)
    out.append(f"  dropped {(~keep).sum()} prompts ({(~keep & y).sum()} lies); kept {keep.sum()}")
    out.append(f"  {'':18} {'all':>7} {'without':>8}")
    for n in names:
        out.append(f"  {n:18} {A[n]:7.3f} {stratified_auroc(sc[n][keep], y[keep], lev[keep]):8.3f}")

    out.append("\n== §6 evasive replies: mean score after subtracting each level's mean (honest and lie) ==")
    out.append(f"  {'':18} {'honest':>8} {'evasive':>8} {'lie':>8}   evasive on the honest (0) -> lie (1) scale")
    for n in names:
        c = s[n].copy()
        for L in ROLE_LEVELS:
            c[level == L] -= s[n][(level == L) & (label != "evasive")].mean()
        m = {lab: c[label == lab].mean() for lab in ("honest", "evasive", "lie")}
        out.append(f"  {n:18} {m['honest']:8.3f} {m['evasive']:8.3f} {m['lie']:8.3f}   "
                   f"{(m['evasive'] - m['honest']) / (m['lie'] - m['honest']):+.2f}")
    out.append(f"  (n: honest {(label == 'honest').sum()}, evasive {(label == 'evasive').sum()}, "
               f"lie {(label == 'lie').sum()})")

    plot(A, ci, rand, len(set(ids)))
    print("\n".join(out))
    REPORT_PATH.write_text("\n".join(out) + "\n")
    print(f"\nwrote {REPORT_PATH}, {DETECTORS_PATH}, {FIG_PATH}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("step", choices=["states", "report"])
    {"states": states, "report": report}[parser.parse_args().step]()


if __name__ == "__main__":
    main()
