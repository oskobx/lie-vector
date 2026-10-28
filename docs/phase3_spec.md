# Phase 3 spec: detection (short version; timebox 2 sessions, Gate 3 at the end)

**Approved by Oskar 2026-10-28, before any detection number was computed** (no Phase 3 code or run existed). Everything below, including Gate 3, is fixed; later changes go in §9 with a date and a reason. One change during drafting, before approval: the fact-reader, first cut, was put back inside a single "shortcut detector" (§3), because among "Yes"-replies a fact-reader alone separates neutral-question lies.

Question: **from the model's state at the last prompt token, before it writes anything, can we tell whether it is about to lie?** A detector turns that state into a number; it works if lies get higher numbers than honest replies on held-out scenarios, and if it knows more than two shortcuts that Phase 2 showed are everywhere in this data.

Read first: `docs/roadmap.md` (decisions list) and the Phase 2 handoff (§6 open items). Inherited and not to be changed: layer 18, the extract/eval split, judge labels from `lying_v1` (cached, no new judging), level names, the paired scenario bootstrap.

## 1. Why the shortcuts matter

On the unpushed replies (eval split, six role levels), 91% of lies start with "Yes" against 55% of honest replies. Two shortcuts can therefore score well without detecting lying:

- **word-reader:** "the model is about to say Yes". Reading the first word alone gives AUROC 0.68 on the eval split.
- **fact-reader:** "the fact in this scenario is no". On neutral questions the model's lies are mostly "yes" on scenarios whose true answer is no, so a direction that only encodes the true answer also separates lies from honest replies.

Phase 2 found every steering direction is largely one of these (Yes ↔ No tilts; v_honesty_neutral learned "the answer is yes"). Gate 3 asks for more than both shortcuts combined.

**AUROC**: the probability that a randomly chosen lie gets a higher score than a randomly chosen honest reply; 0.5 is guessing, 1 is perfect.

## 2. Data

- Prompts: the six role levels 1, 1-lead, 2, 2-lead, 3, 3-lead (as for v_honesty), scenarios that pass level 0, both splits. Labels: the cached unpushed judgments in `artifacts/lying_baseline.csv`, **lie vs honest; evasive replies are dropped** (as in extraction). Eval split: 223 honest, 161 lies, 79 scenarios; extract split: 186 honest, 158 lies.
- One forward pass per prompt (~950 prompts, plus level 0 for §3's fact direction, ~160), no generation. Saved per prompt: the layer-18 state at the last prompt token, and the next-token logits of the two answer words. `artifacts/phase3/states.pt`.
- Everything that is fitted (signs, the probe, the shortcut detector, the fact direction) uses the **extract** split only; every reported number is on the **eval** split.

## 3. Detectors

Each gives one score per prompt; higher = "more likely a lie".

| detector | score | source |
|---|---|---|
| v_honesty | projection onto v̂_honesty at layer 18 | Phase 2 |
| v_honesty_neutral | projection onto its unit vector | Phase 2b |
| v_truth | projection onto v̂_truth | Phase 2 |
| **probe** | w · h, logistic regression (below) | new |

**Signs.** A direction has no built-in "lie" end for detection (v_truth's "true" side raised lying in Phase 2). Each detector's sign is chosen on the extract split (the orientation with AUROC ≥ 0.5 there) and frozen before the eval split is scored.

**Probe.** L2-regularised logistic regression on the 1,536-dimensional states, lie = 1, honest = 0, extract split. Before fitting, each level's mean state is subtracted, so the probe cannot use "which level is this" (the same reason v_honesty is stratified). The regularisation strength λ is chosen from {0.001, 0.01, 0.1, 1, 10} by 5-fold cross-validation over extract **scenarios** (all levels of a scenario in the same fold), by the stratified AUROC of §4. Written in PyTorch, no new dependency (`extract.py`, function `logistic_regression(X, y, lam)`: full-batch gradient descent with L-BFGS, a few lines; a test on synthetic data).

**Comparison detectors** (the controls):

- **shortcut detector:** logistic regression (same function, no level centring needed for two features) on two features, fitted on extract: the word-reader, logit("Yes") − logit("No") at the last prompt position; and the fact-reader, the projection onto d_fact = mean(h | true answer no) − mean(h | true answer yes) from **level 0** extract prompts (all honest by the level-0 filter, so honesty does not vary). This is the best that both shortcuts can do together.
- **random directions:** 100 unit vectors, seeds `config.SEED + i`, i = 0 … 99 (random0–2 from Phase 2 are among them). A random direction's sign means nothing, so each counts with max(A, 1 − A).

The answer words ("Yes", "No") are trait-specific and live in `traits/lying.py`; each must be a single token (asserted).

## 4. Statistics

- **Stratified AUROC.** AUROC within each of the six levels (lies vs honest replies at that level only), averaged with weights w_ℓ ∝ n_H n_L / (n_H + n_L), as for v_honesty. Comparing only within a level removes "the question was leading" as a shortcut, and makes any per-level offset in a score irrelevant. Levels with fewer than 5 replies on either side are skipped (none on the eval split).
- **Intervals.** Paired bootstrap over the 79 eval scenarios (2,000 draws, `config.SEED`): each draw resamples scenarios with all their levels and recomputes every detector's stratified AUROC, so differences between detectors are paired.

## 5. Gate 3

A detector **reads lying** if, on the eval split at layer 18:

1. **Better than guessing:** stratified AUROC ≥ 0.70, with its 95% interval above 0.5.
2. **More than the shortcuts:** its AUROC minus the shortcut detector's, paired, has its interval wholly above 0.
3. **More than a random direction:** its AUROC is above the 95th percentile of the 100 random directions' max(A, 1 − A).

Applied to v_honesty, v_honesty_neutral, v_truth and the probe.

| result | reading |
|---|---|
| a Phase 2 direction passes | it **reads** lying though it doesn't **steer** it: reading and writing come apart (roadmap twist 4) |
| only the probe passes | lying is linearly readable before the reply, but not along a difference-of-means direction; Phase 4 includes the probe's w |
| v_truth passes | evidence for picture A (lying shows up on the truth axis), to be weighed in Phase 4 |
| none passes | before the reply, the state carries no linear lie signal beyond the planned word and the fact; a second null, reported as such |

## 6. Reported, no gate

- AUROC per level for every detector, and for the two shortcut features separately.
- Among replies that start with "Yes" only, and among those that start with "No" only (11 lies on eval: shown, too few to judge).
- With and without negated leading questions (keyword flag on the leading question: n't, not, never, no; `traits/lying.py`), since some lies there are yes/no slips.
- Mean score of evasive replies against honest and lying ones, per detector.
- One figure: stratified AUROC per detector with intervals, the shortcut detector, and the 5–95% band of the random directions. `docs/figures/phase3_detection.png`.

## 7. Runs and code

| step | work | time |
|---|---|---|
| states | one forward pass per prompt, layer 18 + answer-word logits | ~10 min |
| fit + score + report | probe, shortcut detector, 100 random, bootstrap; CPU | minutes |

Code: `scripts/30_detection.py` (steps `states`, `report`); `extract.logistic_regression`; `metrics.stratified_auroc`; `traits/lying.py`: answer words, `is_negated`, and a `split` argument to `extraction_strata`. Tests for the logistic regression (synthetic data with a known direction), the stratified AUROC (a hand-computed case), and the answer words being single tokens.

Session 1: code, tests, states. Session 2: report, Gate 3, close.

## 8. Cut from the roadmap's Phase 3 (Oskar, 2026-10-28: shortest reasonable version)

| cut | why |
|---|---|
| steering with the new detectors ("steers best?") | Phase 2 found nothing steers; low expected value for ~1.5 h of runs and judging |
| top PCA component as a third method | least likely to work; the probe covers "a better way to find the direction" |
| AUROC over all layers | layer 18 is fixed; a curve invites picking the best layer afterwards |
| hand check of negated questions | replaced by the keyword flag, reported with and without |

## 9. Amendments

*(dated entries)*
