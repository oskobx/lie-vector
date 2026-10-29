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

**2026-10-28, before any detection number (agreed with Oskar).** Four points the spec left open, found when the states were counted (no detector had been fitted or scored):

1. *Cross-validation scoring (§3).* Within one held-out fold (16 scenarios) almost no level has 5 replies on each side, so a per-fold stratified AUROC is undefined under §4's rule. Instead each extract prompt is scored by the model fitted without its fold, and one stratified AUROC is computed over all extract prompts (out-of-fold predictions). Folds: extract scenario ids shuffled with `config.SEED`, dealt into 5. A tie in λ goes to the larger λ. Level centring uses the training folds' level means.
2. *λ of the shortcut detector (§3).* Chosen like the probe's: same grid, same 5-fold cross-validation. d_fact (from level-0 prompts, which are never scored) is computed once on the extract split.
3. *"Starts with No" (§6).* No level has 5 lies (eval: 11 lies over six levels), so this subset gets one AUROC pooled over levels, labelled "pooled, not stratified; too few to judge". The "starts with Yes" subset is stratified as usual; only levels 1, 2, 3 have ≥ 5 honest replies there, so it covers the neutral levels only.
4. *"Without negated leading questions" (§6).* Drops the -lead prompts of flagged scenarios; their neutral-level prompts never contain the leading question and stay.

Also fixed here: §6's evasive means are computed after subtracting each level's mean score (honest and lie prompts), since evasive replies concentrate at levels 3 and 3-lead; reported with their position on the honest (0) → lie (1) scale.

**2026-10-28, AFTER the eval split was scored (agreed with Oskar): the shortcut detector is fitted per level.** The §3 shortcut detector (one logistic regression on the two features, all levels pooled, "no level centring needed") does not work as a control. It scored 0.335 on eval, below guessing and below the word-reader alone (0.668), so criterion 2 was passed by every detector. The defect is visible on the extract split alone: 0.562 in-sample and 0.51 cross-validated, against 0.623 for the word-reader alone. Cause, also on extract: the fact-reader's sign flips between level types. On neutral levels "true answer no" goes with lying (AUROC 0.51–0.57). On -lead levels nearly every honest reply has true answer no (1-lead: 10 of 12), so "true answer yes" predicts lying (AUROC 0.10–0.14, i.e. 0.86–0.90 reversed). One pooled weight per feature cannot follow this; the pooled fit mostly learns the between-level lie rates (0.2 vs 0.8), which the stratified AUROC ignores.

- *Change:* the shortcut detector is a separate two-feature logistic regression within each level, extract split only, λ by the same 5-fold cross-validation (out-of-fold scores, §9.1). It is "the best both shortcuts can do together" for a stratified AUROC.
- *Status:* the as-specified Gate 3 stays on record, marked "criterion 2 vacuous: control broken". The per-level version is labelled post hoc, decided after eval was scored, and the Phase 3 reading rests on it.
- *What had been seen first* (said so in advance; the fix could cut either way): at -lead levels the word-reader alone was at v_honesty's level (0.94 / 0.93 / 0.86 vs 0.90 / 0.90 / 0.76). Among "Yes"-starting replies the fact-reader alone (0.909) was above every detector.
- *Also noted:* one scenario per split has only evasive replies at all six role levels, so it contributes no prompts; the bootstrap resamples 78 eval scenarios, not 79.

**2026-10-28, Gate 3 decision (Oskar): passes for v_honesty and the probe, with the shortcut caveat.**

Eval split, layer 18; 78 scenarios, 223 honest, 161 lies; stratified AUROC with paired bootstrap 95% intervals. Random directions, max(A, 1 − A): median 0.573, 95th percentile 0.679, max 0.765.

| detector | AUROC | minus per-level shortcut (paired) | 1 | 2 | 3 | |
|---|---|---|---|---|---|---|
| v_honesty | 0.814 [0.734, 0.882] | +0.138 [+0.064, +0.206] | ✓ | ✓ | ✓ | **passes** |
| v_honesty_neutral | 0.697 [0.602, 0.789] | +0.021 [−0.088, +0.126] | ✗ | ✗ | ✓ | fails |
| v_truth | 0.648 [0.561, 0.737] | −0.028 [−0.098, +0.047] | ✗ | ✗ | ✗ | fails |
| probe (λ = 0.1 by CV) | 0.778 [0.682, 0.864] | +0.102 [+0.020, +0.189] | ✓ | ✓ | ✓ | **passes** |
| shortcut detector, per level | 0.676 [0.605, 0.742] | | | | | control |
| shortcut detector as specified | 0.335 [0.257, 0.410] | | | | | broken |

As specified (pooled shortcut, criterion 2 vacuous) the same two pass.

**Reading (§5 table):** a Phase 2 direction passes. v_honesty **reads** lying before the reply though it does not **steer** it (Phase 2): reading and writing come apart. v_truth fails, so there is no evidence for picture A. The probe passes but does not beat v_honesty.

**The shortcut caveat:**
1. *The control is under strength.* The per-level shortcut detector is fitted on about 15 lies per neutral level. On extract it barely beats the word-reader alone (cross-validated 0.628 vs 0.623). On eval it falls below the fact-reader alone at the neutral levels (0.49 / 0.41 / 0.58 vs 0.64). A stronger two-shortcut detector would narrow v_honesty's margin.
2. *Where v_honesty's margin comes from.* At every -lead level the word-reader alone is ahead of v_honesty (0.94 / 0.93 / 0.86 vs 0.90 / 0.90 / 0.76), so the margin comes from the neutral levels. Even there, among replies starting "Yes", the fact-reader alone (0.909) is above v_honesty (0.782).

So the defensible claim is narrower: before the reply, v_honesty carries lie information beyond the planned word on neutral questions. Whether that is more than "the true answer is no" is not settled here.

**Also reported (§6):**
- Without the -lead prompts of negated questions, v_honesty is 0.815 (no change).
- Evasive replies sit at +0.09 on the honest → lie scale for v_honesty and +0.93 for v_honesty_neutral.
- v_truth is reversed among "Yes" replies (0.25), consistent with a Yes/agree dial.

**Proposed, not run:**
- a per-level "AND" shortcut (word, fact, word × fact), or a fresh pre-registered test, for the open question above;
- in Phase 4, the geometry of v_honesty against d_fact and the word-reader direction.

Outputs: `artifacts/phase3/report.txt`, `detectors.pt`; `docs/figures/phase3_detection.png`.
