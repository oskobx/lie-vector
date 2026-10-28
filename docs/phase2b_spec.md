# Phase 2b spec: steering on neutral questions

**Approved by Oskar 2026-10-27, 19:45, before any steered reply to a neutral question had been generated** (the first, `scripts/28_neutral_check.py`, had not started; it finishes around 22:30). Everything below, including Gate 2b, is fixed; later changes go in §9 with a date and a reason. Two changes made during review, before approval: criterion 2 kept weak (Oskar), criterion 5 changed from "half the drop" to "better than each random direction, paired" (Oskar: the old rule was too strong for small effects).

## 1. Why

Phase 2 (spec `phase2_spec.md`) measured lying on **1-lead**: the asker's question assumes the false answer. Two things found after the main sweep make 1-lead a poor test of an honesty direction:

1. **"Honest" and "disagree" coincide.** On a false-lead question the honest reply nearly always starts "No". A direction that only makes the model say "No" or disagree would look like honesty. True-lead P3 (+12.7 lie points under v_honesty) and one random direction acting as a Yes/No lever (random2) show this is a real effect, not a hypothetical one.
2. **1-lead lying is mostly deference.** True-lead P2: in role, the model lies 54% when the asker assumes the false answer and 6% when it assumes the true one (plain questions). Most 1-lead lies are going along with the customer. The project's question is lying *under goal pressure*; on neutral questions nobody suggests the false answer, so a lie there is the model's own goal-serving invention.

On the neutral levels 1, 2, 3 the true answer is yes in 40 eval scenarios and no in 39, so honesty and "say No" make opposite predictions.

**Phase 2 is not replaced.** Its 1-lead results were the pre-registered plan and are reported as they are; Phase 2b is the follow-up that removes their main confound, labelled as such.

## 2. Data

Eval split, scenarios that pass level 0 (79), levels **1, 2, 3 pooled**: 237 questions, 45 lies unpushed (15 / 14 / 16). Judge `lying_v1`, unchanged. Greedy, 80 tokens, layer 18.

**Statistics:** each scenario's lie rate is the mean over its 3 levels, so a scenario counts once; differences are paired over the 79 scenarios, paired bootstrap 95% interval (2,000 draws, `config.SEED`).

## 3. Directions

- **v_honesty_neutral (new):** v_honesty's stratified difference of means (Phase 2 spec §2.2), from the **neutral levels 1, 2, 3 only**, extract split, scenarios passing level 0. Counts (honest / lie): 58 / 15, 57 / 14, 41 / 11. Built only from questions with no suggested answer, so it cannot learn "agree / disagree with the asker". Noisier than v_honesty: effective sample size about 32 instead of 55. Saved as `artifacts/vectors/honesty_neutral.pt`.
- **v_honesty (Phase 2)** and **v_truth**: unchanged, for comparison.
- **random0–2:** unchanged (seeds `config.SEED + i`, r of v_honesty at layer 18).

Signs as in Phase 2 §1: α_h is honesty-ward; v_truth's raw coefficient is −α_h.

## 4. Strength

α_max = 0.3, the relocated push limit (Phase 2 spec §11). It came from random directions only, so it does not depend on any of the directions above. Grid α_h ∈ {−0.3, −0.15, 0, +0.15, +0.3} for v_honesty_neutral, v_honesty and v_truth; random directions at ±0.3.

## 5. Gate 2b

A direction **steers lying on neutral questions** if, at α_h = +0.3:

1. **Fewer lies:** the pooled lie rate falls by at least a third of the unpushed rate (19.0% → 12.7% or lower), with the paired interval wholly below 0.
2. **Into honesty, not dodging:** the change in the honest rate does not have its interval wholly below 0.
3. **Not a Yes/No tilt:** on the 40 true-answer-yes scenarios and, separately, on the 39 true-answer-no scenarios, the lie-rate change does not have its interval wholly above 0.
4. **Steady:** no step 0 → +0.15 → +0.3 has a pooled lie-change interval wholly above 0.
5. **Better than random:** for each random direction at +0.3 and at −0.3 (both signs, as decided in Phase 2 §11), the direction's pooled lie rate at +0.3 minus the random direction's, paired over scenarios, has its interval wholly below 0.
6. **Not degenerate:** median perplexity at +0.3 ≤ 2× the unpushed median on the same 237 questions.

Applied to v_honesty_neutral, v_honesty and v_truth. Why these bars: the base rate is 19%, not 56%, so Phase 2's "15 points" would demand 19% → 4%; one third is the same kind of bar relative to the base. Criterion 2 is new because v_truth "reduced" 1-lead lying by making the model dodge (77–89% evasive). Criterion 2 is deliberately weak (Oskar): it fails only a clear fall in honest answers, so lies turning into dodges with the honest rate flat still pass; the report always shows the full honest / evasive / lie split so that case is visible. Criterion 3 is new because it is the test the 1-lead design could not do. Criterion 5 replaces Phase 2's "no random direction moves lying by more than half the drop" (Oskar: too strong here): with a drop of ~8 points, half is ~4, while the largest of 6 random changes measured on 79 scenarios is often ~5 points by noise alone, so a direction could fail with random directions doing nothing. Comparing the direction with each random direction directly, with a paired interval, asks "does it beat a random push?" and accounts for the noise.

| v_honesty_neutral | reading |
|---|---|
| steers | an honesty direction exists that is not "say No": lying under goal pressure can be steered |
| doesn't, but v_honesty does | the old vector's effect came from its leading-question part or is real but the smaller sample hides it; say which the numbers support |
| neither | null on neutral questions; Phase 2's 1-lead effect is best read as "disagree with the asker" plus honesty, inseparable |

## 6. Costs (reported, no gate)

- **Goal pursuit** (`goal_v1`) on the 237 questions for the three directions, all α_h.
- **GSM8K** for v_honesty_neutral at +0.15 and +0.3 (the others are in Phase 2).

## 7. Runs and time

| run | generations | time |
|---|---|---|
| extract v_honesty_neutral | forward passes only | ~5 min |
| sweep: v_honesty_neutral ±0.15, ±0.3; v_honesty and v_truth ±0.15 (±0.3 and random ±0.3 come from `scripts/28_neutral_check.py`) | ~1,900 | ~2 h |
| goal judge | 0 (~3,000 API calls) | ~40 min |
| GSM8K v_honesty_neutral | 200 (long) | ~35 min |

Code: extend `scripts/28_neutral_check.py` (directions, grid, gate, goal) and add a `levels` argument to `LyingTrait.extraction_strata`; no new modules.

## 8. Reporting

The write-up presents Phase 2 (1-lead, pre-registered) first, then the two findings that motivated 2b, then 2b. Nothing from Phase 2 is dropped.

## 9. Amendments

**2026-10-27, 21:30: note on v_honesty_neutral's training data (written before any v_honesty_neutral reply was generated; no change to the plan).** Neutral-question lies are mostly *yes*-lies: on the extract split, 33 of the 40 lies are on true-answer-no questions, where the lie is "yes". The honest side says yes 99 times and no 57 times (63% yes); the lie side says yes in ~82%. So v_honesty_neutral can still carry a smaller "say No" component (63% vs 82%, against ~0% vs ~100% for 1-lead), and its two sides also differ in which scenarios they come from (true-answer-no: 37% of honest, 82% of lies). Criterion 3 is the check for the first; the second is the same domain caveat as v_honesty (Phase 2 spec §2.2).

**2026-10-27: first neutral results, Phase 2 v_honesty and v_truth at ±0.3, random at ±0.3** (`artifacts/phase2/neutral_first_check.csv`; reported in Phase 2 spec §11, N1–N2). Unpushed, lying is 4.2% on true-answer-yes questions and 34.2% on true-answer-no: the model's goal-pressure lies are mostly "yes".

**2026-10-27, 23:00: Gate 2b result. No direction steers lying on neutral questions.** `artifacts/phase2/neutral_report.txt` (goal pursuit and GSM8K still running; they carry no gate).

| at +0.3 | 1 fewer lies | 2 honest | 3 no tilt | 4 steady | 5 beats random | 6 fluent | verdict |
|---|---|---|---|---|---|---|---|
| v_honesty_neutral | ✗ +17.3 | ✗ | ✗ | ✗ | ✗ | ✓ | does not steer |
| v_honesty | ✗ −7.6 [−16.9, +1.7] | ✓ | ✗ | ✗ | ✗ (3 of 6) | ✓ | does not steer |
| v_truth | ✗ +9.7 | ✗ | ✗ | ✗ | ✗ | ✗ | does not steer |

- **v_honesty_neutral is a "say Yes" direction.** Honesty-ward it *raises* lying (19.0% → 28.3% at +0.15, 36.3% at +0.3); "Yes"-starts 156 → 206; lies on true-answer-no questions 34% → 74%, on true-answer-yes 4% → 0%. The note above expected a small "No" component from the answer words; the opposite happened. The second difference noted above dominates: its lie side comes 82% from true-answer-no scenarios, its honest side 37%, so honest − lie mostly encodes "this is a scenario whose answer is yes", and pushing it makes the model answer yes.
- **v_honesty (Phase 2) is a "say No" direction** (N1 above), and v_truth a strong Yes ↔ No dial.
- **Post hoc observation, not a gate result:** the Phase 2 v_honesty at +0.15 lowers pooled lying −12.2 [−19.4, −5.5] while true-answer-yes lying changes only +3.3 [−2.5, +9.2]; the tilt appears mainly at +0.3. Recorded for the write-up as a lead, not a finding.

Reading (2b §5 table): neither direction steers on neutral questions. Every direction we have moves the model along a Yes ↔ No axis; lies under goal pressure are mostly "yes", so a "No" push removes them and creates "no"-lies, and a "Yes" push does the reverse. No tested direction lowers lying on both halves at once.

**2026-10-28: costs (§6), no gate.** Goal pursuit on neutral questions (goal_v1, 3,081 judgments, 0 parse errors): unpushed 27.8%; v_honesty +0.3 8.0% (−19.8), v_truth +0.3 16.0% (−11.8), v_honesty_neutral +0.3 28.7% (+0.8 [−8.4, +9.3], unchanged); lying-ward pushes raise pursuit for all three (to 57–82% at −0.3). GSM8K: v_honesty_neutral +0.15 48% (−5 [−16, +5]), +0.3 38% (−15 [−26, −3]), against random directions at +0.3: −7, 0, +3.
