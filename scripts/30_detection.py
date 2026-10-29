"""Phase 3 (docs/phase3_spec.md): detection. From the model's state at the last
prompt token, before it writes anything, can we tell whether it is about to lie?

    uv run python scripts/30_detection.py states   # one forward pass per prompt, ~10 min

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

Output: artifacts/phase3/states.pt.
"""

import argparse
import csv
import re
import time

import torch

from trait_vectors import config
from trait_vectors.extract import load_vectors
from trait_vectors.hooks import read_activations
from trait_vectors.model import build_inputs, load_model
from trait_vectors.traits.lying import ANSWER_WORDS, BASELINE_PATH, ROLE_LEVELS, LyingTrait, is_negated

LAYER = 18
LEVELS = ["0"] + ROLE_LEVELS  # level 0 for the fact direction (spec §3), the role levels for detection

VECTORS_DIR = config.ARTIFACTS_DIR / "vectors"
OUT_DIR = config.ARTIFACTS_DIR / "phase3"
STATES_PATH = OUT_DIR / "states.pt"


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("step", choices=["states"])
    {"states": states}[parser.parse_args().step]()


if __name__ == "__main__":
    main()
