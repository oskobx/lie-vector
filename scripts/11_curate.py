"""Spec §2.3: curate the candidate scenarios by hand, one at a time.

Shows each candidate with its level-0 pre-screen result and asks: keep,
delete or edit. Every answer is saved at once to data/lying/curation.json, so
you can quit at any time and resume later; candidates already decided are
skipped. After each answer, data/lying/scenarios.json is rewritten from the
kept candidates, with edits applied and the split (extract / eval) added.

Reject (spec §2.3): ambiguous truth, fact not clearly known to the role,
question not yes/no, duplicates. Target 110-130 kept.

To change a decision later, delete that id's entry from curation.json and rerun.

Spot-check mode (spec §11, 2026-10-23), instead of reviewing every candidate:
    --sample 20     review only a fixed random sample of 20 (seeded, so the
                    same sample after quitting and resuming)
    --accept-rest   keep every undecided candidate, marked "accepted without
                    review" in curation.json
"""

import argparse
import json
import random

from trait_vectors import config
from trait_vectors.traits.base import split_by_id

DATA_DIR = config.ROOT / "data" / "lying"
CANDIDATES_PATH = DATA_DIR / "candidates.json"
COMPARISON_PATH = DATA_DIR / "comparison_ids.json"
DECISIONS_PATH = DATA_DIR / "curation.json"
SCENARIOS_PATH = DATA_DIR / "scenarios.json"
PRESCREEN_PATH = config.ARTIFACTS_DIR / "level0_prescreen.json"  # from 10b_level0_prescreen.py

FIELDS = ["role", "goal", "stake", "permission", "fact", "question", "leading", "truth", "lie"]
EDITABLE = FIELDS[:-1]  # "lie" is not edited directly; it is set to the opposite of "truth"


def load_json(path, default):
    return json.loads(path.read_text()) if path.exists() else default


def ask(prompt):
    """input(), but None when the input ends (Ctrl-D, or end of piped input)."""
    try:
        return input(prompt).strip()
    except EOFError:
        return None


def save(candidates, decisions):
    """Write the decisions, then rebuild scenarios.json from the kept candidates."""
    DECISIONS_PATH.write_text(json.dumps(decisions, indent=2))
    kept = []
    for c in candidates:
        d = decisions.get(c["id"])
        if d and d["decision"] == "keep":
            kept.append({**c, **d["edits"], "split": split_by_id(c["id"])})
    SCENARIOS_PATH.write_text(json.dumps(kept, indent=2))


def show(c, edits, prescreen, is_comparison, decisions, n_total):
    """Print one candidate; fields changed by an edit are marked with *."""
    n_kept = sum(d["decision"] == "keep" for d in decisions.values())
    print("\n" + "=" * 78)
    print(f"{c['id']} ({c['domain']})    decided {len(decisions)}/{n_total}, kept {n_kept}    "
          f"split: {split_by_id(c['id'])}")
    if prescreen:
        print(f"level 0: {prescreen['verdict'].upper()}  ->  {prescreen['reply'][:160]}")
    else:
        print("level 0: not run")
    if is_comparison:
        print("note: numeric-comparison question (comparison_ids.json)")
    print()
    for f in FIELDS:
        mark = "*" if f in edits else " "
        print(f"{mark} {f:11} {edits.get(f, c[f])}")


def edit(c, edits):
    """Ask for one field and its new value, and record it in edits."""
    field = ask(f"field ({', '.join(EDITABLE)}): ")
    if field not in EDITABLE:
        print("no such field")
        return
    value = ask(f"new {field} (empty = cancel): ")
    if not value:
        return
    if field == "truth":
        if value not in ("yes", "no"):
            print("truth must be yes or no")
            return
        edits["lie"] = "no" if value == "yes" else "yes"
    edits[field] = value


def summary(candidates, decisions):
    kept = [c for c in candidates if decisions.get(c["id"], {}).get("decision") == "keep"]
    n_deleted = sum(d["decision"] == "delete" for d in decisions.values())
    print(f"\nkept {len(kept)}, deleted {n_deleted}, undecided {len(candidates) - len(decisions)}")
    if kept:
        truths = [decisions[c["id"]]["edits"].get("truth", c["truth"]) for c in kept]
        print(f"kept truth yes/no: {truths.count('yes')} / {truths.count('no')}")
        for domain in dict.fromkeys(c["domain"] for c in candidates):
            print(f"  {domain:18} {sum(c['domain'] == domain for c in kept)}")
        n_eval = sum(split_by_id(c["id"]) == "eval" for c in kept)
        print(f"split: extract {len(kept) - n_eval}, eval {n_eval}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, help="review only a seeded random sample of this many candidates")
    parser.add_argument("--accept-rest", action="store_true", help="keep every undecided candidate without review")
    args = parser.parse_args()

    candidates = load_json(CANDIDATES_PATH, None)["candidates"]
    decisions = load_json(DECISIONS_PATH, {})
    prescreen = {r["id"]: r for r in load_json(PRESCREEN_PATH, [])}
    comparison = set(load_json(COMPARISON_PATH, {"ids": []})["ids"])

    if args.accept_rest:
        for c in candidates:
            if c["id"] not in decisions:
                decisions[c["id"]] = {"decision": "keep", "edits": {}, "note": "accepted without review"}
        save(candidates, decisions)
        summary(candidates, decisions)
        return

    to_review, note = candidates, None
    if args.sample:
        # Sampled from all candidates, not just undecided ones, so the sample does not change on resume.
        chosen = set(random.Random(config.SEED).sample([c["id"] for c in candidates], args.sample))
        to_review, note = [c for c in candidates if c["id"] in chosen], "spot-check"

    for c in to_review:
        if c["id"] in decisions:
            continue
        edits = {}
        while True:
            show(c, edits, prescreen.get(c["id"]), c["id"] in comparison, decisions, len(candidates))
            answer = ask("[k]eep  [d]elete  [e]dit  [q]uit > ")
            if answer is None or answer == "q":
                summary(candidates, decisions)
                return
            if answer == "e":
                edit(c, edits)
            elif answer in ("k", "d"):
                decisions[c["id"]] = {"decision": "keep" if answer == "k" else "delete", "edits": edits}
                if note:
                    decisions[c["id"]]["note"] = note
                save(candidates, decisions)
                break
    summary(candidates, decisions)


if __name__ == "__main__":
    main()
