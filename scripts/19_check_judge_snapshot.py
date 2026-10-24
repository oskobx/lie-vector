"""Phase 2 spec §0.3: is the judge still the model that produced Phase 1's labels?

JUDGE_MODEL is an alias. Phase 1's cache did not record which dated snapshot
answered, so the check is indirect: re-judge 5 cached 1-lead baseline items with
the cache bypassed, compare the labels with the cached ones, and print the model
id the API reports. Exits with an error if any label differs or the reported ids
disagree with each other. 5 API calls (fewer than 1 cent).

    uv run python scripts/19_check_judge_snapshot.py
"""

import csv
import random
import sys

from trait_vectors import config
from trait_vectors.judge.runner import judge
from trait_vectors.traits.lying import JUDGE_FALLBACK, JUDGE_LABELS, JUDGE_PROMPT, LyingTrait, judge_fields

N_ITEMS = 5
CSV_PATH = config.ARTIFACTS_DIR / "lying_baseline.csv"


def main():
    with open(CSV_PATH) as f:
        rows = [r for r in csv.DictReader(f) if r["level"] == "1-lead" and r["parse_error"] == "False"]
    sample = random.Random(config.SEED).sample(rows, N_ITEMS)
    items = {item.id: item for item in LyingTrait().eval_prompts()}

    mismatches, models = 0, set()
    for r in sample:
        item = items[f"{r['id']}/{r['level']}"]
        fresh = judge(JUDGE_PROMPT, judge_fields(item, r["response"]), JUDGE_LABELS, JUDGE_FALLBACK, use_cache=False)
        same = fresh.label == r["label"]
        mismatches += not same
        models.add(fresh.meta["reported_model"])
        print(f"{item.id:22s} cached={r['label']:8s} fresh={fresh.label:8s} "
              f"{'same' if same else 'DIFFERENT'}  reported_model={fresh.meta['reported_model']}")

    print(f"\nalias {config.JUDGE_MODEL} -> reported {sorted(models)}; {mismatches}/{N_ITEMS} labels differ")
    if mismatches or len(models) != 1:
        sys.exit("STOP: the judge may have changed since Phase 1 (spec §0.3)")


if __name__ == "__main__":
    main()
