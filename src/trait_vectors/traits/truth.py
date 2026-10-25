"""Truth: true vs false statements for v_truth. The only file that knows Geometry of Truth.

Data is six CSVs from The Geometry of Truth (data/truth/, see its README):
plain factual statements labelled true (1) or false (0), about cities,
Spanish words and number comparisons, each with a negated version. Each
statement is a user message with no framing: no role and no goal anywhere,
which is the point of the research question (Phase 1 spec §6).

    extraction_pairs()  150 true + 150 false, 25 of each per file
    eval_prompts()      a disjoint 100 (50 true + 50 false), for v_truth's held-out AUROC (Phase 2 spec §2.1)

Both are drawn with config.SEED, so they are the same on every run.
"""

import csv
import random

from trait_vectors import config
from trait_vectors.traits.base import EvalItem, split_by_id

DATA_DIR = config.ROOT / "data" / "truth"
FILES = ["cities", "neg_cities", "sp_en_trans", "neg_sp_en_trans", "larger_than", "smaller_than"]
EXTRACT_PER_FILE = 25  # per label: 6 files x 25 = 150 true and 150 false
N_EVAL_PER_LABEL = 50


def _load():
    """All statements as dicts {id, file, statement, label}, label 1 = true, 0 = false."""
    rows = []
    for name in FILES:
        with open(DATA_DIR / f"{name}.csv") as f:
            for i, r in enumerate(csv.DictReader(f)):
                rows.append({"id": f"{name}-{i:04d}", "file": name,
                             "statement": r["statement"], "label": int(r["label"])})
    return rows


def _partition():
    """(extraction rows, eval rows), disjoint and fixed by config.SEED."""
    rng = random.Random(config.SEED)
    rows = _load()
    extract, rest = [], []
    for name in FILES:
        for label in (1, 0):
            group = [r for r in rows if r["file"] == name and r["label"] == label]
            rng.shuffle(group)
            extract += group[:EXTRACT_PER_FILE]
            rest += group[EXTRACT_PER_FILE:]
    rng.shuffle(rest)
    evaluation = ([r for r in rest if r["label"] == 1][:N_EVAL_PER_LABEL]
                  + [r for r in rest if r["label"] == 0][:N_EVAL_PER_LABEL])
    return extract, evaluation


class TruthTrait:
    """Satisfies the Trait protocol (see traits/base.py)."""

    def extraction_pairs(self):
        """(P_plus, P_minus): 150 true and 150 false statements as plain user messages."""
        extract, _ = _partition()
        return ([r["statement"] for r in extract if r["label"] == 1],
                [r["statement"] for r in extract if r["label"] == 0])

    def eval_prompts(self):
        """100 statements not used for extraction; condition is "true" or "false"."""
        _, evaluation = _partition()
        return [EvalItem(id=r["id"], condition="true" if r["label"] else "false",
                         messages=[{"role": "user", "content": r["statement"]}],
                         meta={"file": r["file"], "label": r["label"]})
                for r in evaluation]

    def score(self, item, response):
        raise NotImplementedError("statements are not judged; v_truth is checked by AUROC (Phase 2 spec §2.1)")

    def split(self, item):
        return split_by_id(item.id)
