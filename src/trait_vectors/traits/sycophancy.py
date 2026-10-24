"""Sycophancy: contrast pairs for v_syc. The only file that knows about CAA.

Data is the CAA sycophancy set (data/sycophancy/, see its README): A/B
questions from users who describe themselves, where one answer agrees with
the persona's likely view. These are opinion items, a different flavour from
the lying scenarios (agreeing with a false belief). Extraction is Phase 2;
the direction is used for the Phase 4 geometry (spec §5).

Sycophancy has no eval set in this project (spec §11, 2026-10-23): it is
measured only as the lying scenarios' level 0-lead rate. So eval_prompts()
is empty and score() is not implemented.
"""

import json

from trait_vectors import config
from trait_vectors.traits.base import split_by_id

CAA_PATH = config.ROOT / "data" / "sycophancy" / "caa_sycophancy_generate.json"


class SycophancyTrait:
    """Satisfies the Trait protocol (see traits/base.py)."""

    def extraction_pairs(self):
        """(P_plus, P_minus) from CAA: each a chat whose assistant turn starts with the chosen letter.

        P_plus starts the reply with the persona-matching (sycophantic) letter,
        P_minus with the other one, e.g. [{"role": "user", ...}, {"role": "assistant", "content": "(A"}].
        Note: these are chats, not plain user strings as for sentiment; model.build_inputs
        leaves the partial assistant turn open, so the last token read is the letter
        (Phase 2 spec §2.3).
        """
        data = json.loads(CAA_PATH.read_text())
        plus, minus = [], []
        for d in data:
            user = {"role": "user", "content": d["question"].strip()}
            plus.append([user, {"role": "assistant", "content": d["answer_matching_behavior"].strip().rstrip(")")}])
            minus.append([user, {"role": "assistant", "content": d["answer_not_matching_behavior"].strip().rstrip(")")}])
        return plus, minus

    def eval_prompts(self):
        """No eval set: sycophancy is measured through the lying scenarios' level 0-lead."""
        return []

    def score(self, item, response):
        raise NotImplementedError("sycophancy has no eval set (spec §11, 2026-10-23)")

    def split(self, item):
        return split_by_id(item.id)
