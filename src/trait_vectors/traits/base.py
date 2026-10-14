"""The Trait interface: what extract.py and the scripts need from a trait.

A trait is a behaviour we want to steer, such as sentiment. This module says
what any trait must provide; it knows nothing about any particular one.
Concrete traits live alongside it in this package, and nothing outside
`trait_vectors.traits` may refer to a specific trait by name.

`Protocol` is an interface defined by shape: a class is a Trait if it has
these methods with these signatures. It does not need to inherit from
Trait or register anywhere. Type checkers verify the match; at runtime it is
ordinary duck typing.

`@dataclass` (used for EvalItem and Score) writes the constructor for us: a
class with annotated fields gets an __init__ taking those fields, plus a
readable repr and field-by-field ==. Roughly a Java record.
"""

import hashlib
from dataclasses import dataclass, field
from typing import Protocol

from trait_vectors import config


@dataclass
class EvalItem:
    """One evaluation prompt.

    id         unique within the trait, stable across runs (the split hashes it)
    condition  the experimental condition; all aggregation groups by it
               (a pressure level for lying, "none" for sentiment)
    messages   the chat to generate a reply to: a list of
               {"role": "system" | "user", "content": str}
    meta       anything else the trait's score() needs, e.g. the true answer
    """

    id: str
    condition: str
    messages: list[dict]
    meta: dict = field(default_factory=dict)


@dataclass
class Score:
    """The result of scoring one response.

    A trait fills `value` (a number, e.g. P(positive) for sentiment) or
    `label` (a category, e.g. "honest" / "evasive" / "lie"); the other stays
    None. `meta` holds extras such as a judge's quote or a parse-error flag.
    """

    label: str | None = None
    value: float | None = None
    meta: dict = field(default_factory=dict)


def split_by_id(item_id: str) -> str:
    """Return "extract" or "eval", deterministically from a seeded hash of the id.

    sha256 of "<seed>:<id>" is uniform over its outputs, so the parity of the
    digest is a fair coin: about half the ids land in each split, and the same
    id always lands in the same one, on any machine. Changing config.SEED
    reshuffles everything.
    """
    digest = hashlib.sha256(f"{config.SEED}:{item_id}".encode()).digest()
    return "extract" if digest[-1] % 2 == 0 else "eval"


class Trait(Protocol):
    def extraction_pairs(self) -> tuple[list[str], list[str]]:
        """Return (P_plus, P_minus): two lists of user messages.

        P_plus exhibits the trait, P_minus exhibits its opposite. The
        steering vector at each layer is mean(h over P_plus) minus
        mean(h over P_minus), read at the last token of the templated
        prompt. These prompts are for extraction only and must be disjoint
        from anything used for evaluation.
        """
        ...

    def eval_prompts(self) -> list[EvalItem]:
        """Return the items to steer on when evaluating.

        Neutral with respect to the trait, and off-domain relative to the
        extraction data, so that a steering effect is evidence the vector
        encodes the trait rather than the extraction domain.
        """
        ...

    def score(self, item: EvalItem, response: str) -> Score:
        """Score how strongly `response` (to `item`) exhibits the trait.

        Higher `value` means more of the trait; for a labelled trait the
        label set is the trait's own. Results only need to be comparable
        across responses of the same trait.
        """
        ...

    def split(self, item: EvalItem) -> str:
        """Return "extract" or "eval" for `item`, via split_by_id.

        Nothing may extract a direction from an "eval" item.
        """
        ...
