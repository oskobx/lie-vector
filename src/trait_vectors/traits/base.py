"""The Trait interface: what extract.py and the scripts need from a trait.

A trait is a behaviour we want to steer, such as sentiment. This module says
what any trait must provide; it knows nothing about any particular one.
Concrete traits live alongside it in this package, and nothing outside
`trait_vectors.traits` may refer to a specific trait by name.

`Protocol` is an interface defined by shape: a class is a Trait if it has
these three methods with these signatures. It does not need to inherit from
Trait or register anywhere. Type checkers verify the match; at runtime it is
ordinary duck typing.
"""

from typing import Protocol


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

    def eval_prompts(self) -> list[str]:
        """Return the user messages to steer on when evaluating.

        Neutral with respect to the trait, and off-domain relative to the
        extraction data, so that a steering effect is evidence the vector
        encodes the trait rather than the extraction domain.
        """
        ...

    def score(self, prompt: str, response: str) -> float:
        """Return how strongly `response` (to `prompt`) exhibits the trait.

        Higher means more of the trait. The scale is the trait's own; the
        sweep only needs it to be comparable across responses.
        """
        ...
