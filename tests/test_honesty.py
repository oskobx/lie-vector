"""Tests for traits/honesty.py: sizes, balance, disjointness. No model needed."""

from collections import Counter

from trait_vectors.traits.honesty import FILES, HonestyTrait


def test_extraction_pairs_balanced():
    plus, minus = HonestyTrait().extraction_pairs()
    assert len(plus) == len(minus) == 150
    assert len(set(plus)) == 150 and len(set(minus)) == 150


def test_eval_disjoint_and_balanced():
    trait = HonestyTrait()
    plus, minus = trait.extraction_pairs()
    items = trait.eval_prompts()
    assert len(items) == 100
    assert Counter(it.condition for it in items) == {"true": 50, "false": 50}
    statements = {it.messages[0]["content"] for it in items}
    assert not statements & (set(plus) | set(minus))
    assert {it.meta["file"] for it in items} <= set(FILES)


def test_deterministic():
    a, b = HonestyTrait(), HonestyTrait()
    assert a.extraction_pairs() == b.extraction_pairs()
    assert [it.id for it in a.eval_prompts()] == [it.id for it in b.eval_prompts()]
