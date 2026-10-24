"""Tests for traits/sycophancy.py: the CAA contrast pairs. No model needed."""

from trait_vectors.traits.sycophancy import SycophancyTrait


def test_caa_pairs():
    plus, minus = SycophancyTrait().extraction_pairs()
    assert len(plus) == len(minus) == 1000
    for p, m in zip(plus, minus):
        assert p[0] == m[0]                                        # same question
        assert {p[1]["content"], m[1]["content"]} == {"(A", "(B"}  # the two letters
