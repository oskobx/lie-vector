"""Tests for traits/sycophancy.py: answer parsing, the pushback rule, item sets, CAA pairs. No model needed."""

import pytest

from trait_vectors.traits.sycophancy import N_EVAL, SycophancyTrait, parse_answer, wrong_answer


def test_parse_answer():
    assert parse_answer("so 16 - 3 - 4 = 9 eggs, 9 * 2 = 18.\nFinal answer: 18") == 18
    assert parse_answer("Final answer: $1,250.") == 1250
    assert parse_answer("**Final answer:** 42") == 42
    assert parse_answer("Final answer: 3. Wait, Final answer: 5") == 5   # the last one counts
    assert parse_answer("The total is 7 apples and 12 pears, so 19.") == 19  # fallback: last number
    assert parse_answer("No numbers here.") is None


def test_wrong_answer():
    assert wrong_answer(18) == 27      # 18 × 1.5
    assert wrong_answer(15) == 23      # 22.5 rounds up
    assert wrong_answer(4) == 11       # below 10: + 7
    assert all(wrong_answer(g) != g for g in range(0, 500))


@pytest.fixture(scope="module")
def trait():
    return SycophancyTrait()


def test_eval_and_audit_sets_disjoint(trait):
    eval_ids = [it.id for it in trait.eval_prompts()]
    audit_ids = [it.id for it in trait.audit_prompts()]
    assert len(eval_ids) == len(audit_ids) == N_EVAL
    assert not set(eval_ids) & set(audit_ids)
    assert eval_ids == [it.id for it in trait.eval_prompts()]  # same order every time


def test_turn2_and_score(trait):
    item = trait.eval_prompts()[0]
    turn2 = trait.turn2_item(item, "... Final answer: 5")
    assert [m["role"] for m in turn2.messages] == ["user", "assistant", "user"]
    assert str(item.meta["wrong"]) in turn2.messages[-1]["content"]
    assert trait.score(item, f"Final answer: {item.meta['gold']}").label == "correct"
    assert trait.score(item, f"Final answer: {item.meta['wrong']}").label == "incorrect"


def test_caa_pairs(trait):
    plus, minus = trait.extraction_pairs()
    assert len(plus) == len(minus) == 1000
    for p, m in zip(plus[:20], minus[:20]):
        assert p[0] == m[0]                                  # same question
        assert {p[1]["content"], m[1]["content"]} == {"(A", "(B"}  # the two letters
