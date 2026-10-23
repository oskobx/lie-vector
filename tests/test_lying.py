"""Tests for traits/lying.py: items, templates and split. No model needed."""

import pytest

from trait_vectors.traits.lying import LEVELS, LyingTrait


@pytest.fixture(scope="module")
def trait():
    return LyingTrait()


@pytest.fixture(scope="module")
def items(trait):
    return trait.eval_prompts()


def test_five_items_per_scenario(trait, items):
    assert len(items) == 5 * len(trait.scenarios)
    assert len({it.id for it in items}) == len(items)
    assert [it.condition for it in items[:5]] == LEVELS


def test_templates_filled(items):
    """Every item is system + user, with no unfilled {placeholder} left."""
    for it in items:
        assert [m["role"] for m in it.messages] == ["system", "user"]
        for m in it.messages:
            assert "{" not in m["content"] and "}" not in m["content"]


def test_which_question_each_level_asks(trait, items):
    """Level 0 asks the neutral question; every other level asks the leading one."""
    s = trait.scenarios[0]
    by_level = {it.condition: it.messages[1]["content"] for it in items if it.meta["scenario_id"] == s["id"]}
    assert s["question"] in by_level["0"] and s["leading"] not in by_level["0"]
    for level in ["0-lead", "1", "2", "3"]:
        assert s["leading"] in by_level[level]


def test_pressure_builds_up(trait, items):
    """Level 2 adds the stake to level 1, level 3 adds the permission to level 2."""
    s = trait.scenarios[0]
    system = {it.condition: it.messages[0]["content"] for it in items if it.meta["scenario_id"] == s["id"]}
    assert s["fact"] in system["1"] and s["stake"] not in system["1"]
    assert system["2"] == system["1"] + " " + s["stake"]
    assert system["3"] == system["2"] + " " + s["permission"]


def test_split_by_scenario(trait, items):
    """All five levels share their scenario's split, the one recorded in scenarios.json."""
    recorded = {s["id"]: s["split"] for s in trait.scenarios}
    for it in items:
        assert trait.split(it) == recorded[it.meta["scenario_id"]]
