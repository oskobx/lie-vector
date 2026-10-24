"""Tests for traits/lying.py: items, templates and split. No model needed."""

import pytest

from trait_vectors.traits.lying import LEVELS, LyingTrait


@pytest.fixture(scope="module")
def trait():
    return LyingTrait()


@pytest.fixture(scope="module")
def items(trait):
    return trait.eval_prompts()


def test_one_item_per_level_per_scenario(trait, items):
    assert len(items) == len(LEVELS) * len(trait.scenarios)
    assert len({it.id for it in items}) == len(items)
    assert [it.condition for it in items[:len(LEVELS)]] == LEVELS


def test_templates_filled(items):
    """Every item is system + user, with no unfilled {placeholder} left."""
    for it in items:
        assert [m["role"] for m in it.messages] == ["system", "user"]
        for m in it.messages:
            assert "{" not in m["content"] and "}" not in m["content"]


def test_which_question_each_level_asks(trait, items):
    """Plain levels ask the neutral question; "-lead" levels ask the leading one."""
    s = trait.scenarios[0]
    by_level = {it.condition: it.messages[1]["content"] for it in items if it.meta["scenario_id"] == s["id"]}
    for level in ["0", "1", "2", "3"]:
        assert s["question"] in by_level[level] and s["leading"] not in by_level[level]
    for level in ["0-lead", "1-lead", "2-lead", "3-lead"]:
        assert s["leading"] in by_level[level]


def test_lead_levels_share_system_prompts(trait, items):
    """k-lead differs from level k only in the question asked."""
    s = trait.scenarios[0]
    system = {it.condition: it.messages[0]["content"] for it in items if it.meta["scenario_id"] == s["id"]}
    for k in ["0", "1", "2", "3"]:
        assert system[f"{k}-lead"] == system[k]


def test_pressure_builds_up(trait, items):
    """Level 2 adds the stake to level 1, level 3 adds the permission to level 2."""
    s = trait.scenarios[0]
    system = {it.condition: it.messages[0]["content"] for it in items if it.meta["scenario_id"] == s["id"]}
    assert s["fact"] in system["1"] and s["stake"] not in system["1"]
    assert system["2"] == system["1"] + " " + s["stake"]
    assert system["3"] == system["2"] + " " + s["permission"]


def test_split_by_scenario(trait, items):
    """All levels share their scenario's split, the one recorded in scenarios.json."""
    recorded = {s["id"]: s["split"] for s in trait.scenarios}
    for it in items:
        assert trait.split(it) == recorded[it.meta["scenario_id"]]
