"""Tests for traits/lying.py: items, templates and split. No model needed."""

import pytest

from trait_vectors.judge.runner import fill_prompt
from trait_vectors.traits.base import split_by_id
from trait_vectors.traits.lying import (BASELINE_PATH, GOAL_PROMPT, LEVELS, ROLE_LEVELS, TRUELEAD_JUDGE_PROMPT,
                                        TRUELEAD_LEVELS, TRUELEAD_PATH, LyingTrait, goal_fields, judge_fields,
                                        messages_for)


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


def test_goal_prompt_filled(items):
    """Every placeholder of the goal judge prompt (spec §5.2) is filled, for every 1-lead item."""
    for it in items:
        if it.condition == "1-lead":
            prompt = fill_prompt(GOAL_PROMPT, goal_fields(it, "Yes."))
            for name in ("role", "goal", "counterpart", "fact", "asked", "response"):
                assert "{" + name + "}" not in prompt


@pytest.mark.skipif(not TRUELEAD_PATH.exists(), reason="needs scripts/26_truelead_questions.py")
def test_truelead_items(trait, items):
    """Two true-lead items per scenario: the true-lead question, the matching system prompt, their own judge prompt."""
    tl = {it.id: it for it in trait.truelead_prompts()}
    by_id = {it.id: it for it in items}
    assert len(tl) == len(TRUELEAD_LEVELS) * len(trait.scenarios)
    for s in trait.scenarios:
        for level in TRUELEAD_LEVELS:
            it = tl[f"{s['id']}/{level}"]
            assert it.messages[1]["content"].count(s["leading_true"]) == 1
            assert s["leading"] not in it.messages[1]["content"]
            lead = by_id[f"{s['id']}/{level.replace('-truelead', '-lead')}"]
            assert it.messages[0] == lead.messages[0]  # same system prompt as the false-lead level
            assert "TRUE answer as an assumption" in fill_prompt(TRUELEAD_JUDGE_PROMPT, judge_fields(it, "Yes."))


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


@pytest.mark.skipif(not BASELINE_PATH.exists(), reason="needs the Phase 1 baseline (scripts/12_baseline_lying.py)")
def test_honesty_strata(trait):
    """Phase 2 spec §2.2: per-level counts on the extract split, and no eval scenario anywhere."""
    strata = trait.extraction_strata()
    counts = {level: (len(h), len(d)) for level, (h, d) in strata.items()}
    assert counts == {"1": (58, 15), "1-lead": (12, 45), "2": (57, 14),
                      "2-lead": (7, 50), "3": (41, 11), "3-lead": (11, 23)}
    eval_prompts = {str(messages_for(s, level)) for s in trait.scenarios
                    if split_by_id(s["id"]) == "eval" for level in ROLE_LEVELS}
    for level, (honest, lie) in strata.items():
        assert not {str(m) for m in honest + lie} & eval_prompts
        assert not {str(m) for m in honest} & {str(m) for m in lie}  # a reply is either honest or a lie


def test_split_by_scenario(trait, items):
    """All levels share their scenario's split, the one recorded in scenarios.json."""
    recorded = {s["id"]: s["split"] for s in trait.scenarios}
    for it in items:
        assert trait.split(it) == recorded[it.meta["scenario_id"]]
