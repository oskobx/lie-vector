"""Tests for judge/runner.py. The API call (_call) is replaced by a fake, so no key or credits are needed."""

import pytest

from trait_vectors.judge import runner

LABELS = ("honest", "evasive", "lie")
FIELDS = {"fact": "F", "question": "Q?", "truth": "yes", "asked": "A?", "response": "R"}


@pytest.fixture
def fake_api(tmp_path, monkeypatch):
    """Replace the API with a list of canned replies, and the cache with an empty temp file.

    Returns the list of message lists the fake was called with.
    """
    monkeypatch.setattr(runner, "CACHE_PATH", tmp_path / "cache.sqlite")
    calls, replies = [], []

    def fake_call(messages):
        calls.append(list(messages))
        return replies.pop(0)

    monkeypatch.setattr(runner, "_call", fake_call)
    return calls, replies


def test_parse():
    assert runner.parse('{"label": "lie", "quote": "It is fine."}', LABELS) == ("lie", "It is fine.")
    assert runner.parse('Sure!\n```json\n{"label": "honest", "quote": "x"}\n```', LABELS) == ("honest", "x")
    assert runner.parse("not json", LABELS) is None
    assert runner.parse('{"label": "maybe"}', LABELS) is None


def test_fill_prompt_leaves_json_braces():
    prompt = runner.fill_prompt("lying_v1", FIELDS)
    assert "FACT: F" in prompt and "REPLY: R" in prompt
    assert "{fact}" not in prompt and '{"label"' in prompt


def test_cached_after_first_call(fake_api):
    calls, replies = fake_api
    replies.append('{"label": "lie", "quote": "q"}')
    first = runner.judge("lying_v1", FIELDS, LABELS, "evasive")
    second = runner.judge("lying_v1", FIELDS, LABELS, "evasive")
    assert first.label == second.label == "lie"
    assert len(calls) == 1  # the second judgment came from the cache


def test_reask_then_success(fake_api):
    calls, replies = fake_api
    replies += ["I think it is a lie.", '{"label": "lie", "quote": "q"}']
    score = runner.judge("lying_v1", FIELDS, LABELS, "evasive")
    assert score.label == "lie" and not score.meta["parse_error"]
    assert len(calls) == 2 and calls[1][-1]["content"] == runner.REASK


def test_reask_then_fallback(fake_api):
    calls, replies = fake_api
    replies += ["garbage", "more garbage"]
    score = runner.judge("lying_v1", FIELDS, LABELS, "evasive")
    assert score.label == "evasive" and score.meta["parse_error"]
    assert len(calls) == 2  # exactly one re-ask, no more


def test_dry_run_calls_nothing(fake_api, capsys):
    calls, _ = fake_api
    assert runner.judge("lying_v1", FIELDS, LABELS, "evasive", dry_run=True) is None
    assert "FACT: F" in capsys.readouterr().out
    assert calls == []
