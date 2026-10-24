"""Tests for judge/runner.py. The API call (_call) is replaced by a fake, so no key or credits are needed."""

from types import SimpleNamespace

import pytest

from trait_vectors.judge import runner

LABELS = ("honest", "evasive", "lie")
FIELDS = {"fact": "F", "question": "Q?", "truth": "yes", "asked": "A?", "response": "R"}
FAKE_SNAPSHOT = "claude-haiku-4-5-fake"


@pytest.fixture
def fake_api(tmp_path, monkeypatch):
    """Replace the API with a list of canned replies, and the cache with an empty temp file.

    Returns the list of message lists the fake was called with.
    """
    monkeypatch.setattr(runner, "CACHE_PATH", tmp_path / "cache.sqlite")
    calls, replies = [], []

    def fake_call(messages):
        calls.append(list(messages))
        return replies.pop(0), FAKE_SNAPSHOT

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
    assert first.meta["reported_model"] == second.meta["reported_model"] == FAKE_SNAPSHOT


def test_use_cache_false_always_asks(fake_api):
    calls, replies = fake_api
    replies += ['{"label": "lie", "quote": "q"}', '{"label": "honest", "quote": "q"}']
    runner.judge("lying_v1", FIELDS, LABELS, "evasive")  # cached as "lie"
    fresh = runner.judge("lying_v1", FIELDS, LABELS, "evasive", use_cache=False)
    assert fresh.label == "honest" and len(calls) == 2
    assert runner.judge("lying_v1", FIELDS, LABELS, "evasive").label == "lie"  # cache untouched


def test_old_cache_gets_reported_model_column(fake_api):
    # A cache file in Phase 1's format (no reported_model column) keeps its rows and gains the column.
    db = runner.sqlite3.connect(runner.CACHE_PATH)
    db.execute("""CREATE TABLE judgments (judge_model TEXT, prompt_version TEXT, input_sha256 TEXT,
                  label TEXT, quote TEXT, parse_error INTEGER, raw TEXT,
                  PRIMARY KEY (judge_model, prompt_version, input_sha256))""")
    sha = runner.hashlib.sha256(runner.fill_prompt("lying_v1", FIELDS).encode()).hexdigest()
    db.execute("INSERT INTO judgments VALUES (?, ?, ?, ?, ?, ?, ?)",
               (runner.config.JUDGE_MODEL, "lying_v1", sha, "lie", "q", 0, "{}"))
    db.commit()
    db.close()
    calls, _ = fake_api
    score = runner.judge("lying_v1", FIELDS, LABELS, "evasive")
    assert score.label == "lie" and score.meta["reported_model"] is None and calls == []


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


def test_wrong_snapshot_raises(monkeypatch):
    # A stand-in for the SDK client whose messages.create answers from a different snapshot.
    def create(**kwargs):
        return SimpleNamespace(model="claude-haiku-4-5-29990101", stop_reason="end_turn",
                               content=[SimpleNamespace(type="text", text='{"label": "lie"}')])

    monkeypatch.setattr(runner, "_client", SimpleNamespace(messages=SimpleNamespace(create=create)))
    with pytest.raises(RuntimeError, match="snapshot changed"):
        runner._call([{"role": "user", "content": "x"}])


def test_dry_run_calls_nothing(fake_api, capsys):
    calls, _ = fake_api
    assert runner.judge("lying_v1", FIELDS, LABELS, "evasive", dry_run=True) is None
    assert "FACT: F" in capsys.readouterr().out
    assert calls == []
