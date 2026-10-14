"""Tests for the Phase 1 section 1 pipeline changes: EvalItem, Score, split, cache.

`monkeypatch` and `tmp_path` are built-in pytest fixtures: tmp_path is a fresh
empty directory per test, and monkeypatch.setattr changes an attribute for
the duration of one test and restores it afterwards.
"""

import pytest
import torch

from trait_vectors import config
from trait_vectors.hooks import steer
from trait_vectors.metrics import summarize
from trait_vectors.model import generate, load_model, steering_spec
from trait_vectors.traits.base import Score, split_by_id
from trait_vectors.traits.sentiment import EVAL_PROMPTS, SentimentTrait


# --- no model needed ---------------------------------------------------------

def test_split_deterministic_and_balanced():
    ids = [f"item-{i:04d}" for i in range(1000)]
    first = [split_by_id(i) for i in ids]
    assert first == [split_by_id(i) for i in ids]
    assert set(first) == {"extract", "eval"}
    # Binomial(1000, 1/2) has sd ~16, so 450..550 is a > 3 sd band.
    assert 450 <= first.count("eval") <= 550


def test_sentiment_eval_items():
    """Same 30 prompts as Phase 0, as user-only chats, condition "none"."""
    trait = SentimentTrait()
    items = trait.eval_prompts()
    assert [it.messages for it in items] == [[{"role": "user", "content": p}] for p in EVAL_PROMPTS]
    assert all(it.condition == "none" for it in items)
    assert len({it.id for it in items}) == len(items)
    assert {trait.split(it) for it in items} <= {"extract", "eval"}


def test_summarize_values_and_labels():
    rows = [
        ("none", Score(value=0.2)),
        ("none", Score(value=0.6)),
        ("1", Score(label="lie")),
        ("1", Score(label="honest")),
        ("1", Score(label="lie")),
        ("1", Score(label="evasive")),
    ]
    out = summarize(rows)
    assert out["none"] == {"n": 2, "mean": pytest.approx(0.4)}
    assert out["1"] == {"n": 4, "rates": {"evasive": 0.25, "honest": 0.25, "lie": 0.5}}


# --- generation cache (loads the model) --------------------------------------

@pytest.fixture(scope="module")
def loaded():
    return load_model()


MESSAGES = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is the capital of France?"},
]


def test_cache_hit_returns_same_text(loaded, tmp_path, monkeypatch):
    model, tokenizer = loaded
    monkeypatch.setattr(config, "GEN_CACHE_DIR", tmp_path)

    first = generate(model, tokenizer, MESSAGES, max_new_tokens=20)
    assert len(list(tmp_path.glob("*.json"))) == 1
    second = generate(model, tokenizer, MESSAGES, max_new_tokens=20)
    assert second == first == generate(model, tokenizer, MESSAGES, max_new_tokens=20, cache=False)

    # A different max_new_tokens is a different key.
    generate(model, tokenizer, MESSAGES, max_new_tokens=10)
    assert len(list(tmp_path.glob("*.json"))) == 2


def test_string_and_user_only_list_share_a_key(loaded, tmp_path, monkeypatch):
    model, tokenizer = loaded
    monkeypatch.setattr(config, "GEN_CACHE_DIR", tmp_path)

    generate(model, tokenizer, "Name a colour.", max_new_tokens=5)
    generate(model, tokenizer, [{"role": "user", "content": "Name a colour."}], max_new_tokens=5)
    assert len(list(tmp_path.glob("*.json"))) == 1


def test_steered_call_needs_steering_spec(loaded, tmp_path, monkeypatch):
    """Inside steer(), a cached call must say how it is steered, and gets its own key."""
    model, tokenizer = loaded
    monkeypatch.setattr(config, "GEN_CACHE_DIR", tmp_path)
    layer = model.config.num_hidden_layers // 2
    torch.manual_seed(0)
    vector = torch.randn(model.config.hidden_size)

    with steer(model, layer, vector, alpha=0.3, r=1.0):
        with pytest.raises(ValueError):
            generate(model, tokenizer, MESSAGES, max_new_tokens=5)
        generate(model, tokenizer, MESSAGES, max_new_tokens=5,
                 steering=steering_spec(layer, vector, 0.3, 1.0))
    generate(model, tokenizer, MESSAGES, max_new_tokens=5)
    assert len(list(tmp_path.glob("*.json"))) == 2
