"""Tests for the Phase 1 section 1 pipeline changes: EvalItem, Score, split, cache.

`monkeypatch` and `tmp_path` are built-in pytest fixtures: tmp_path is a fresh
empty directory per test, and monkeypatch.setattr changes an attribute for
the duration of one test and restores it afterwards.
"""

import math
import statistics

import pytest
import torch

from trait_vectors import config
from trait_vectors.hooks import steer
from trait_vectors.metrics import auroc, paired_bootstrap, stratified_auroc, summarize
from trait_vectors.model import generate, load_model, reply_perplexity, steering_spec
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


def test_auroc():
    assert auroc([3, 4], [1, 2]) == 1.0      # every pos above every neg
    assert auroc([1, 2], [3, 4]) == 0.0      # every pos below
    assert auroc([1, 3], [2]) == 0.5         # one pair above, one below
    assert auroc([1], [1]) == 0.5            # a tie counts half


def test_stratified_auroc():
    """Hand-computed (Phase 3 spec §4).

    Stratum a: positives 4, 2; negatives 1, 3. Pairs 4>1, 4>3, 2>1, 2<3: AUROC 3/4, weight 2·2/4 = 1.
    Stratum b: positive 5; negatives 6, 6, 0. Only 5>0: AUROC 1/3, weight 1·3/4 = 3/4.
    Weighted mean: (1·3/4 + 3/4·1/3) / (1 + 3/4) = 4/7.
    """
    scores = [4, 2, 1, 3, 5, 6, 6, 0]
    positive = [True, True, False, False, True, False, False, False]
    strata = ["a", "a", "a", "a", "b", "b", "b", "b"]
    assert stratified_auroc(scores, positive, strata, min_per_side=1) == pytest.approx(4 / 7)
    assert auroc([4, 2, 5], [1, 3, 6, 6, 0]) == pytest.approx(8 / 15)  # pooled, for contrast
    # Shifting one stratum's scores changes the pooled AUROC (to 6/15) but not the stratified one.
    shifted = scores[:4] + [s + 100 for s in scores[4:]]
    assert auroc([4, 2, 105], [1, 3, 106, 106, 100]) == pytest.approx(6 / 15)
    assert stratified_auroc(shifted, positive, strata, min_per_side=1) == pytest.approx(4 / 7)
    # min_per_side=2 skips stratum b (one positive), leaving a alone.
    assert stratified_auroc(scores, positive, strata, min_per_side=2) == pytest.approx(3 / 4)
    with pytest.raises(ValueError):
        stratified_auroc(scores, positive, strata, min_per_side=3)


def test_paired_bootstrap():
    # No difference anywhere: the interval collapses to 0.
    assert paired_bootstrap([1, 0, 1], [1, 0, 1]) == (0.0, 0.0, 0.0)
    # d = 1 for 60 scenarios, 0 for 40: mean 0.6; the normal interval is 0.6 ± 1.96·sd/√n ≈ ± 0.096,
    # and with n = 100 the bootstrap interval should agree with it to about 0.02.
    a, b = [1] * 60 + [0] * 40, [0] * 100
    mean, lo, hi = paired_bootstrap(a, b)
    half = 1.96 * statistics.stdev([x - y for x, y in zip(a, b)]) / math.sqrt(100)
    assert mean == pytest.approx(0.6)
    assert lo == pytest.approx(0.6 - half, abs=0.02) and hi == pytest.approx(0.6 + half, abs=0.02)
    assert paired_bootstrap(a, b) == (mean, lo, hi)  # seeded: same interval every time
    with pytest.raises(ValueError):
        paired_bootstrap([1, 0], [1])


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


def test_reply_perplexity(loaded):
    """The model's own greedy reply is unsurprising to it; the same words scrambled are not."""
    model, tokenizer = loaded
    prompt = "Describe a sunny morning in two sentences."  # a reply of several words, so scrambling changes it
    reply = generate(model, tokenizer, prompt, max_new_tokens=40, cache=False)
    scrambled = " ".join(reversed(reply.split()))
    own, garbage = reply_perplexity(model, tokenizer, prompt, reply), reply_perplexity(model, tokenizer, prompt, scrambled)
    assert own < 3.0 and garbage > 10 * own
    with steer(model, model.config.num_hidden_layers // 2, torch.ones(model.config.hidden_size), alpha=0.1, r=1.0):
        with pytest.raises(ValueError):
            reply_perplexity(model, tokenizer, prompt, reply)


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
