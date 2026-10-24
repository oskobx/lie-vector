"""Tests for traits/sycophancy.py: the CAA contrast pairs, and how they are templated. No model needed."""

from transformers import AutoTokenizer

from trait_vectors import config
from trait_vectors.model import build_inputs
from trait_vectors.traits.sycophancy import SycophancyTrait


def test_caa_pairs():
    plus, minus = SycophancyTrait().extraction_pairs()
    assert len(plus) == len(minus) == 1000
    for p, m in zip(plus, minus):
        assert p[0] == m[0]                                        # same question
        assert {p[1]["content"], m[1]["content"]} == {"(A", "(B"}  # the two letters


def test_partial_assistant_turn_ends_on_the_letter():
    """Phase 2 spec §2.3: position -1 of a CAA prompt is the letter token, not an end-of-turn marker."""
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_ID)
    plus, minus = SycophancyTrait().extraction_pairs()
    for chat in plus[:20] + minus[:20]:
        ids = build_inputs(tokenizer, chat)["input_ids"][0]
        assert tokenizer.decode(ids[-1:]) == chat[-1]["content"]  # "(A" or "(B" is one token


def test_user_only_chat_still_ends_on_assistant_header():
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_ID)
    ids = build_inputs(tokenizer, "Hello")["input_ids"][0]
    assert tokenizer.decode(ids[-3:]) == "<|im_start|>assistant\n"
