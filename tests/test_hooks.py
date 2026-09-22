"""Tests for the read hook (spec step 4).

The model is loaded once for the whole test session (a module-scoped fixture),
because loading takes ~15 s. Tests must therefore not leave the model in a
modified state.
"""

import pytest
import torch

from trait_vectors.hooks import read_activations
from trait_vectors.model import load_model


@pytest.fixture(scope="module")
def loaded():
    """(model, tokenizer, inputs) shared by every test in this file."""
    model, tokenizer = load_model()
    inputs = tokenizer.apply_chat_template(
        [{"role": "user", "content": "What is the capital of France?"}],
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    ).to(model.device)
    return model, tokenizer, inputs


def hook_counts(model):
    """Number of forward hooks currently attached to each block."""
    return [len(block._forward_hooks) for block in model.model.layers]


def test_matches_output_hidden_states(loaded):
    """h_ell from our hook equals output_hidden_states[ell + 1].

    HF returns L + 1 hidden states: index 0 is the embedding output, so block
    ell's output sits at index ell + 1. The last block is excluded because HF
    applies the final norm before returning it (see README conventions).
    """
    model, _, inputs = loaded
    L = model.config.num_hidden_layers
    layers = [0, L // 2, L - 2]

    with read_activations(model, layers) as acts:
        with torch.inference_mode():
            out = model(**inputs, output_hidden_states=True)

    assert sorted(acts) == sorted(layers)
    for ell in layers:
        expected = out.hidden_states[ell + 1].detach().cpu().float()
        # atol suits bfloat16: ~2-3 significant digits, values of order 1-100.
        assert torch.allclose(acts[ell], expected, atol=1e-2), f"mismatch at ell={ell}"


def test_stored_tensors_are_cpu_float32(loaded):
    """Stored activations are detached, on CPU, in float32, shape (batch, n, d)."""
    model, _, inputs = loaded
    d = model.config.hidden_size
    n = inputs["input_ids"].shape[1]

    with read_activations(model, [0]) as acts:
        with torch.inference_mode():
            model(**inputs)

    h = acts[0]
    assert h.shape == (1, n, d)
    assert h.dtype == torch.float32
    assert h.device.type == "cpu"
    assert h.grad_fn is None


def test_hooks_removed_on_exit(loaded):
    """Every hook we registered is gone once the context manager exits.

    We compare against the count *before* entering rather than against zero:
    transformers 5.x implements output_hidden_states=True with its own forward
    hooks on every block and never removes them, so another test may have left
    library hooks attached. We only claim ours are gone.
    """
    model, _, inputs = loaded
    L = model.config.num_hidden_layers
    layers = [0, L // 2, L - 2]

    before = hook_counts(model)

    with read_activations(model, layers) as acts:
        during = hook_counts(model)
        with torch.inference_mode():
            model(**inputs)

    after = hook_counts(model)

    # inside the block, exactly the requested layers gained one hook each
    for ell in range(L):
        expected = before[ell] + (1 if ell in layers else 0)
        assert during[ell] == expected, f"unexpected hook count at ell={ell}"

    assert after == before


def test_hooks_removed_after_exception(loaded):
    """The finally block runs even when the with-body raises."""
    model, _, _ = loaded
    before = hook_counts(model)

    with pytest.raises(ValueError):
        with read_activations(model, [0, 1]):
            raise ValueError("boom")

    assert hook_counts(model) == before
