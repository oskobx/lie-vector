"""Tests for the read hook (spec step 4).

The model is loaded once for the whole test session (a module-scoped fixture),
because loading takes ~15 s. Tests must therefore not leave the model in a
modified state.
"""

import pytest
import torch

from trait_vectors.hooks import read_activations, steer
from trait_vectors.model import generate, load_model


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


# --- steering hook (spec step 5) --------------------------------------------

PROMPT = "Describe your commute this morning."


@pytest.fixture(scope="module")
def direction(loaded):
    """A fixed random direction in R^d. Any direction will do for these tests."""
    model, _, _ = loaded
    torch.manual_seed(0)
    return torch.randn(model.config.hidden_size)


def test_steer_alpha_zero_is_identity(loaded, direction):
    """With alpha = 0 the generated text is identical to unsteered.

    Greedy decoding is deterministic, so any difference would mean the hook
    changed the output even when asked to add nothing.
    """
    model, tokenizer, _ = loaded
    L = model.config.num_hidden_layers

    unsteered = generate(model, tokenizer, PROMPT)
    with steer(model, L // 2, direction, alpha=0.0, r=1.0):
        steered = generate(model, tokenizer, PROMPT)

    assert steered == unsteered


def test_steer_changes_downstream_not_upstream(loaded, direction):
    """Steering at layer ell changes h_{ell+1} and leaves h_{ell-1} untouched.

    Blocks run in order, so a change at ell can only be seen after ell. If
    ell-1 changed, the hook is attached to the wrong block or leaked; if ell+1
    did not change, the hook is not actually replacing the output.
    """
    model, _, inputs = loaded
    L = model.config.num_hidden_layers
    ell = L // 2

    with read_activations(model, [ell - 1, ell, ell + 1]) as clean:
        with torch.inference_mode():
            model(**inputs)

    # r_ell for this one prompt: the norm of h_ell at the last token. With
    # alpha = 1 the shift is then the size of a typical activation, so it
    # cannot vanish in bfloat16 rounding.
    r = clean[ell][0, -1].norm().item()

    with steer(model, ell, direction, alpha=1.0, r=r):
        with read_activations(model, [ell - 1, ell + 1]) as steered:
            with torch.inference_mode():
                model(**inputs)

    assert torch.equal(steered[ell - 1], clean[ell - 1])
    assert not torch.equal(steered[ell + 1], clean[ell + 1])


def test_steer_hooks_removed(loaded, direction):
    """The steering hook is gone after exit, both normally and after an exception."""
    model, _, inputs = loaded
    L = model.config.num_hidden_layers
    before = hook_counts(model)

    with steer(model, L // 2, direction, alpha=1.0, r=1.0):
        during = hook_counts(model)
        with torch.inference_mode():
            model(**inputs)
    assert during[L // 2] == before[L // 2] + 1
    assert hook_counts(model) == before

    with pytest.raises(ValueError):
        with steer(model, L // 2, direction, alpha=1.0, r=1.0):
            raise ValueError("boom")
    assert hook_counts(model) == before
