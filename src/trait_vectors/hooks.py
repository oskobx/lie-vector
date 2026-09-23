"""Forward hooks on the decoder blocks: read the residual stream, or add to it.

Conventions (see README): block ell is model.model.layers[ell] for
ell = 0, ..., L-1, and h_ell is the *output* of block ell, a tensor of shape
(batch, n, d). Never hook the last block (ell = L-1).

A forward hook is a function hook(module, inputs, output) that PyTorch calls
right after module.forward returns. It is registered with
module.register_forward_hook(hook), which returns a handle; handle.remove()
detaches it. If the hook returns a value, that value replaces the module's
output; if it returns None the output is left as is.

Every registration in this project goes through the context managers below,
which remove their hooks in a `finally` block. No bare register_forward_hook
calls anywhere else.
"""

from contextlib import contextmanager


def _hidden_states(output):
    """Return the hidden-states tensor h from a decoder block's output.

    Depending on the transformers version, a block returns either h itself
    (shape (batch, n, d)) or a tuple whose first element is h. This helper
    accepts both and returns the tensor. (transformers 5.17, installed here,
    returns the bare tensor; older versions returned a tuple.)
    """
    if isinstance(output, tuple):
        return output[0]
    return output


@contextmanager
def read_activations(model, layers):
    """Record h_ell for every ell in `layers` during forward passes in the block.

    Usage:
        with read_activations(model, [0, 5]) as acts:
            model(**inputs)
        acts[5]   # h_5, shape (batch, n, d), float32, on CPU

    Yields a dict {ell: tensor} that the hooks fill in. Each stored tensor is
    detached from autograd, moved to CPU, and cast to float32. If several
    forward passes happen inside the block (e.g. during generate), the last
    one wins.

    Registers one forward hook per layer on model.model.layers[ell] and
    removes all of them in a `finally` block, so they are gone after the
    `with` block exits, whether normally or by exception.
    """
    acts = {}
    handles = []

    for ell in layers:
        def hook(module, inputs, output, ell=ell):
            acts[ell] = _hidden_states(output).detach().cpu().float()
        handles.append(model.model.layers[ell].register_forward_hook(hook))

    try:
        yield acts
    finally:
        for handle in handles:
            handle.remove()


@contextmanager
def steer(model, layer, vector, alpha, r):
    """Add alpha * r * v_hat to the output of block `layer`, at every position.

    Usage:
        with steer(model, 14, v, alpha=0.3, r=r_14):
            text = generate(model, tokenizer, "Describe your commute.")

    With v_hat = vector / ||vector||, the hook replaces h_layer by
    h_layer + alpha * r * v_hat on every forward pass, at every position. So
    alpha is dimensionless: with r = r_ell (the mean ||h_ell|| at the last
    token over the extraction prompts), alpha = 1 is a shift the size of a
    typical activation. See the steering-scale convention in the README.

    Notes for the implementation:

    - The hook must *return* the modified output; returning a value from a
      forward hook replaces the module's output. If the block returned a
      tuple, return a tuple with the new tensor first, so the block's
      contract is preserved. Use _hidden_states to read it either way.
    - Normalise the vector and move it to the model's device and dtype once,
      here, outside the hook. The hook runs once per generated token, so it
      should do nothing but the addition.
    - v_hat has shape (d,) and h has shape (batch, n, d), so a plain `+`
      broadcasts over batch and positions. During generation the hook sees
      (1, n, d) on the first pass and (1, 1, d) afterwards (the KV cache means
      later passes carry only the new token); broadcasting handles both, so
      no special casing is needed to steer every token.
    - One hook on one block, removed in a `finally` block as above.
    """
    v_hat = vector / vector.norm()
    v_hat = v_hat.to(model.device, model.dtype)
    shift = alpha * r * v_hat

    def hook(module, inputs, output):
        h = _hidden_states(output)
        if isinstance(output, tuple):
            return (h + shift, ) + output[1:]
        else:     
            return h + shift

    handle = model.model.layers[layer].register_forward_hook(hook)

    try:
        yield
    finally:
        handle.remove()