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
