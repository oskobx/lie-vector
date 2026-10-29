"""Difference-of-means extraction: one direction per layer, plus the scale r_ell.

For a trait with prompt sets P_plus and P_minus, the vector at layer ell is

    v_ell = mean over P_plus of h_ell  -  mean over P_minus of h_ell

where h_ell is read at the last token of the chat-templated prompt. r_ell is
the mean of ||h_ell|| at that same token over all extraction prompts, and is
what makes the steering scale alpha dimensionless (see README).

Nothing here knows which trait it is working on.
"""

import torch

from trait_vectors import config
from trait_vectors.hooks import read_activations
from trait_vectors.model import build_inputs
from trait_vectors.traits.base import Trait


def last_token_activations(model, tokenizer, prompts, layers):
    """Return a float32 tensor of shape (len(prompts), len(layers), d).

    Entry [i, j] is h_{layers[j]} at the last token of prompts[i]. One forward
    pass per prompt, batch size 1: with a single unpadded prompt, position -1
    is guaranteed to be the last real token. (Batching would need left
    padding to keep that true; deferred to Phase 1.)
    """
    rows = []
    for prompt in prompts:
        inputs = build_inputs(tokenizer, prompt).to(model.device)
        with read_activations(model, layers) as acts:
            with torch.inference_mode():
                model(**inputs)
        # acts[ell] is (1, n, d); [0, -1] picks batch 0, last position -> (d,)
        rows.append(torch.stack([acts[ell][0, -1] for ell in layers]))
    return torch.stack(rows)


def extract(model, tokenizer, trait: Trait):
    """Return {"layers": [...], "vectors": (len(layers), d), "r": (len(layers),)}.

    Layers are 0 .. L-2: the last block is never hooked (README conventions).
    """
    L = model.config.num_hidden_layers
    layers = list(range(L - 1))

    plus, minus = trait.extraction_pairs()
    h_plus = last_token_activations(model, tokenizer, plus, layers)    # (n_plus, L-1, d)
    h_minus = last_token_activations(model, tokenizer, minus, layers)  # (n_minus, L-1, d)

    vectors = h_plus.mean(dim=0) - h_minus.mean(dim=0)  # (L-1, d)
    h_all = torch.cat([h_plus, h_minus])                # (n_plus + n_minus, L-1, d)
    r = h_all.norm(dim=-1).mean(dim=0)                  # (L-1,)

    return {"layers": layers, "vectors": vectors, "r": r}


def logistic_regression(X, y, lam):
    """L2-regularised logistic regression (Phase 3 spec §3). Returns (w, b): w of shape (d,), b a float.

    X is (n, d), y is (n,) with 1 = positive, 0 = negative. Minimises

        mean_i log(1 + exp(-s_i (w·x_i + b)))  +  (lam / 2) ||w||^2,     s_i = 2 y_i - 1,

    a strictly convex function of w (b is not penalised), so there is one
    minimum. Solved full-batch with L-BFGS, a quasi-Newton method, in float64
    on the CPU. Raises if the gradient at the end is not close to zero.
    """
    X = X.detach().cpu().double()
    y = y.detach().cpu().double()
    # requires_grad=True makes PyTorch record operations on w and b, so that
    # loss.backward() can fill w.grad and b.grad with the gradient of the loss.
    w = torch.zeros(X.shape[1], dtype=torch.float64, requires_grad=True)
    b = torch.zeros((), dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([w, b], max_iter=1000, tolerance_grad=1e-10, tolerance_change=1e-14,
                            history_size=20, line_search_fn="strong_wolfe")

    def loss_fn():
        # binary_cross_entropy_with_logits(z, y) is mean_i -[y_i log σ(z_i) + (1 - y_i) log(1 - σ(z_i))],
        # which equals the mean in the docstring, computed without overflow for large |z|.
        return torch.nn.functional.binary_cross_entropy_with_logits(X @ w + b, y) + 0.5 * lam * (w @ w)

    def closure():
        """L-BFGS calls this whenever it needs the loss and gradient at the current (w, b)."""
        opt.zero_grad()
        loss = loss_fn()
        loss.backward()
        return loss

    opt.step(closure)
    grad_w, grad_b = torch.autograd.grad(loss_fn(), [w, b])  # gradient at the final point
    grad_norm = (grad_w.norm() ** 2 + grad_b ** 2).sqrt().item()
    if grad_norm > 1e-6:
        raise RuntimeError(f"logistic regression did not converge: gradient norm {grad_norm:.2e} (lam={lam})")
    return w.detach(), b.item()


def save_vectors(path, result, description):
    """Write the extraction result plus provenance to a .pt file."""
    payload = {
        **result,
        "model_id": config.MODEL_ID,
        "dtype": str(config.DTYPE),
        "seed": config.SEED,
        "dataset": description,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_vectors(path):
    return torch.load(path)
