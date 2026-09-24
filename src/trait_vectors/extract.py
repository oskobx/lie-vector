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
