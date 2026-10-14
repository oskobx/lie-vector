"""Load the model and generate text, with an on-disk cache of generations.

Nothing here registers hooks or knows about traits. The cache key does
include a description of any steering the caller has set up, which the
caller passes in (see generate and steering_spec).
"""

import hashlib
import json

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel

from trait_vectors import config


def load_model():
    """Return (model, tokenizer): the model in eval mode, on config.DEVICE, in config.DTYPE.

    Block ell is model.model.layers[ell] for ell = 0, ..., L-1, with
    L = model.config.num_hidden_layers and d = model.config.hidden_size.
    """
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_ID)
    # The annotation is only for the type checker: transformers does not declare a
    # return type on from_pretrained, so Pylance infers a class rather than an
    # instance and flags model.to(...) / model.eval().
    model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
        config.MODEL_ID, dtype=config.DTYPE
    )
    model.to(config.DEVICE)
    model.eval()
    return model, tokenizer


def as_messages(messages):
    """Accept a chat list, or a bare string meaning a single user message."""
    if isinstance(messages, str):
        return [{"role": "user", "content": messages}]
    return messages


def build_inputs(tokenizer, messages):
    """Chat-template `messages` into model inputs (input_ids, attention_mask).

    `messages` is a list of {"role": ..., "content": ...} dicts, or a bare
    string for a single user message.

    add_generation_prompt=True appends the assistant header, so position -1
    is the token the model predicts its first reply token from. That is the
    "last token" of the README conventions, used both for generation and
    for reading activations in extract.py.
    """
    return tokenizer.apply_chat_template(
        as_messages(messages),
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    )


def steering_spec(layer, vector, alpha, r):
    """Describe a steer(model, layer, vector, alpha, r) call, for the cache key.

    The vector enters as the sha256 of its float32 bytes, so a re-extracted
    vector with different values gets different cache entries even if it
    keeps the same name.
    """
    vector_bytes = vector.detach().cpu().float().numpy().tobytes()
    return {
        "layer": layer,
        "alpha": alpha,
        "r": r,
        "vector_sha256": hashlib.sha256(vector_bytes).hexdigest(),
    }


def _steer_hook_active(model):
    """True if a hook made by hooks.steer is attached to any block.

    block._forward_hooks is PyTorch's dict of the forward hooks currently on
    that module. steer's inner function is named "hook" and defined inside
    steer, so Python gives it the qualified name "steer.<locals>.hook".
    """
    for block in model.model.layers:
        for fn in block._forward_hooks.values():
            if fn.__qualname__.startswith("steer."):
                return True
    return False


def generate(model, tokenizer, messages, max_new_tokens=config.MAX_NEW_TOKENS,
             steering="none", cache=True):
    """Greedy-decode a reply to `messages` and return only the newly generated text.

    `messages` is a chat list or a bare user string (see build_inputs).

    With cache=True the reply is stored in config.GEN_CACHE_DIR, keyed by
    (model id, dtype, sha256 of the messages, steering, max_new_tokens), and
    returned from there on a repeat call. Greedy decoding is deterministic,
    so the cached reply is the one the model would produce again.

    `steering` must describe any steering active during the call: "none", or
    steering_spec(...) with the same arguments given to hooks.steer. The
    cache cannot see what a hook adds, so a wrong value here would return an
    unsteered reply for a steered call. As a safety net, a cached call with
    steering="none" while a steer hook is attached raises. Tests of the hooks
    themselves pass cache=False so the model really runs.
    """
    messages = as_messages(messages)

    if cache:
        if steering == "none" and _steer_hook_active(model):
            raise ValueError("a steer() hook is active: pass steering=steering_spec(...) "
                             "or cache=False")
        # json.dumps with sort_keys gives one canonical string per value, so
        # equal inputs always hash to the same key.
        key_fields = {
            "model_id": model.name_or_path,
            "dtype": str(model.dtype),
            "messages_sha256": hashlib.sha256(json.dumps(messages, sort_keys=True).encode()).hexdigest(),
            "steering": steering,
            "max_new_tokens": max_new_tokens,
        }
        key = hashlib.sha256(json.dumps(key_fields, sort_keys=True).encode()).hexdigest()
        path = config.GEN_CACHE_DIR / f"{key}.json"
        if path.exists():
            return json.loads(path.read_text())["response"]

    inputs = build_inputs(tokenizer, messages).to(model.device)

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    # generate() returns prompt + reply; keep only the reply.
    prompt_len = inputs["input_ids"].shape[1]
    new_ids = output_ids[0, prompt_len:]
    response = tokenizer.decode(new_ids, skip_special_tokens=True)

    if cache:
        config.GEN_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        record = {**key_fields, "messages": messages, "response": response}
        path.write_text(json.dumps(record, indent=2))

    return response
