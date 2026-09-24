"""Load the model and generate text. Nothing here knows about hooks or traits."""

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


def build_inputs(tokenizer, user_message, system=None):
    """Chat-template a user message into model inputs (input_ids, attention_mask).

    add_generation_prompt=True appends the assistant header, so position -1
    is the token the model predicts its first reply token from. That is the
    "last token" of the README conventions, used both for generation and
    for reading activations in extract.py.
    """
    messages = []
    if system is not None:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user_message})

    return tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    )


def generate(model, tokenizer, user_message, system=None):
    """Greedy-decode a reply to user_message and return only the newly generated text."""
    inputs = build_inputs(tokenizer, user_message, system).to(model.device)

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=config.MAX_NEW_TOKENS,
            do_sample=False,
        )

    # generate() returns prompt + reply; keep only the reply.
    prompt_len = inputs["input_ids"].shape[1]
    new_ids = output_ids[0, prompt_len:]
    return tokenizer.decode(new_ids, skip_special_tokens=True)
