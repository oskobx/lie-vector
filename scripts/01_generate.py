"""Step 2 check: load the model on MPS and answer three fixed questions."""

import torch

from trait_vectors import config
from trait_vectors.model import generate, load_model

QUESTIONS = [
    "What is the capital of France?",
    "Explain in two sentences what a prime number is.",
    "Describe your commute this morning.",
]


def main():
    torch.manual_seed(config.SEED)
    model, tokenizer = load_model()

    L = model.config.num_hidden_layers
    d = model.config.hidden_size
    assert len(model.model.layers) == L

    print(f"model:   {config.MODEL_ID}")
    print(f"device:  {model.device}   dtype: {model.dtype}")
    print(f"L = {L} blocks,  d = {d}")
    print(f"memory footprint: {model.get_memory_footprint() / 1e9:.2f} GB")

    for q in QUESTIONS:
        print("\n> " + q)
        print(generate(model, tokenizer, q))


if __name__ == "__main__":
    main()
