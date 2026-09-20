# trait-vectors

Steering vectors in the residual stream of `Qwen/Qwen2.5-1.5B-Instruct`.
We read directions out of the model's activations and add them back during
generation to change its behaviour. No training, only forward passes with
PyTorch hooks. Phase 0 (plumbing) is specified in `docs/phase0_spec.md`.

## Setup

```
uv sync
uv run scripts/00_check_env.py   # prints torch version and MPS availability
uv run scripts/01_generate.py    # loads the model and answers three questions
```

## Conventions

- Blocks are indexed ℓ = 0, ..., L−1, matching `model.model.layers[ℓ]`.
  L and d (hidden size) are read from `model.config`, never hardcoded.
- **h_ℓ** is the *output* of block ℓ, a tensor of shape (batch, n, d). It equals
  `output_hidden_states[ℓ+1]` from Hugging Face (index 0 there is the embedding
  output). Exception: for ℓ = L−1, HF applies the final norm before returning
  it, so they differ. We never hook the last block.
- "Last token" means position −1 of the chat-templated prompt built with
  `add_generation_prompt=True`.
- **Steering scale.** Let v̂ = v/‖v‖ and r_ℓ = mean of ‖h_ℓ‖ at the last token
  over the extraction prompts. Steering is h_ℓ ↦ h_ℓ + α · r_ℓ · v̂, applied at
  every position on every forward pass. So α is dimensionless: α = 1 is a shift
  the size of a typical activation. r_ℓ is computed at the last token only,
  because the first position in Qwen models has an abnormally large norm.
- Decoding is greedy (`do_sample=False`), `max_new_tokens=60`, fixed seeds.
- Precision is bfloat16 on Apple MPS, float32 as fallback. Never float16,
  never quantization.
- All hook registration goes through context managers in `hooks.py` that
  remove hooks in a `finally` block.
- Generated files go in `artifacts/` (gitignored).
