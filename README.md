# trait-vectors

Steering vectors in the residual stream of `Qwen/Qwen2.5-1.5B-Instruct`.
We read directions out of the model's activations and add them back during
generation to change its behaviour. No training, only forward passes with
PyTorch hooks. Phase 0 (plumbing, closed) is specified in `docs/phase0_spec.md`;
Phase 1 (data and scoring) in `docs/phase1_spec.md`.

## Phase 0 results: steering sentiment

Phase 0 built the pipeline and proved it end to end on a toy trait.

- **Read and write the residual stream** with PyTorch forward hooks
  (`hooks.py`): `read_activations` captures h_ℓ at chosen blocks,
  `steer` adds α·r_ℓ·v̂ to h_ℓ on every forward pass. Both are context managers
  that remove their hooks in `finally`; seven tests cover them, including a
  cross-check against Hugging Face's own `output_hidden_states`.
- **Extract a direction by difference of means** (`extract.py`): 100 positive
  and 100 negative SST-2 sentences, one forward pass each, h_ℓ read at the last
  prompt token; v_ℓ = mean(P+) − mean(P−) at every layer, plus the scale r_ℓ.
  The class separation ‖v_ℓ‖/r_ℓ jumps from ~0.02 to ~0.10 at layer 16 and the
  direction stabilises (consecutive-layer cosine > 0.9) late in the network.
- **Steer and measure** (`03_steer_sentiment.py`): 4 layers × 7 values of α
  × 30 neutral writing prompts, none about films, scored with an independent
  DistilBERT sentiment classifier, against three random directions of the same
  length as controls.

![mean P(positive) vs alpha, four layers, random controls](docs/figures/phase0_sentiment_sweep.png)

At layer 21 the score is strictly monotone in α (spread 0.61 between α = −0.6
and +0.6); layer 18 has the largest spread (0.81). Random directions stay
within ±0.13 of the baseline. The text stays fluent across the range — the
same "hospital waiting room" prompt gives *"tense and anxious"* at α = 0 and
*"a warm and welcoming space where patients can feel safe"* at α = +0.6 —
and no reply mentions films. Pushing further (`04_degeneration_probe.py`):
|α| ≤ 0.6 is clean, 1.0 is grammatical but semantically broken, and text
collapses from |α| ≈ 1.5. The whole pipeline is deterministic: a clean copy of
the repo reproduces the vectors bit-for-bit and the sweep byte-for-byte.

The first sweep did not show this. Its prompts asked about the model's own
experience, and 359 of 420 replies opened with "As an AI language model, I
don't have a physical body…", leaving little text to carry tone; the scorer
also saturates, so means over 10 prompts were as noisy as the controls. The
prompts were rephrased as writing tasks and tripled, and the layer grid moved
to where the extraction statistics said the direction existed. The amendment
and its reasons are recorded in the spec.

## Setup

```
uv sync
uv run scripts/00_check_env.py   # prints torch version and MPS availability
uv run scripts/01_generate.py    # loads the model and answers three questions
uv run scripts/02_extract_sentiment.py   # extracts sentiment vectors -> artifacts/sentiment.pt
uv run scripts/03_steer_sentiment.py     # steering sweep -> artifacts/sentiment_sweep.{csv,png}
uv run scripts/04_degeneration_probe.py  # pushes alpha past the sweep's range; read the output
uv run pytest                    # hook tests
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
