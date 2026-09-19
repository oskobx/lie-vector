# Phase 0 spec: plumbing (days 1-2)

Goal: load a small instruct model on MPS, read the residual stream at any layer, add a vector to it during generation, and prove the whole path works by steering sentiment. Nothing here knows about lying or sycophancy.

## 0. Decisions fixed for this phase

- **Model:** `Qwen/Qwen2.5-1.5B-Instruct`. The model id lives in one config value; nothing else may hardcode it, the layer count L, or the width d (read them from `model.config`).
- **Hooking:** plain Hugging Face `transformers` + PyTorch forward hooks. No TransformerLens, no nnsight.
- **Decoding:** greedy (`do_sample=False`), `max_new_tokens=60`. Deterministic outputs make before/after comparisons meaningful.
- **Precision:** `bfloat16` on MPS. If outputs contain NaN or garbage, fall back to `float32` (about 6 GB, still fits). Never float16 (Qwen2.5 overflows in fp16) and never 4-bit.
- **Python 3.12, uv.** Dependencies: `torch`, `transformers`, `datasets`, `numpy`, `pytest`. Nothing else without asking.

## 1. Conventions (put these in the README and in docstrings)

- Blocks are indexed ℓ = 0, ..., L−1, matching `model.model.layers[ℓ]`.
- **h_ℓ** means the *output* of block ℓ: a tensor of shape (batch, n, d). This equals `output_hidden_states[ℓ+1]` from Hugging Face (index 0 there is the embedding output). Exception: for ℓ = L−1, HF applies the final norm before returning it, so they differ. Never hook the last block.
- "Last token" means position −1 of the chat-templated prompt built with `add_generation_prompt=True`.
- **Steering scale.** Let v̂ = v/‖v‖ and r_ℓ = mean of ‖h_ℓ‖ at the last token over the extraction prompts. Steering is h_ℓ ↦ h_ℓ + α · r_ℓ · v̂, applied at every position on every forward pass. So α is dimensionless: α = 1 is a shift the size of a typical activation. Compute r_ℓ at the last token only, because the first position in Qwen models has an abnormally large norm and would distort a mean over positions.

## 2. Repo layout

```
lie-vector/
  pyproject.toml
  README.md
  src/trait_vectors/
    config.py        # model id, dtype, device, paths
    model.py         # load_model(), generate()
    hooks.py         # read_hook, steer_hook, context managers
    extract.py       # difference of means, r_ell, save/load vectors
    traits/
      base.py        # Trait protocol
      sentiment.py   # Phase 0 toy trait
  scripts/
    01_generate.py
    02_extract_sentiment.py
    03_steer_sentiment.py
  tests/
    test_hooks.py
  artifacts/         # vectors, results (gitignored)
```

## 3. Steps

Do them in order. Stop after each step and show me the output before continuing.

### Step 1: repo and environment
`uv init`, pin Python 3.12, add dependencies, `.gitignore` (include `artifacts/`, `.venv/`, HF cache is outside the repo anyway). A script that prints torch version and `torch.backends.mps.is_available()`.

### Step 2: load and generate (`model.py`, `01_generate.py`)
- `load_model()` returns `(model, tokenizer)`, model in eval mode on MPS. Check the installed transformers version for whether the dtype argument is `dtype=` or the older `torch_dtype=`.
- `generate(model, tokenizer, user_message, system=None) -> str` builds the prompt with `tokenizer.apply_chat_template`, generates greedily under `torch.inference_mode()`, and returns only the newly generated text (slice off the prompt tokens before decoding).
- Script prints L, d, memory footprint, and the answers to three fixed questions.

### Step 3: read hook (`hooks.py`)
- A forward hook is a function `hook(module, inputs, output)` that PyTorch calls after `module.forward`. Register with `module.register_forward_hook(hook)`, which returns a handle; `handle.remove()` detaches it.
- Depending on the transformers version, a decoder block returns either a tensor or a tuple whose first element is the hidden states. Write one helper that handles both, and use it in both hooks.
- `read_activations(model, layers: list[int])`: a context manager that registers read hooks on the given blocks, yields a dict `{ℓ: tensor}` filled during the forward pass, and removes the hooks in a `finally` block. Stored tensors are detached, moved to CPU, cast to float32.
- Leaked hooks are the classic bug here (a hook left attached silently steers every later run). All hook registration goes through context managers; no bare `register_forward_hook` calls outside `hooks.py`.

### Step 4: tests for the read hook (`tests/test_hooks.py`)
1. For ℓ in {0, L//2, L−2}: hooked h_ℓ equals `output_hidden_states[ℓ+1]` (allclose, tolerance suitable for bfloat16).
2. After the context manager exits, `len(block._forward_hooks) == 0` for every block.

### Step 5: steering hook
- `steer(model, layer, vector, alpha, r)`: context manager. The hook returns a modified output (returning a value from a forward hook replaces the module's output). It adds `alpha * r * v_hat`, with the vector moved to the model's device and dtype once, outside the hook.
- During generation with a KV cache (the cache of past attention keys/values, so each new token needs only one new forward pass), the hook sees shape (1, n, d) on the first pass and (1, 1, d) afterwards. Adding to all positions in both cases implements "every token". No special casing.
- Tests: (3) with α = 0 the generated text is identical to unsteered; (4) with α ≠ 0 at layer ℓ, a read hook at ℓ+1 shows changed activations and a read hook at ℓ−1 shows unchanged ones; (5) hooks are gone after exit, including when an exception is raised inside the `with` block.

### Step 6: trait interface (`traits/base.py`)
A `Protocol` with three methods, kept minimal:
- `extraction_pairs() -> tuple[list[str], list[str]]` (P+, P− as user messages)
- `eval_prompts() -> list[str]`
- `score(prompt: str, response: str) -> float`

`extract.py` and the scripts accept any `Trait`. Nothing outside `traits/` may mention sentiment.

### Step 7: sentiment trait and extraction
- Data: SST-2 (`stanfordnlp/sst2`, train split). Keep sentences with at least 8 words, sample 100 positive and 100 negative with a fixed seed. Extraction set only; disjoint from anything used for evaluation.
- Each sentence becomes a user message: `Here is a movie review: "{sentence}"`.
- `extract.py`: for each prompt, one forward pass, batch size 1 (batching needs left-padding to keep "last token" at index −1; defer that to Phase 1), read h_ℓ at the last token for all ℓ in one pass. Output per layer: v_ℓ = mean(P+) − mean(P−) and r_ℓ. Save one `.pt` file containing the vectors, r_ℓ, model id, dtype, seed, and dataset description.
- Print per layer: ‖v_ℓ‖ / r_ℓ, and cosine(v_ℓ, v_{ℓ+1}).

### Step 8: sanity sweep (`03_steer_sentiment.py`)
- 10 neutral evaluation prompts, none about movies (e.g. "Describe your commute this morning.", "Tell me about the last meal you cooked."). Testing off-domain is deliberate: the vector should encode sentiment, not movie-ness.
- Grid: layers {L//4, L//2, 3L//4} × α ∈ {−0.6, −0.3, −0.15, 0, 0.15, 0.3, 0.6}. The α range is a guess; widen or narrow after the first run.
- Scorer: `distilbert/distilbert-base-uncased-finetuned-sst-2-english` on CPU, P(positive) of the response.
- **Random control:** repeat the sweep at the best layer with 3 random unit vectors, same α and r_ℓ.
- Output: a CSV of (layer, α, direction, prompt, response, score), a plot of mean score vs. α per layer with the random controls overlaid, and a printed table of responses for one prompt across α.

## 4. Definition of done (day-2 checkpoint)

1. All five tests pass.
2. At some layer, mean P(positive) is monotone in α across the grid, with a spread of at least 0.3 between the extremes.
3. Random directions show no monotone trend at the same scale.
4. At the α values that satisfy (2), responses are still fluent English that address the prompt (checked by reading them). Record the α at which text degenerates; that bound is needed in Phase 2.
5. A fresh clone plus `uv sync` plus the three scripts reproduces the plot.

If (2) fails: check in order (a) hook actually modifies output (test 4), (b) α range too small (print ‖α r v̂‖ / ‖h‖), (c) dtype problems (rerun in float32), (d) extraction position (last token of the templated prompt, not of the raw sentence).

## 5. Out of scope

Batching, judges, lying scenarios, GSM8K, perplexity, the 3B model, any demo code.
