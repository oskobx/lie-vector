# Project: steering vectors on a small language model

Research-style portfolio project. We extract directions in the residual stream of Qwen2.5-1.5B-Instruct and add them during generation to change behavior. No training anywhere, only forward passes with PyTorch hooks.

Current phase: **Phase 0** (plumbing). The spec is `docs/phase0_spec.md`. Follow it exactly.

## How to work with me

- Do one spec step at a time. After each step, stop, show me the output, and wait for me to say continue.
- I am a mathematician, new to PyTorch and ML engineering. When you write code, briefly explain anything PyTorch- or transformers-specific. Define ML jargon the first time you use it.
- I want to understand every line I ship. Prefer short, plain code over clever code. No abstractions the spec doesn't ask for.
- Do not add dependencies beyond the spec without asking.
- Do not run git commands. I commit myself through VS Code Source Control. Tell me when a step is a good commit point and suggest a commit message.
- Keep terminal command explanations to one line each.
- **`src/trait_vectors/hooks.py` is mine to write.** For spec steps 3 and 5, do not write the hook bodies. Write the tests (steps 4 and 5) and a skeleton with function signatures and docstrings, then let me fill in the bodies. Give hints if I ask, full solutions only if I ask explicitly.

## Technical rules

- Python 3.12, uv. Run things with `uv run`.
- Device is Apple MPS. dtype bfloat16, fallback float32. Never float16, never quantization.
- Never hardcode the model id, number of layers, or hidden size outside `config.py`; read L and d from `model.config`.
- All hook registration goes through context managers in `hooks.py` that remove hooks in a `finally` block.
- Nothing outside `src/trait_vectors/traits/` may refer to a specific trait (sentiment, lying, sycophancy).
- Fixed seeds everywhere. Greedy decoding unless the spec says otherwise.
- Generated files go in `artifacts/` (gitignored).
