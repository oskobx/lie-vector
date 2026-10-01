# Project: steering vectors on a small language model

Research-style portfolio project. We extract directions in the residual stream of Qwen2.5-1.5B-Instruct and add them during generation to change behavior. No training anywhere, only forward passes with PyTorch hooks.

Current phase: **Phase 1** (data and scoring). The spec is `docs/phase1_spec.md`. Follow it exactly. Phase 0 is closed; its spec is `docs/phase0_spec.md` and its handoff is in `docs/private/handoffs/`.

## How to work with me

- Do one spec step at a time. After each step, stop, show me the output, and wait for me to say continue.
- I am a mathematician, new to PyTorch and ML engineering. When you write code, briefly explain anything PyTorch- or transformers-specific. Define ML jargon the first time you use it.
- I want to understand every line I ship. Prefer short, plain code over clever code. No abstractions the spec doesn't ask for.
- Do not add dependencies beyond the spec without asking.
- Do not run git commands. I commit myself through VS Code Source Control. Tell me when a step is a good commit point and suggest a commit message.
- Keep terminal command explanations to one line each.
- `src/trait_vectors/hooks.py` is finished and tested; change it only if the spec requires it and say why.

## Technical rules

- Python 3.12, uv. Run things with `uv run`.
- Device is Apple MPS. dtype bfloat16, fallback float32. Never float16, never quantization.
- Never hardcode the model id, number of layers, or hidden size outside `config.py`; read L and d from `model.config`.
- All hook registration goes through context managers in `hooks.py` that remove hooks in a `finally` block.
- Nothing outside `src/trait_vectors/traits/` may refer to a specific trait (sentiment, lying, sycophancy).
- Fixed seeds everywhere. Greedy decoding unless the spec says otherwise.
- Generated files go in `artifacts/` (gitignored).

## Project journal

`docs/private/journal.md` (gitignored, local only) is the running record (timeline, decision log, per-phase results, interview question bank). When a phase closes, write the phase handoff to `docs/private/handoffs/phaseN_handoff_<date>.md` (gitignored) and then update the journal: timeline row, phase entry in the existing format, new decision-log rows, new interview questions. Keep journal entries short; detail belongs in the handoff.

## API access

The judge and the scenario generator use the Anthropic API, model `claude-haiku-4-5`. The key lives in `.env` as `ANTHROPIC_API_KEY` (gitignored); read it with `python-dotenv`, never hardcode it, never print it. Credits are prepaid and small, so every API call path must be cached and must fail loudly rather than retry forever.
