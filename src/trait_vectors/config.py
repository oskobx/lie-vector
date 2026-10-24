"""Project-wide constants. The only place the model id, dtype and device are written down."""

from pathlib import Path

import torch

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"

# bfloat16 on MPS. If outputs contain NaN or garbage, switch this to torch.float32.
# Never float16 (Qwen2.5 overflows) and never quantization.
DTYPE = torch.bfloat16
DEVICE = "mps"

# Anthropic API models (spec Phase 1 §0): the judge and the scenario generator.
# The generator runs once, so it uses the strongest model (amendment 2026-10-22).
JUDGE_MODEL = "claude-haiku-4-5"
GENERATOR_MODEL = "claude-opus-5-5"
# The dated snapshot the JUDGE_MODEL alias must resolve to (Phase 2 spec §0.3, checked 2026-10-24
# by scripts/19_check_judge_snapshot.py). The judge stops if the API reports anything else.
JUDGE_SNAPSHOT = "claude-haiku-4-5-20251001"

SEED = 0
MAX_NEW_TOKENS = 60  # default for generate(); callers may pass their own

# Repo root is two levels up from this file: src/trait_vectors/config.py
ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS_DIR = ROOT / "artifacts"
GEN_CACHE_DIR = ARTIFACTS_DIR / "gen_cache"  # one JSON file per cached generation
