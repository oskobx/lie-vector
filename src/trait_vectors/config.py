"""Project-wide constants. The only place the model id, dtype and device are written down."""

from pathlib import Path

import torch

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"

# bfloat16 on MPS. If outputs contain NaN or garbage, switch this to torch.float32.
# Never float16 (Qwen2.5 overflows) and never quantization.
DTYPE = torch.bfloat16
DEVICE = "mps"

SEED = 0
MAX_NEW_TOKENS = 60

# Repo root is two levels up from this file: src/trait_vectors/config.py
ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS_DIR = ROOT / "artifacts"
