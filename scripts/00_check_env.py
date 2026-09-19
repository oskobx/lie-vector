"""Print the torch version and whether the Apple MPS backend is usable."""

import torch

print("torch version:", torch.__version__)
print("mps available:", torch.backends.mps.is_available())
