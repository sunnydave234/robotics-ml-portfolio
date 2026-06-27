"""
Seed all RNG sources for reproducible training runs.

Call seed_everything(cfg.seed) as the FIRST thing in train() -
before model init, before DataLoader creation, before any tensor ops.

On MPS: use_deterministic_algorithms may warn for ops without
deterministic MPS implementations. This is expected and documented.
On CUDA (Week 3): add CUBLAS_WORKSPACE_CONFIG=:4096:8 as an env var
for full determinism.
"""
import os
import random

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    """ Seed all RNG sources. Call before any tensor or model creation. """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    
    os.environ["PYTHONHASHSEED"] = str(seed)

    # warn_only=True because some MPS ops lack deterministic implementations.
    # This warns rather than errors - training can proceed.
    # On the Week 3 CUDA box, this can be Ture without warn_only.
    torch.use_deterministic_algorithms(True, warn_only=True)