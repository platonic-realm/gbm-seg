"""Reproducibility / deterministic-CUDA regression. Locks in §E1.

After `seed_everything()` is called, three invariants must hold:

1. `CUBLAS_WORKSPACE_CONFIG` env var is set (required for cuBLAS deterministic
   workspace; set at module-import in both `train.py` and `gbm.py`).
2. `torch.are_deterministic_algorithms_enabled()` returns True.
3. The same seed produces the same Python/numpy/torch RNG state on a
   subsequent invocation (i.e., seeding is idempotent and complete).
"""

import os
import random

import numpy as np
import torch


def test_cublas_workspace_config_env_var_is_set():
    """The env var must be set before any CUDA op. train.py + gbm.py set it
    at module-import time via os.environ.setdefault."""
    # Importing train.py should be sufficient — it does the setdefault on import.
    import train  # noqa: F401
    assert os.environ.get("CUBLAS_WORKSPACE_CONFIG") == ":4096:8"


def test_seed_everything_enables_deterministic_algorithms():
    from train import SEED, seed_everything

    seed_everything(SEED)
    assert torch.are_deterministic_algorithms_enabled() is True


def test_seed_everything_is_reproducible():
    """Two seed_everything() calls with the same seed must produce identical
    RNG draws across python/numpy/torch."""
    from train import SEED, seed_everything

    seed_everything(SEED)
    py = random.random()
    np_val = np.random.rand()
    torch_val = torch.rand(()).item()

    seed_everything(SEED)
    assert random.random() == py
    assert np.random.rand() == np_val
    assert torch.rand(()).item() == torch_val


def test_seed_everything_is_isolated_per_seed():
    """Different seeds must produce different RNG draws (sanity)."""
    from train import seed_everything

    seed_everything(42)
    a = torch.rand(())
    seed_everything(43)
    b = torch.rand(())
    assert not torch.equal(a, b)
