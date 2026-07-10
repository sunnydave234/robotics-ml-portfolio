"""
Saves everything needed to resume training exactly: model weights,
optimizer state, RNG state (including MPS, which LeRobot's own code
does not save - see W2D4 finding in PREFLIGHT.md), and lineage info.
"""
import random
from pathlib import Path

import numpy as np
import torch

from robot_policy_lab.utils.lineage import (
    get_config_hash,
    get_dvc_dataset_hash,
    get_git_hash,
)


def _collect_rng_states() -> dict:
    states = {
        "python":       random.getstate(),
        "numpy":        np.random.get_state(),
        "torch_cpu":    torch.get_rng_state(),
        "torch_mps":    None,
        "torch_cuda":   None,
    }
    if torch.backends.mps.is_available():
        states["torch_mps"] = torch.mps.get_rng_state()
    if torch.cuda.is_available():
        states["torch_cuda"] = torch.cuda.get_rng_state_all()
    return states


def _restore_rng_states(states: dict) -> None:
    random.setstate(states["python"])
    np.random.set_state(states["numpy"])
    torch.set_rng_state(states["torch_cpu"])
    if states.get("torch_mps") is not None and torch.backends.mps.is_available():
        torch.mps.set_rng_state(states["torch_mps"])
    if states.get("torch_cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(states["torch_cuda"])


def save_checkpoint(
    path,
    *,
    step: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler=None,
    cfg=None,
    dvc_path: str | None = None,
    wandb_run_id: str | None = None,
) -> dict:
    ckpt = {
        "step":                 step,
        "model_state_dict":     model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "rng_states":           _collect_rng_states(),
        "lineage": {
            "git_hash":         get_git_hash(),
            "dataset_dvc_hash": get_dvc_dataset_hash(dvc_path) if dvc_path else None,
            "config_hash":      get_config_hash(cfg) if cfg is not None else None,
            "wandb_run_id":     wandb_run_id,
        },
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, path)
    return ckpt


def load_checkpoint(
    path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler=None,
    restore_rng: bool = True,
) -> dict:
    # weights_only=False is required here: Python/NumPy RNG state is a
    # pickled tuple, not a tensor. Recent PyTorch defaults to
    # weights_only=True and will refuese to load our own file without this.
    ckpt = torch.load(path, weights_only=False)

    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if scheduler is not None and ckpt["shceduler_state_dict"] is not None:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    if restore_rng:
        _restore_rng_states(ckpt["rng_states"])
    return ckpt