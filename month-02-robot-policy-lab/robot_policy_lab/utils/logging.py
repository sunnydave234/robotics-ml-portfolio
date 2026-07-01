"""
robot_policy_lab/utils/logging.py

Production-grade W&B logging: every training run gets stamped with 7
reproducibility fields so a run ID alone answers "what code, what data,
what config, what hardware produced this?"

Design rule: init_wandb_run() takes dvc_path as an explicit keyword-only
argument rather than reading cfg.dataset_dvc_path. dataset_dvc_path is a
custom key that lives in conf/pusht_act.json but is NOT a declared field
on LeRobot's TrainPipelineConfig dataclass -- draccus parses the JSON
without complaint, but never attaches unrecognized top-level keys to cfg.
cfg.dataset_dvc_path raises AttributeError even though the key sits right
there in the file.

dataset name is NOT a separate parameter -- it's read straight off
cfg.dataset.repo_id, which IS a real, draccus-bound field (the required
--dataset.repo_id CLI flag confirmed back in W1D1). Reusing it means the
W&B tag can never drift out of sync with the dataset actually training.

Call order in scripts/train.py:
    1. configure_mps_env()
    2. seed_everything(cfg.seed)
    3. init_wandb_run(cfg, dvc_path=...)      <- this file
    4. training loop
"""
import dataclasses
import platform
import torch
import wandb

from robot_policy_lab.utils.lineage import(
    get_config_hash,
    get_dvc_dataset_hash,
    get_git_hash,
)


def get_device_profile() -> dict:
    """
    Minimal device info for the W&B config panel.
    Covers MPS (M4 Mac), CUDA, and CPU fallback -- the CUDA branch won't
    fire on your machine today, but it's what classifies your runs once
    you're on the Week 3 GPU box.
    """
    if torch.backends.mps.is_available():
        return {
            "device_type":      "mps",
            "platform":         platform.platform(),
            "torch_version":    torch.__version__,
        }
    elif torch.cuda.is_available():
        return {
            "device_type":      "cuda",
            "gpu_name":         torch.cuda.get_device_name(),
            "gpu_count":        torch.cuda.device_count(),
            "torch_version":    torch.__version__,
        }
    else:
        return {
            "device_type":      "cpu",
            "platform":         platform.platform(),
            "torch_version":    torch.__version__,
        }


def init_wandb_run(cfg, *, dvc_path) -> None:
    """
    Initialize a W&B run with all 7 reproducibility fields.

    Args:
        cfg:      draccus-parsed TrainPipelineConfig.
        dvc_path: path to the DVC-tracked artifact to hash for lineage
                  (see module docstring for why this can't come from cfg).

    Must be called after seed_everything() and configure_mps_env().
    """
    # dataclasses.asdict() recurses into nested dataclasses -- this
    # snapshots every hyperparameter draccus resolved, not just the
    # handful referenced explicitly below.
    config = dataclasses.asdict(cfg)

    # Group the 7 fields under _meta so they're easy to find in the W&B
    # Config panel instead of buried among 80+ flat hyperparameter keys.
    config["_meta"] = {
        "git_hash":         get_git_hash(),
        "dataset_dvc_hash": get_dvc_dataset_hash(dvc_path),
        "config_hash":      get_config_hash(cfg),
        **get_device_profile(),
    }

    wandb.init(
        project=cfg.wandb.project,
        config=config,
        resume="allow",                     # resume by run ID if a run crashes
        tags=[cfg.dataset.repo_id, "act"]
    )


def log_train_step(loss: float, lr: float, grad_norm: float,
                    steps_per_sec: float, global_step: int) -> None:
    wandb.log({
        "train/loss":           loss,
        "train/lr":             lr,
        "train/grad_norm":      grad_norm,
        "perf/steps_per_sec":   steps_per_sec,
    }, step=global_step)


def log_eval(success_rate: float, n_rollouts: int, global_step: int) -> None:
    """
    Wilson CI bounds are a Week 4 deliverable -- stubbed as None for now.
    wandb omits None-valued keys from the UI until a real value lands
    under the same key, so these two just won't show up until Week 4.
    """
    wandb.log({
        "eval/success_rate":          success_rate,
        "eval/n_rollouts":            n_rollouts,
        "eval/success_rate_ci_lower": None,
        "eval/success_rate_ci_upper": None,
    }, step=global_step)


def log_best_checkpoint(checkpoint_path: str, global_step: int,
                        success_rate: float, dataset_dvc_hash: str) -> None:
    """
    Log the best checkpoint as a W&B artifact. Call only when success_rate
    improves -- artifacts are immutable once logged, so logging every
    checkpoint wastes storage.
    """
    artifact = wandb.Artifact(
        name=f"model-{wandb.run.id}",
        type="model",
        metadata={
            "step":                 global_step,
            "success_rate":         success_rate,
            "dataset_dvc_hash":     dataset_dvc_hash,   # pins which data trained this
        },
    )
    artifact.add_file(checkpoint_path)
    wandb.log_artifact(artifact)