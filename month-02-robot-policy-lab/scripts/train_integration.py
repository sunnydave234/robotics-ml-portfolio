"""
Runs real LeRobot training against RobotForgeAdapter instead of the
default HuggingFace-hosted LerobotDataset, by swapping in our dataset
right before training starts.
"""
from unittest.mock import patch

import wandb

import lerobot.scripts.lerobot_train as lerobot_train_module
from robot_policy_lab.datasets.adapter import RobotForgeAdapter
from robot_policy_lab.paths import DATASET_PARQUET_PATH, DATASET_STATS_PATH, DATASET_DVC_PATH
from robot_policy_lab.utils.logging import (
    get_device_profile,
    get_git_hash,
    get_dvc_dataset_hash,
    get_config_hash,
)

_original_make_datasets = lerobot_train_module.make_train_eval_datasets


def patched_make_train_eval_datasets(cfg):
    # Call the real one first — gives us LeRobot's normal eval_dataset,
    # AND the original train dataset's .meta (features, stats, camera_keys —
    # including ImageNet image stats already injected by use_imagenet_stats,
    # datasets/factory.py lines 205-209).
    original_train_ds, eval_ds = _original_make_datasets(cfg)

    # W2D5 bug: LeRobot's own WandBLogger.__init__ (wandb_utils.py:99) already
    # called wandb.init() at lerobot_train.py:226 — BEFORE this function ever
    # runs (line 249). It logs cfg.to_dict() with no _meta at all — our
    # init_wandb_run() from W2D3 is correct but never gets called by the real
    # entry point, since LeRobot has its own native wandb integration.
    # Fix: push _meta into the ALREADY-LIVE run via wandb.config.update()
    # rather than calling wandb.init() a second time (which would either
    # error or silently create an orphaned second run).
    if wandb.run is not None:
        wandb.config.update({
            "_meta": {
                "git_hash":         get_git_hash(),
                "dataset_dvc_hash": get_dvc_dataset_hash(DATASET_DVC_PATH),
                "config_hash":      get_config_hash(cfg),
                **get_device_profile(),
            }
        }, allow_val_change=True)
        print("[RobotForge] _meta fields pushed into live W&B run")
    else:
        print("[RobotForge] WARNING: wandb.run is None — _meta not logged. "
              "Check cfg.wandb.enable and cfg.wandb.mode.")

    print(f"[RobotForge] Replacing default train dataset "
          f"({len(original_train_ds)} samples) with RobotForgeAdapter")
    train_ds = RobotForgeAdapter(
        parquet_path=DATASET_PARQUET_PATH,
        stats_path=DATASET_STATS_PATH,
        n_action_steps=cfg.policy.n_action_steps,
    )

    # make_policy() (policies/factory.py:518,536,548) requires dataset.meta
    # with .features and .stats. Reuse the ORIGINAL dataset's .meta directly
    # rather than hand-building one: both datasets are the same underlying
    # lerobot/pusht data (ours just reads it from our own HDF5 store instead
    # of LeRobot's cache), so features/stats/camera_keys are identical.
    train_ds.meta = original_train_ds.meta

    print(f"[RobotForge] RobotForgeAdapter loaded: {len(train_ds)} samples")

    return train_ds, eval_ds


if __name__ == "__main__":
    # Same registration step main() normally does before calling train() —
    # skipped automatically since we're calling train() directly instead of main().
    lerobot_train_module.register_third_party_plugins()

    with patch.object(
        lerobot_train_module,
        "make_train_eval_datasets",
        patched_make_train_eval_datasets,
    ):
        lerobot_train_module.train()