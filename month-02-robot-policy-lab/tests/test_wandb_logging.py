"""
tests/test_wandb_logging.py

Uses WANDB_MODE=disabled -- no network call, no API key needed, no run
actually created on wandb.ai's servers.

test_init_wandb_run() calls the REAL init_wandb_run() rather than
reimplementing its _meta-building logic inline. A test that rebuilds the
function's logic in parallel can't catch a bug *inside* that function --
a typo in a dict key, a broken cfg access -- it would just reproduce the
same bug and still pass. Calling the real thing is what actually proves
the 7 fields land in the run's config.
"""

import os

os.environ.setdefault("WANDB_MODE", "disabled")  # must be set before wandb.init() runs

import dataclasses

import wandb

from robot_policy_lab.utils.logging import get_device_profile, init_wandb_run


# FakeCfg is shaped like the REAL TrainPipelineConfig, not like the plan's
# original draft. dataset.repo_id and wandb.project mirror real,
# draccus-bound fields. There is no dataset_dvc_path or dataset_name field
# here -- that absence is the point. If init_wandb_run() ever regresses to
# reading cfg.dataset_dvc_path directly, this test fails with a real
# AttributeError instead of quietly passing.

@dataclasses.dataclass
class FakeDatasetConfig:
    repo_id: str = "lerobot/pusht"


@dataclasses.dataclass
class FakeWandbConfig:
    enable: bool = True
    project: str = "robot-policy-lab"


@dataclasses.dataclass
class FakeCfg:
    seed: int = 42
    steps: int = 500
    batch_size: int = 64
    dataset: FakeDatasetConfig = dataclasses.field(default_factory=FakeDatasetConfig)
    wandb: FakeWandbConfig = dataclasses.field(default_factory=FakeWandbConfig)


def test_device_profile():
    profile = get_device_profile()
    assert "device_type" in profile
    assert profile["device_type"] in ("mps", "cuda", "cpu")
    assert "torch_version" in profile
    print(f"test_device_profile ✓  device={profile['device_type']}")


def test_init_wandb_run():
    cfg = FakeCfg()
    dvc_path = "../month-01-robot-data-forge/outputs/metadata.parquet"

    init_wandb_run(cfg, dvc_path=dvc_path)

    meta = dict(wandb.run.config["_meta"])

    assert meta["git_hash"] != "unknown",                       "git_hash missing"
    assert not meta["dataset_dvc_hash"].startswith("unknown"),  "dvc_hash missing"
    assert len(meta["config_hash"]) == 8,                        "config_hash wrong length"
    assert meta["device_type"] in ("mps", "cuda", "cpu"),        "device_type missing"

    print("test_init_wandb_run ✓")
    print(f"  git_hash:         {meta['git_hash'][:12]}...")
    print(f"  dataset_dvc_hash: {meta['dataset_dvc_hash']}")
    print(f"  config_hash:      {meta['config_hash']}")
    print(f"  device_type:      {meta['device_type']}")

    wandb.finish()


if __name__ == "__main__":
    test_device_profile()
    test_init_wandb_run()
    print("\nAll W&B logging tests passed ✓")
