# robotics-ml-portfolio

A 3-month build toward robotics ML and data infrastructure roles — going from
zero robotics experience to a trained policy, a self-provisioned distributed
GPU training run, and (if the timing works out) inference running on a real
robot arm.

Background: Senior Data Engineer (Python, PySpark, AWS, distributed systems,
NLP pipelines) applying that systems experience to robot learning data and
training infrastructure.

## The arc

| Month | Project | Status |
|---|---|---|
| 1 | [`month-01-robot-data-forge`](./month-01-robot-data-forge) | ✅ Complete |
| 2 | `month-02-robot-policy-lab` | 🚧 In progress |
| 3 | `month-03-edge-policy` | ⏳ Not started |

## Month 1 — robot-data-forge ✅

A production-shaped robot learning data pipeline built on the LeRobot
`pusht` dataset (206 episodes, 25,650 frames): HDF5 ingestion → validation →
Parquet metadata index → PyTorch `Dataset`/`DataLoader` (benchmarked to
~26K samples/sec) → weighted episode sampling → a behavioral-cloning MLP
baseline with W&B experiment tracking → a DVC pipeline with full
reproducibility. Full details in the [project README](./month-01-robot-data-forge).

## Month 2 — robot-policy-lab 🚧

Training ACT and Diffusion Policy on the Month-1 data engine, then
provisioning real multi-GPU infrastructure with Terraform to run
distributed (DDP) training — provision, train, profile, tear down.

## Month 3 — edge-policy ⏳

Quantizing a trained policy for edge inference on a Raspberry Pi 5, with a
fleet data collector closing the loop back into the Month-1 pipeline. Capstone:
real sim-to-real training and deployment on an SO-101 robot arm, if it arrives
in time.

## Why this exists

Robotics companies care about the training-data engine and the
infrastructure around it as much as the model itself. This portfolio is
built to demonstrate both: the data pipeline a model actually learns from,
and the infrastructure that trains and deploys it at scale.

## Real training with RobotForgeAdapter

Wires Month-1 HDF5 data into LeRobot's actual training entry point
(`lerobot_train.train()`), not the config-inspection debug harness
(`scripts/train.py`).

```bash
PYTHONPATH=$(pwd) python scripts/train_integration.py \
  --config_path=conf/pusht_act.json \
  --steps=1000 \
  --save_freq=500 \
  --seed=42 \
  --wandb.project=robot-policy-lab-test
```

**Before running:** activate `~/envs/robotics-policy-lab` and confirm
`robot_policy_lab/paths.py`'s `DATASET_PARQUET_PATH`, `DATASET_STATS_PATH`,
and `DATASET_DVC_PATH` resolve correctly (env-var overridable if your
layout differs).

**Expect to see, in order:**
1. `[RobotForge] _meta fields pushed into live W&B run`
2. `[RobotForge] Replacing default train dataset (25650 samples) with RobotForgeAdapter`
3. Standard LeRobot training log lines, loss decreasing
4. A W&B run with all 6 `_meta` fields (`git_hash`, `dataset_dvc_hash`, `config_hash`, `device_type`, `platform`, `torch_version`) in the Config panel
5. Checkpoints on disk under `outputs/train/<date>/<time>_act/` at every `save_freq` interval

### Hyperparameter sweep

Bash loop — no Hydra multirun in this system (draccus + flat JSON only).

```bash
for seed in 42 43; do
  for lr in 1e-4 3e-4; do
    PYTHONPATH=$(pwd) python scripts/train_integration.py \
      --config_path=conf/pusht_act.json \
      --seed=$seed \
      --policy.optimizer_lr=$lr \
      --steps=200 \
      --wandb.project=robot-policy-lab-test &
  done
done
wait
```

**⚠ Use `--policy.optimizer_lr`, not `--optimizer.lr`.** When
`use_policy_training_preset=True` (the default), LeRobot rebuilds the
optimizer from `cfg.policy.optimizer_lr` after CLI flags are applied —
`--optimizer.lr` gets silently overwritten and has no effect.