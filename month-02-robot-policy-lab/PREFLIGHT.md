# Month 2 Preflight — robot-policy-lab

## W1D1 Findings

- Pinned commit: 2d7a42011a4f8e05a8c85d5fb908da258d4cc7b1
- Python: 3.12.12 (separate venv: ~/envs/robotics-policy-lab).
  Month 1 env (3.11.9) untouched.
- CLI: lerobot-train --policy.type=act --dataset.repo_id=<data> --policy.repo_id=<hub target>
  Both repo_ids required and distinct: dataset.repo_id = training data source,
  policy.repo_id = hub push target (required by cfg.validate() even if not pushing).
- Config system: Python dataclasses (ACTConfig), not Hydra YAML.
  Location: src/lerobot/policies/act/configuration_act.py
- ACTConfig defaults (line 85-86): chunk_size=100, n_action_steps=100
- Validation (__post_init__, line 138-146): n_action_steps must be 1 if temporal
  ensembling is enabled; n_action_steps must be <= chunk_size.
- MPS: auto-detected ("Device 'None' is not available. Switching to 'mps'") —
  did NOT silently fall back to CPU as expected. Training ran successfully,
  lo.595 -> 0.397 over 7.3K/100K steps, no MPS op errors.
- Throughput baseline (MPS, batch_size=8): ~202 samples/sec, ~25 steps/sec.
  Compare against Week 3 4xT4 DDP numbers.
- Missing extra: 'accelerate' not included by default — required `pip install -e ".[training]"`.
- Known warning (non-blocking): objc duplicate libavdevice symbol conflict between
  `av` package's bundled libavdevice.61 and Homebrew ffmpeg's libavdevice.62.
  Did not crash training; flagged as possible future instability source.
- Week 2 contract: RobotForgeAdapter.__getitem__ must return action[t : t+n_action_steps].

## W1D2 — Config system: draccus, not Hydra

- Confirmed: draccus ChoiceRegistry entries (`PreTrainedConfig._choice_registry`)
  are populated by decorator side-effect on import — `"act"` is NOT resolvable
  until `lerobot.policies.act.configuration_act` has been imported somewhere
  in the process. Fixed in scripts/train.py with explicit import before
  draccus.wrap() entry point.
- Settings file format is JSON (`--config_path=conf/pusht_act.json`), not YAML.
  Precedence: file defaults < CLI flags. No Hydra multirun — sweeps need bash loop.
- TrainPipelineConfig has NO top-level `device` field. Device lives at
  cfg.policy.device — auto-resolves to 'mps' with zero code, confirmed live
  in smoke test: 'device': 'mps' in full config dump.
- Flat field names confirmed from real source + live dump: `steps` (not
  `training.offline_steps`), `wandb.enable` (not a shorthand), `batch_size`,
  `seed` — all top-level on TrainPipelineConfig, not nested.
- device.py design decision: TWO separate functions, not one.
  configure_mps_env() sets env vars + global torch state (real gap — LeRobot
  doesn't set these). get_device() is a pure standalone resolver for code
  outside TrainPipelineConfig. Side effects must not be hidden inside a getter.
  configure_mps_env() must be called before draccus parses config — before
  any tensor/model creation.
- Files created W1D2: conf/pusht_act.json, scripts/train.py,
  robot_policy_lab/__init__.py, robot_policy_lab/utils/ (empty, ready for
  device.py tomorrow). Committed: 457ca1a.

## W1D3 - Source path correction (carry into all future sessions)

500-step smoke test: PASSED
- Loss trajectory: 5.157 (step 200) → 2.140 (step 400) — clear decrease, correct
- Throughput (MPS, batch_size=64): ~6.4 steps/sec, ~374–411 smp/s
- Note: steps/sec lower than W1D1 (~25) because batch_size=64 vs 8 — smp/s is the fair metric

MPS vs CPU benchmark (100 steps, batch_size=64):
- MPS:  23.95s wall → 6.40 steps/sec,  ~374 smp/s
- CPU: 175.55s wall → 0.59 steps/sec,  ~38 smp/s
- MPS speedup: 10.8× over CPU
- Reference for Week 3: 4×T4 DDP target is >10× over this MPS baseline

Post-training 403 root cause: push_to_hub=True in config, HF token read-only.
Fix: added push_to_hub=false + repo_id to conf/pusht_act.json permanently.
libavdevice objc warning: non-blocking, already flagged W1D1, unchanged.

- LeRobot source is at src/lerobot/, not lerobot/common/
- Any plan referencing lerobot/common/policies/ or lerobot/common/datasets/
  needs the prefix corrected to src/lerobot/policies/ and src/lerobot/datasets/
- Applies to W1D4 plan: modeling_act.py is at src/lerobot/policies/act/modeling_act.py

## W1D4 — Architecture contract (carry into all future sessions)
- ARCHITECTURE.md committed — read it before any Week 2 adapter work
- lerobot_train.py is the training entry point (not train.py)
- action_is_pad is a required key in every __getitem__ return dict
- Two eval modes: loss-only (is_eval_step) vs rollout (is_env_eval_step)
## W1D5 — Reproducibility
- seed_everything() confirmed working on MPS: identical loss curves across runs
- robot_policy_lab package requires PYTHONPATH set to month-02-robot-policy-lab/ to resolve without install
- Run tests as: PYTHONPATH=$(pwd) python tests/test_reproducibility.py from month-02-robot-policy-lab/

## W2D1 — RobotForgeAdapter (carry into all future sessions)

- `h5py`, `pandas`, `pyarrow` must be installed in `robotics-policy-lab` env — not just Month-1 env.
  Always activate `~/envs/robotics-policy-lab` before running any Month-2 code.
- HDF5 path rewrite in adapter `__init__`: parquet stores shallow paths (missing one `robotics-ml-portfolio/`
  segment). Rewrite applied at load time via `str.replace`. Existence check on `file_paths[0]` in `__init__`
  catches stale paths loudly before training starts.
- Two HDF5 copies on disk — parquet correctly indexes `data/hdf5/`, not `data/` (early ingest artifact).
- Stats keys: `dataset_stats.json["states"]["mean"]` and `["actions"]["mean"]` — plural, confirmed.
- `action_is_pad` shape: `(n_action_steps,)` bool — required by loss fn at `lerobot_train.py:145`.
- Run tests: `PYTHONPATH=$(pwd) python tests/test_adapter.py` from `month-02-robot-policy-lab/` root.
- All 4 tests passing: schema, boundary padding, DataLoader multiprocessing (num_workers=2), total length.
- Commit: `W2D1.1: RobotForgeAdapter — HDF5→ACT contract, action_is_pad, path rewrite, all tests pass`

## W2D2 — Lineage utilities: pre-execution corrections (carry into next session)

Plan was written against Hydra/OmegaConf — three things corrected before execution:

- NO omegaconf. Config system is draccus + Python dataclasses (confirmed W1D2).
  get_config_hash() uses dataclasses.asdict() + json.dumps(), not OmegaConf.to_yaml().
- NO YAML config files. conf/ uses flat JSON (conf/pusht_act.json). No conf/dataset/
  subfolder — that's Hydra config group syntax. Add dataset path fields directly to
  conf/pusht_act.json.
- NO ??? syntax. That's OmegaConf required-field syntax. Not valid in draccus JSON.
- yaml package needed: get_dvc_dataset_hash() uses yaml.safe_load() to read .dvc pointer
  files. Install if missing: pip install pyyaml
- DVC pointer files confirmed from Month-1 W4D1:
    ../month-01-robot-data-forge/data/hdf5.dvc
    ../month-01-robot-data-forge/outputs/metadata.parquet.dvc
  Pass path WITHOUT .dvc suffix to get_dvc_dataset_hash() — function appends it.
  Use metadata.parquet path — that's what RobotForgeAdapter reads.
- /tmp/dvc-cache (Month-1 DVC remote) may have been purged on reboot. If
  get_dvc_dataset_hash() returns "unknown": run from month-01-robot-data-forge/:
  dvc repro && dvc add outputs/metadata.parquet to regenerate the .dvc pointer file.

## W2D2 — Lineage utilities (carry into all future sessions)

- metadata.parquet is a dvc.yaml pipeline output (build_index stage) — NOT a
  standalone dvc add artifact. No metadata.parquet.dvc file exists or should exist.
- get_dvc_dataset_hash() reads from dvc.lock, not a .dvc pointer file.
  Path arg is the full path to the file; function computes dvc_root as
  tracked.parent.parent and searches all stages for the matching output path.
- dvc.lock must stay committed to git — it's the source of truth for dataset MD5s.
  If it goes stale: cd month-01-robot-data-forge && dvc repro && git add dvc.lock && git commit
- Confirmed hashes (W2D2 run):
    git:    1a42f14023d4cda484dc7ffc1474df6842b61bef
    dvc:    4cb77b4b4ba3ac54e4edcb70b6d4aeba
    config: 67c34877