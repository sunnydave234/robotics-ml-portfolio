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

## W2D3 — Production-grade W&B logging + draccus strict-schema correction

Two separate bugs surfaced today, in order of discovery — the second was bigger
than the first and is the one that actually cost debugging time.

**Bug 1 — cfg attribute access.** init_wandb_run(cfg, *, dvc_path: str) —
dvc_path passed explicitly, NOT read from cfg. dataset name is NOT a separate
param — read from cfg.dataset.repo_id, which IS a real, draccus-bound field.
dataset_dvc_path / dataset_name are NOT declared fields on TrainPipelineConfig,
so cfg.dataset_dvc_path would AttributeError even if the JSON parsed cleanly.

**Bug 2 — NameError, Python scoping.** cfg only exists inside the function
draccus.wrap() calls — nothing at module level, and nothing before that
function is invoked, can reference it. Hit as `NameError: name 'cfg' is not
defined` when seed_everything(cfg.seed) and init_wandb_run(cfg, ...) were
left at module scope in scripts/train.py instead of inside main(cfg). Fix:
moved both calls inside main(), after configure_mps_env() (module-level,
correctly has no cfg dependency) and after draccus resolves cfg.

**Bug 3 — draccus is strict, not lenient. Bigger than Bug 1, and invalidates
a W2D2 assumption that had gone unverified for two weeks.** "Add custom
fields directly to conf/pusht_act.json" (W2D2 note above) was wrong.
draccus.wrap() validates every top-level JSON key against TrainPipelineConfig's
declared fields and rejects the ENTIRE file — not per-field — if even one key
doesn't match. Confirmed live:
  DecodingError: The fields eval_freq, dataset_parquet_path, dataset_stats_path,
  dataset_dvc_path are not valid for TrainPipelineConfig
It does not silently ignore unknown keys. This had been sitting latent since
W2D2 because nothing had actually run conf/pusht_act.json through
draccus.wrap() with those custom fields present until this exact session.
Fix: those three custom paths now live in robot_policy_lab/paths.py
(env-var overridable, same convention as Month 1's config.py) — never in
conf/pusht_act.json. eval_freq was removed outright; it was never a real
field (see confirmed field list below).

**Confirmed full TrainPipelineConfig field list** (via
`dataclasses.fields(TrainPipelineConfig)`):
  dataset, env, policy, reward_model, output_dir, job_name, resume, seed,
  cudnn_deterministic, num_workers, batch_size, prefetch_factor,
  persistent_workers, steps, env_eval_freq, log_freq, eval_steps,
  max_eval_samples, tolerance_s, save_checkpoint, save_freq,
  use_policy_training_preset, optimizer, scheduler, eval, wandb, peft,
  sample_weighting, rename_map, checkpoint_path

No top-level eval_freq — ever was one. Real fields: env_eval_freq
(rollout/sim eval cadence, default 20000) and eval_steps (held-out loss
eval, currently 0/off).

**Note for W2D4:** save_checkpoint, save_freq, checkpoint_path, resume are
all real, native TrainPipelineConfig fields. LeRobot's own lerobot_train.py
almost certainly already checkpoints model/optimizer/scheduler/step —
verify what it saves before writing a checkpoint function from scratch.

500-step live run confirmed successful:
  https://wandb.ai/sunnydave234-student/robot-policy-lab/runs/hr4fnmb8 (absurd-lake-1)

_meta fields confirmed live in the W&B Config panel:
    git_hash:         51d65e54c2cf218a8f734acf6d9791795aaef63a
    dataset_dvc_hash: 4cb77b4b4ba3ac54e4edcb70b6d4aeba
    config_hash:      892531ac
    device_type:      mps
    platform:         macOS-26.5.1-arm64-arm-64bit
    torch_version:    2.11.0

dataset_dvc_hash matches W2D2's confirmed hash (4cb77b4b...) exactly —
same metadata.parquet, byte-identical, DVC lineage confirmed stable across
sessions. git_hash and config_hash both differ from W2D2's values, and
that's expected, not a bug: more commits landed between W2D2 and this run
(different git_hash), and this run's actual cfg — with a --steps=500 CLI
override, live TrainPipelineConfig — has different content than whatever
produced W2D2's 67c34877 (different config_hash). Same data, different
code and config, and the three hashes correctly say so independently.

scripts/train.py is still a config-resolution debug harness, no training
loop — train/loss, eval, and checkpoint-artifact logging remain unverified
until the real training loop exists.

## W2D4 — Checkpointing (carry into all future sessions)

- Shape verdict: HYBRID. LeRobot native checkpoint (random_utils.py) saves
  model/opt/sched/step + python/numpy/torch-cpu/torch-cuda RNG as
  rng_state.safetensors — MORE than plan assumed. It does NOT save torch.mps
  RNG (no MPS branch in serialize_torch_rng_state, line 85-88). Native
  resume on MPS is therefore not bit-exact for stochastic ops (dropout,
  ACT CVAE noise). Verified at pinned 2d7a420.
- Future sidecar for real runs needs ONLY: torch_mps RNG + lineage
  (git/dvc/config hashes, wandb_run_id). Everything else is LeRobot's.
- torch.mps RNG API confirmed on torch 2.11: get_rng_state / set_rng_state.
- Real line numbers: save_checkpoint() call at lerobot_train.py:643 (held);
  resume path at 375 (plan said 380). Line 408: DataLoader resume is
  sample-exact by design — no fast-forward machinery needed, ever.
- torch.load needs weights_only=False for our checkpoint (pickled py/np
  RNG tuples) — torch 2.11 defaults to True and raises otherwise.
- Harness model now includes nn.Dropout(p=0.1) — without an RNG consumer
  the resume test passes even with RNG restore deleted (vacuous).
- ★ PR-shaped: MPS branch in random_utils.py serialize/deserialize —
  mirrors the existing CUDA branch exactly. Check current main before filing.

## W2D5 — Real training integration: dataset swap, meta reuse, W&B fix, optimizer bug (complete)

**checkpoint.py bug (fixed first, unrelated to the rest of today):**
load_checkpoint() read ckpt["shceduler_state_dict"] (typo) while
save_checkpoint() writes ckpt["scheduler_state_dict"]. One-line fix,
re-verified against test_checkpoint_resume.py.

**Real dataset construction hook — confirmed via grep, not assumed:**
make_train_eval_datasets is imported into lerobot_train.py from
lerobot.datasets.factory (line 51), called at lines 249 (main process)
and 255 (non-main-process, distributed only — irrelevant on single Mac).
patch.object(lerobot_train_module, "make_train_eval_datasets", ...) is
the correct target — imports bind a local name in the importing module,
and that's what the calling code references, regardless of where the
function is actually defined.

**Entry point mechanics — confirmed via direct grep:**
train(cfg) is decorated `@parser.wrap()` — NOT `@draccus.wrap()`.
`parser` is LeRobot's own name for their draccus-based wrapper
(`from lerobot.configs import parser`, line 48). Grepping literally for
"draccus" in lerobot_train.py returns nothing — caused real confusion
mid-session. main() is just:
    def main():
        register_third_party_plugins()
        train()
scripts/train_integration.py replicates this manually — calls
register_third_party_plugins() then train() directly, inside a
patch.object() context manager around make_train_eval_datasets — since
main() itself can't be called with our patch active around it.

**RobotForgeAdapter needed 5 attributes beyond __getitem__ before real
training would run — discovered one crash at a time, should be found in
one grep pass next time:**

    .meta                       — AttributeError, make_policy() needs it
                                   (policies/factory.py:518,536,548)
    .num_frames                 — AttributeError, lerobot_train.py:395
    .num_episodes                — AttributeError, lerobot_train.py:396
    .episodes                    — AttributeError, lerobot_train.py:413
    .absolute_to_relative_idx    — AttributeError, lerobot_train.py:417

Fix for .meta: reuse the ORIGINAL dataset's .meta directly. The patch
still calls _original_make_datasets(cfg) first (gives us LeRobot's normal
eval_dataset too) — both datasets are the same underlying lerobot/pusht
data, ours just reads from our own HDF5 store instead of LeRobot's cache,
so features/stats/camera_keys are identical by construction, including
ImageNet image-normalization stats already injected by use_imagenet_stats
(datasets/factory.py:128-131, 205-209) — confirmed via a real
dataset.meta.stats["observation.image"] lookup before trusting the reuse.

Fix for the other four: 4 new @property methods on RobotForgeAdapter
(between __init__ and __len__):
  num_frames               -> len(self)
  num_episodes              -> len(self.ep_lengths)
  episodes                  -> None  (no episode subsetting — confirmed
                                      real LeRobotDataset also returns
                                      None under the same conditions)
  absolute_to_relative_idx  -> None  (no distributed sharding — only
                                      populated on the non-main-process
                                      path, lerobot_train.py:253.
                                      ⚠ MUST REVISIT Week 3 real DDP.)

**LESSON for next adapter-writing session:**
    grep -n "dataset\.\w\+" "$LR_TRAIN" | sort -u -t. -k2
Run this BEFORE writing an adapter, not after 5 separate crashes — returns
the full attribute contract in one shot. A pre-training conformance check
(assert hasattr(ds, attr) for attr in REQUIRED_ATTRS) is the right
permanent fix — not yet built as a real file, worth adding before Week 3.

**adapter.py path-doubling bug, separate from today's other work:**
The SHALLOW→DOUBLED rewrite from W2D1 wasn't idempotent. build_index.py
had been re-run since W2D1 and started writing already-doubled paths —
the old unconditional .replace() call then tripled the segment on top of
an already-correct path. Fixed:
    DOUBLED = "robotics-ml-portfolio/robotics-ml-portfolio/month-01-robot-data-forge"
    p if DOUBLED in p else p.replace(SHALLOW, DOUBLED)
Same idempotency principle as the `done` sentinel in ingest.py and the
DVC pipeline — a recurring pattern worth remembering, not a one-off fix.

**W&B _meta fields — silent absence, not a crash, only caught by manually
reading the Config panel:**
LeRobot's own WandBLogger.__init__ (common/wandb_utils.py:99) calls
wandb.init() at lerobot_train.py:226 — BEFORE make_train_eval_datasets
runs (line 249). It logs cfg.to_dict() with zero _meta fields.
init_wandb_run() (W2D3, robot_policy_lab/utils/logging.py) is correct and
unit-tested but was NEVER called by the real entry point — LeRobot has
its own native wandb integration with no knowledge our custom one exists.
Undetected through an entire successful-looking 1000-step run (loss
decreasing, checkpoints saving) — a missing _meta field never crashes
anything.
Fix: inside patched_make_train_eval_datasets, after confirming
wandb.run is not None, push _meta via:
    wandb.config.update({"_meta": {...}}, allow_val_change=True)
NOT wandb.init() again — would either error (run already live) or create
an orphaned second run. Confirmed working across 5 separate runs today
(1×1000-step, 4×sweep): all 6 _meta fields present every time, git_hash
cross-verified against W&B's own independent "Git state" field (both show
4e05bc1...), dataset_dvc_hash held at 4cb77b4b... across every single run
regardless of which code path built it.

**optimizer.lr CLI override silently does nothing when
use_policy_training_preset=True (the default) — found via the sweep,
would NOT have been caught by any single-run smoke test:**
TrainPipelineConfig.__post_init__ (configs/train.py:210-213):
    elif self.use_policy_training_preset and not self.resume:
        self.optimizer = active_cfg.get_optimizer_preset()
        self.scheduler = active_cfg.get_scheduler_preset()
Runs AFTER draccus applies CLI flags. --optimizer.lr=X sets
cfg.optimizer.lr correctly, then gets silently discarded and rebuilt from
cfg.policy.optimizer_lr instead (configuration_act.py:126,155) — a
completely separate field your flag never touched.

CORRECT FLAG for ACT sweeps: --policy.optimizer_lr=X, not --optimizer.lr=X.

Confirmed: first sweep attempt showed lr:1.0e-05 (JSON default) in ALL 4
runs regardless of --optimizer.lr value passed. Corrected sweep with
--policy.optimizer_lr produced 2 distinct lr values across seeds,
confirmed independently via config dump, training-loop log line, AND
W&B Config panel (three sources agreeing).

Untested alternative: --use_policy_training_preset=false might also make
--optimizer.lr take effect directly, per the __post_init__ branch logic —
not verified, don't state as fact.

**Sweep result — a real finding, not just a mechanics check:**
At 200 steps: lr=1e-4 -> loss ~2.83-2.93 (both seeds, consistent).
lr=3e-4 -> loss ~5.06-5.33 (both seeds, consistent). Pattern holds across
both seeds — not noise. 3e-4 may be too aggressive this early in
training. Revisit at Week 4 alongside the ACT/mining/DiffusionPolicy
comparison — this is a real data point, not just a mechanics check.

**⚠ Process lesson — cost real debugging time:**
A `python3 << 'EOF' ... EOF` heredoc pasted into the terminal silently
truncated scripts/train_integration.py mid-edit (cut the entire
if __name__ == "__main__": block AND the RobotForgeAdapter construction
inside patched_make_train_eval_datasets). File remained valid Python — it
just ended after a complete statement — so `python -m py_compile` passed
clean, and running it produced exit code 0 with ZERO output (defining a
function ≠ calling it). Looked identical to several other real failure
modes investigated first (PYTHONPATH, a local wandb/ folder shadowing the
real package, a hanging import) and took multiple rounds to isolate.
RULE: after ANY heredoc-based file edit, verify with BOTH
`python -m py_compile <file>` AND `cat -n <file> | tail -20` to confirm
the file actually ends where expected. Clean compile is necessary but not
sufficient proof the edit landed completely. Prefer single-block
str_replace-style edits over multi-line heredocs.

**Final confirmed state (all fixes applied, all verified):**
1000-step run: loss 4.262 (step 200) -> 0.712 (step 1000), 6.40 steps/sec
(matches W1D3 MPS baseline exactly), checkpoints at step 500 and 1000.
4-run sweep: correct seed AND lr differentiation confirmed three ways.
All 6 W&B _meta fields present on every run. dataset_dvc_hash stable at
4cb77b4b4ba3ac54e4edcb70b6d4aeba across every run this session.

**Still outstanding:**
- MPS RNG PR (random_utils.py, flagged since W2D4) — now flagged 5
  sessions running. Needs an explicit decision (file it vs. consciously
  defer past Month 2) before Week 3 Terraform work consumes attention.
- check_adapter_conformance.py — discussed, not yet built as a real file.
  Worth adding before writing any future adapter-style code.
