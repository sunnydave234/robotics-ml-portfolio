# Progress Log

---

## Month 1 — robot-data-forge
**Goal:** Production robot learning data engine over Open-X / LeRobot data.  
**Stack:** LeRobot, HDF5, DVC, PyTorch, OpenCV  
**Repo:** [robotics-ml-portfolio/month-01-robot-data-forge](https://github.com/sunnydave234/robotics-ml-portfolio/)

---

### ✅ Week 1 — Get a Robot Episode on Screen
**Completed:** April 27, 2026  
**Deliverable:** Rendered episode MP4 with action overlays, pushed to GitHub.

#### What was built
| File | Description |
|---|---|
| `extract_episode.py` | Extracts one episode as `{images: (T,C,H,W), actions: (T,2)}` CPU tensors |
| `visualize_episode.py` | Renders episode as MP4 with per-frame action overlays |
| `profile_dataset.py` | Computes dataset stats → JSON + histogram PNG |
| `config.py` | Central path config — all scripts read paths from here |
| `scripts/check_episodes.py` | Smoke-tests episode extraction across indices 0, 5, 10, 50, 100 |
| `tests/test_extract.py` | Pytest suite for `extract_episode` |
| `dataset_profile.json` | Committed dataset artifact |

#### Key things learned
- LeRobot V3 API: `ds.meta.episodes` is a pandas DataFrame; `ds.meta.total_episodes` for count
- Episodes return channel-first `(C, H, W)` tensors — require `.permute(1,2,0)` before OpenCV
- `.permute()` breaks memory contiguity — always call `.contiguous()` before `.numpy()`
- Source frames are 96×96px — 4× upscale (`INTER_NEAREST`) required for readable overlays
- `mp4v` is the reliable codec on M4 Mac; `avc1` requires a licensed OpenCV build
- All tensor ops stay in PyTorch until the single conversion boundary: `tensor_to_bgr_frame()`
- Paths centralized in `config.py` — overridable via env vars, no hardcoded strings in scripts

#### Dataset facts (lerobot/pusht)
- 206 episodes, 25,650 total frames, 10 FPS
- Episode length: min=49, max=246, mean=124.5, std=35.7
- Action dim: 2 (x, y end-effector target position)
- Image shape: 3×96×96, float32 [0.0, 1.0]
- Success rate column: 0.0 (sparse reward — not meaningful at this stage)

---

### ✅ Week 2 — Build the Ingestion Pipeline: Raw → Validated HDF5
**Completed:** May 23, 2026  
**Deliverable:** 206 episodes ingested, validated, indexed. One-command pipeline runner. README with architecture diagram. Fresh-clone verified.

#### What was built
| File | Description |
|---|---|
| `write_one_episode.py` | Validated single-episode write pattern — chunked (1,H,W,C), gzip level 4 |
| `ingest.py` | Full ETL loop — idempotent, exception-safe, progress-logged, `done` sentinel |
| `validate.py` | Per-file checks: shape, dtype, NaN, timestamp monotonicity, action/frame parity |
| `build_index.py` | Scans HDF5 root attrs → writes `metadata.parquet` episode index |
| `query.py` | Filters index by task/success/frame count; benchmarks query time |
| `run_pipeline.sh` | One-command runner: `bash run_pipeline.sh lerobot/pusht 206` |
| `config.py` | Added `HDF5_DIR`, `DATASET_NAME`, `episode_hdf5_path()` |
| `requirements.txt` | Pinned deps — fresh-clone verified |

#### HDF5 schema (per episode file `ep_{idx:06d}.hdf5`)
```
/
├── observations/
│   ├── images   (T, H, W, C)  uint8    chunked (1,H,W,C)  gzip-4
│   └── state    (T, 2)        float32
└── actions      (T, 2)        float32
/ [root attrs]   episode_id, task, frame_count, success, done
```

#### Pipeline results
- 206 / 206 episodes ingested, 0 failures
- 56 MB total on disk, 23.6 ep/s (data cached locally)
- `validate.py` — 206 / 206 passed, 0 failures
- `metadata.parquet` — 206 rows, query time <100ms
- Idempotency verified — second ingest run skips all 206 in <1s
- Fresh-clone verified: `git clone → pip install -r requirements.txt → bash run_pipeline.sh lerobot/pusht 5`

#### Key things learned
- HDF5 `done` sentinel attr: safer than checking file existence — guards against mid-write crashes
- `set -euo pipefail` in shell scripts: aborts the pipeline on any stage failure, same as a checked exception chain
- Chunk shape `(1, H, W, C)`: optimized for sequential frame reads — one chunk I/O per frame
- `gzip` level 4: good compression/speed tradeoff for uint8 image data; level 9 is rarely worth the CPU
- `metadata.parquet` as a lightweight index: avoids opening HDF5 files to filter by episode attributes — same pattern as an OpenSearch metadata layer over raw records
- `build_index.py` runs after `validate.py`: index only reflects episodes that passed validation

---

### ✅ Week 3 — PyTorch Dataset + DataLoader over HDF5
**Goal:** `RobotEpisodeDataset(Dataset)` → normalization → DataLoader benchmark
**Completed:** 06/09/2026
| File | Description |
|---|---|
| `benchmark_dataloader.py` | 16 configs × 100 batches, end-to-end throughput timing |
| `profile_dataloader.py` | PyTorch profiler on workers=4, bs=64 → Chrome trace |
| `outputs/benchmark_results.json` | Full benchmark results |
| `outputs/dataloader_trace.json` | Open at chrome://tracing |

**Benchmark results (lerobot/pusht, 25,650 samples, normalize=True):**

| num_workers | pin_memory | batch_size | samples/sec | p95 batch (ms) |
|-------------|------------|------------|-------------|----------------|
| 0 | False | 16 | 5,307 | 3.0 |
| 0 | False | 64 | 5,375 | 11.9 |
| 0 | True | 16 | 5,331 | 3.0 |
| 0 | True | 64 | 5,409 | 11.8 |
| 2 | False | 16 | 8,464 | 1.9 |
| 2 | False | 64 | 9,402 | 6.8 |
| 2 | True | 16 | 8,292 | 1.9 |
| 2 | True | 64 | 9,387 | 6.8 |
| 4 | False | 16 | 13,998 | 1.1 |
| 4 | False | 64 | 15,931 | 4.0 |
| 4 | True | 16 | 14,037 | 1.1 |
| 4 | True | 64 | 15,681 | 4.1 |
| 8 | False | 16 | 20,973 | 0.8 |
| 8 | False | 64 | 25,831 | 2.5 |
| 8 | True | 16 | 21,019 | 0.8 |
| **8** | **True** | **64** | **25,905** | **2.5** |

Key findings:
- `pin_memory` effect: <1% — confirmed no-op on M4 unified memory
- `num_workers` scaling: ~1.7× per doubling (0→2→4→8) — healthy parallel I/O curve
- Bottleneck: HDF5 reads + gzip decompression in worker subprocesses (93.7% of profiled CPU time)
- High throughput caveat: 56MB dataset fits in macOS page cache after first pass — numbers reflect RAM reads, not NVMe. At 10,000+ episodes this curve flattens as I/O bus saturates.
- `persistent_workers=True` essential on macOS — avoids repeated spawn overhead between configs

#### W3D4 — WeightedEpisodeSampler (complete)
| File | Description |
|---|---|
| `add_weights.py` | Adds `weight` column to metadata.parquet (1.0 success / 0.3 failure) |
| `weighted_sampler.py` | WeightedEpisodeSampler — frame-level weights via np.repeat + torch.multinomial |

- Verified: 10,000-draw sampler test matches expected episode-level distribution
- pusht quirk: all 206 episodes have success=0.0 → uniform weight=0.3 → uniform sampling (correct, not a bug)
- Wired into DataLoader via batch_sampler — 5 batches iterated successfully, batch shapes correct
- This is the placeholder architecture for Month 2 hard-example mining: updating the `weight` column from eval results requires zero training-loop changes

#### W3D5 — visualize_batch.py + context_window (complete)
| File | Description |
|---|---|
| `visualize_batch.py` | Loads one DataLoader batch (`normalize=False`), renders 2×4 grid — image + `(x,y)` action bar chart per sample → `outputs/batch_visualization.png` |
| `robot_dataset.py` | Added `context_window: int = 1` param — `K>1` returns `(K,H,W,C)` image stack (no permute, matches HDF5 on-disk layout); `K==1` path unchanged |

- Boundary padding verified: `frame_offset=0, K=3` → `[frame0, frame0, frame0]`; `frame_offset=1, K=3` → `[frame0, frame0, frame1]` — repeat-pads the earliest available frame, never crosses episode boundaries
- `context_window=1` (default) fully backward-compatible — no existing callers affected
- README updated: benchmark table, `batch_visualization.png`, Dataset/DataLoader usage snippet (incl. `context_window` example)
- Fresh-install check passed: `python visualize_batch.py` from a clean clone reproduces `outputs/batch_visualization.png`

---

### Week 3 — Final State (handoff to Week 4)

**`RobotEpisodeDataset` final interface:**
```python
RobotEpisodeDataset(
    parquet_path: str | Path,   # positional — always OUTPUTS_DIR / "metadata.parquet"
    normalize: bool = False,    # z-score action/state via outputs/dataset_stats.json
    context_window: int = 1,    # K>1 -> image is (K,H,W,C), frame-0 padded at episode bounds
)

# __getitem__ returns:
{
    "image":        (C,H,W) float32 [0,1]   # or (K,H,W,C) if context_window > 1
    "action":       (2,) float32             # pusht x/y end-effector target, raw or normalized
    "state":        (2,) float32             # placeholder — same as action for pusht
    "episode_idx":  int
    "frame_offset": int
}
```

**Key files (Month 1, current state):**
- `config.py` — `OUTPUTS_DIR`, `HDF5_DIR`, `DATASET_NAME`, path helpers
- `write_one_episode.py`, `ingest.py`, `validate.py` — Week 2 ETL
- `build_index.py` → `outputs/metadata.parquet` (206 episodes)
- `compute_stats.py` → `outputs/dataset_stats.json` (Welford action/state mean/std)
- `robot_dataset.py` — `RobotEpisodeDataset` (normalize, context_window)
- `weighted_sampler.py` — `WeightedEpisodeSampler` (weight col: 1.0 success / 0.3 unsuccessful)
- `benchmark_dataloader.py`, `profile_dataloader.py` — peak 25,905 samples/sec @ workers=8, bs=64
- `visualize_batch.py` → `outputs/batch_visualization.png`

**Known quirks carried into Week 4:**
- All 206 pusht episodes have `success=0.0` — sampler weights are uniform (0.3) by design, not a bug
- `pin_memory` confirmed no-op on M4 unified memory
- `context_window > 1` reads K separate HDF5 chunks (chunk shape `(1,H,W,C)`) — throughput will be lower than the W3D3 benchmark table for K>1; not yet re-benchmarked

---

### ✅ Week 4 — DVC Pipeline + End-to-End Reproducibility
**Goal:** `dvc.yaml` (ingest → validate → build_index → compute_stats) + `reproduce.sh` + trivial MLP behavior-cloning baseline + polished README

#### W4D1 — DVC init + track existing artifacts (complete)
| File | Description |
|---|---|
| `.dvc/config` | Local remote `local_remote` → `/tmp/dvc-cache` (throwaway — proves mechanics only) |
| `data/hdf5.dvc` | Pointer for HDF5 store (206 episodes) |
| `outputs/metadata.parquet.dvc` | Pointer for Parquet index |

- `dvc init` + `dvc remote add -d local_remote /tmp/dvc-cache`
- Tracked `data/hdf5` and `outputs/metadata.parquet` with `dvc add` → `.dvc` pointers committed to Git, raw bytes auto-added to `.gitignore`
- Lineage proven: `dvc push` → `rm -rf data/hdf5` + `rm outputs/metadata.parquet` → `dvc pull` restored 206 episodes byte-identical → `dvc status` clean
- Kept small JSON stats (`dataset_stats.json`, `benchmark_results.json`) in Git, not DVC — text diffs well; only binary/large artifacts (Parquet, HDF5) go to DVC

#### Key things learned
- `.dvc` pointer = S3 object pointer; DVC remote = the bucket; MD5 hash = the content-addressable join key between Git pointer and remote bytes
- `dvc add` auto-appends to `.gitignore` — but it's a no-op for files Git *already* tracks; `git rm --cached` first if so
- Plan's `data/` / `metadata.parquet` shorthand ≠ actual paths: real layout is `data/hdf5` and `outputs/metadata.parquet`
- macOS purges `/tmp` on reboot — fine for this throwaway remote, never for a real one

### ✅ W4D2 — dvc.yaml pipeline + DAG
- `dvc.yaml`: 4 stages — ingest, validate, build_index, compute_stats
- DAG shape (verified via `dvc dag`): ingest → {validate, build_index}; build_index → compute_stats
  - validate is a leaf (nothing downstream reads validation_report.json — by design)
- Resolved two "output already tracked by SCM" conflicts: validation_report.json and
  dataset_stats.json were plain git-tracked files from W2/W3; both `git rm --cached`'d
  and are now DVC-managed pipeline outputs
- `dvc repro` clean, `dvc.lock` committed
- Remote is still /tmp/dvc-cache (W4D1 throwaway) — `dvc push` does nothing useful;
  pipeline reproducibility (not dvc pull) is the safety net
- Pending: cache-invalidation experiments (edit ingest.py vs edit build_index.py,
  observe what reruns)
---

### ✅ W4D3 — BC smoke test + W&B logging (complete)
| File | Description |
|---|---|
| `model.py` | BCMLP: 3-layer MLP, input_dim=4 (state+prev_action), hidden=256, output_dim=2 |
| `bc_dataset.py` | BCDataset: wraps RobotEpisodeDataset, adds prev_action with episode-boundary repeat-pad |
| `train.py` | 10-epoch BC training loop, MSE loss, Adam, W&B logging, best-checkpoint artifact |

W&B run: https://wandb.ai/sunnydave234-student/robot-data-forge/runs/c1o22yul

Loss curve:
- train/loss: drops sharply epoch 1 → near-zero by epoch 2, stays flat
- val/loss: bottoms at 0.000010 by epoch 4, mild noise after (0.000093 final)

Key finding: val/loss near zero is expected — pusht writes actions into both
/observations/state and /actions, so model input contains the target. The MLP
learned a near-identity mapping. Confirms pipeline correctness, not model quality.
pin_memory confirmed no-op (M4 unified memory). MPS backend confirmed working.

## Month 2 — robot-policy-lab
### ★ LeRobot PR — fix(cameras): fix mypy type errors in cameras module
**Date:** June 20, 2026 (during Month 2 preflight / Week 1)
**PR:** https://github.com/huggingface/lerobot/pull/3839
**Status:** Open, awaiting review

#### Context
While doing preflight for Month 2 — checking LeRobot's CLI and install extras —
went looking for the LeRobot PR opportunity the roadmap had slotted for Week 4+.
Found `#1724` ("Ensure the cameras module passes MyPy type checks"), closed by
stale-bot in February with two prior unmerged attempts (`#1788`, `#2036`).
Verified directly against current `main` rather than trusting the closed status:
`mypy src/lerobot/cameras/` failed with 15 real errors. Decided to take it on
immediately rather than wait for the planned Week 4 slot.

#### What was wrong (root-caused, not guessed)
- **`camera_zmq.py` / `image_server.py`** — `pyzmq`'s own `zmq/__init__.pyi`
  re-exports its `sugar` submodule via a bare `from .sugar import *`, which
  mypy can't statically expand into concrete names. Runtime `import zmq` works
  fine (Python's real star-import resolves it); mypy's static analysis can't.
  Confirmed via a minimal repro isolated from the project entirely.
- **`camera_opencv.py`** — `cv2.VideoWriter_fourcc` is a real runtime
  attribute with zero stub coverage in `opencv-python`'s `.pyi` files (verified
  directly — no `.pyi` in the package even mentions it).
- **`camera_opencv.py`** — also found a real (minor) logic bug along the way:
  `self.fps` was being assigned the raw `float` return of
  `cv2.VideoCapture.get()` while declared `int | None` — fixed with an
  explicit `int()` cast, not just suppressed.
- **Both files** — `width`/`height` inherited from `Camera.__init__` weren't
  resolving across the subclass boundary in mypy's inference; fixed with
  explicit class-level type annotations on both `ZMQCamera` and `OpenCVCamera`.

#### Fix approach
- `TYPE_CHECKING`-only import of `zmq.sugar.context.Context` /
  `zmq.sugar.socket.Socket` for the two type annotations mypy couldn't
  resolve through the broken re-export.
- `# type: ignore[attr-defined]` with an explanatory comment (no bare
  ignores, per the contributing guide) for the runtime `zmq.*` / `cv2.*`
  attribute accesses with no valid typed alternative.
- Discovered along the way: a pytest failure in `tests/cameras/` traced back
  to Git LFS pointer files never being pulled in the local clone (fixture
  PNGs were literal `version https://git-lfs...` text, not image data) —
  unrelated to the PR itself, fixed locally with `git lfs install && git lfs pull`.

#### Verification
- `mypy src/lerobot/cameras/` (and via the project's actual `pre-commit`
  mypy hook, which runs stricter than a standalone scoped call): clean,
  0 errors, down from 15
- `pytest tests/cameras/`: 39 passed, 3 skipped, 0 failed
- Did a second pass cleaning up the `type: ignore` comments themselves
  before merge — dropped a vague link that pointed at pyzmq's whole issues
  list instead of anything specific, fixed one truncated comment

#### Key things learned
- A stale-closed GitHub issue is not the same as a resolved bug — always
  verify against current `main` directly before trusting the tracker state
- `pyzmq` and `opencv-python` both have real, verifiable gaps in their type
  stubs despite working fine at runtime — `ignore_missing_imports = true`
  in `pyproject.toml` doesn't cover attribute-level gaps on packages with
  *partial* stub coverage, only packages with *no* stubs at all
- Standalone `mypy <path>` and the project's real `pre-commit` mypy hook can
  give different results — the hook is the source of truth, not a manual
  scoped call, since it may enable stricter checks
- Git LFS pointer files (plain text starting with `version https://git-lfs...`)
  silently masquerade as the real binary file with the same extension —
  `file <path>` is the fast way to tell the difference
- Fork-based contribution workflow: `origin` = your fork (push target),
  `upstream` = the real repo (pull updates from); a repo not existing yet
  shows as "Repository not found" on push, not an auth error

### Week 1 — ACT training on MPS + config system + device verification

#### W1D1 — Install from source + run unmodified train command + find n_action_steps (complete)
**Completed:** June 21, 2026

| File | Description |
|---|---|
| `PREFLIGHT.md` | Commit pin, config system findings, MPS baseline, Week 2 contract |

- Cloned LeRobot from source, pinned commit `2d7a42011a4f8e05a8c85d5fb908da258d4cc7b1`
- Plan was written against the old Hydra/YAML architecture (`lerobot/configs/*.yaml`,
  `pip install -e ".[dev]"`) — corrected mid-session against actual `main`:
  - Config system is now Python dataclasses (`ACTConfig`), not Hydra YAML — no `conf/` to `cat`
  - Source layout is `src/lerobot/`, not `lerobot/`
  - `lerobot-train` is the current CLI; `.[dev]` extra no longer exists
- Python version drift: LeRobot `main` now requires `>=3.12` (Day 0 setup pinned 3.11.9).
  Created new venv `~/envs/robotics-policy-lab` (Python 3.12.12, already available via
  pyenv — no install needed). Month 1 env (`~/envs/robotics`, 3.11.9) untouched.
- `pip install -e .` missing the `[training]` extra by default — `accelerate` not installed,
  required `pip install -e ".[training]"`
- `cfg.validate()` requires both `--dataset.repo_id` (training data source) AND
  `--policy.repo_id` (hub push target) — two distinct, both-required fields, easy to conflate
- Ran the unmodified `lerobot-train --policy.type=act --dataset.repo_id=lerobot/pusht
  --policy.repo_id=<placeholder>` command. MPS was auto-detected
  (`Device 'None' is not available. Switching to 'mps'`) — did **not** silently fall back
  to CPU as the original plan assumed, and did **not** error on MPS either. Training ran
  successfully: loss `6.595 → 0.397` over ~7.3K/100K steps, no MPS op errors. Stopped early
  (proof-of-execution only, not a full run).
- Found `n_action_steps` and `chunk_size` in `src/lerobot/policies/act/configuration_act.py`
  (line 85-86, not a YAML file): both default to `100`. `__post_init__` validation
  (line 138-146): `n_action_steps` must be `1` if temporal ensembling is enabled, and must
  be `<= chunk_size`.

#### Key things learned
- A roadmap/plan doc can drift from upstream reality even within the same project —
  verify against the actual repo (`web_fetch` the GitHub root, grep the real source file)
  rather than trusting cached plan assumptions, same discipline as the LeRobot PR work
- Two Python venvs, one pyenv install: a venv is permanently bound to the interpreter it
  was created from — there's no in-place upgrade. New venv > new minor Python version,
  every time. Old env stays untouched and keeps working.
- ACT's CLI flags map 1:1 onto `ACTConfig` dataclass field names (`--policy.chunk_size=N`)
  — no string-based YAML key matching, fully typed, overridable in Python directly without
  any config-composition system
- `lerobot-train` trains against `LeRobotDataset` fetched straight from the HF Hub by
  default — Month 1's HDF5/Parquet pipeline is not yet wired in. `RobotForgeAdapter`
  (Week 2) is the bridge: must replicate `LeRobotDataset`'s `__getitem__` contract,
  specifically returning `action[t : t+n_action_steps]` slices, not single actions
- MPS throughput baseline (`batch_size=8`, ACT, ResNet-18 backbone): ~202 samples/sec,
  ~25 steps/sec, `updt_s` (gradient step) dominates over `data_s` (data loading) —
  dataloader is not the bottleneck on MPS. Reference point for Week 3 DDP comparison.
- Known non-blocking issue: `objc[...]: Class AVFFrameReceiver is implemented in both...`
  — duplicate `libavdevice` symbols between the `av` package's bundled build and Homebrew
  `ffmpeg`. Did not crash training; flagged in `PREFLIGHT.md` as a possible future
  instability source, not yet resolved.

#### W1D2 — Config system: draccus + package structure (complete)
**Completed:** June 21, 2026

| File | Description |
|---|---|
| `conf/pusht_act.json` | draccus config file — dataset, policy type, seed, batch_size, steps, wandb settings |
| `scripts/train.py` | Debug harness — loads config via draccus, prints resolved config dict, not a real training entry point |
| `robot_policy_lab/__init__.py` | Package marker — makes robot_policy_lab importable for Week 2's RobotForgeDataset |
| `robot_policy_lab/utils/` | Empty folder — ready for device.py (Day 3) and reproducibility.py (Day 5) |

#### Key things learned
- Plan drift: W1D2 was written against Hydra/YAML — corrected mid-session against actual pinned commit (`2d7a420`). Config system is draccus + JSON, not Hydra + YAML. No `conf/policy/act.yaml` tree, no `omegaconf`, no `hydra.main`.
- draccus ChoiceRegistry: `"act"` is not resolvable until `lerobot.policies.act.configuration_act` has been imported — the `@register_subclass("act")` decorator only fires on import. Fixed with explicit import before `@draccus.wrap()` entry point. `lerobot-train`'s real entry point handles this via `register_third_party_plugins()`.
- JSON config precedence: file defaults < CLI flags — same order Hydra had. `--config_path=conf/pusht_act.json` loads the file; any flag after it (e.g. `--seed=99`) overrides that field only. Verified live: seed changed from 42 → 99, batch_size from 64 → 32 in a single override run.
- `TrainPipelineConfig` field names (confirmed from real source + live dump): `steps` (not `training.offline_steps`), `wandb.enable`, `batch_size`, `seed` — all flat top-level fields, not nested under a `training` block.
- No top-level `device` field on `TrainPipelineConfig`. Device lives at `cfg.policy.device` — already auto-resolves to `'mps'` with zero code. Confirmed live in smoke-test output: `'device': 'mps'` in full config dump.
- `device.py` design decision (Day 3 pre-decided): TWO functions, not one. `configure_mps_env()` sets env vars + global torch state (LeRobot doesn't set these). `get_device()` is a pure standalone resolver for code outside `TrainPipelineConfig`. Side effects must not be hidden inside a getter. `configure_mps_env()` must be called before draccus parses config.
- Repo layout confirmed: real git repo is `~/robotics-ml-portfolio/robotics-ml-portfolio/` (doubled path from cloning inside same-named folder). Outer `~/robotics-ml-portfolio/` has no `.git` — leaving as-is, always `cd` two levels deep.

#### Smoke test results
- `python scripts/train.py --config_path=conf/pusht_act.json` → full resolved config dict printed, `'device': 'mps'` confirmed, all 16 policy types in registry including `'act'`
- CLI override verified: `--seed=99 --batch_size=32` → `seed: 99`, `batch_size: 32` in output, JSON defaults (`42`, `64`) correctly overridden

#### W1D4 — ACT architecture + training loop annotation (complete)
**Completed:** [today's date]

| File | Description |
|---|---|
| `ARCHITECTURE.md` | ACT config, action shape contract, eval execution model, required dict keys, lerobot_train.py hook points, Week 2 checklist |

Key things learned:
- Training script is `lerobot_train.py` (not `train.py`) — entry point for all
  future hook point work
- Two eval modes: `is_eval_step` (loss on held-out split, no env) vs
  `is_env_eval_step` (rollout in sim, returns pc_success + video)
- Week 4 weighted loss path already exists: `policy.forward(batch, reduction="none")`
  at line 121 — hard-example mining just needs to wire sample_weights in
- `action_is_pad` is a required key — missing it = KeyError at training start
  (loud failure, catches it early)
- `accelerate` handles device detection, not LeRobot directly — explains why
  MPS auto-detected in W1D1 without patching
- `pin_memory=device.type=="cuda"` at line 460 — correctly False on MPS,
  consistent with W3 benchmark finding that pin_memory is a no-op on M4

#### W1D5 — seed_everything + reproducibility verification (complete)

| File | Description |
|---|---|
| `robot_policy_lab/utils/reproducibility.py` | Seeds all RNG sources: Python random, NumPy, PyTorch, MPS, PYTHONHASHSEED |
| `tests/test_reproducibility.py` | Runs 20-step training loop twice with seed=42, asserts identical loss curves |

- Reproducible ✓ confirmed on MPS: loss range 0.074695 → 0.204569, device: mps
- Two separate functions kept consistent with W1D3 design: `seed_everything()` is pure side-effect, no return value
- `torch.mps.manual_seed()` (not `manual_seed_all`) — MPS-specific, confirmed working
- `torch.use_deterministic_algorithms(True, warn_only=True)` — warns on non-deterministic MPS ops, does not error
- Tests run from `month-02-robot-policy-lab/` root with `PYTHONPATH=$(pwd) python tests/test_reproducibility.py`
- `seed_everything(cfg.seed)` wired into `scripts/train.py` as first call after `configure_mps_env()`
- On Week 3 CUDA box: add `CUBLAS_WORKSPACE_CONFIG=:4096:8` env var for full determinism

### Week 2 — RobotForgeAdapter + W&B + checkpoint-resume

#### ✅ W2D1 — RobotForgeAdapter: HDF5 → ACT __getitem__ contract (complete)
**Completed:** June 28, 2026

| File | Description |
|---|---|
| `robot_policy_lab/datasets/__init__.py` | Package marker |
| `robot_policy_lab/datasets/adapter.py` | RobotForgeAdapter — HDF5 → ACT sample dict translation layer |
| `tests/test_adapter.py` | Schema, boundary padding, DataLoader multiprocessing, total length |

Test results:
test_schema ✓          image (3,96,96) float32, state (2,) float32, action (100,2) float32, action_is_pad (100,) bool

test_episode_boundary ✓  last frame idx=160, padded steps=99/100

test_dataloader_smoke ✓  batch action (8,100,2), num_workers=2 no h5py collision

test_length ✓           25,650 total samples = sum of 206 episode frame_counts

#### Key things learned
- HDF5 path rewrite required at adapter load time: parquet stores paths from when `ingest.py` ran
  (`~/robotics-ml-portfolio/month-01-robot-data-forge/...`) but real files are one level deeper in the
  doubled-repo structure (`~/robotics-ml-portfolio/robotics-ml-portfolio/month-01-robot-data-forge/...`).
  Fix: `str.replace("robotics-ml-portfolio/month-01-robot-data-forge", "robotics-ml-portfolio/robotics-ml-portfolio/month-01-robot-data-forge")`
  applied to every file_path at load time. Existence check fires loudly in `__init__`, not silently inside `__getitem__`.
- `h5py` and `pandas`/`pyarrow` must be installed in the `robotics-policy-lab` env (Python 3.12).
  They were only in the Month-1 `robotics` env (Python 3.11). Fix: `pip install h5py pandas pyarrow`.
- `action_is_pad` is required — `False` for real steps, `True` for repeat-padded steps past episode end.
  Missing key = `KeyError` at `lerobot_train.py:145`. Not a silent failure, but caught in test before training.
- Action chunk padding at episode boundary: `actions[frame_t : min(frame_t + n_action_steps, ep_len)]`
  gives a short array near episode end — `np.tile(actions[-1], (pad_count, 1))` fills to `(n_action_steps, action_dim)`.
- h5py file handles opened inside `__getitem__`, never `__init__` — multiprocessing DataLoader safety.
  Confirmed: `num_workers=2` smoke test passes with no handle collision.
- Stats key names verified: `dataset_stats.json` uses `"states"` and `"actions"` (plural) with nested `"mean"`/`"std"`.
- pusht action values are raw pixel coordinates (~228, ~294 mean) not [0,1] — z-score normalization still correct,
  output centers near 0 with ~unit variance.
- image min is 0.239, not 0.0 — pusht background is not pure black. Range assertion uses `>= 0.0` not `== 0.0`.
- Two HDF5 copies exist on disk: `data/ep_000000.hdf5` (early ingest run, before HDF5_DIR was configured)
  and `data/hdf5/ep_000000.hdf5` (correct, what parquet indexes). Parquet correctly points to `data/hdf5/`.

#### Known scalability limitation (documented, not fixed)
Random HDF5 access at 100K+ episodes is a throughput bottleneck. Each `__getitem__` opens a file,
seeks to one frame, closes. At 206 episodes / 56MB the dataset fits in page cache — not felt yet.
At scale, production systems pre-shard into WebDataset `.tar` or Arrow files and stream sequentially.
This adapter is correct for the portfolio; the limitation is named explicitly.

#### W2D2 — Lineage utilities (complete)
| File | Description |
|---|---|
| `robot_policy_lab/utils/lineage.py` | get_git_hash(), get_dvc_dataset_hash(), get_config_hash() |
| `tests/test_lineage.py` | All 3 lineage tests passing |

- metadata.parquet is a dvc.yaml pipeline output — get_dvc_dataset_hash() reads
  from dvc.lock (not a .dvc pointer file). Searches all stages for matching output path.
- Confirmed hashes: git=1a42f14..., dvc=4cb77b4b4ba3ac54e4edcb70b6d4aeba, config=67c34877

### 🚧 W2D3 — Production-grade W&B logging + draccus strict-schema correction (mostly complete)
**Date:** June 30, 2026

| File | Description |
|---|---|
| `robot_policy_lab/utils/logging.py` | `init_wandb_run()`, `get_device_profile()`, `log_train_step()`, `log_eval()`, `log_best_checkpoint()` — 7-field run reproducibility metadata |
| `robot_policy_lab/paths.py` | New — custom Month-1 output paths (`DATASET_PARQUET_PATH`, `DATASET_STATS_PATH`, `DATASET_DVC_PATH`), env-var overridable. Moved here after the draccus strict-schema bug below |
| `tests/test_wandb_logging.py` | `test_device_profile`, `test_init_wandb_run` — calls the real `init_wandb_run()` under `WANDB_MODE=disabled`, not a reimplementation of its logic |
| `scripts/train.py` | `init_wandb_run()` wired in inside `main()`, right after `seed_everything(cfg.seed)`. Still the W1D2 config-resolution debug harness — no training loop yet |
| `conf/pusht_act.json` | Reverted to LeRobot-native fields only — `eval_freq`, `dataset_parquet_path`, `dataset_stats_path`, `dataset_dvc_path` removed (see below) |

Test results: `test_device_profile ✓`, `test_init_wandb_run ✓` — `git_hash`, `dataset_dvc_hash`, `config_hash`, `device_type` all populated.

500-step live run: https://wandb.ai/sunnydave234-student/robot-policy-lab/runs/hr4fnmb8 (absurd-lake-1)

`_meta` fields confirmed live in the W&B Config panel:
```
git_hash:         51d65e54c2cf218a8f734acf6d9791795aaef63a
dataset_dvc_hash: 4cb77b4b4ba3ac54e4edcb70b6d4aeba
config_hash:      892531ac
device_type:      mps
platform:         macOS-26.5.1-arm64-arm-64bit
torch_version:    2.11.0
```

`dataset_dvc_hash` matches W2D2's confirmed hash (`4cb77b4b...`) exactly — same
`metadata.parquet`, byte-identical, DVC lineage confirmed stable across sessions.
`git_hash` and `config_hash` both differ from W2D2's values, and correctly so —
more commits landed between W2D2 and this run (different `git_hash`), and this
run's actual `cfg` (`--steps=500` override, live `TrainPipelineConfig`) has
different content than whatever produced W2D2's `67c34877` (different
`config_hash`). Same data, different code and config — the three hashes say
so independently, which is the whole point of keeping them separate.

`train/loss`, `eval/success_rate`, checkpoint artifact: **not verified** — `scripts/train.py` has no training loop, so nothing calls `log_train_step()` / `log_eval()` / `log_best_checkpoint()` yet. Deferred to whichever day wires in the real loop.

#### Key things learned
- **draccus is strict, not lenient — the real bug, bigger than the original attribute-access issue.** W2D2 assumed custom fields could be added directly to `conf/pusht_act.json` on the theory that unrecognized keys just wouldn't bind to `cfg`. Wrong: `draccus.wrap()` validates every top-level key in the JSON against `TrainPipelineConfig`'s declared fields and raises `DecodingError` on anything it doesn't recognize — confirmed live: `DecodingError: The fields eval_freq, dataset_parquet_path, dataset_stats_path, dataset_dvc_path are not valid for TrainPipelineConfig`. It doesn't silently drop unknown keys; it refuses to parse the file at all. Any custom, non-LeRobot config now lives in `robot_policy_lab/paths.py` (env-var overridable, same convention as Month 1's `config.py`), never in `conf/pusht_act.json`.
- `cfg` only exists inside `draccus.wrap()`'s wrapped function, after draccus calls it — module-level code before the wrapped function can't reference it. Hit as `NameError: name 'cfg' is not defined`; fixed by moving `seed_everything()` and `init_wandb_run()` inside `main(cfg)`.
- `init_wandb_run(cfg, *, dvc_path: str)` — `dvc_path` is an explicit keyword-only argument, not read off `cfg`. Dataset name is *not* a separate parameter — read straight from `cfg.dataset.repo_id`, which is a real, draccus-bound field, so the W&B tag can never drift out of sync with the dataset actually training.
- No top-level `eval_freq` field ever existed on `TrainPipelineConfig` — confirmed via `dataclasses.fields(TrainPipelineConfig)`. Real fields: `env_eval_freq` (rollout/sim eval cadence, default `20000`) and `eval_steps` (held-out loss eval, currently `0`/off).
- Test design matters: `test_init_wandb_run()` calls the real `init_wandb_run()` rather than reconstructing its `_meta` dict inline — a test that rebuilds the logic in parallel can't catch a bug *inside* the function under test.
- `FakeCfg` in the test is deliberately shaped like the real `TrainPipelineConfig` (`dataset.repo_id`, `wandb.project`) with no `dataset_dvc_path`/`dataset_name` field — if `init_wandb_run()` ever regresses to reading those off `cfg`, the test fails loudly instead of silently passing.

#### Known gap (documented, not fixed)
`scripts/train.py` now wires in W&B logging correctly but still has no training loop — it's the W1D2 debug harness (parses config, prints the dict), not `lerobot-train`. `log_train_step()`, `log_eval()`, and `log_best_checkpoint()` are implemented and unit-tested in isolation but unverified end-to-end until a real training loop calls them. **Note for W2D4:** `save_checkpoint`, `save_freq`, `checkpoint_path`, `resume` are all real, native `TrainPipelineConfig` fields — LeRobot's own `lerobot_train.py` almost certainly already checkpoints model/optimizer/scheduler/step. Verify what it saves before writing a checkpoint function from scratch.

### ✅ W2D5 — Real training integration: RobotForgeAdapter → lerobot_train.py (complete)
**Date:** July 10–12, 2026

| File | Description |
|---|---|
| `scripts/train_integration.py` | Monkeypatches `make_train_eval_datasets` via `unittest.mock.patch.object`, wires `RobotForgeAdapter` into the real `lerobot_train.train()` entry point, pushes `_meta` lineage fields into LeRobot's native W&B run |
| `robot_policy_lab/datasets/adapter.py` | Added 4 `@property` methods (`num_frames`, `num_episodes`, `episodes`, `absolute_to_relative_idx`) for `LeRobotDataset` compatibility; fixed non-idempotent path-doubling bug in the HDF5 path rewrite |
| `robot_policy_lab/utils/checkpoint.py` | Fixed typo: `load_checkpoint()` was reading `ckpt["shceduler_state_dict"]` instead of `ckpt["scheduler_state_dict"]` |

**1000-step real integration run:** loss 4.262 → 0.712, 6.40 steps/sec (matches W1D3 MPS baseline exactly), checkpoints at step 500 and 1000, all 6 W&B `_meta` fields confirmed present, `dataset_dvc_hash` stable at `4cb77b4b4ba3ac54e4edcb70b6d4aeba`.

**4-run hyperparameter sweep** (seed ∈ {42, 43} × lr ∈ {1e-4, 3e-4}, 200 steps each): confirmed correct seed and lr differentiation across three independent sources (config dump, training-loop log, W&B Config panel). Loss at 200 steps: lr=1e-4 → ~2.83–2.93, lr=3e-4 → ~5.06–5.33 — consistent across both seeds, a real (not noise) early signal that 3e-4 may be too aggressive for this setup.

#### What was built
Bridged `RobotForgeAdapter` (Month-1 HDF5 data) into LeRobot's actual `lerobot_train.py` training script end to end, for the first time — every prior Week 2 day tested the adapter, lineage utils, W&B logging, and checkpointing in isolation. This was the integration test, and it surfaced three real bugs that isolated unit tests couldn't have caught.

The bridge: `unittest.mock.patch.object` swaps `make_train_eval_datasets` for the duration of the run. The patched version calls LeRobot's real function first (for a normal `eval_dataset` and, critically, the real dataset's `.meta` object), then substitutes `RobotForgeAdapter` as the training dataset before returning.

#### Three bugs found, in order of discovery
1. **5 missing `LeRobotDataset`-compatibility attributes** on `RobotForgeAdapter` (`.meta`, `.num_frames`, `.num_episodes`, `.episodes`, `.absolute_to_relative_idx`) — discovered one crash at a time by running real training against it. Fixed by reusing the original dataset's `.meta` directly (both datasets are the same underlying pusht data) and adding 4 new `@property` methods matching what the real `LeRobotDataset` returns under the same single-Mac, no-sharding conditions.
2. **W&B `_meta` fields silently absent** — LeRobot's own native `WandBLogger` calls `wandb.init()` before our dataset patch ever runs, using its own plain config with no knowledge our custom `init_wandb_run()` (built and unit-tested back in W2D3) exists. Never crashed — undetected through an entire successful 1000-step run until manually checking the W&B Config panel. Fixed by pushing `_meta` into the already-live run via `wandb.config.update(..., allow_val_change=True)`.
3. **`--optimizer.lr` CLI override silently discarded** — `TrainPipelineConfig.__post_init__` rebuilds `cfg.optimizer` from `cfg.policy.optimizer_lr` whenever `use_policy_training_preset=True` (the default), running *after* draccus applies CLI flags. Only caught because the sweep task explicitly required comparing `lr` values across runs — a single-run smoke test would never have revealed it. Correct flag: `--policy.optimizer_lr`.

#### Key things learned
- `grep` the full attribute contract of a class you're duck-typing against *before* writing the adapter, not one crash at a time after — `grep -n "dataset\.\w\+" file.py` returns the whole list in one pass.
- A crash-free, loss-decreasing run is not proof everything is correct. Two of today's three bugs (missing `_meta`, discarded `lr` override) produced zero errors — they were only caught by deliberately checking the actual output (the W&B panel, the sweep's `lr` values) against what was expected, not by trusting a clean exit code.
- Reusing a real, working object (`original_train_ds.meta`) instead of hand-building a stand-in sidesteps an entire category of subtle bugs (wrong dtype, missing keys, tensor-vs-list mismatches) that would otherwise need discovering one at a time.
- `python -m py_compile` passing is not proof a heredoc-based edit landed completely — always re-view the actual file after a multi-line heredoc edit.
- `@parser.wrap()` is LeRobot's own name for their draccus-based decorator — grepping literally for `"draccus"` can come back empty and be misleading.

## Month 3 — edge-policy *(not started)*