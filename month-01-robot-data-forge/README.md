# robot-data-forge
### A Production Robot Learning Data Engine over LeRobot / Open-X

A production robot learning data engine that ingests raw LeRobot demonstration episodes,
normalizes them into a validated HDF5 store, versions everything with DVC, and serves
training-ready batches through a PyTorch DataLoader at 25,900+ samples/sec. Built as
Month 1 of a 3-month robotics ML portfolio by a senior data engineer transitioning into
robot learning infrastructure roles.

**Dataset:** `lerobot/pusht` — 206 episodes, 25,650 frames, 2D end-effector control

---

## Output

[![Episode 0 — PushT](outputs/episode_000_thumb.png)](outputs/episode_000.mp4)

*Episode 0 rendered at 384×384 (4× upscaled from 96×96 source).
Action overlays show [x, y] target position per frame.*

---

## Architecture

```
LeRobot / HuggingFace
        │
        │  lerobot/pusht  (206 episodes, 25,650 frames)
        ▼
┌─────────────────┐
│   ingest.py     │  Extract episodes via LeRobot V3 API
│                 │  Normalize to HDF5 schema (uint8 images, float32 actions)
│                 │  Write done sentinel — guards against mid-write crashes
└────────┬────────┘
         │  data/hdf5/ep_000000.hdf5 … ep_000205.hdf5
         │  /observations/images  (T, H, W, C) uint8   chunked (1,H,W,C)  gzip-4
         │  /observations/state   (T, 2)        float32
         │  /actions              (T, 2)        float32
         │
         ├──────────────────────────────────┐
         ▼                                  ▼
┌─────────────────┐              ┌─────────────────────┐
│  validate.py    │              │   build_index.py    │
│                 │              │                     │
│  Shape / dtype  │              │  Scan HDF5 attrs    │
│  NaN check      │              │  → metadata.parquet │
│  Action bounds  │              │  (episode_id, task, │
│  Frame parity   │              │   success,          │
└────────┬────────┘              │   frame_count,      │
         │                      │   file_path)        │
         │  validation_          └──────────┬──────────┘
         │  report.json                     │
         │                                  ▼
         │                       ┌─────────────────────┐
         │                       │  compute_stats.py   │
         │                       │                     │
         │                       │  Welford streaming  │
         │                       │  mean / std over    │
         │                       │  all episodes       │
         │                       │  → dataset_stats    │
         │                       │    .json            │
         │                       └──────────┬──────────┘
         │                                  │
         └──────────────┬───────────────────┘
                        │
                        ▼
           ┌────────────────────────┐
           │   robot_dataset.py     │
           │   RobotEpisodeDataset  │
           │                       │
           │   flat index           │
           │   → (episode, frame)  │
           │   normalize=True/False │
           │   context_window=K     │
           └───────────┬───────────┘
                       │
                       ▼
           ┌────────────────────────┐
           │   weighted_sampler.py  │
           │   WeightedEpisodeSampler│
           │                       │
           │   episode weights      │
           │   → frame-level repeat │
           │   → torch.multinomial  │
           └───────────┬───────────┘
                       │
                       ▼
           ┌────────────────────────┐
           │      train.py          │
           │      BCMLP baseline    │
           │                       │
           │   state + prev_action  │
           │   → MLP → action pred  │
           │   MSE loss, Adam       │
           │   W&B logging          │
           └────────────────────────┘

All stages versioned by DVC. Fully reproducible via: bash reproduce.sh
```

---

## Dataset Profile

| Metric | Value |
|---|---|
| Total episodes | 206 |
| Total frames | 25,650 |
| FPS | 10 |
| Episode length (mean) | 124.5 frames |
| Episode length (min / max) | 49 / 246 frames |
| Episode length std | 35.7 frames |
| Action dim | 2 (x, y target position) |
| Image shape | 3 × 96 × 96 (C, H, W) |
| Image dtype | float32 [0.0, 1.0] |
| HDF5 store size | 56 MB (206 episodes, gzip-4 compressed) |
| Success rate | 0% (demonstrator data — sparse reward, not used for filtering) |

![Episode Length Distribution](outputs/episode_length_hist.png)

---

## Usage

### Full pipeline reproduction (recommended for fresh clones)

```bash
git clone https://github.com/sunnydave234/robotics-ml-portfolio.git
cd robotics-ml-portfolio/month-01-robot-data-forge

bash reproduce.sh
```

`reproduce.sh` creates an isolated `.venv/`, installs dependencies, runs `dvc repro`
(ingest → validate → build_index → compute_stats), and runs the BC training smoke test.
Never touches your global environment.

```bash
CI=true bash reproduce.sh             # skip wandb login (set WANDB_API_KEY in env)
SKIP_DVC_PULL=true bash reproduce.sh  # skip dvc pull, rebuild pipeline from scratch
```

### Run individual scripts

```bash
# Set up environment (Python 3.11, MPS backend for M4 Mac)
python -m venv ~/envs/robotics && source ~/envs/robotics/bin/activate
pip install -r requirements.txt

# Render a single episode as MP4 with action overlays
python visualize_episode.py --episode 0

# Profile the full dataset → outputs/dataset_profile.json + episode_length_hist.png
python profile_dataset.py

# Smoke-test episode extraction across multiple indices
python scripts/check_episodes.py

# Run the full ingestion pipeline manually
bash run_pipeline.sh lerobot/pusht 206

# Query the episode index
python query.py --task pusht --min_frames 100

# Visualize a DataLoader batch → outputs/batch_visualization.png
python visualize_batch.py

# Run the BC training smoke test
python train.py
```

Override any output path without touching code:
```bash
OUTPUTS_DIR=/tmp/renders python visualize_episode.py --episode 5
HDF5_DIR=/Volumes/SSD/hdf5 bash run_pipeline.sh lerobot/pusht 206
```

**Inspect the DVC pipeline DAG:**
```bash
dvc dag
```

---

## Ingestion Pipeline

### Benchmarks

| Metric | Value |
|---|---|
| Episodes ingested | 206 / 206 |
| Total disk usage | 56 MB |
| Avg ingestion rate | 23.6 ep/s (data cached locally) |
| Total wall time | 7.9s |
| Compression | gzip level 4 |
| Chunk shape (images) | (1, 96, 96, 3) — one chunk = one frame |
| Chunk shape (actions) | (1, 2) |
| Failed episodes | 0 |
| Idempotency | ✅ skip-if-done via `done` sentinel attribute |

### HDF5 Schema (per episode file `ep_{idx:06d}.hdf5`)

```
/
├── observations/
│   ├── images   (T, H, W, C)  uint8    chunked (1,H,W,C)  gzip-4
│   └── state    (T, 2)        float32  chunked (1,2)       gzip-4
└── actions      (T, 2)        float32  chunked (1,2)       gzip-4
/ [root attrs]   episode_id, task, frame_count, success, done
```

---

## PyTorch Dataset & DataLoader

`RobotEpisodeDataset` is a flat-indexed `Dataset` over the per-episode HDF5
files, addressed via `outputs/metadata.parquet`. Each item returns:

- `image`: `(C, H, W)` float32 in `[0, 1]` — or `(K, H, W, C)` if `context_window > 1`
- `action`: `(2,)` float32 — pusht's 2D end-effector target position
- `state`: `(2,)` float32 — placeholder (pusht has no proprioceptive state)
- `episode_idx`: int
- `frame_offset`: int

```python
from torch.utils.data import DataLoader
from robot_dataset import RobotEpisodeDataset
from config import OUTPUTS_DIR

# normalize=True applies z-score normalization to action/state using
# stats from outputs/dataset_stats.json (compute_stats.py)
dataset = RobotEpisodeDataset(OUTPUTS_DIR / "metadata.parquet", normalize=True)

loader = DataLoader(
    dataset,
    batch_size=64,
    num_workers=8,             # see benchmark table below
    persistent_workers=True,
    shuffle=True,
)

batch = next(iter(loader))
print(batch["image"].shape)   # torch.Size([64, 3, 96, 96])
print(batch["action"].shape)  # torch.Size([64, 2])

# Temporal context window — returns (K, H, W, C), frame-0 padded at
# episode boundaries
ctx_dataset = RobotEpisodeDataset(
    OUTPUTS_DIR / "metadata.parquet", normalize=False, context_window=3
)
print(ctx_dataset[0]["image"].shape)  # torch.Size([3, 96, 96, 3])
```

### DataLoader Benchmark

Measured on Mac Studio M4 Max (48GB unified memory). Dataset: 25,650 samples,
`normalize=True`, `persistent_workers=True`.

| num_workers | pin_memory | batch_size | samples/sec | p95 batch (ms) |
|---|---|---|---|---|
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

- `pin_memory` has <1% effect — confirmed no-op on M4 unified memory (CPU and GPU share the same physical memory; no PCIe DMA transfer needed)
- `num_workers` scaling is ~1.7× per doubling (0→2→4→8) — healthy parallel I/O curve
- Bottleneck is HDF5 reads + gzip decompression in worker subprocesses (93.7% of profiled CPU time, measured via `profile_dataloader.py` → Chrome trace at `chrome://tracing`)
- Numbers reflect macOS page-cache hits after first epoch — at 10,000+ episodes the dataset no longer fits in RAM and this curve flattens as NVMe I/O becomes the ceiling

### Batch Visualization

![Batch visualization](outputs/batch_visualization.png)

8 samples from the DataLoader (`normalize=False`), each with its raw
`(x, y)` action vector plotted as a bar chart below.

---

## W&B Run

BC baseline training (10 epochs, MSE loss, MPS backend):
**https://wandb.ai/sunnydave234-student/robot-data-forge/runs/c1o22yul**

Loss curve: `train/loss` drops sharply epoch 1 → near-zero by epoch 2 and stays flat.
This is expected — pusht writes actions into both `/observations/state` and `/actions`,
so the model input contains the prediction target. The MLP learned a near-identity
mapping. This confirms pipeline correctness, not model quality.

---

## Design Decisions

**HDF5 over raw image files.** Storing episodes as individual JPEG/PNG frames scatters one episode across 50–250 files. A DataLoader reading episode 42 frame 17 would open a directory, stat a filename, and open a file — three filesystem calls per sample. HDF5 collapses this to one file open and one dataset read. At 25,650 frames across 206 episodes, the directory-scatter approach requires managing 25,000+ files with no schema enforcement. HDF5 gives a schema, compression, and chunked random access in a single artifact.

**Images stored as uint8, not float32.** uint8 storage reduces the HDF5 store from ~224 MB (float32) to 56 MB — a 4× reduction with no training impact. Normalization to `[0.0, 1.0]` float32 happens in `RobotEpisodeDataset.__getitem__` at read time, in the DataLoader worker process. The cost is a vectorized divide-by-255 per batch, which is negligible compared to gzip decompression. Storing float32 on disk would 4× the I/O per training step with zero benefit.

**Chunk shape `(1, H, W, C)` — one chunk per frame.** HDF5 chunk shape determines the minimum unit of I/O. A chunk shape of `(T, H, W, C)` (full episode) means reading any single frame decompresses the entire episode. A chunk shape of `(1, H, W, C)` means reading frame 17 only decompresses frame 17. Since `__getitem__` reads one frame at a time (or K frames for `context_window > 1`), per-frame chunking minimizes wasted decompression. The tradeoff is slightly worse compression ratio, which gzip-4 handles well for uint8 image data.

**HDF5 files opened per `__getitem__`, not per `__init__`.** h5py file handles are not serializable across process boundaries. PyTorch DataLoader with `num_workers > 0` forks worker processes — any file handle opened in `__init__` on the main process is invalid in the worker. Opening the file inside `__getitem__` means each worker opens its own handle on demand. This is the standard pattern for HDF5 in multi-worker DataLoaders and is why the 8-worker config works correctly.

**Parquet index over querying HDF5 directly.** Filtering episodes by task type, success rate, or frame count would require opening every HDF5 file and reading root attributes — O(N) file opens per query. `metadata.parquet` stores those same attributes in a 206-row columnar file that fits in memory and filters in under 100ms via pandas. This is the same pattern as an OpenSearch metadata layer over raw records: the index is cheap to query, the raw store is only opened when you need the actual data.

**`done` sentinel attribute over file-existence checks.** `ingest.py` writes the `done` attribute as the last operation before closing each HDF5 file. A crash mid-write produces a valid HDF5 file with `done=False`. `build_index.py` skips any file without `done=True`. This means a partial ingest is safe to re-run — the crashed episode gets re-ingested, completed ones are skipped via idempotency check. File-existence checks can't distinguish "wrote successfully" from "started writing."

**All paths centralized in `config.py`, overridable via env vars.** No hardcoded paths anywhere in the codebase. Every script imports `OUTPUTS_DIR`, `HDF5_DIR`, and path helpers from `config.py`. Any path can be overridden at runtime without touching code:

```bash
OUTPUTS_DIR=/tmp/renders python visualize_episode.py --episode 0
HDF5_DIR=/Volumes/SSD/hdf5 bash run_pipeline.sh lerobot/pusht 206
```

---

## File Reference

| File | Description |
|---|---|
| `config.py` | Central path config — all scripts import from here, no hardcoded paths |
| `extract_episode.py` | Extracts one episode from LeRobot as `{images, actions}` tensors |
| `visualize_episode.py` | Renders episode as MP4 with per-frame action overlays |
| `profile_dataset.py` | Computes dataset statistics → JSON + histogram PNG |
| `scripts/check_episodes.py` | Smoke-tests episode extraction across multiple indices |
| `tests/test_extract.py` | Pytest suite for `extract_episode` |
| `ingest.py` | Full ETL loop — idempotent, exception-safe, progress-logged |
| `write_one_episode.py` | Single-episode write pattern (used in dev/testing) |
| `validate.py` | Per-file checks: shape, dtype, NaN, action bounds, frame parity |
| `build_index.py` | Scans HDF5 root attrs → `outputs/metadata.parquet` |
| `query.py` | Filter episode index by task / success / frame count |
| `compute_stats.py` | Welford streaming mean/std → `outputs/dataset_stats.json` |
| `robot_dataset.py` | `RobotEpisodeDataset(Dataset)` — normalize, context_window |
| `add_weights.py` | Adds `weight` column to `metadata.parquet` for sampler |
| `weighted_sampler.py` | `WeightedEpisodeSampler` — episode weights → frame-level sampling |
| `benchmark_dataloader.py` | 16 configs × 100 batches throughput benchmark |
| `profile_dataloader.py` | PyTorch profiler → `outputs/dataloader_trace.json` (open at `chrome://tracing`) |
| `visualize_batch.py` | 2×4 batch grid with action bar charts → `outputs/batch_visualization.png` |
| `model.py` | `BCMLP` — 3-layer MLP behavior cloning baseline |
| `bc_dataset.py` | `BCDataset` — wraps `RobotEpisodeDataset`, adds `prev_action` |
| `train.py` | 10-epoch BC training loop, MSE loss, Adam, W&B logging |
| `reproduce.sh` | Full pipeline reproduction in isolated `.venv` from a fresh clone |
| `run_pipeline.sh` | One-command ingestion pipeline runner |
| `dvc.yaml` | 4-stage DVC pipeline: ingest → validate + build_index → compute_stats |

---

## Architecture Note

All tensor operations stay in PyTorch until the OpenCV boundary.
`tensor_to_bgr_frame()` in `visualize_episode.py` is the single conversion point:
`(C,H,W) float32 tensor → (H,W,3) uint8 BGR numpy array`.
No numpy until OpenCV needs it.

**Background:** The `lerobot/pusht` task is a 2D manipulation benchmark where a robot
end-effector must push a T-shaped block into a goal zone. The 2D action space makes it
a clean starting point for understanding the data structure before moving to 6-DOF arm tasks.
