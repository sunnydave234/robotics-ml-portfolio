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

## Month 2 — robot-policy-lab *(not started)*
## Month 3 — edge-policy *(not started)*