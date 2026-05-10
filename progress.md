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

### ⬜ Week 2 — Build the Ingestion Pipeline: Raw → Validated HDF5
**Goal:** `ingest.py` → `validate.py` → `metadata.parquet` index → `query.py`
### ✅ Week 2 Day 2 — Full HDF5 Ingestion Pipeline
**Completed:** May 9, 2026

#### What was built
| File | Description |
|---|---|
| `write_one_episode.py` | Writes one episode to HDF5 — chunked (1,H,W,C), gzip level 4 |
| `ingest.py` | Full ETL loop — idempotent, exception-safe, progress-logged |
| `config.py` | Added HDF5_DIR, DATASET_NAME, episode_hdf5_path() |

#### Results
- 206 / 206 episodes ingested, 0 failures
- 56 MB total on disk, 23.6 ep/s (data cached locally)
- Schema: /observations/images (uint8), /observations/state (float32), /actions (float32)
- Idempotency verified — second run skipped all 206 in <1s

### ⬜ Week 3 — PyTorch Dataset + DataLoader over HDF5
### ⬜ Week 4 — DVC Pipeline + End-to-End Reproducibility

---

## Month 2 — robot-policy-lab *(not started)*
## Month 3 — edge-policy *(not started)*