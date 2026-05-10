"""
ingest.py - Converts all LeRobot episodes to chunked, compressed HDF5.

Resulting schema:
/
├── observations/
│   ├── images (T, H, W, C)
│   └── state (T, state_dim)
└── actions (T, action_dim)
/ [root attrs]   episode_id, task, frame-count, success

images: (T, C, H, W) float32 tensor, values in [0.0, 1.0]
actions: (T, A) float32 tensor

Usage:
    python ingest.py                    # process all episodes
    python ingest.py --max-episodes 20  # process first N
"""

import argparse
import logging
import os
import time
import h5py

from config import HDF5_DIR, DATASET_NAME
from extract_episode import extract_episode
from write_one_episode import write_episode
from pathlib import Path

logging.basicConfig(
  level=logging.INFO,
  format='%(asctime)s - %(levelname)s - %(message)s',
  datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def episode_path(episode_idx: int) -> Path:
  return os.path.join(HDF5_DIR, f"ep_{episode_idx:06d}.hdf5")


def get_episode_success(ds, episode_idx: int) -> float:
  """
  Read episode-level success from dataset metadata.
  For pusht this is always 0.0 - sparse reward, not meaningful yet.
  Falls back to 0.0 if the column doesn't exist on other datasets.
  """
  try:
    row = ds.meta.episodes.iloc[episode_idx]
    for col in ("success", "index_success", "task_success"):
      if col in row.index:
        return float(row[col])
  except Exception:
    pass
  return 0.0


def is_done(episode_idx: int) -> bool:
  """
  Return True only if this episode was successfullly written.
  Checks for the 'done' sentinel attribute - not just whether the file exists.
  A file can exist but be incomplete if the process crashed mid-write.
  """
  path = episode_path(episode_idx)
  if not os.path.exists(path):
    return False
  try:
    with h5py.File(path, 'r') as f:
      return bool(f.attrs.get('done', False))
  except Exception:
    return False


def ingest_episode(ds, episode_idx: int) -> None:
  """ Extract one episode from the dataset and write it to HDF5. """
  ep = extract_episode(ds, episode_idx)
  images = ep['images']
  actions = ep['actions']
  success = get_episode_success(ds, episode_idx)

  with h5py.File(episode_path(episode_idx), 'w') as f:
    write_episode(f, episode_idx, images, actions, success)
    # 'done' is set AFTER write_episode completes successfully.
    f.attrs['done'] = True


def _hdf5_dir_bytes() -> int:
  total = 0
  for entry in os.scandir(HDF5_DIR):
    if entry.name.endswith(".hdf5"):
      total += entry.stat().st_size
  return total


def run(max_episodes: int | None = None) -> None:
  from lerobot.datasets.lerobot_dataset import LeRobotDataset

  os.makedirs(HDF5_DIR, exist_ok=True)
  ds = LeRobotDataset(DATASET_NAME)
  total = ds.meta.total_episodes
  target = min(total, max_episodes) if max_episodes else total

  log.info(f"Dataset: {DATASET_NAME} | total episodes: {total} | target:  {target}")

  succeeded = 0
  skipped = 0
  failed = 0
  t_start = time.perf_counter()
  bytes_before = _hdf5_dir_bytes()

  for idx in range(target):

    if is_done(idx):
      skipped += 1
      continue
    
    try:
      ingest_episode(ds, idx)
      succeeded += 1
    except Exception as e:
      log.warning(f"Episode {idx:06d} FAILED - {type(e).__name__}: {e}")
      failed += 1
      continue
    
    processed = succeeded + failed
    if processed % 10 == 0:
      elapsed = time.perf_counter() - t_start
      rate = processed / elapsed
      eta = (target - idx - 1 - skipped) / rate if rate > 0 else 0
      log.info(
        f"[{idx + 1}/{target}]  done={succeeded}  skipped={skipped}   "
        f"failed={failed} rate={rate:1f} ep/s   eta={eta:.0f}s"
      )
  
  elapsed = time.perf_counter() - t_start
  bytes_written = _hdf5_dir_bytes() - bytes_before

  log.info("=" * 60)
  log.info(f"DONE in {elapsed:.1f}s")
  log.info(f"  Succeeded : {succeeded}")
  log.info(f"  Skipped   : {skipped}  (already written)")
  log.info(f"  Failed    : {failed}")
  log.info(f"  Disk used : {bytes_written / 1024**2:.1f} MB")
  if succeeded > 0:
      log.info(f"  Avg rate  : {succeeded / elapsed:.2f} ep/s")
  log.info("=" * 60)


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--max-episodes", type=int, default=None)
  args = parser.parse_args()
  run(max_episodes=args.max_episodes)
