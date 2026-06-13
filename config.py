"""
Central path configuratio nfor robot-data-forge.
All scripts import from here - no hardcoded paths anywhere else.

Override any value with an env variable:
    OUTPUTS_DIR=/tmp/renders python visulize_episode.py --episode 0
"""
import os
from pathlib import Path

# ── Project root ────────────────────────────────────────────────────────────
# Always resolved relative to this file -safe regarless of cwd
PROJECT_ROOT = Path(__file__).parent.resolve()

# ── Directories ─────────────────────────────────────────────────────────────
OUTPUTS_DIR = Path(os.environ.get("OUTPUTS_DIR", PROJECT_ROOT / "outputs"))
CACHE_DIR   = Path(os.environ.get("CACHE_DIR", PROJECT_ROOT / ".cache"))

# ── Week 2: HDF5 storage ──────────────────────────────────────────────────────
# One .hdf5 file per episode lands here after ingest.py runs.
# Override with HDF5_DIR env var if you want files on a different drive.
HDF5_DIR = Path(os.environ.get("HDF5_DIR", PROJECT_ROOT / "data" / "hdf5"))


# ── Dataset ───────────────────────────────────────────────────────────────────
# W1 kept this as DEFAULT_DATASET — W2 scripts import it as DATASET_NAME.
# Both names point at the same value so neither set of scripts breaks.
DEFAULT_DATASET = os.environ.get("LEROBOT_DATASET", "lerobot/pusht")
DATASET_NAME    = DEFAULT_DATASET   # alias used by ingest.py and validate.py

DEFAULT_FPS     = int(os.environ.get("DEFAULT_FPS",     "10"))
DEFAULT_UPSCALE = int(os.environ.get("DEFAULT_UPSCALE", "4"))

# ── Output file conventions ──────────────────────────────────────────────────
def episode_video_path(episode_idx: int) -> Path:
    """Canonical path for a rendered episode MP4 (used by visualize_episode.py)."""
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUTS_DIR / f"ep_{episode_idx:06d}.mp4"

def episode_hdf5_path(episode_idx: int) -> Path:
    """Canonical path for a written episode HDF5 (used by ingest.py and validate.py)."""
    HDF5_DIR.mkdir(parents=True, exist_ok=True)
    return HDF5_DIR / f"ep_{episode_idx:06d}.hdf5"

def profile_json_path() -> Path:
    return OUTPUTS_DIR / "dataset_profile.json"

def histogram_path() -> Path:
    return OUTPUTS_DIR / "episode_length_hist.png"

def checkpoint_path(name: str = "bc_mlp_best.pt") -> Path:
    """ Canonical path for a saved model checkpoint. """
    ckpt_dir = OUTPUTS_DIR / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    return ckpt_dir / name
