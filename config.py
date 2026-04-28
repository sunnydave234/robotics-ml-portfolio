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

# ── Dataset ──────────────────────────────────────────────────────────────────
DEFAULT_DATASET = os.environ.get("LEROBOT_DATASET", "lerobot/pusht")
DEFAULT_FPS     = int(os.environ.get("DEFAULT_FPS", "10"))
DEFAULT_UPSCALE = int(os.environ.get("DEFAULT_UPSCALE", "4"))

# ── Output file conventions ──────────────────────────────────────────────────
def episode_video_path(episode_idx: int) -> Path:
    """ Canonical path for a rendered episode MP4. """
    return OUTPUTS_DIR / f"episode_{episode_idx:03d}.mp4"

def profile_json_path() -> Path:
    return OUTPUTS_DIR / "dataset_profile.json"

def histogram_path() -> Path:
    return OUTPUTS_DIR / "episode_length_hist.png"
