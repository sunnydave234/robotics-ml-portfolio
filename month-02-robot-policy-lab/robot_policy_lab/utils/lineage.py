import dataclasses
import hashlib
import json
import subprocess
from pathlib import Path


def get_git_hash() -> str:
    """Return the current HEAD commit SHA, or 'unknown' if git is unavailable."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


def get_dvc_dataset_hash(tracked_path: str) -> str:
    """
    Read the MD5 hash from dvc.lock for a pipeline-managed output.

    tracked_path: path to the tracked file, relative to the dvc.yaml directory.
    Example: "../month-01-robot-data-forge/outputs/metadata.parquet"

    Reads dvc.lock from the same directory as dvc.yaml (the Month-1 root).
    Searches all stages for the matching output path.
    """
    try:
        import yaml
        from pathlib import Path

        tracked = Path(tracked_path)
        dvc_root = tracked.parent.parent  # month-01-robot-data-forge/
        lock_file = dvc_root / "dvc.lock"

        with open(lock_file) as f:
            lock = yaml.safe_load(f)

        # The path stored in dvc.lock is relative to the dvc.yaml root
        # e.g. "outputs/metadata.parquet" not the full path
        relative_path = str(tracked.relative_to(dvc_root))

        for stage in lock.get("stages", {}).values():
            for out in stage.get("outs", []):
                if out.get("path") == relative_path:
                    return out["md5"]

        return "unknown (path not found in dvc.lock)"
    except Exception as e:
        return f"unknown ({e})"


def get_config_hash(cfg) -> str:
    """
    Short hash of the fully resolved draccus config (Python dataclass).

    cfg: any draccus/dataclass config object (TrainPipelineConfig, ACTConfig, etc.)
    Uses dataclasses.asdict() — works on any nested dataclass, no OmegaConf needed.
    sort_keys=True ensures identical configs always produce identical hashes
    regardless of field insertion order.
    """
    d = dataclasses.asdict(cfg)
    return hashlib.md5(
        json.dumps(d, sort_keys=True, default=str).encode()
    ).hexdigest()[:8]
