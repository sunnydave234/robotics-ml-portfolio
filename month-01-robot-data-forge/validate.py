"""
validate.py — Data quality layer for the HDF5 episode store.

Runs 6 checks on every episode file:
    1. done sentinel present (write completed without crash)
    2. Required datasets exist at correct paths
    3. Frame count parity: actions.shape[0] == images.shape[0] == state.shape[0]
    4. frame_count attribute matches actual array length
    5. images dtype is uint8, values in [0, 255]
    6. No NaN or Inf in actions or state
    7. Action values within empirically derived bounds (computed from your own data)

Usage:
    python validate.py                    # validate all episodes
    python validate.py --max-episodes 20  # validate first N
    python validate.py --episode 42       # validate one episode (debug)
"""

import argparse
import json
import logging
import numpy as np
import h5py
from pathlib import Path
from typing import Any

from config import HDF5_DIR, OUTPUTS_DIR, episode_hdf5_path, DATASET_NAME

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# --- Action bounds (computed from data, not hardcoded) -----------------------
def compute_action_bounds(hdf5_dir: Path, margin: float = 0.10) -> tuple[float, float]:
    """
    Scan all HDF5 files in hdf5_dir, read only the actions dataset from each,
    return (global_min - margin%, global_max + margin%) as plausibility bonds.

    This reads actions only - no images loaded - so it's fast even at 200+ episodes.
    THe margin prevents the bounds from being so tight that floating-point noise
    from a legitimate episode triggers a flase positive.

    margin=0.1 means +-10% of the observed range on each side.
    """
    global_min =  float("inf")
    global_max = -float("inf")
    files_scanned = 0

    for path in sorted(hdf5_dir.glob("ep_*.hdf5")):
        try:
            with h5py.File(path, "r") as f:
                if not bool(f.attrs.get("done", False)):
                    continue
                actions = f["actions"][:]
                finite_vals = actions[np.isfinite(actions)]   # mask out inf/NaN
                if finite_vals.size == 0:
                    continue
                global_min = min(global_min, float(finite_vals.min()))
                global_max = max(global_max, float(finite_vals.max()))
                files_scanned += 1
        except Exception:
            continue

    if files_scanned == 0:
        raise RuntimeError(f"No valid HDF5 files found in {hdf5_dir}.")

    observed_range = global_max - global_min
    lo = global_min - margin * observed_range
    hi = global_max + margin * observed_range

    log.info(
        f"Action bounds from {files_scanned} files — "
        f"observed=[{global_min:.3f}, {global_max:.3f}]  "
        f"with {margin*100:.0f}% margin → [{lo:.3f}, {hi:.3f}]"
    )
    return lo, hi


# --- Per-episode validator ----------------------------------------

def validate_episode(
    episode_idx: int,
    action_min: float,
    action_max: float,
) -> dict[str, Any]:
    """
    Open one HDF5 file, run all quality checks, return a report dict.

    Keys:
        episode_idx : int
        path        : str
        passed      : bool
        failures    : list[str]     - empty if passed
        frame_count : int | None    - None if file unreadable
    """
    path = episode_hdf5_path(episode_idx)
    report: dict[str, Any] = {
        "episode_idx": episode_idx,
        "path": str(path),
        "passed": False,
        "failures" : [],
        "frame_count": None
    }

    # File accessible?
    if not path.exists():
        report["failures"].append(f"file_missing")
        return report
    
    try:
        f = h5py.File(path, "r")
    except Exception as e:
        report["failures"].append(f"file_unreadable: {e}")
        return report
    
    with f:
        # Check 1: done sentinel
        # Absent means ingest.py crashed mid-write. Fileis structurally corrupt.
        if not bool(f.attrs.get("done", False)):
            report["failures"].append("missing_done_sentinel")
            return report           # no point continuing - arrays may be truncated

        # Check 2: required datasets exist
        required = ["observations/images", "observations/state", "actions"]
        for ds_path in required:
            if ds_path not in f:
                report["failures"].append(f"missing_dataset: {ds_path}")

        if report["failures"]:
            return report

        images_ds  = f["observations/images"]       # (T, H, W, C) unit8
        actions_ds = f["actions"]                   # (T, A) float32
        state_ds   = f["observations/state"]        # (T, A) float32

        T_images    = images_ds.shape[0]
        T_actions   = actions_ds.shape[0]
        T_state     = state_ds.shape[0]
        report["frame_count"] = T_images

        # Check 3: frame count parity
        # actions and state must have exactly as many rows as images has frames.
        # A mismatch means the write was truncated or source data was misaligned.
        if T_actions != T_images:
            report["failures"].append(
                f"frame_counts_mismatch: images={T_images} actions={T_actions}"
            )
        if T_state != T_images:
            report["failures"].append(
                f"frame_count_mismatch: images={T_images} actions={T_state}"
            )
        
        # Check 4: frame_count attirbute matches actual array length
        attr_fc = f.attrs.get("frame_count", None)
        if attr_fc is None:
            report["failures"].append("missing_frame_count_attr")
        elif int(attr_fc) != T_images:
            report["failures"].append(
                f"frame_count_attr_mismatch: attr={attr_fc} images:{T_images}"
            )
        
        # Check 5: image dtype and value range
        # uint8 is the contract from write_one_episode.py.
        # Float32 images won't crash anything, but downstream normalization
        # will be wrong and storage is 4x what it shuold be.
        if images_ds.dtype != np.uint8:
            report["failures"].append(
                f"images_wrong_dtype: expected unit8, got {images_ds.dtype}"
            )
        else:
            images = images_ds[:]           # load only after dtype confirmed
            img_min, img_max = int(images.min()), int(images.max())
            if img_min < 0 or img_max > 255:
                report["failures"].append(
                    f"images_out_of_range: min={img_min} max={img_max}"
                )

        # Check 6: NaN / Inf in actions and state
        # np.isfinite catches both NaN and Inf in one pass.
        # A single NaN will poison the training loss silently on step 1.
        actions = actions_ds[:]
        state   = state_ds[:]

        if not np.all(np.isfinite(actions)):
            n_bad = int(np.sum(~np.isfinite(actions)))
            report["failures"].append(f"actions_nan_or_inf: {n_bad} bad values")

        if not np.all(np.isfinite(state)):
            n_bad = int(np.sum(~np.isfinite(state)))
            report["failures"].append(f"state_nan_or_inf: {n_bad} bad values")

        # Check 7: action bounds
        # Bounds were computed empirically from your data with a margin.
        # Catches values that are finite but physically impossible -
        # e.g. 1e6 from a dtype promotion bug or a corrupt float.
        if np.all(np.isfinite(actions)):    # skip if NaN already flagged above
            a_min, a_max = float(actions.min()), float(actions.max())
            if a_min < action_min or a_max > action_max:
                report["failures"].append(
                    f"action_out_of_bounds: min={a_min:.3f} max={a_max:.3f} "
                    f"(expected [{action_min:.3f}, {action_max:.3f}])"
                )

    report["passed"] = len(report["failures"]) == 0
    return report


# Aggregation
def aggregate(reports: list[dict]) -> dict:
    """ Roll up per-episode reports into a summary dict """
    passed = [r for r in reports if r["passed"]]
    failed = [r for r in reports if not r["passed"]]

    failure_counts: dict[str, int] = {}
    for r in failed:
        for reason in r["failures"]:
            key = reason.split(":")[0]      # bucket by prefix, e.g. "actions_nan_or_inf"
            failure_counts[key] = failure_counts.get(key, 0) + 1

    return {
        "total_episodes": len(reports),
        "passed": len(passed),
        "failed": len(failed),
        "pass_rate": len(passed) / len(reports) if reports else 0.0,
        "failure_reasons": failure_counts,
        "failed_episodes": [r["episode_idx"] for r in failed],
    }


def run(max_episodes: int | None = None, episode: int | None = None) -> None:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    # Compute action bounds before validation loop
    # One fast pass over actions-only (no images). DOne once, reused per episode.
    action_min, action_max = compute_action_bounds(HDF5_DIR)

    if episode is not None:
        indices = [episode]
    else:
        ds = LeRobotDataset(DATASET_NAME)
        total = ds.meta.total_episodes
        n = min(total, max_episodes) if max_episodes else total
        indices = list(range(n))

    log.info(f"Validating {len(indices)} episode(s) ...")

    reports = []
    for idx in indices:
        report = validate_episode(idx, action_min, action_max)
        reports.append(report)

        if not report["passed"]:
            log.warning(f"ep_{idx:06d} FAILED - {report['failures']}")
        elif idx % 50 == 0:
            log.info(f"ep_{idx:06d} OK (frame_count={report['frame_count']})")

    summary = aggregate(reports)

    log.info("=" * 60)
    log.info("VALIDATION COMPLETE")
    log.info(f"  Total   : {summary['total_episodes']}")
    log.info(f"  Passed  : {summary['passed']}")
    log.info(f"  Failed  : {summary['failed']}")
    log.info(f"  Pass %  : {summary['pass_rate'] * 100:.1f}%")
    if summary["failure_reasons"]:
        log.info(f"  Failures: {summary['failure_reasons']}")
    log.info("=" * 60)

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = OUTPUTS_DIR / "validation_report.json"
    report_path.write_text(
        json.dumps(
            {
                "summary": summary,
                "action_bounts_used": {"min": action_min, "max": action_max},
                "episodes": reports
            },
            indent=2,
        )
    )
    log.info(f"Report saved -> {report_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument(
        "--episode", type=int, default=None,
        help="Validate a single episode (debug mode)"
    )
    args = parser.parse_args()
    run(max_episodes=args.max_episodes, episode=args.episode)
