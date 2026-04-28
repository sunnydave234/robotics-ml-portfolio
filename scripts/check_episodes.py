"""
Smoke-test extract_episode + visualizer on a spread of episode indices.
Run this before pushing. If any episode fails, fix extract_episode first.
"""
import sys
from pathlib import Path

# Add project root to path so imports resolve from anywhere
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from extract_episode import extract_episode
from visualize_episode import write_episode_video_with_actions
from config import DEFAULT_DATASET, episode_video_path

PROBE_EPISODES = [0, 5, 10, 50, 100]

def check_episode(ds: LeRobotDataset, idx: int) -> dict:
    """Extract one episode and validate its shape contract."""
    ep = extract_episode(ds, episode_idx=idx)

    images  = ep["images"]   # (T, C, H, W)
    actions = ep["actions"]  # (T, action_dim)

    # Invariant that must hold for every episode
    assert images.ndim == 4, f"ep {idx}: images.ndim={images.ndim}, expected 4"
    assert actions.ndim == 2, f"ep {idx}: actions.ndim={actions.ndim}, expected 2"
    assert images.shape[0] == actions.shape[0], (
        f"ep {idx}: frame/action count mismatch "
        f"({images.shape[0]} vs {actions.shape[0]})"
    )
    assert images.dtype  == torch.float32, f"ep {idx}: images dtype=={images.dtype}"
    assert actions.dtype == torch.float32, f"ep {idx}: images dtype=={actions.dtype}"
    assert images.device.type   == "cpu", f"ep {idx}: images on {images.device} - expected CPU"
    assert actions.device.type  == "cpu", f"ep {idx}: images on {images.device} - expected CPU"

    T = images.shape[0]

    # Cross-check against metadata - catch index bugs before they corrupt a video
    expected_T = int(ds.meta.episodes["length"][idx])
    assert T == expected_T, (
        f"ep {idx}: expected {T} frames but metadata says {expected_T}"
    )

    return {
        "episode": idx,
        "frames":   T,
        "action_dim": actions.shape[1],
        "img_shape": list(images.shape[1:]),    # [C, H, W]
    }


def main():
    print(f"Loading {DEFAULT_DATASET}...")
    ds = LeRobotDataset(DEFAULT_DATASET)

    total = ds.meta.total_episodes
    print(f"Dataset: {total} episodes\n")
    
    results = []
    for idx in PROBE_EPISODES:
        if idx >= total:
            print(f"    ep {idx:>3}     SKIP (dataset only has {total} episodes)")
            continue
        try:
            info = check_episode(ds, idx)
            print(
                f"  ep {info['episode']:>3}   "
                f"T={info['frames']:>4} "
                f"action_dim={info['action_dim']}   "
                f"image_shape={info['img_shape']}   OK"
            )
            results.append(info)
        except Exception as e:
            print(f"    ep {idx:>3} FAIL - {e}")
            sys.exit(1)

    print(f"\nAll {len(results)} probes passed.")


    # Render episode 0 as the canonical README thumbnail source
    print("\nRendering episode 0 for README thumbnail...")
    ep0 = extract_episode(ds, episode_idx=0)
    out = episode_video_path(0)
    write_episode_video_with_actions(ep0, str(out), fps=10, upscale=4)
    print(f"Video -> {out}")


if __name__ == "__main__":
    main()
