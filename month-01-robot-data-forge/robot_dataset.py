import json
import numpy as np
import pandas as pd
import torch
import h5py
from config import OUTPUTS_DIR
from torch.utils.data import Dataset


class RobotEpisodeDataset(Dataset):
    """
    Flat-indexed PyTorch Dataset over a collection of per-episode HDF5 files.

    Each integer index maps to one (frame, action) pair somewhere across all episodes.
    The episode/frame lookup is solved at __init__ time using a cumulative frame index -
    same pattern as Hive partition manifest.
    """

    def __init__(self, parquet_path: str, normalize: bool = False, context_window: int = 1):
        # Load the episode index built by build_index.py
        # Each row: episode_id, frame_count, file_path, success, task
        self.df = pd.read_parquet(parquet_path)

        # Build the cumulative frame boundry array.
        # frame_counts = [49, 124, 87, ...]
        # cumulative = [49, 173, 260, ...] <- exclusive upper bound per episode

        # This is the partition mainfest. Given any flat idx, searchsorted returns
        # the first episode whose upper bound exceeds idx - that's the right episode.
        frame_counts = self.df["frame_count"].values    # shape: (N_episodes,)
        self.cumulative = np.cumsum(frame_counts)       # shape: (N_episodes,)
        self.total_frames = int(self.cumulative[-1])

        # Keep file paths as a plain list - fast index access in __getitem__
        self.file_paths = self.df["file_path"].tolist()

        self.normalize = normalize
        if context_window < 1:
            raise ValueError(f"context_window must be >= 1, got {context_window}")
        self.context_window = context_window
        self._norm_stats = None

        if normalize:
            stats_path = OUTPUTS_DIR / "dataset_stats.json"

            # Give a clear error if someone forgot to run compute_stats.py first
            if not stats_path.exists():
                raise FileNotFOundError(
                    f"Stats file not found at {stats_path}. "
                    "Run compute_stats.py first."
                )
            
            with open(stats_path) as f:
                raw = json.load(f)
            
            # Convert to tensors once here - not every time __getitem__ is called
            self._norm_stats = {
                "action_mean":  torch.tensor(raw["actions"]["mean"], dtype=torch.float32),
                "action_std":   torch.tensor(raw["actions"]["std"], dtype=torch.float32),
                "state_mean":   torch.tensor(raw["states"]["mean"], dtype=torch.float32),
                "state_std":    torch.tensor(raw["states"]["std"], dtype=torch.float32),
            }

    def __len__(self) -> int:
        return self.total_frames

    def __getitem__(self, idx: int) -> dict:
        if idx < 0 or idx >= self.total_frames:
            raise IndexError(f"Index {idx} out of range [0, {self.total_frames})")

        # ── Step 1: find which episode this flat index falls into ──────────────
        # searchsorted returns the first position where cumulative[pos] > idx.
        # side='right' means: for an idx exactly equal to a boundary,
        # we land in the *next* episode — which is correct, since boundaries
        # are exclusive upper bounds.
        ep_idx = int(np.searchsorted(self.cumulative, idx, side="right"))

        # ── Step 2: compute within-episode frame offset ────────────────────────
        # cumulative[ep_idx - 1] is the start of this episode in flat space.
        # For ep_idx == 0 (the very first episode), the start is 0.
        ep_start = int(self.cumulative[ep_idx - 1]) if ep_idx > 0 else 0
        frame_offset = idx - ep_start   # local frame index inside this episode

        # ── Step 3: open the HDF5 file and read exactly one frame ─────────────
        # CRITICAL: open here, not in __init__.
        # h5py file handles are not fork-safe. Opening per-call is cheap —
        # HDF5 caches pages internally and our chunk shape is (1, H, W, C),
        # so one chunk I/O = exactly one frame read.
        with h5py.File(self.file_paths[ep_idx], "r") as f:
            # images stored as (T, H, W, C) uint8 in [0, 255]
            # Read only the frame we need — HDF5 fetches one chunk (one frame)
            if self.context_window == 1:
                # images stored as (T, H, W, C) uint8 in [0, 255]
                # Read only the frame we need — HDF5 fetches one chunk (one frame)
                raw_image = f["observations/images"][frame_offset]   # (H, W, C) uint8
            else:
                K = self.context_window
                start = frame_offset - K + 1
                pad_count = max(0, -start)
                start = max(start, 0)

                # (window_len, H, W, C) uint8, window_len = K - pad_count
                window = f["observations/images"][start:frame_offset + 1]

                if pad_count > 0:
                    # Repeat the earliest available frame to fill in the missing
                    # leading context — keeps the window inside this episode,
                    # never reaching into the previous one.
                    pad = np.repeat(window[0:1], pad_count, axis=0)
                    window = np.concatenate([pad, window], axis=0)

                raw_image = window   # (K, H, W, C) uint8

            # actions stored as (T, action_dim) float32
            action = f["actions"][frame_offset]                  # (action_dim,) float32
            state = f["observations/state"][frame_offset]

        # ── Step 4: convert to tensors ─────────────────────────────────────────
        # Normalize image: uint8 [0, 255] → float32 [0.0, 1.0]
        # PyTorch convention is channel-first (C, H, W), so permute.
        if self.context_window == 1:
            image_tensor = (
                torch.from_numpy(raw_image.copy())   # (H, W, C) uint8
                .permute(2, 0, 1)                    # (C, H, W) uint8
                .float()                             # (C, H, W) float32
                .div(255.0)                          # (C, H, W) float32 in [0.0, 1.0]
            )
        else:
            # (K, H, W, C) uint8 -> (K, H, W, C) float32 [0.0, 1.0]
            # No permute — matches HDF5 on-disk layout, and is the (K,H,W,C)
            # shape manipulation policies expect for observation history.
            image_tensor = (
                torch.from_numpy(raw_image.copy())
                .float()
                .div(255.0)
            )

        action_tensor = torch.from_numpy(action.copy())  # (action_dim,) float32
        state_tensor  = torch.from_numpy(state.copy())

        # Apply normalization if the flag is set
        if self.normalize and self._norm_stats is not None:
            s = self._norm_stats
            action_tensor = (action_tensor - s["action_mean"]) / (s["action_std"] + 1e-8)
            state_tensor  = (state_tensor  - s["state_mean"])  / (s["state_std"]  + 1e-8)
            # Note: 1e-8 is a tiny safety number to avoid dividing by zero
        
        return {
            "image":      image_tensor,    # (C,H,W) if context_window==1, else (K,H,W,C)
            "action":     action_tensor,   # (action_dim,) float32
            "state":      state_tensor,
            "episode_idx": ep_idx,         # int — useful for debugging boundary cases
            "frame_offset": frame_offset,  # int — local position within episode
        }
if __name__ == "__main__":
    from config import OUTPUTS_DIR

    parquet_path = OUTPUTS_DIR / "metadata.parquet"

    # Default behavior, unchanged
    ds = RobotEpisodeDataset(parquet_path)
    print("context_window=1, idx=0 image shape:", ds[0]["image"].shape)  # (3, 96, 96)

    # K=3 — check the boundary-padding case at the very start of an episode
    ds_ctx = RobotEpisodeDataset(parquet_path, context_window=3)
    sample0 = ds_ctx[0]
    sample1 = ds_ctx[1]
    print("context_window=3, idx=0 image shape:", sample0["image"].shape)  # (3, 96, 96, 3)
    print("idx=0 frame_offset:", sample0["frame_offset"])  # 0 -> all 3 frames are frame 0
    print("idx=1 frame_offset:", sample1["frame_offset"])  # 1 -> [frame0, frame0, frame1]

    # Sanity check: at frame_offset=0, all K frames in the window should be identical
    import torch
    assert torch.equal(sample0["image"][0], sample0["image"][1])
    assert torch.equal(sample0["image"][1], sample0["image"][2])
    print("Boundary padding check: passed")
