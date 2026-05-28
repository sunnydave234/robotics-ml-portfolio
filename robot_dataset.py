import numpy as np
import pandas as pd
import torch
import h5py
from torch.utils.data import Dataset


class RobotEpisodeDataset(Dataset):
    """
    Flat-indexed PyTorch Dataset over a collection of per-episode HDF5 files.

    Each integer index maps to one (frame, action) pair somewhere across all episodes.
    The episode/frame lookup is solved at __init__ time using a cumulative frame index -
    same pattern as Hive partition manifest.
    """

    def __init__(self, parquet_path: str):
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
            raw_image = f["observations/images"][frame_offset]   # (H, W, C) uint8

            # actions stored as (T, action_dim) float32
            action = f["actions"][frame_offset]                  # (action_dim,) float32

        # ── Step 4: convert to tensors ─────────────────────────────────────────
        # Normalize image: uint8 [0, 255] → float32 [0.0, 1.0]
        # PyTorch convention is channel-first (C, H, W), so permute.
        image_tensor = (
            torch.from_numpy(raw_image.copy())   # (H, W, C) uint8
            .permute(2, 0, 1)                    # (C, H, W) uint8
            .float()                             # (C, H, W) float32
            .div(255.0)                          # (C, H, W) float32 in [0.0, 1.0]
        )

        action_tensor = torch.from_numpy(action.copy())  # (action_dim,) float32

        return {
            "image":      image_tensor,    # (C, H, W) float32
            "action":     action_tensor,   # (action_dim,) float32
            "episode_idx": ep_idx,         # int — useful for debugging boundary cases
            "frame_offset": frame_offset,  # int — local position within episode
        }
