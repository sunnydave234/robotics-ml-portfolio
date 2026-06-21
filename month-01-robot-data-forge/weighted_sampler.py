"""
WeightedEpisodeSampler: a PyTorch Sampler that draws flat frame indices
proportionally to per-episode weights stored in the Parquet index.

Architecture:
  metadata.parquet  →  episode weights
                    ↓
  frame_weights[i]  =  episode_weight[ep(i)]  for every flat frame index i
                    ↓
  torch.multinomial(frame_weights, N)  →  batch of frame indices
                    ↓
  DataLoader  →  RobotEpisodeDataset.__getitem__(flat_idx)

Run: python weighted_sampler.py
"""

import collections
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Sampler

from config import OUTPUTS_DIR
# RobotEpisodeDataset must be importable — adjust path if yours lives elsewhere
from robot_dataset import RobotEpisodeDataset


# ─────────────────────────────────────────────────────────────────────────────
class WeightedEpisodeSampler(Sampler):
    """
    Draws flat frame indices weighted by per-episode scores in a Parquet DataFrame.

    Args:
        df:           metadata DataFrame with columns [frame_count, <weight_col>]
        weight_col:   column name to use as sampling weight (e.g. 'weight')
        num_samples:  total number of flat frame indices to draw per epoch.
                      Defaults to the full dataset size (sum of frame_counts).
        batch_size:   if > 0, __iter__ yields batches (lists) of indices.
                      if == 0 (default), yields individual indices — standard Sampler mode.
        generator:    optional torch.Generator for reproducible sampling
    """

    def __init__(
        self,
        df: pd.DataFrame,
        weight_col: str,
        num_samples: int | None = None,
        batch_size: int = 0,
        generator: torch.Generator | None = None,
    ) -> None:
        self.batch_size  = batch_size
        self.generator   = generator

        # ── Total frames in the dataset ───────────────────────────────────────
        # frame_counts[i] = number of frames in episode i
        frame_counts = df["frame_count"].to_numpy(dtype=np.int64)
        self._total_frames = int(frame_counts.sum())
        self.num_samples = num_samples if num_samples is not None else self._total_frames

        # ── Broadcast episode weights → frame weights ─────────────────────────
        # np.repeat([w0, w1, w2, ...], [T0, T1, T2, ...])
        # → array of length sum(T_i) where every frame in episode i gets weight w_i
        #
        # This preserves episode-length proportionality:
        # a 246-frame episode gets 246× the influence of a 1-frame episode
        # at any given weight value, not the same influence as a 49-frame episode.
        episode_weights = df[weight_col].to_numpy(dtype=np.float64)
        frame_weights   = np.repeat(episode_weights, frame_counts)

        # Normalise to a probability distribution — required by torch.multinomial
        frame_weights /= frame_weights.sum()

        # Store as float32 tensor — multinomial doesn't need float64 precision
        self._weights = torch.from_numpy(frame_weights).float()

    # ── Core sampling logic ───────────────────────────────────────────────────
    def _draw_indices(self) -> torch.Tensor:
        """Draw self.num_samples frame indices with replacement."""
        return torch.multinomial(
            self._weights,
            num_samples=self.num_samples,
            replacement=True,         # with-replacement: same frame can appear twice in one epoch
            generator=self.generator, # None → use global RNG
        )

    def __iter__(self):
        indices = self._draw_indices()

        if self.batch_size <= 0:
            # Standard Sampler mode: yield one index at a time
            yield from indices.tolist()
        else:
            # batch_sampler mode: yield complete batches as lists
            # Drop the last partial batch to keep all batches equal size
            n_full_batches = len(indices) // self.batch_size
            for i in range(n_full_batches):
                start = i * self.batch_size
                yield indices[start : start + self.batch_size].tolist()

    def __len__(self) -> int:
        if self.batch_size <= 0:
            return self.num_samples
        return self.num_samples // self.batch_size


# ─────────────────────────────────────────────────────────────────────────────
def verify_sampler(df: pd.DataFrame, n_draws: int = 10_000) -> None:
    """
    Draw n_draws frame indices and verify the episode-level distribution
    matches the expected weight ratio.

    Expected for pusht (all success=0.0, weight=0.3 uniform):
      All episodes equally weighted → uniform draw → each episode ~n_draws/206 frames.

    If you manually set one episode to weight=1.0, it should appear ~3.3× more often
    than the rest (1.0 / 0.3 = 3.33). This test verifies that ratio.
    """
    print("=" * 60)
    print(f"Sampler verification — {n_draws:,} draws")
    print("=" * 60)

    sampler = WeightedEpisodeSampler(df, weight_col="weight", num_samples=n_draws)

    # Draw all indices at once
    indices = list(sampler)
    assert len(indices) == n_draws, f"Expected {n_draws} indices, got {len(indices)}"

    # ── Map flat frame index → episode index ──────────────────────────────────
    # Same cumsum trick as RobotEpisodeDataset: cumulative frame offsets
    frame_counts = df["frame_count"].to_numpy(dtype=np.int64)
    episode_ends = np.cumsum(frame_counts)   # episode i ends at episode_ends[i]

    # np.searchsorted: for each flat index, find which episode it belongs to
    episode_hits = np.searchsorted(episode_ends, indices, side="right")

    # ── Count draws per episode ───────────────────────────────────────────────
    hit_counts = collections.Counter(episode_hits.tolist())

    # ── Aggregate by weight class ─────────────────────────────────────────────
    success_episodes = set(df.index[df["success"] > 0.0].tolist())
    failure_episodes = set(df.index[df["success"] == 0.0].tolist())

    success_draws = sum(hit_counts[ep] for ep in success_episodes)
    failure_draws = sum(hit_counts[ep] for ep in failure_episodes)
    total_draws   = success_draws + failure_draws

    print(f"\nEpisodes — success: {len(success_episodes)}, failure: {len(failure_episodes)}")
    print(f"\nDraws from success episodes: {success_draws:,}  ({100*success_draws/total_draws:.1f}%)")
    print(f"Draws from failure episodes: {failure_draws:,}  ({100*failure_draws/total_draws:.1f}%)")

    # ── Expected ratios ───────────────────────────────────────────────────────
    # Per-frame weight total for each class:
    #   class_weight_mass = sum(episode_weight * frame_count) for episodes in class
    success_mass = sum(
        df.loc[ep, "weight"] * df.loc[ep, "frame_count"]
        for ep in success_episodes
    ) if success_episodes else 0.0

    failure_mass = sum(
        df.loc[ep, "weight"] * df.loc[ep, "frame_count"]
        for ep in failure_episodes
    )

    total_mass = success_mass + failure_mass
    expected_success_pct = 100 * success_mass / total_mass if total_mass > 0 else 0.0
    expected_failure_pct = 100 * failure_mass / total_mass if total_mass > 0 else 0.0

    print(f"\nExpected success %: {expected_success_pct:.1f}%")
    print(f"Expected failure %: {expected_failure_pct:.1f}%")

    # For pusht: all episodes are failure, so expect ~100% failure draws
    # If you added a success episode for testing, expect ~77% / ~23% split
    print("\n✓ Verification complete — check observed vs expected match above")
    print("  (Small variance is normal for 10k draws — re-run to confirm)")


# ─────────────────────────────────────────────────────────────────────────────
def run_dataloader_demo(df: pd.DataFrame, n_batches: int = 5) -> None:
    """Wire WeightedEpisodeSampler into a DataLoader and iterate n_batches."""
    print("\n" + "=" * 60)
    print("DataLoader demo — 5 batches with WeightedEpisodeSampler")
    print("=" * 60)

    # RobotEpisodeDataset takes parquet_path as its first positional arg —
    # it loads dataset_stats.json internally from OUTPUTS_DIR when normalize=True
    ds = RobotEpisodeDataset(
        OUTPUTS_DIR / "metadata.parquet",
        normalize=True,
    )

    batch_size = 32
    sampler = WeightedEpisodeSampler(
        df,
        weight_col="weight",
        num_samples=len(ds),   # one full epoch worth of draws
        batch_size=batch_size,
    )

    loader = DataLoader(
        ds,
        batch_sampler=sampler,   # batch_sampler replaces both sampler + batch_size args
        num_workers=4,
        persistent_workers=True,
    )

    print(f"\nDataset: {len(ds):,} frames  |  Sampler draws: {sampler.num_samples:,}")
    print(f"Batch size: {batch_size}  |  Batches per epoch: {len(sampler)}")
    print(f"\nIterating {n_batches} batches...\n")

    for i, batch in enumerate(loader):
        if i >= n_batches:
            break

        images  = batch["image"]
        actions = batch["action"]
        state   = batch["state"]

        print(
            f"  Batch {i+1}: "
            f"images {tuple(images.shape)}  "
            f"actions {tuple(actions.shape)}  "
            f"state {tuple(state.shape)}  "
            f"img mean={images.float().mean():.4f}"
        )

    print("\n✓ DataLoader ran without errors")


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parquet_path = OUTPUTS_DIR / "metadata.parquet"
    df = pd.read_parquet(parquet_path, engine="pyarrow")

    if "weight" not in df.columns:
        raise RuntimeError(
            "No 'weight' column found in metadata.parquet.\n"
            "Run  python add_weights.py  first."
        )

    verify_sampler(df)
    run_dataloader_demo(df)
