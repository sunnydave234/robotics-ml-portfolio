"""
Single-pass streaming stats over the full dataset using Welford's algorithm.
Iterates via DataLoader (num_workers=4, batch_size=64) - never loads full
dataset into RAM.

Run: python compute_stats.py
Out: outputs/dataset_stats.json
"""

import json
import math

import torch
from torch.utils.data import DataLoader

from config import OUTPUTS_DIR
from robot_dataset import RobotEpisodeDataset


# Welford state: one per tensor key

class WelfordAccumulator:
    """
    Tracks running mean and variance for a 1-D or 2-D stream of values.
    Each call to .update(batch) accepts shape (B, D) or (B,).
    You feed it batches one at a time. It never stores the full data.
    """
    def __init__(self):
        self.count  = 0
        self.mean   = None
        self.M2     = None      # running tracker for variance
    
    def update(self, batch: torch.Tensor):
        # Make sure we're working with a 2D tensor: (batch_size, num_dimensions)
        if batch.dim() == 1:
            batch = batch.unsqueeze(1)

        batch = batch.float()
        B = batch.shape[0]  # how many samples in this batch

        # Set up on the very first batch
        if self.mean is None:
            D = batch.shape[1]
            self.mean = torch.zeros(D)
            self.M2   = torch.zeros(D)
        
        self.count += B

        # Update the running mean and variance tracker
        # Two deltas are needed because the mean itself changes with each update
        delta = batch - self.mean               # distance from OLD mean
        self.mean += delta.sum(dim=0) / self.count
        delta2 = batch - self.mean              # distance from NEW mean
        self.M2 += (delta * delta2).sum(dim=0)

    def finalize(self):
        """ Return mean and std as plain Python lists (so they can be saved to JSON). """
        variance = self.M2 / self.count
        std = torch.sqrt(variance)
        return self.mean.tolist(), std.tolist()


def compute_stats():
    # normalize=False because we're computing the stats - we don't have them yet!
    ds = RobotEpisodeDataset(
        parquet_path=OUTPUTS_DIR / "metadata.parquet",
        normalize=False
    )
    print(f"Dataset loaded: {len(ds)} samples, {len(ds.df)} episodes")

    loader = DataLoader(
        ds,
        batch_size=64,      # process 64 samples at a time - never more in memory
        num_workers=4,      # 4 parallel workers reading HDF5 files
        shuffle=False,      # order doesn't matter for computing stats
        pin_memory=False,   # not needed on M4 Mac (unified memory)
    )

    action_acc  = WelfordAccumulator()
    state_acc   = WelfordAccumulator()

    for i, batch in enumerate(loader):
        action_acc.update(batch["action"])
        state_acc.update(batch["state"])

        # Print progress every 50 batches
        if i % 50 == 0:
            pct = 100 * (i * 64) / len(ds)
            print(f"    {pct:.0f}% done - {action_acc.count} samples seen so far")

    action_mean, action_std = action_acc.finalize()
    state_mean, state_std   = state_acc.finalize()

    stats = {
        "actions":      {"mean": action_mean, "std": action_std},
        "states":       {"mean": state_mean, "std": state_std},
        "n_samples":    action_acc.count
    }

    # Save to disk
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUTS_DIR / "dataset_stats.json"
    with open(out_path, "w") as f:
        json.dump(stats, f, indent=2)
    
    print(f"\nDone. Stats saved to: {out_path}")
    print(f"Action mean:    {[round(v, 2) for v in action_mean]}")
    print(f"Action std:     {[round(v, 2) for v in action_std]}")

    return stats


if __name__ == "__main__":
    compute_stats()
