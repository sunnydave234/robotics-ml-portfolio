# verify_normalization.py
"""
Grab 100 samples from the normalized dataset.
Check that mean is near 0 and std is near 1.

Run: python verify_normalization.py
"""

import torch
from torch.utils.data import DataLoader
from robot_dataset import RobotEpisodeDataset
from config import OUTPUTS_DIR


def verify():
    ds = RobotEpisodeDataset(
        parquet_path=OUTPUTS_DIR / "metadata.parquet",
        normalize=True,
    )

    # Load 100 random samples
    loader = DataLoader(ds, batch_size=100, shuffle=True, num_workers=0)
    batch = next(iter(loader))   # grab the first (and only) batch

    actions = batch["action"]   # shape: (100, 2)
    states  = batch["state"]    # shape: (100, 2)

    print("Action stats after normalization:")
    print(f"  mean: {actions.mean(dim=0).tolist()}   (should be near [0, 0])")
    print(f"  std:  {actions.std(dim=0).tolist()}    (should be near [1, 1])")

    print("\nState stats after normalization:")
    print(f"  mean: {states.mean(dim=0).tolist()}    (should be near [0, 0])")
    print(f"  std:  {states.std(dim=0).tolist()}     (should be near [1, 1])")

    # Check: mean should be within 0.2 of zero, std within 0.2 of 1.0
    mean_ok = (actions.mean(dim=0).abs() < 0.2).all()
    std_ok  = ((actions.std(dim=0) - 1.0).abs() < 0.2).all()

    if mean_ok and std_ok:
        print("\n✓  Normalization is working correctly")
    else:
        print("\n✗  Something is wrong — re-run compute_stats.py and check the math")


if __name__ == "__main__":
    verify()
