"""
visualize_batch.py

W3D5 — render one DataLoader batch as images + action bar charts.
This is a visual sanity check: does the pipeline produce real,
correctly-shaped, correctly-aligned data?

Run: python visualize_batch.py
Out: outputs/batch_visualization.png
"""

import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

from config import OUTPUTS_DIR
from robot_dataset import RobotEpisodeDataset


def visualize_batch(batch_size: int = 8, seed: int = 42) -> None:
    # normalize=False -> action values stay in raw units (pusht: pixel-space
    # x/y target), so the bar chart shows something physically meaningful
    dataset = RobotEpisodeDataset(OUTPUTS_DIR / "metadata.parquet", normalize=False)

    # num_workers=0: this is a one-shot sanity check, not a throughput test
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=generator)

    batch = next(iter(loader))
    images = batch["image"]    # (B, C, H, W) float32 [0,1]
    actions = batch["action"]  # (B, action_dim)

    B = images.shape[0]
    action_dim = actions.shape[1]
    action_labels = ["x", "y"][:action_dim]  # pusht: 2D end-effector target

    # Shared y-axis scale across all bar charts so magnitudes are comparable
    a_min, a_max = actions.min().item(), actions.max().item()
    margin = (a_max - a_min) * 0.1 or 1.0
    ylim = (a_min - margin, a_max + margin)

    n_cols = 4
    n_sample_rows = (B + n_cols - 1) // n_cols  # 8 samples / 4 cols = 2 rows
    n_fig_rows = n_sample_rows * 2              # each sample row -> image row + bar row

    fig, axes = plt.subplots(n_fig_rows, n_cols, figsize=(n_cols * 3, n_fig_rows * 3))

    for i in range(B):
        sample_row, col = divmod(i, n_cols)
        img_ax = axes[sample_row * 2, col]
        bar_ax = axes[sample_row * 2 + 1, col]

        # (C,H,W) -> (H,W,C); .contiguous() required after .permute() before .numpy()
        img = images[i].permute(1, 2, 0).contiguous().numpy()
        img_ax.imshow(np.clip(img, 0, 1))
        img_ax.set_title(f"sample {i}", fontsize=9)
        img_ax.axis("off")

        action_vals = actions[i].numpy()
        bar_ax.bar(action_labels, action_vals)
        bar_ax.set_ylim(*ylim)
        bar_ax.tick_params(labelsize=8)

    fig.suptitle("RobotEpisodeDataset batch sample (normalize=False)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    out_path = OUTPUTS_DIR / "batch_visualization.png"
    fig.savefig(out_path, dpi=120)
    plt.close(fig)

    print(f"Saved: {out_path}")
    print(f"Batch shapes -> images: {tuple(images.shape)}, actions: {tuple(actions.shape)}")


if __name__ == "__main__":
    visualize_batch()
