import json
import torch
import argparse
from pathlib import Path
from lerobot.datasets.lerobot_dataset import LeRobotDataset


def get_episode_lenghts(dataset: LeRobotDataset) -> torch.Tensor:
    """
    Pull per-episode frame counts directly from episode metadata.
    In V3, dataset.meta.episodes is a pandas DataFrame with a 'lenght' column.
    This is O(1) - no frame iteration required.
    """
    # .tolist() converts the pandas Series to a plain Python list
    # torch.tensor() then buids the tensor without going through numpy
    return torch.tensor(
        dataset.meta.episodes["length"],
        dtype=torch.long,
    )


def get_success_rate(dataset: LeRobotDataset) -> float | None:
    """
    Read the terminal success flag for each episode.
    LeRobot stores 'next.success' as a per-frame bool - the episode-level
    result lives on the final frame of each episode (dataset_to_index - 1).
    """

    if "next.success" not in dataset.features:
        return None

    # dataset.meta.episodes["dataset_to_index"] gives the exclusve upper bound
    # so the last frame of episode i is at (dataset_to_index[i] - 1)
    last_indices = (
        torch.tensor(dataset.meta.episodes["dataset_to_index"]) - 1
    ).tolist()

    success = torch.tensor(
        [bool(dataset[int(i)]["next.success"]) for i in last_indices],
        dtype=torch.float32,
    )
    return success.mean().item()


def build_profile(dataset: LeRobotDataset) -> dict:
    lengths   = get_episode_lenghts(dataset)     # (num_episodes,) int64
    lenghts_f = lengths.float()

    # Pull one sample to inspect image and action shapes
    sample = dataset[0]

    # Image key varies by datset: pusht uses 'observation.image',
    # aloha uses 'observation.images.top' etc. - find it dynamically
    image_key = next(k for k in dataset.features if "image" in k.lower())
    image: torch.Tensor = sample[image_key]     # (C, H, W), float32
    action: torch.Tensor = sample["action"]     # (action_dim,)

    return{
        "dataset_id":       dataset.repo_id,
        "total_episodes":   dataset.meta.total_episodes,
        "total_frames":     dataset.meta.total_frames,
        "fps":              dataset.meta.fps,
        "episode_length": {
            "min":      int(lengths.min().item()),
            "max":      int(lengths.max().item()),
            "mean":     round(lenghts_f.mean().item(), 2),
            "std":      round(lenghts_f.std().item(), 2),
            "median":   round(lenghts_f.median().item(), 2),
        },
        "action_dim":       int(action.shape[0]),
        "image_key":        image_key,
        "image_shape_chw":  list(image.shape),      # [C, H, W] - PyTorch convention
        "image_dtype":     str(image.dtype),
        "image_value_range": [
            round(image.min().item(), 4),
            round(image.max().item(), 4),
        ],
        "available_features":   list(dataset.features.keys()),
        "success_rate":         get_success_rate(dataset),
    }


def print_profile_table(profile: dict) -> None:
    """Formatted terminal output — no external table library needed."""
    w   = 46
    sep = "─" * w

    def row(label, value):
        print(f"  {label:<30} {str(value):>10}")

    print(f"\n{'Dataset Profile':^{w}}")
    print(sep)
    row("Dataset ID",       profile["dataset_id"])
    row("Total episodes",   profile["total_episodes"])
    row("Total frames",     profile["total_frames"])
    row("FPS",              profile["fps"])
    print(sep)

    el = profile["episode_length"]
    row("Episode length — min",    el["min"])
    row("Episode length — mean",   el["mean"])
    row("Episode length — median", el["median"])
    row("Episode length — max",    el["max"])
    row("Episode length — std",    el["std"])
    print(sep)

    row("Action dim",       profile["action_dim"])
    row("Image key",        profile["image_key"])
    row("Image shape (C,H,W)", profile["image_shape_chw"])
    row("Image dtype",      profile["image_dtype"])
    row("Image value range", profile["image_value_range"])
    print(sep)

    sr = profile["success_rate"]
    row("Success rate", f"{sr:.1%}" if sr is not None else "n/a")
    print(sep)
    print(f"\n  Features: {', '.join(profile['available_features'])}")

def save_histogram(lengths: torch.Tensor, output_path: Path) -> None:
    """
    Render an episode-length histogram and save as PNG.
    matplotlib only touches data at the very end — all stats computed in PyTorch first.
    """
    import matplotlib
    matplotlib.use("Agg")           # non-interactive backend — no display required
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)

    lengths_f = lengths.float()

    fig, ax = plt.subplots(figsize=(9, 4))

    # .tolist() → plain Python list is the matplotlib-safe path (no numpy dependency)
    ax.hist(lengths.tolist(), bins=40, color="#4C8BF5", edgecolor="white", linewidth=0.4)

    ax.set_title("Episode Length Distribution — lerobot/pusht", fontsize=13, pad=12)
    ax.set_ylabel("Episode count")

    stats_line = (
        f"n={lengths.shape[0]}  |  "
        f"μ={lengths_f.mean():.0f}  |  "
        f"median={lengths_f.median():.0f}  |  "
        f"σ={lengths_f.std():.0f}  |  "
        f"min={lengths.min()}  |  max={lengths.max()}"
    )
    ax.set_xlabel(f"Frames per episode\n{stats_line}", fontsize=9)

    # Annotate mean as a vertical line — makes skew obvious at a glance
    mean_val = lengths_f.mean().item()
    ax.axvline(mean_val, color="#FF4B4B", linewidth=1.5, linestyle="--", label=f"mean = {mean_val:.0f}")
    ax.legend(fontsize=9)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Histogram → {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Profile a LeRobot dataset.")
    parser.add_argument("--dataset",   default="lerobot/pusht",
                        help="HuggingFace dataset ID (default: lerobot/pusht)")
    parser.add_argument("--output",    default="dataset_profile.json",
                        help="Path for the JSON profile output")
    parser.add_argument("--histogram", default="outputs/episode_length_hist.png",
                        help="Path for the histogram PNG")
    args = parser.parse_args()

    print(f"Loading {args.dataset}...")
    ds = LeRobotDataset(args.dataset)

    print("Building profile...")
    profile = build_profile(ds)
    print_profile_table(profile)

    # ── Save JSON ──────────────────────────────────────────────────────────
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(profile, indent=2))
    print(f"\n  Profile JSON → {out}")

    # ── Save histogram ─────────────────────────────────────────────────────
    lengths = get_episode_lenghts(ds)
    save_histogram(lengths, Path(args.histogram))


if __name__ == "__main__":
    main()
