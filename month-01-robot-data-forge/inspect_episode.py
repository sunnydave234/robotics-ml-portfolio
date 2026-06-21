import argparse
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset

def get_episode_frames(ds, episode_idx):
    start = ds.meta.episodes["dataset_from_index"][episode_idx]
    end = ds.meta.episodes["dataset_to_index"][episode_idx]
    return [ds[i] for i in range(start, end)]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episode", type=int, required=True)
    args = parser.parse_args()

    ds = LeRobotDataset("lerobot/pusht")
    frames = get_episode_frames(ds, args.episode)

    print(f"Episode {args.episode}: {len(frames)} frames")

    for i, frame in enumerate(frames):
        print(f"\n--- Frame {i} ---")
        for key, val in frame.items():
            if hasattr(val, "shape"):
                print(f"  {key}: shape={val.shape}, dtype={val.dtype}")
            else:
                print(f"  {key}: {val}")
    actions = torch.stack([f["action"] for f in frames])
    states  = torch.stack([f["observation.state"] for f in frames])

    print(f"\n=== Episode {args.episode} Summary ===")
    print(f"  frames:         {len(frames)}")
    print(f"  action min/max: {actions.min():.3f} / {actions.max():.3f}")
    print(f"  action mean:    {actions.mean(dim=0)}")
    print(f"  state min/max:  {states.min():.3f} / {states.max():.3f}")
    print(f"  timestamps:     {frames[0]['timestamp'].item():.3f} → {frames[-1]['timestamp'].item():.3f}")