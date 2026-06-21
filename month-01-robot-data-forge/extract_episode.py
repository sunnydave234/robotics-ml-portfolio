import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset

def extract_episode(ds: LeRobotDataset, episode_idx: int) -> dict:
    start = ds.meta.episodes["dataset_from_index"][episode_idx]
    end = ds.meta.episodes["dataset_to_index"][episode_idx]

    frames = []
    actions = []

    for i in range(start, end):
        sample = ds[i]
        frames.append(sample["observation.image"])
        actions.append(sample["action"])

    return {
        "images":       torch.stack(frames),
        "actions":      torch.stack(actions),
        "episode_idx":  episode_idx,
        "length":      end - start,
    }

if __name__ == "__main__":
    ds = LeRobotDataset("lerobot/pusht")
    ep = extract_episode(ds, episode_idx=0)
    print("image shape:", ep['images'].shape)
    print("action shape:", ep['actions'].shape)
    print("image dtype:", ep['images'].dtype)
    print("device:      ", ep['images'].device)
    print("length:      ", ep['length'])
