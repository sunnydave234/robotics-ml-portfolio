"""
bc_dataset.py — W4D3
Adapter: wraps RobotEpisodeDataset and adds `prev_action` as a feature.

prev_action[t] = action[t-1], or action[t] if t==0 (episode start —
repeat-pad, same convention as context_window's boundary handling).

Does not modify robot_dataset.py or its on-disk schema.
"""
from torch.utils.data import Dataset
from robot_dataset import RobotEpisodeDataset

class BCDataset(Dataset):
    def __init__(self, base: RobotEpisodeDataset):
        self.base = base

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int) -> dict:
        sample = self.base[idx]

        if sample["frame_offset"] == 0:
            # Episode start - no real t-1. so repeat the start action.
            # matching the boundary convention from context_window's __getitem__..
            prev_action = sample["action"]
        else:
            # idx-1 is guaranteed to be frame t-1 of THIS episode,
            # because frame_offset > 0 mean idx isn't an episode boundary.
            prev_action = self.base[idx - 1]["action"]
        
        return {
            "state": sample["state"],
            "prev_action": prev_action,
            "action": sample["action"]
        }


if __name__ == "__main__":
    from config import OUTPUTS_DIR

    base = RobotEpisodeDataset(OUTPUTS_DIR / "metadata.parquet", normalize=True)
    ds = BCDataset(base)

    s0 = ds[0]  # episode 0, frame 0 - boundary case
    print("idx 0    shapes:", {k: v.shape for k, v in s0.items()})
    print("idx 0    prev_action == action", (s0["prev_action"] == s0["action"]).all().item())

    s1 = ds[1]  # episode 0, frame 1 -  normal case
    print("idx 1    prev_action == ds[0].action:", (s1["prev_action"] == s0["action"]).all().item())
