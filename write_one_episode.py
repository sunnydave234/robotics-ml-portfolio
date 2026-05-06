import h5py
import numpy as np
from pathlib import Path
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from extract_episode import extract_episode

DATA_DIR =  Path("data")

def images_to_hwc_uint8(images: "torcch.Tensor") -> np.ndarray:
    """
    Convert (T, C, H, W) float32 [0,1] -> (T, H, W, C) uint8 [0,255].
    This is the single onversion boundry for image data.
    """
    # permute: (T, C, H, W) -> (T, H, W, C)
    hwc = images.permute(0, 2, 3, 1)
    # contiguous() required - permute breaks memory layout
    hwc = hwc.contiguous()
    # scale and cast, then to numpy
    return (hwc * 255).byte().numpy()


def write_episode(ds: LeRobotDataset, episode_idx: int,out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"ep_{episode_idx:06d}.hdf5"

    ep = extract_episode(ds, episode_idx)

    images_uint8 = images_to_hwc_uint8(ep["images"])
    actions_np = ep["actions"].numpy()

    # LeRobot pusht has no state obs - use actions as a stand-in for shape
    # In week 3 this slot will hold the agent_pos observation
    state_np = ep["actions"].numpy()

    with h5py.File(out_path, "w") as f:
        # Groups
        obs = f.create_group("observations")

        # Datasets - no chucking or compression yet
        obs.create_dataset("images", data=images_uint8)         # (T, 96, 96, 3) uint8
        obs.create_dataset("state", data=state_np)              # (T, 2)         float32
        f.create_dataset("actions", data=actions_np)            # (T, 2)         float32

        # Episode-level attributes on the root group
        f.attrs["episode_id"]   = episode_idx
        f.attrs["task"]         = "pusht"
        f.attrs["frame_count"]  = ep["length"]
        f.attrs["success"]      = 0.0   # pusht sucess is sparse; always 0.0 at this stage

    return out_path


if __name__ == "__main__":
    ds = LeRobotDataset("lerobot/pusht")
    path = write_episode(ds,episode_idx=0, out_dir=DATA_DIR)
    print(f"Written: {path}   ({path.stat().st_size / 1024:.1f} KB)")
