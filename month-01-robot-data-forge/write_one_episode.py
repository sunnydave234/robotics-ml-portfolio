import h5py
import torch

def write_episode(
    f: h5py.File,
    episode_idx: int,
    images: torch.Tensor,
    actions: torch.Tensor,
    success: float = 0.0,
) -> None:
    """
    Write one episode into an open HDF5 file following the W2D1 schema.

    Resulting structure:
    /
    ├── observations/
    │   ├── images (T, H, W, C)
    │   └── state (T, state_dim)
    └── actions (T, action_dim)
    /[root attrs]   episode_id, task, frame-count, success

    images: (T, C, H, W) float32 tensor, values in [0.0, 1.0]
    actions: (T, A) float32 tensor
    success: episode success label from source dataset (0.1 for pusht)
    """

    T, C, H, W = images.shape

    # (T, C, H, W) float32 -> (T, H, W, C) uint8
    # .permute() breaks memory contiguity, so .contiguous() is requried before .numpy()
    frames = (images.permute(0, 2, 3, 1).contiguous() * 255).byte().numpy()

    # /observations/ group holds all sensor data, separate from actions
    obs = f.create_group("observations")

    # chunks=(1, H, W, C): one chunk = one frame
    # This means HDF5 can jump directly to any frame without reading the others
    # compression_opts=4: mid-range gzip - good ratio withrout being CPU-heavy
    obs.create_dataset(
        "images",
        data=frames,
        chunks=(1, H, W, C),    # one chunk = one frame = one DataLoader read
        compression="gzip",
        compression_opts=4,
    )

    # pusht has no true proprioceptive state — actions used as placeholder.
    # Documented in schema. Replace if switching to a dataset with joint encoders.
    obs.create_dataset(
        'state',
        data=actions.numpy(),
        chunks=(1, actions.shape[1]),
        compression='gzip',
        compression_opts=4,
    )    

    # Actions stay float32 - no uint8, no precision loss
    # Chunk on time axis to keep frame-aligned access consistent across both datasets
    f.create_dataset(
        'actions',
        data=actions.numpy(),
        chunks=(1, actions.shape[1]),
        compression="gzip",
        compression_opts=4,
    )

    # Root attributes - episode-level metadata
    # query.py can read these without touching the image datasets.
    f.attrs["episode_id"] = episode_idx
    f.attrs["task"] = 'pusht'
    f.attrs["frame_count"] = T
    f.attrs["success"] = success
