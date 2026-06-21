import h5py
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from extract_episode import extract_episode
from write_one_episode import write_episode
import os

ds = LeRobotDataset("lerobot/pusht")
ep = extract_episode(ds, 0)
images = ep["images"]
actions = ep["actions"]

os.makedirs("data/hdf5", exist_ok=True)

with h5py.File("data/hdf5/ep_000000.hdf5", "w") as f:
    write_episode(f, 0, images, actions, success=0.0)
    f.attrs["done"] = True

with h5py.File("data/hdf5/ep_000000.hdf5", "r") as f:
    print("=== top-level keys ===")
    print(list(f.keys()))                       # ['actions', 'observations']

    print("\n=== observations keys ===")
    print(list(f["observations"].keys()))       # ['images', 'state']

    print("\n=== observations/images ===")
    img = f["observations/images"]
    print(f"  shape:       {img.shape}")        # (T, 96, 96, 3)
    print(f"  dtype:       {img.dtype}")        # uint8
    print(f"  chunks:      {img.chunks}")       # (1, 96, 96, 3)
    print(f"  compression: {img.compression}")  # gzip

    print("\n=== observations/state ===")
    state = f["observations/state"]
    print(f"  shape:       {state.shape}")      # (T, 2)
    print(f"  dtype:       {state.dtype}")      # float32

    print("\n=== actions ===")
    act = f["actions"]
    print(f"  shape:       {act.shape}")        # (T, 2)
    print(f"  dtype:       {act.dtype}")        # float32

    print("\n=== root attributes ===")
    for k, v in f.attrs.items():
        print(f"  {k}: {v}")
    # episode_id:  0
    # task:        pusht
    # frame_count: T
    # success:     0.0
    # done:        True



This is the W2D3 plan

```
Day 3
Build validate.py — data quality layer
3.5 hrs
- Write validate.py that opens every HDF5 file and checks: (1) no NaN or Inf values in actions or state, (2) images are uint8 in [0,255], (3) actions.shape[0] == images.shape[0] (frame count parity), (4) frame_count attribute matches actual array length, (5) action values within physically plausible bounds (dataset-specific — inspect your profile for reasonable ranges).
- Validator returns a per-episode report dict. Aggregate into a summary: total episodes, N passed, N failed, failure reasons. Print and save to validation_report.json.
- Deliberately corrupt one file (inject a NaN manually) and verify your validator catches it. This is your unit test.
- Run on all 20 ingested episodes. Any failures? Fix them in your ingestion logic, not in the validator.

DE SHORTCUT
This is your FHIR pipeline data quality layer. You've written frame-count parity checks before (source record count vs loaded count). The only robotics-specific check is action bounds validation — and you can derive those bounds directly from your dataset_profile.json min/max stats.
```

Normally I share the plan and mentioned I have completed the previous plan. 

Could you create me a prompt for the W2D3 session with the required details that you mentioned above? (leave the space for me to share the .py scripts that you mentioned above.)