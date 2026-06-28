"""
Tests for RobotForgeAdapter.

Run from month-02-robot-policy-lab/:
    PYTHONPATH=$(pwd) python tests/test_adapter.py

Four tests:
  test_schema          — all keys, correct shapes, dtypes, image range, no NaN
  test_episode_boundary — padding at episode end, action_is_pad correctness
  test_dataloader_smoke — num_workers=2 multiprocessing safety
  test_length          — total sample count matches sum of frame_count column
"""
import torch
from torch.utils.data import DataLoader
from robot_policy_lab.datasets.adapter import RobotForgeAdapter

PARQUET_PATH   = "../month-01-robot-data-forge/outputs/metadata.parquet"
STATS_PATH     = "../month-01-robot-data-forge/outputs/dataset_stats.json"
N_ACTION_STEPS = 100
ACTION_DIM     = 2
STATE_DIM      = 2


def test_schema() -> None:
    """All required keys, correct shapes, dtypes, image range, no NaN."""
    adapter = RobotForgeAdapter(PARQUET_PATH, STATS_PATH, n_action_steps=N_ACTION_STEPS)
    sample  = adapter[0]

    # All four required keys must be present — missing any = KeyError at training start
    for key in ("observation.image", "observation.state", "action", "action_is_pad"):
        assert key in sample, f"Missing required key: '{key}'"

    # Shape assertions
    assert sample["observation.image"].shape == (3, 96, 96), \
        f"image: {sample['observation.image'].shape} ≠ (3, 96, 96)"
    assert sample["observation.state"].shape == (STATE_DIM,), \
        f"state: {sample['observation.state'].shape} ≠ ({STATE_DIM},)"

    # THE critical assertion — PyTorch silently broadcasts (action_dim,) and training
    # appears to work but the model learns garbage. Shape must be (n_action_steps, action_dim).
    assert sample["action"].shape == (N_ACTION_STEPS, ACTION_DIM), \
        f"ACTION SHAPE WRONG: {sample['action'].shape} — ACT expects ({N_ACTION_STEPS}, {ACTION_DIM})"

    assert sample["action_is_pad"].shape == (N_ACTION_STEPS,), \
        f"action_is_pad: {sample['action_is_pad'].shape} ≠ ({N_ACTION_STEPS},)"

    # Dtypes
    assert sample["observation.image"].dtype  == torch.float32, "image must be float32"
    assert sample["observation.state"].dtype  == torch.float32, "state must be float32"
    assert sample["action"].dtype             == torch.float32, "action must be float32"
    assert sample["action_is_pad"].dtype      == torch.bool,    "action_is_pad must be bool"

    # Image range — NOT checking action/state range (z-scored, will be outside [0,1])
    assert sample["observation.image"].min() >= 0.0, "image min < 0.0"
    assert sample["observation.image"].max() <= 1.0, "image max > 1.0"

    # No NaN in numeric tensors
    for key in ("observation.image", "observation.state", "action"):
        assert not torch.isnan(sample[key]).any(), f"NaN found in '{key}'"

    print("test_schema ✓")
    print(f"  image:         {tuple(sample['observation.image'].shape)}  {sample['observation.image'].dtype}")
    print(f"  state:         {tuple(sample['observation.state'].shape)}  {sample['observation.state'].dtype}")
    print(f"  action:        {tuple(sample['action'].shape)}  {sample['action'].dtype}")
    print(f"  action_is_pad: {tuple(sample['action_is_pad'].shape)}  {sample['action_is_pad'].dtype}")
    print(f"  image range:   [{sample['observation.image'].min():.3f}, {sample['observation.image'].max():.3f}]")


def test_episode_boundary() -> None:
    """
    At the last frame of episode 0, the adapter can read only 1 real action
    (the final frame) and must pad 99 more. Verify:
      - action shape is still (100, 2)  — padding filled the gap
      - action_is_pad[0]  is False       — the one real step
      - action_is_pad[1:] are all True   — the 99 padded steps
    """
    adapter    = RobotForgeAdapter(PARQUET_PATH, STATS_PATH, n_action_steps=N_ACTION_STEPS)
    last_frame = int(adapter._cumulative[1]) - 1   # last frame index of episode 0

    sample = adapter[last_frame]

    assert sample["action"].shape == (N_ACTION_STEPS, ACTION_DIM), \
        f"Boundary padding failed: {sample['action'].shape}"

    # At the very last frame: 1 real step, 99 padded
    assert sample["action_is_pad"][0]  == False, \
        "action_is_pad[0] should be False (last real step)"
    assert sample["action_is_pad"][1:].all() == True, \
        "action_is_pad[1:] should all be True (99 padded steps)"
    assert sample["action_is_pad"].sum().item() == N_ACTION_STEPS - 1, \
        f"Expected 99 padded steps, got {sample['action_is_pad'].sum().item()}"

    print("test_episode_boundary ✓")
    print(f"  last frame idx: {last_frame}")
    print(f"  padded steps:   {sample['action_is_pad'].sum().item()} / {N_ACTION_STEPS}")


def test_dataloader_smoke() -> None:
    """
    Two DataLoader workers must iterate without h5py handle collisions.

    If h5py is opened in __init__ (wrong), both workers inherit the same
    file descriptor and race. The failure mode is either a deadlock or a
    RuntimeError about a closed file. This test catches that.
    """
    adapter = RobotForgeAdapter(PARQUET_PATH, STATS_PATH, n_action_steps=N_ACTION_STEPS)
    loader  = DataLoader(adapter, batch_size=8, num_workers=2, shuffle=False)

    batch = next(iter(loader))

    assert batch["action"].shape       == (8, N_ACTION_STEPS, ACTION_DIM)
    assert batch["action_is_pad"].shape == (8, N_ACTION_STEPS)
    assert batch["action_is_pad"].dtype == torch.bool

    print("test_dataloader_smoke ✓")
    print(f"  batch action shape:        {tuple(batch['action'].shape)}")
    print(f"  batch action_is_pad shape: {tuple(batch['action_is_pad'].shape)}")
    print(f"  num_workers=2: no h5py collision")


def test_length() -> None:
    """Total sample count must equal sum of all episode frame_counts (25,650 for pusht)."""
    import pandas as pd
    adapter = RobotForgeAdapter(PARQUET_PATH, STATS_PATH, n_action_steps=N_ACTION_STEPS)
    df      = pd.read_parquet(PARQUET_PATH)

    expected = int(df["frame_count"].sum())
    assert len(adapter) == expected, \
        f"len(adapter)={len(adapter)}, expected {expected}"

    print("test_length ✓")
    print(f"  total samples: {len(adapter)} (= sum of 206 episode frame_counts)")


if __name__ == "__main__":
    test_schema()
    test_episode_boundary()
    test_dataloader_smoke()
    test_length()
    print("\nAll tests passed ✓")
