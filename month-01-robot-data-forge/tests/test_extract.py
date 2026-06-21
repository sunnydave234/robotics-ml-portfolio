import sys
import os
import torch
import pytest
from lerobot.datasets.lerobot_dataset import LeRobotDataset


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from extract_episode import extract_episode

@pytest.fixture(scope="module")
def ds():
    return LeRobotDataset("lerobot/pusht")

@pytest.fixture(scope="module")
def episode(ds):
    return extract_episode(ds, episode_idx=0)

def test_meta_api_column_names(ds):
    # Guart agains LeRobot renaming the column names in the furture releases
    cols = ds.meta.episodes.column_names
    assert "dataset_from_index" in cols, \
        f"Expected 'dataset_from_index' in meta columns, got:{cols}"
    assert "dataset_to_index" in cols, \
        f"Expected 'dataset_to_index' in meta columns, got: {cols}"

def test_episode_bound_valid(ds):
    start = ds.meta.episodes["dataset_from_index"][0]
    end = ds.meta.episodes["dataset_to_index"][0]

    assert isinstance(start, int), \
        f"dataset_from_index returned {type(start)}, expected int"
    assert isinstance(end, int), \
        f"dataset_to_index returned {type(end)}, expected int"

def test_image_shape_consistent(episode):
    imgs = episode["images"]
    assert imgs.ndim == 4, f"Expected 4D tensor, got {imgs.ndim}"
    assert imgs.shape[0] == episode["length"]

def test_action_shape_consistent(episode):
    acts = episode["actions"]
    assert acts.ndim == 2
    assert acts.shape[0] == episode["length"]
    assert acts.shape[1] > 0, "action_dim is 0 — extraction bug"

def test_tensors_on_cpu(episode):
    assert episode["images"].device.type == "cpu", \
        f"Images on wrong device: {episode['images'].device}"
    assert episode["actions"].device.type == "cpu", \
        f"Actions on wrong device: {episode['actions'].device}"

def test_image_dtype(episode):
    assert episode["images"].dtype == torch.float32, \
        f"Unexpected image dtype: {episode['images'].dtype}"

def test_action_dtype(episode):
    assert episode["actions"].dtype == torch.float32, \
        f"Unexpected action dtype: {episode['actions'].dtype}"
