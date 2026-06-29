import dataclasses
from pathlib import Path
from robot_policy_lab.utils.lineage import get_git_hash, get_dvc_dataset_hash, get_config_hash

DVC_TRACKED_PATH = "../month-01-robot-data-forge/outputs/metadata.parquet"


def test_git_hash():
    h = get_git_hash()
    assert h != "unknown", f"git hash returned 'unknown' - is this a git repo?"
    assert len(h) == 40, f"Expected 40-char SHA, got:{h!r}"
    print(f"test_git_hash ✓  {h}")


def test_dvc_hash():
    h = get_dvc_dataset_hash(DVC_TRACKED_PATH)
    assert not h.startswith("unknown"), \
        f"DVC hash failed: {h}\nCheck that {DVC_TRACKED_PATH}.dvc exists."
    assert len(h) == 32, f"Expected 32-char MD5, got: {h!r}"
    print(f"test_dvc_hash ✓  {h}")


def test_config_hash():
    @dataclasses.dataclass
    class FakeCfg:
        seed: int = 42
        steps: int = 1000
        batch_size: int = 64

    h = get_config_hash(FakeCfg())
    assert len(h) == 8, f"Expected 8-char hash, got: {h!r}"

    # Same config must always produce same hash
    assert get_config_hash(FakeCfg()) == get_config_hash(FakeCfg())

    # Different config must produce different hash
    assert get_config_hash(FakeCfg(seed=42)) != get_config_hash(FakeCfg(seed=99))

    print(f"test_config_hash ✓  {h}")


if __name__ == "__main__":
    test_git_hash()
    test_dvc_hash()
    test_config_hash()
    print("\nAll lineage tests passed ✓")