# corrupt_test.py — run once to verify the validator catches injected NaN
import shutil
import h5py
import numpy as np
from pathlib import Path
from config import episode_hdf5_path, HDF5_DIR
from validate import compute_action_bounds, validate_episode

TARGET_EP = 0
path = episode_hdf5_path(TARGET_EP)
backup = path.with_suffix(".hdf5.bak")

action_min, action_max = compute_action_bounds(HDF5_DIR)

# Back up the clean file BEFORE touching it
shutil.copy2(path, backup)
print(f"Backup saved → {backup}")

print(f"Injecting NaN into ep_{TARGET_EP:06d} actions[5, 0] ...")
with h5py.File(path, "r+") as f:
    actions = f["actions"][:]
    actions[5, 0] = float("nan")
    del f["actions"]
    f.create_dataset(
        "actions", data=actions,
        chunks=(1, actions.shape[1]), compression="gzip", compression_opts=4,
    )

print("Corruption injected. Running validator ...")
report = validate_episode(TARGET_EP, action_min, action_max)
print(f"passed  : {report['passed']}")
print(f"failures: {report['failures']}")
assert not report["passed"], "Validator did NOT catch the NaN — something is wrong"
assert any("nan" in f.lower() or "inf" in f.lower() for f in report["failures"])
print("✓ Validator correctly caught the injected NaN")

# Restore from the backup — guaranteed clean, no in-memory roundtrip
print(f"\nRestoring ep_{TARGET_EP:06d} from backup ...")
shutil.copy2(backup, path)
backup.unlink()

report_after = validate_episode(TARGET_EP, action_min, action_max)
assert report_after["passed"], f"Restore failed: {report_after['failures']}"
print("✓ File restored and re-validated clean")
