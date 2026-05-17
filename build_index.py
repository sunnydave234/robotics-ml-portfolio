"""
Scan all validated HDF5 files, read root attributes,
and write a Parquet metadata index.

Run: python build_index.py
Out: outputs/metadata.parquet
"""

import time
from datetime import datetime, timezone
from pathlib import Path

import h5py
import pandas as pd

from config import HDF5_DIR, OUTPUTS_DIR, episode_hdf5_path, DATASET_NAME


def read_episode_attrs(hdf5_path: Path) -> dict | None:
    """
    Open one HDF5 file, read its root attributes, return a dict.
    Returns None if the file is missing the 'done' sentinel or done=False.
    """
    with h5py.File(hdf5_path, "r") as f:
        attrs = dict(f.attrs)   # copies all root attributes into memory

    # Fast pre-check: skip any episode that didn't finish ingestion cleanly
    if not attrs.get("done", False):
        return None
    
    return {
        "episode_id":   int(attrs["episode_id"]),
        "task":         str(attrs["task"]),
        "success":      float(attrs["success"]),
        "frame_count":  int(attrs["frame_count"]),
        "file_path":    str(hdf5_path),
        # Snapshot time in UTC ISO format - same convention as your FHIR ingestion timestamp
        "ingested_at": datetime.now(timezone.utc).isoformat(),
    }


def build_index() -> pd.DataFrame:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    hdf5_files = sorted(HDF5_DIR.glob("ep_*.hdf5"))
    if not hdf5_files:
        raise FileNotFound(f"No HDF5 files found in {HDF5_DIR}")
    
    print(f"Scanning {len(hdf5_files)} HDF5 files...")
    t0 = time.perf_counter()

    rows = []
    skipped = 0
    for path in hdf5_files:
        record = read_episode_attrs(path)
        if record is None:
            skipped += 1
            continue
        rows.append(record)
    
    elapsed = time.perf_counter() - t0

    df = pd.DataFrame(rows)
    # Explicit dtypes - avoids silent object columns that bloat Parquet files
    df = df.astype({
        "episode_id":   "int32",
        "success":      "float32",
        "frame_count":  "int32",
    })

    out_path = OUTPUTS_DIR / "metadata.parquet"
    df.to_parquet(out_path, engine="pyarrow", index=False)

    print(f"Index built: {len(df)} episodes, {skipped} skipped")
    print(f"Elapsed: {elapsed:.3f}s  ({len(df)/elapsed:.1f} ep/s)")
    print(f"Saved: {out_path}  ({out_path.stat().st_size / 1024:.1f} KB)")
    print(f"\nDataset: {DATASET_NAME}")
    print(df.describe().to_string())

    return df

if __name__ == "__main__":
    build_index()
