"""
Query the Parquet metadata index.

Examples:
  python query.py                          # all episodes
  python query.py --task push              # task contains 'push'
  python query.py --min_frames 100         # frame_count >= 100
  python query.py --success_only           # success > 0.5
  python query.py --task push --min_frames 100 --benchmark
"""


import argparse
import time

import h5py
import pandas as pd

from config import HDF5_DIR, OUTPUTS_DIR


INDEX_PATH = OUTPUTS_DIR / "metadata.parquet"


def load_index() -> pd.DataFrame:
    if not INDEX_PATH.exists():
        raise FileNotFoundError(
            f"Index not found at {INDEX_PATH}. Run build_index.py first."
        )
    return pd.read_parquet(INDEX_PATH, engine="pyarrow")


def query_parquet(
    task: str | None = None,
    min_frames: int | None = None,
    success_only: bool = False,
) -> pd.DataFrame:
    """Filter the Parquet index. Returns matching rows."""
    df = load_index()
    mask = pd.Series([True] * len(df), index=df.index)

    if task:
        mask &= df["task"].str.contains(task, case=False, na=False)
    if min_frames is not None:
        mask &= df["frame_count"] >= min_frames
    if success_only:
        mask &= df["success"] > 0.5

    return df[mask].reset_index(drop=True)


def benchmark_hdf5_scan(
    task: str | None = None,
    min_frames: int | None = None,
    success_only: bool = False,
) -> list[int]:
    """
    Naive baseline: open every HDF5 file and read attributes directly.
    Returns matching episode IDs. Used only for benchmarking.
    """
    hdf5_files = sorted(HDF5_DIR.glob("ep_*.hdf5"))
    results = []
    for path in hdf5_files:
        with h5py.File(path, "r") as f:
            attrs = dict(f.attrs)
        if not attrs.get("done", False):
            continue
        if task and task.lower() not in str(attrs.get("task", "")).lower():
            continue
        if min_frames is not None and int(attrs["frame_count"]) < min_frames:
            continue
        if success_only and float(attrs["success"]) <= 0.5:
            continue
        results.append(int(attrs["episode_id"]))
    return results


def print_results(df: pd.DataFrame, elapsed_ms: float) -> None:
    print(f"\nMatched {len(df)} episodes  ({elapsed_ms:.3f} ms)\n")
    if df.empty:
        print("  (no results)")
        return
    print(df[["episode_id", "task", "success", "frame_count"]].to_string(index=False))
    print(f"\nframe_count  mean={df['frame_count'].mean():.1f}  "
          f"min={df['frame_count'].min()}  max={df['frame_count'].max()}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Query the robot episode index")
    parser.add_argument("--task",         type=str,  default=None)
    parser.add_argument("--min_frames",   type=int,  default=None)
    parser.add_argument("--success_only", action="store_true")
    parser.add_argument("--benchmark",    action="store_true",
                        help="Also run the slow HDF5 scan and print the speedup ratio")
    args = parser.parse_args()

    # --- Parquet query ---
    t0 = time.perf_counter()
    results = query_parquet(args.task, args.min_frames, args.success_only)
    parquet_ms = (time.perf_counter() - t0) * 1000

    print_results(results, parquet_ms)

    if args.benchmark:
        # --- HDF5 scan baseline ---
        t0 = time.perf_counter()
        hdf5_ids = benchmark_hdf5_scan(args.task, args.min_frames, args.success_only)
        hdf5_ms = (time.perf_counter() - t0) * 1000

        speedup = hdf5_ms / parquet_ms if parquet_ms > 0 else float("inf")
        print(f"\n{'─'*40}")
        print(f"Parquet query:  {parquet_ms:.3f} ms")
        print(f"HDF5 scan:      {hdf5_ms:.1f} ms")
        print(f"Speedup:        {speedup:.0f}x")

        # Sanity check: both methods should return same episode IDs
        parquet_ids = sorted(results["episode_id"].tolist())
        assert parquet_ids == sorted(hdf5_ids), (
            f"Result mismatch!\nParquet: {parquet_ids[:5]}...\nHDF5: {sorted(hdf5_ids)[:5]}..."
        )
        print("✓ Results match between Parquet and HDF5 scan")


if __name__ == "__main__":
    main()    
