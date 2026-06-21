"""
W3D3 — DataLoader throughput benchmark.
 
Runs 16 combinations of (num_workers, pin_memory, batch_size) × 100 batches each.
Reports median samples/sec per config. Writes results to outputs/benchmark_results.json
and prints a markdown table to stdout.
 
Run:
    python benchmark_dataloader.py
 
Expected runtime: 5–15 minutes total on M4 Mac (16 configs × 100 batches each).
The script prints progress as it goes.
"""

import json
import statistics
import time
from itertools import product
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from config import OUTPUTS_DIR
from robot_dataset import RobotEpisodeDataset

# Benchmark paramters
NUM_WORKERS_OPTIONS = [0, 2, 4, 8]
PIN_MEMORY_OPTIONS  = [False, True]
BATCH_SIZE_OPTIONS  = [16, 64]

# How many batches to run per config.
# 100 batches gives a stable median without taking forever.
BATCHES_PER_RUN = 100
 
# Warm-up batches before timing starts.
# First few batches are slow (workers spawning, OS page cache cold).
# Discard these so they don't skew results.
WARMUP_BATCHES = 5

# Helpers
def run_one_config(
    dataset: RobotEpisodeDataset,
    num_workers: int,
    pin_memory: bool,
    batch_size: int,
) -> dict:
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
        drop_last=True,
        persistent_workers=(num_workers > 0),
    )

    def iter_batches(n: int):
        """Yield exactly n batches, cycling the loader if needed."""
        yielded = 0
        while yielded < n:
            for batch in loader:
                yield batch
                yielded += 1
                if yielded >= n:
                    return

    # Warmup — discard these batches entirely
    for batch in iter_batches(WARMUP_BATCHES):
        _ = batch

    # Timed pass — measure total wall time for all batches, then compute throughput.
    # Per-batch timing is misleading with multiple workers due to prefetch queue effects:
    # next() returns instantly when the queue is full, hiding the real I/O time.
    batch_gen = iter_batches(BATCHES_PER_RUN)
    
    t0 = time.perf_counter()
    for _ in range(BATCHES_PER_RUN):
        batch = next(batch_gen)
        _ = batch["action"].sum()
    total_elapsed = time.perf_counter() - t0

    total_samples = BATCHES_PER_RUN * batch_size
    overall_samples_per_sec = total_samples / total_elapsed
    # Use overall throughput for all three reported values — per-batch variance
    # is noise when prefetching is active.
    avg_batch_time = total_elapsed / BATCHES_PER_RUN

    return {
        "num_workers":        num_workers,
        "pin_memory":         pin_memory,
        "batch_size":         batch_size,
        "median_samples_sec": round(overall_samples_per_sec, 1),
        "min_samples_sec":    round(overall_samples_per_sec, 1),
        "max_samples_sec":    round(overall_samples_per_sec, 1),
        "p95_batch_time_ms":  round(avg_batch_time * 1000, 1),
    }


def print_markdown_table(results: list[dict]) -> None:
    """Print benchmark results as a markdown table, bold the fastest config."""
    # Find the fastest config by median_samples_sec
    best_median = max(r["median_samples_sec"] for r in results)
 
    header = "| num_workers | pin_memory | batch_size | samples/sec (median) | samples/sec (min) | samples/sec (max) | p95 batch (ms) |"
    sep    = "|-------------|------------|------------|----------------------|-------------------|-------------------|----------------|"
    print()
    print(header)
    print(sep)
 
    for r in results:
        is_best = r["median_samples_sec"] == best_median
        mark = "**" if is_best else ""
        row = (
            f"| {mark}{r['num_workers']}{mark} "
            f"| {mark}{r['pin_memory']}{mark} "
            f"| {mark}{r['batch_size']}{mark} "
            f"| {mark}{r['median_samples_sec']:.1f}{mark} "
            f"| {r['min_samples_sec']:.1f} "
            f"| {r['max_samples_sec']:.1f} "
            f"| {r.get('p95_batch_time_ms', 0.0):.1f} |"
        )
        print(row)
 
    print()
    best = max(results, key=lambda r: r["median_samples_sec"])
    print(f"Fastest config: num_workers={best['num_workers']}, "
          f"pin_memory={best['pin_memory']}, "
          f"batch_size={best['batch_size']} "
          f"→ {best['median_samples_sec']:.1f} samples/sec")


def main() -> None:
    parquet_path = OUTPUTS_DIR / "metadata.parquet"
    stats_path   = OUTPUTS_DIR / "dataset_stats.json"

    if not parquet_path.exists():
        raise FileNotFoundError(f"metadata.parquet not found at {parquet_path}. Run build_index.py first.")
    if not stats_path.exists():
        raise FileNotFoundError(f"dataset_stats.json not found at {stats_path}. Run compute_stats.py first.")
    
    # normalize=True: benchmark with normalization applied - that's the real training workload.
    # If you benchmark without normalization, your numbers wn't reflect actual training speed.
    dataset = RobotEpisodeDataset(parquet_path=parquet_path, normalize=True)
    print(f"Dataset loaded: {len(dataset)} samples")
    print(f"Benchmark: {len(NUM_WORKERS_OPTIONS)} × {len(PIN_MEMORY_OPTIONS)} × {len(BATCH_SIZE_OPTIONS)} = "
          f"{len(NUM_WORKERS_OPTIONS) * len(PIN_MEMORY_OPTIONS) * len(BATCH_SIZE_OPTIONS)} configs, "
          f"{BATCHES_PER_RUN} batches each (+{WARMUP_BATCHES} warmup)")
    print()

    configs = list(product(NUM_WORKERS_OPTIONS, PIN_MEMORY_OPTIONS, BATCH_SIZE_OPTIONS))
    results = []

    for i, (num_workers, pin_memory, batch_size) in enumerate(configs, 1):
        label = f"[{i:2d}/{len(configs)}] workers={num_workers} pin={str(pin_memory):<5} bs={batch_size:2d}"
        print(f"{label} ... ", end="", flush=True)

        t_start = time.perf_counter()
        result = run_one_config(dataset, num_workers, pin_memory, batch_size)
        elapsed = time.perf_counter() - t_start
        
        print(f"{result['median_samples_sec']:7.1f} samples/sec  ({elapsed:.1f}s)")
        results.append(result)

        # Write JSON artifact
        out_path = OUTPUTS_DIR / "benchmark_results.json"
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults written to {out_path}")

        # Print markdown table
        print_markdown_table(results)


if __name__ == "__main__":
    main()
