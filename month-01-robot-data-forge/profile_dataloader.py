
"""
W3D3 — PyTorch profiler on one DataLoader config.
 
Profiles num_workers=4, batch_size=64 for PROFILE_BATCHES batches.
Exports a Chrome trace JSON to outputs/dataloader_trace.json.
 
Open the trace at: chrome://tracing  (paste the file path or drag-and-drop)
 
Run:
    python profile_dataloader.py
"""

import time
from pathlib import Path
 
import torch
import torch.profiler
from torch.utils.data import DataLoader
 
from config import OUTPUTS_DIR
from robot_dataset import RobotEpisodeDataset

# ── Config to profile ─────────────────────────────────────────────────────────
# These match the "interesting" config from the benchmark plan.
# num_workers=4 is where parallelism starts but hasn't saturated I/O yet.
PROFILE_NUM_WORKERS = 4
PROFILE_BATCH_SIZE  = 64
PROFILE_PIN_MEMORY  = False   # False on M4 — pin_memory is a no-op on MPS

# How many batches to profile. Keep small — profiling has overhead.
# 20 batches is enough to see a stable pattern.
PROFILE_BATCHES = 20

# Warmup batches before profiling starts (not recorded).
WARMUP_BATCHES = 3


def main() -> None:
    parquet_path = OUTPUTS_DIR / "metadata.parquet"
    stats_path   = OUTPUTS_DIR / "dataset_stats.json"
 
    if not parquet_path.exists():
        raise FileNotFoundError(f"metadata.parquet not found at {parquet_path}")
    if not stats_path.exists():
        raise FileNotFoundError(f"dataset_stats.json not found at {stats_path}")
    
    dataset = RobotEpisodeDataset(parquet_path=parquet_path, normalize=True)
    print(f"Dataset: {len(dataset)} samples")

    loader = DataLoader(
        dataset,
        batch_size=PROFILE_BATCH_SIZE,
        num_workers=PROFILE_NUM_WORKERS,
        pin_memory=PROFILE_PIN_MEMORY,
        shuffle=False,
        drop_last=True,
        persistent_workers=(PROFILE_NUM_WORKERS>0),
    )

    loader_iter = iter(loader)

    # Warmup
    print(f"Warming up ({WARMUP_BATCHES} batches)...")
    for _ in range(WARMUP_BATCHES):
        _ = next(loader_iter)
    
    # Profile
    # activities: CPU ops. Add ProfilerActivity.CUDA for CUDA-GPU ops (not applicable on M4 MPS).
    # record_shapes=True: captures tensor shapes — shows you whether image vs action tensors dominate.
    # with_stack=True: captures Python call stacks — lets you see WHICH LINE of your code is slow.
    # profile_memory=True: shows memory allocation per op — catches unexpected tensor copies.
    trace_path = OUTPUTS_DIR / "dataloader_trace.json"
 
    print(f"Profiling {PROFILE_BATCHES} batches "
          f"(workers={PROFILE_NUM_WORKERS}, bs={PROFILE_BATCH_SIZE})...")
    
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU],
        record_shapes=True,
        with_stack=True,
        profile_memory=True,
    ) as prof:
        t0 = time.perf_counter()
        for i in range(PROFILE_BATCHES):
            # Label each batch so you can find it in the Chrome trace.
            # The label shows up as a colored bar in the timeline view.
            with torch.profiler.record_function(f"batch_{i:03d}"):
                batch = next(loader_iter)
                # Access the tensors to ensure they're fully materialized.
                _ = batch["image"].shape
                _ = batch["action"].shape
        elapsed = time.perf_counter() - t0
 
    # ── Export Chrome trace ───────────────────────────────────────────────────
    prof.export_chrome_trace(str(trace_path))
    print(f"\nTrace written to: {trace_path}")
    print(f"Open at: chrome://tracing  (drag and drop the file)")
 
    # ── Print top ops by CPU time ─────────────────────────────────────────────
    # sort_by="cpu_time_total" → shows which ops consumed the most cumulative CPU time.
    # This is your first signal: if "aten::from_numpy" is at the top, it's tensor conversion.
    # If something h5py-related is dominant, it's I/O or decompression.
    print("\n── Top 15 ops by CPU time ──────────────────────────────────────────")
    print(prof.key_averages(group_by_stack_n=5).table(
        sort_by="cpu_time_total",
        row_limit=15,
    ))
 
    # ── Summary ───────────────────────────────────────────────────────────────
    total_samples = PROFILE_BATCHES * PROFILE_BATCH_SIZE
    print(f"\n{PROFILE_BATCHES} batches × {PROFILE_BATCH_SIZE} samples = {total_samples} samples")
    print(f"Wall time: {elapsed:.2f}s → {total_samples / elapsed:.1f} samples/sec")
    print(f"\nWhat to look for in chrome://tracing:")
    print("  1. Open the file. Press W/S to zoom, A/D to pan.")
    print("  2. Look for the widest bars — those are your bottlenecks.")
    print("  3. Hover over any bar to see the op name and duration.")
    print("  4. The 'batch_NNN' bars show you per-batch wall time.")
    print("  5. If bars are mostly narrow and frequent → CPU-bound (many small ops).")
    print("  6. If bars are wide and sparse → I/O-bound (waiting on HDF5 reads).")
 
 
if __name__ == "__main__":
    main()