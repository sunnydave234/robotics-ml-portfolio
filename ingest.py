# ingest.py

"""
HDF5 Schema Design — robot-data-forge
======================================
Decided before writing any code. Changes here mean rewriting all existing files.

FILE STRUCTURE
--------------
data/
  ep_000000.hdf5
  ep_000001.hdf5
  ...

PER-FILE STRUCTURE
------------------
/                                   ← root group
├── observations/                   ← group: all sensor data
│   ├── images    (T, H, W, C)     ← dataset, uint8,   shape (T, 96, 96, 3)
│   └── state     (T, state_dim)   ← dataset, float32, shape (T, 2)
└── actions       (T, action_dim)  ← dataset, float32, shape (T, 2)

ROOT ATTRIBUTES (episode-level metadata)
-----------------------------------------
episode_id   int    — index in source LeRobot dataset (0-based)
task         str    — "pusht" (hardcoded for now; parameterized in Week 4)
frame_count  int    — T, number of frames in this episode
success      float  — 0.0 or 1.0 from source dataset

DTYPE DECISIONS
---------------
- images: uint8 (NOT float32)
  float32 96×96×3 × 200 frames = 22MB/episode
  uint8   96×96×3 × 200 frames =  5.5MB/episode
  Conversion: (tensor * 255).byte() — lossy but acceptable for visual obs
  
- state, actions: float32
  These are continuous control values — precision matters, size is trivial
  (T × 2 × 4 bytes = ~1.6KB for a 200-frame episode)

CHANNEL ORDER
-------------
LeRobot returns (C, H, W) — images are stored as (H, W, C) for OpenCV compatibility
Conversion: tensor.permute(1, 2, 0) before writing

NO CHUNKING OR COMPRESSION YET
--------------------------------
Adding these in Day 2 after the write pattern is validated.
"""
