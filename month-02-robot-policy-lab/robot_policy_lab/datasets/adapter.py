"""
RobotForgeAdapter — bridges Month-1 HDF5 store into ACT's __getitem__ contract.

Required return keys (ARCHITECTURE.md, pinned commit 2d7a420):
  "observation.image":  (C, H, W)                   float32, [0.0, 1.0]
  "observation.state":  (state_dim,)                 float32, z-score normalized
  "action":             (n_action_steps, action_dim) float32, z-score normalized
  "action_is_pad":      (n_action_steps,)            bool
                         False = real step, True = repeat-padded past episode end
                         Required by loss fn at lerobot_train.py:145 — missing = KeyError
"""
import h5py
import json
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from torch.utils.data import Dataset


class RobotForgeAdapter(Dataset):
    """
    Reads Month-1 HDF5 episodes via the Parquet index.
    Returns ACT-compatible sample dicts on every __getitem__ call.

    Key design constraints:
    - h5py opens inside __getitem__, never __init__ (multiprocessing safety)
    - action shape is (n_action_steps, action_dim), NOT (action_dim,)
    - action_is_pad marks repeat-padded steps at episode boundaries
    - file paths in parquet may be stale (doubled-repo path bug) — rewritten at load time
    """

    def __init__(
        self,
        parquet_path: str | Path,
        stats_path: str | Path,
        n_action_steps: int = 100,
    ) -> None:
        self.n_action_steps = n_action_steps

        # Load episode index - one row per episode
        df = pd.read_parquet(parquet_path)
        self.ep_lengths = df["frame_count"].tolist()

        # Parquet stores paths from when ingest.py ran — one level too shallow
        # because the git repo is doubled (robotics-ml-portfolio/robotics-ml-portfolio/).
        # Insert the missing segment so paths resolve to where the files actually are.
        # Idempotent: only insert the missing segment if it isn't already
        # present. Guards against re-doubling if build_index.py is ever
        # re-run against paths that are already correct (as happened here).
        SHALLOW = "robotics-ml-portfolio/month-01-robot-data-forge"
        DOUBLED = "robotics-ml-portfolio/robotics-ml-portfolio/month-01-robot-data-forge"
        self.file_paths = [
            p if DOUBLED in p else p.replace(SHALLOW, DOUBLED)
            for p in df["file_path"].tolist()
        ]

        # Verify the first path exists — catch stale paths loudly here,
        # not silently inside __getitem__ during training.
        first = Path(self.file_paths[0])
        if not first.exists():
            raise FileNotFoundError(
                f"HDF5 file not found: {first}\n"
                f"Run: find ~ -name 'ep_000000.hdf5' 2>/dev/null\n"
                f"to locate the real path, then update PARQUET_PATH in the test."
            )
        
        # Build flat-index -> (episode_idx, frame_within_episode) lookup.
        # Example: ep_lengths = [124, 89, 103]
        #   _cumulative = [0, 124, 213, 316]
        #   flat idx=150 -> episode 1, frame 26
        self._cumulative = np.cumsum([0] + self.ep_lengths)

        # Normalization stats from compute_stats.py (Month 1)
        # Key names: stats["states"]["mean"], stats["actions"]["mean"]
        # Note: pusht actions are raw pixel coords (~228, ~294), not [0,1].
        # Z-score normalization still works — output centers near 0 with ~unit variance.
        stats = json.loads(Path(stats_path).read_text())
        self._s_mean = np.array(stats["states"]["mean"],  dtype=np.float32)
        self._s_std  = np.array(stats["states"]["std"],   dtype=np.float32) + 1e-8
        self._a_mean = np.array(stats["actions"]["mean"], dtype=np.float32)
        self._a_std  = np.array(stats["actions"]["std"],  dtype=np.float32) + 1e-8

    # --- LeRobotDataset-compatibility properties -----------------------------
    # lerobot_train.py reads these directly off `dataset` (not dataset.meta) at
    # several points — num_frames/num_episodes for logging (lines 395-396, 536,
    # 696), episodes/absolute_to_relative_idx for building EpisodeAwareSampler
    # (lines 413, 417). RobotForgeAdapter never needed these until the real
    # training entry point was exercised end-to-end (W2D5).

    @property
    def num_frames(self) -> int:
        """Total frame count across all episodes — matches len(self)."""
        return len(self)

    @property
    def num_episodes(self) -> int:
        """Episode count — one entry per row in metadata.parquet."""
        return len(self.ep_lengths)

    @property
    def episodes(self):
        """
        None = use all episodes, no subsetting.
        Confirmed against the real LeRobotDataset: it returns None here too
        under the same conditions (no episode filter applied).
        """
        return None

    @property
    def absolute_to_relative_idx(self):
        """
        None = no distributed rank sharding.
        Only populated on LeRobotDataset for the non-main-process path in
        multi-GPU training (lerobot_train.py ~line 253) — not relevant on a
        single Mac. NEEDS REVISITING in Week 3 once RobotForgeAdapter runs
        under real DDP across multiple GPUs on the Terraform box.
        """
        return None
    # --------------------------------------------------------------------------

    def __len__(self) -> int:
        return int(self._cumulative[-1])    # total frames across all episodes = 25, 650
    
    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        # Step 1: map flat idx → (episode index, frame offset within episode)
        # searchsorted on _cumulative[1:] gives the episode boundary.
        # side="right" means idx exactly equal to a boundary goes to the next episode.
        ep_idx  = int(np.searchsorted(self._cumulative[1:], idx, side="right"))
        frame_t = idx - int(self._cumulative[ep_idx])
        ep_len  = self.ep_lengths[ep_idx]

        # Step 2: read from HDF5 — open here, not in __init__
        # h5py file handles cannot be pickled. If opened in __init__ and stored as
        # self.f, every DataLoader worker inherits the same fd and they collide.
        # One open-per-call costs ~0.1ms — negligible vs the actual read.
        with h5py.File(self.file_paths[ep_idx], "r") as f:
            # /observations/images: (T, H, W, C) uint8 — single camera, no /0 suffix
            image   = f["/observations/images"][frame_t]        # -> (H, W, C) unit 8
            state   = f["/observations/state"][frame_t]         # -> (state_dim,) float32
            chunk_end = min(frame_t + self.n_action_steps, ep_len)
            actions = f["/actions"][frame_t : chunk_end]        # -> (actual_steps, action_dim)
        
        # Step 3: build action_is_pad mask and pad the action chunk
        #
        # Concrete example at episode boundary:
        #   episode length = 124 frames, frame_t = 122, n_action_steps = 100
        #   real steps available: actions[122:124] → shape (2, 2)
        #   need: shape (100, 2)
        #   pad: repeat actions[123] (last real) 98 times → shape (98, 2)
        #   concatenate → (100, 2)
        #   action_is_pad: [False, False, True, True, ..., True]  (2 False, 98 True)
        #
        # The loss function at lerobot_train.py:145 uses:
        #   valid_mask = ~batch["action_is_pad"].unsqueeze(-1)
        # This zeros out loss contributions from padded steps so the model
        # doesn't learn "stay still at episode end" as a spurious signal.
        n_real = len(actions)
        is_pad = np.zeros(self.n_action_steps, dtype=bool)  # all False initially

        if n_real < self.n_action_steps:
            pad_rows = self.n_action_steps - n_real
            pad      = np.tile(actions[-1], (pad_rows, 1))   # repeat last action
            actions  = np.concatenate([actions, pad], axis=0)
            is_pad[n_real:] = True                            # mark the padded suffix
        
        # actions is now guaranteed shape (n_action_steps, action_dim)

        # Step 4: image — uint8 HWC [0,255] → float32 CHW [0.0, 1.0]
        # .astype(float32) before divide — uint8 arithmetic wraps at 255
        # .transpose(2,0,1) = HWC → CHW — ACT's ResNet backbone expects channel-first
        img = image.astype(np.float32) / 255.0   # (H, W, C), values in [0.0, 1.0]
        img = img.transpose(2, 0, 1)              # (C, H, W)

        # Step 5: z-score normalize state and actions
        # Formula: (x - mean) / std  → output centered near 0, ~unit variance
        # pusht state == action (same x/y pixel coords) — both use the same stats
        state_norm   = (state   - self._s_mean) / self._s_std
        actions_norm = (actions - self._a_mean) / self._a_std  # (n_action_steps, action_dim)

        return {
            "observation.image": torch.from_numpy(img),
            "observation.state": torch.from_numpy(state_norm).float(),
            "action":            torch.from_numpy(actions_norm).float(),
            "action_is_pad":     torch.from_numpy(is_pad),   # bool tensor — no .float()
        }