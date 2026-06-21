"""
Add a sampling weight column to the metadata Parquet index.
 
Strategy (Month 1 placeholder):
  success > 0.0  →  weight 1.0   (prefer successful episodes)
  success == 0.0 →  weight 0.3   (still sample failures, just less often)
 
In Month 2 you'll replace this with hard-example mining scores:
  weight = policy_failure_rate_on_episode  (continuous, not binary)
 
Run: python add_weights.py
Out: outputs/metadata.parquet  (overwritten in-place, same schema + new column)
"""

import pandas as pd
from config import OUTPUTS_DIR

# Config
PARQUET_PATH   = OUTPUTS_DIR / "metadata.parquet"
SUCCESS_WEIGHT = 1.0   # weight for episodes where success > 0.0
FAILURE_WEIGHT = 0.3   # weight for episodes where success == 0.0
# For pusht: ALL 206 episodes have success=0.0, so all get FAILURE_WEIGHT.
# That's correct — uniform sampling at 0.3 is identical to uniform at 1.0.
# The weight column is structurally ready for Month 2 hard-example scores.

def add_weights(parquet_path=PARQUET_PATH) -> pd.DataFrame:
    df = pd.read_parquet(parquet_path, engine="pyarrow")

    print(f"Loaded index: {len(df)} episodes")
    print(f"Success distribution:\n{df['success'].value_counts().to_string()}\n")

    # Boolean mask - vecotirzed, no loop
    df["weight"] = df["success"].apply(
        lambda s: SUCCESS_WEIGHT if s > 0.0 else FAILURE_WEIGHT
    ).astype("float32")

    # Write back k- same file, same engine, same index=False convention
    df.to_parquet(parquet_path, engine="pyarrow", index=False)

    n_success = (df["success"] > 0.0).sum()
    n_failure = len(df) - n_success

    print(f"Weights assigned:")
    print(f"  success episodes ({n_success}):  weight = {SUCCESS_WEIGHT}")
    print(f"  failure episodes ({n_failure}): weight = {FAILURE_WEIGHT}")
    print(f"\nColumn added: 'weight'")
    print(df[["episode_id", "success", "frame_count", "weight"]].head(10).to_string(index=False))
 
    return df

if __name__ == "__main__":
    add_weights()