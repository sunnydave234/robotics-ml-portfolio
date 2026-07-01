"""
robot_policy_lab/paths.py

Custom path config for Month 2. These CANNOT live in conf/pusht_act.json --
draccus.wrap() strictly validates every top-level key in that file against
TrainPipelineConfig's declared fields and raises DecodingError on anything
it doesn't recognize (confirmed live: "The fields eval_freq,
dataset_parquet_path, dataset_stats_path, dataset_dvc_path are not valid
for TrainPipelineConfig"). draccus is not the lenient extra-keys-ignored
parser W2D2 assumed -- it's strict.

These three paths point at Month-1 outputs, used by RobotForgeAdapter and
the lineage/W&B utilities -- entirely outside LeRobot's own config schema.
Override with env vars, same convention as Month 1's config.py.
"""
import os

DATASET_PARQUET_PATH = os.environ.get(
    "DATASET_PARQUET_PATH", "../month-01-robot-data-forge/outputs/metadata.parquet"
)
DATASET_STATS_PATH = os.environ.get(
    "DATASET_STATS_PATH", "../month-01-robot-data-forge/outputs/dataset_stats.json"
)
DATASET_DVC_PATH = os.environ.get(
    "DATASET_DVC_PATH", "../month-01-robot-data-forge/outputs/metadata.parquet"
)