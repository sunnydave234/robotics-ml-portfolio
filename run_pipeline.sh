#!/usr/bin/env bash
set -euo pipefail   # -e: exit on error, -u: error on unset vars, -o pipefail: catch pipe failures

DATASET="${1:?Usage: bash run_pipeline.sh <dataset_name> <episode_count>}"
N_EPISODES="${2:?Usage: bash run_pipeline.sh <dataset_name> <episode_count>}"

# Always run relateive to this script's location - not the caller's cwd
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "======================================"
echo " robot-data-forge pipeline"
echo " Dataset : $DATASET"
echo " Episodes: $N_EPISODES"
echo " Dir     : $SCRIPT_DIR"
echo "======================================"

# Activate the virtualenv if not already active.
# If you always activate before running, this is a no-op.
# If someone runs this from a fresh shell, it saves them.
if [[ -z "${VIRTUAL_ENV:-}" ]]; then
    echo "[setup] Activating virtualenv..."
    source ~/envs/robotics/bin/activate
fi

echo ""
echo "[1/3] Ingesting episodes..."
python ingest.py --max-episodes "$N_EPISODES"

echo ""
echo "[2/3] Validating HDF5 files..."
python validate.py

echo ""
echo "[3/3] Building metadata index..."
python build_index.py

echo ""
echo "======================================"
echo " Pipeline complete."
echo "======================================"
