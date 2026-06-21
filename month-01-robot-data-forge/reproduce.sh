#!/usr/bin/env bash
# reproduce.sh — Full pipeline reproduction from a fresh clone.
#
# Usage:
#   bash reproduce.sh                    # full run (prompts for wandb login)
#   CI=true bash reproduce.sh            # skip wandb login (for CI environments)
#   SKIP_DVC_PULL=true bash reproduce.sh # skip dvc pull if remote is unavailable
#
# What this does:
#   1. Install Python dependencies
#   2. Authenticate W&B (skipped in CI)
#   3. Pull DVC artifacts from remote (or skip + rebuild from scratch)
#   4. Rebuild the DVC pipeline
#   5. Run the BC training smoke test

set -euo pipefail   # -e: exist on error -u: error on undefined vars    -o pipefaile: catch pipe failures

# 0. Sanity checks
echo "=== reproduce.sh ==="
echo "Python: $(python --version)"
echo "Working dir: $(pwd)"

# Make sure we're in the repo root (where config.py lives)
# THis prevents the script from silently running from the wrong directory.
if [ ! -f "config.py" ]; then
    echo "ERROR: config.py not found. Run this script from the repo root."
    exist 1
fi

# 1. Install dependencies
echo ""
echo "=== [1/5] Creating isolated virtual environment ==="

VENV_DIR=".venv"

if [ -d "$VENV_DIR" ]; then
    echo "Existing .venv found — removing for clean reproduction"
    rm -rf "$VENV_DIR"
fi

python3 -m venv "$VENV_DIR"

# Activate it — everything after this point runs inside .venv
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

echo "Active Python: $(which python) — $(python --version)"

pip install --upgrade pip --quiet
pip install -r requirements.txt --quiet
echo "Dependencies installed into .venv (global env untouched)"

# 2. W&B autentication
# Skip in CI environment - set CI=true or WANDB_API_KEY in env instead.
echo ""
echo "=== [2/5] W&B authentication ==="
if [ "${CI:-false}" = "true" ]; then
    echo "CI mode - skipping wandb login (expects WANDB_API_KEY in env)"
else
    # --relogin: doesn't prompt if already logged in, re-auths if token is stale
    wandb login --relogin
fi

# 3. DVC pull
# Try to pull artifacts from DVC remote.
# If remote is unavailable (e.g. /tmp purged on reboot), fall through to dvc repro.
echo ""
echo "=== [3/5] DVC pull ==="
if [ "${SKIP_DVC_PULL:-false}" = "true" ]; then
    echo "SKIP_DVC_PULL=true - skipping dvc pull, pipeline will rebuild from scratch"
else
    # || true: don't fail the script if pull fails — dvc repro in step 4 handles it
    dvc pull --quiet || {
        echo "WARNING: dvc pull failed (remote may be unavailable). Pipeline will rebuild from scratch."
    }
fi

# 4. DVC pipeline reproduction
# dvc repro runs only the stages whose inputs have changed.
# If dvc pull restored everything, this is a no-op (all stages cached).
# If dvc pull failed, this rebuilds ingest → validate → build_index → compute_stats.
echo ""
echo "=== [4/5] DVC pipeline ==="
dvc repro

# Verify the pipeline outputs exist before proceeding
if [ ! -f "outputs/metadata.parquet" ]; then
    echo "ERROR: outputs/metadata.parquet missing after dvc repro"
    exit 1
fi
if [ ! -f "outputs/dataset_stats.json" ]; then
    echo "ERROR: outputs/dataset_stats.json missing after dvc repro"
    exit 1
fi
echo "Pipeline outputs verified."

# 5. training smoke test
echo ""
echo "=== [5/5] BC training smoke test ==="
python train.py

echo ""
echo "=== reproduce.sh complete ==="
echo "Run 'dvc dag' to inspect the pipeline graph."
