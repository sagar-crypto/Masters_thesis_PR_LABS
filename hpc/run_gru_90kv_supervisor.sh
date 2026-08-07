#!/bin/bash -l
#SBATCH --job-name=gru_90kv_supervisor
#SBATCH --ntasks=1
#SBATCH --gres=gpu:v100:1
#SBATCH --partition=v100
#SBATCH --time=8:00:00
#SBATCH --output=./hpc/hpc_logs/%x/%x-%j-on-%N.out

set -euo pipefail

# ============================================================
# Setup
# ============================================================

echo "Running on: $(hostname)"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Start: $(date)"
echo "TMPDIR: ${TMPDIR:-UNSET}"

source /home/hpc/iwi5/iwi5305h/miniconda3/etc/profile.d/conda.sh
conda activate Masters_thesis_env_py312

# ============================================================
# Configuration
# ============================================================

TOPOLOGY="hv_double_line_90kv"
TOPOLOGY_CONFIG="hv_double_line_90kv"

WINDOW_LENGTH="0p060"
WINDOW_LENGTH_FLOAT="${WINDOW_LENGTH/p/.}"

TARGET="y_fault_location"
MODEL_NAME="gru_regressor"

DATASET_DIR="/home/vault/iwi5/iwi5305h"
THESIS_DIR="/home/hpc/iwi5/iwi5305h/Masters_thesis_PR_LABS"
PROJECT_DIR="$THESIS_DIR/third_party/dl_fault_repo"

SOURCE_WINDOWS_DIR="$DATASET_DIR/windows_tmp"

JOB_TMP_DIR="$TMPDIR/$SLURM_JOB_ID"
WINDOWS_TMP_PATH="$JOB_TMP_DIR/windows_tmp"

MLRUNS_DIR="file://${THESIS_DIR}/mlruns"

export PYTHONPATH="${PROJECT_DIR}/src:${THESIS_DIR}:${PYTHONPATH:-}"
export https_proxy="http://proxy.rrze.uni-erlangen.de:80"
export MLFLOW_TRACKING_URI="$MLRUNS_DIR"

# ============================================================
# Validate paths
# ============================================================

if [[ -z "${TMPDIR:-}" ]]; then
    echo "ERROR: TMPDIR is not set."
    exit 1
fi

if [[ ! -d "$PROJECT_DIR" ]]; then
    echo "ERROR: Project directory not found:"
    echo "$PROJECT_DIR"
    exit 1
fi

if [[ ! -d "$SOURCE_WINDOWS_DIR" ]]; then
    echo "ERROR: Window directory not found:"
    echo "$SOURCE_WINDOWS_DIR"
    exit 1
fi

# ============================================================
# Prepare node-local windows
# ============================================================

rm -rf "$JOB_TMP_DIR"
mkdir -p "$WINDOWS_TMP_PATH"

echo
echo "Copying 90 kV window files to:"
echo "$WINDOWS_TMP_PATH"

COPY_START="$(date +%s)"

for EXTENSION in raw parquet json; do

    FILE_PATTERN="*${TOPOLOGY}_W${WINDOW_LENGTH}_*.${EXTENSION}"

    echo "Looking for: ${FILE_PATTERN}"

    if find "$SOURCE_WINDOWS_DIR" \
        -maxdepth 1 \
        -type f \
        -name "$FILE_PATTERN" \
        -print -quit |
        grep -q .
    then
        find "$SOURCE_WINDOWS_DIR" \
            -maxdepth 1 \
            -type f \
            -name "$FILE_PATTERN" \
            -exec cp -t "$WINDOWS_TMP_PATH" {} +

        echo "Copied ${EXTENSION} files."
    else
        echo "WARNING: No ${EXTENSION} files found."
    fi

done

STAGED_FILE_COUNT="$(
    find "$WINDOWS_TMP_PATH" \
        -maxdepth 1 \
        -type f |
    wc -l
)"

if [[ "$STAGED_FILE_COUNT" -eq 0 ]]; then
    echo "ERROR: No files were copied for ${TOPOLOGY}."
    exit 1
fi

COPY_DURATION="$(( $(date +%s) - COPY_START ))"

echo "Staged files: ${STAGED_FILE_COUNT}"
echo "Copy duration: ${COPY_DURATION} seconds"

# ============================================================
# Run supervisor-repository plain GRU
# ============================================================

cd "$PROJECT_DIR" || exit 1

echo
echo "============================================================"
echo "Starting 90 kV supervisor GRU"
echo "Topology: ${TOPOLOGY_CONFIG}"
echo "Model:    ${MODEL_NAME}"
echo "Target:   ${TARGET}"
echo "Window:   ${WINDOW_LENGTH_FLOAT} s"
echo "============================================================"
echo

python src/dl_psp/models/run_dl_experiment.py \
    dataset="$TOPOLOGY_CONFIG" \
    model.model_name="$MODEL_NAME" \
    window_extraction.window_length="$WINDOW_LENGTH_FLOAT" \
    training.target_label="$TARGET" \
    tracking.project="dl_comparison_${TOPOLOGY}" \
    window_extraction.windows_local_dir="$WINDOWS_TMP_PATH"

echo
echo "============================================================"
echo "90 kV GRU run completed successfully"
echo "End: $(date)"
echo "============================================================"

# ============================================================
# Cleanup
# ============================================================

echo "Cleaning temporary directory:"
echo "$JOB_TMP_DIR"

rm -rf "$JOB_TMP_DIR"

echo "Temporary directory cleaned."
