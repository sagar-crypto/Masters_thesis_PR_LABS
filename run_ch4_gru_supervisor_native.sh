#!/bin/bash -l
#SBATCH --job-name=ch4_gru_supervisor
#SBATCH --ntasks=1
#SBATCH --gres=gpu:v100:1
#SBATCH --partition=v100
#SBATCH --time=4:00:00
#SBATCH --array=0-1
#SBATCH --output=./hpc/hpc_logs/ch4_gru_supervisor-%A_%a-on-%N.out

set -euo pipefail

# ============================================================
# Environment
# ============================================================

echo "Running on $(hostname)"
echo "Array job ID: ${SLURM_ARRAY_JOB_ID}"
echo "Task ID: ${SLURM_ARRAY_TASK_ID}"
echo "SLURM job ID: ${SLURM_JOB_ID}"
echo "Start: $(date)"
echo "TMPDIR: ${TMPDIR:-UNSET}"

source /home/hpc/iwi5/iwi5305h/miniconda3/etc/profile.d/conda.sh
conda activate Masters_thesis_env_py312

DATASET_DIR="/home/vault/iwi5/iwi5305h"

THESIS_DIR="/home/hpc/iwi5/iwi5305h/Masters_thesis_PR_LABS"

PROJECT_DIR="$THESIS_DIR/third_party/dl_fault_repo"

SOURCE_WINDOWS_DIR="$DATASET_DIR/windows_tmp"

WINDOW_LENGTH_TAG="0p060"
WINDOW_LENGTH_FLOAT="0.060"

export PYTHONPATH="${PROJECT_DIR}/src:${THESIS_DIR}:${PYTHONPATH:-}"

# W&B is disabled by the command below, but this avoids accidental
# online initialization if the local configuration changes.
export WANDB_MODE="disabled"

# ============================================================
# Select topology
# ============================================================

case "${SLURM_ARRAY_TASK_ID}" in

    0)
        EXPERIMENT_ID="G90_SUPERVISOR_NATIVE"
        TOPOLOGY="hv_double_line_90kv"
        ;;

    1)
        EXPERIMENT_ID="G110_SUPERVISOR_NATIVE"
        TOPOLOGY="hv_double_line_110kv"
        ;;

    *)
        echo "ERROR: Unknown array task ID: ${SLURM_ARRAY_TASK_ID}"
        exit 1
        ;;

esac

echo
echo "============================================================"
echo "Experiment: ${EXPERIMENT_ID}"
echo "Topology:   ${TOPOLOGY}"
echo "Model:      gru_regressor"
echo "Target:     y_fault_location"
echo "Window:     ${WINDOW_LENGTH_FLOAT} s"
echo "============================================================"
echo

# ============================================================
# Separate output locations
# ============================================================

OUTPUT_BASE="$THESIS_DIR/outputs/chapter4/gru_supervisor_native"

ARRAY_OUTPUT_DIR="$OUTPUT_BASE/${SLURM_ARRAY_JOB_ID}"

EXPERIMENT_OUTPUT_DIR="$ARRAY_OUTPUT_DIR/$EXPERIMENT_ID"

CHECKPOINT_DIR="$EXPERIMENT_OUTPUT_DIR/checkpoints"

TUNING_DIR="$EXPERIMENT_OUTPUT_DIR/tuning"

mkdir -p \
    "$EXPERIMENT_OUTPUT_DIR" \
    "$CHECKPOINT_DIR" \
    "$TUNING_DIR"

# Record the exact submission environment.
{
    echo "experiment_id=${EXPERIMENT_ID}"
    echo "topology=${TOPOLOGY}"
    echo "model=gru_regressor"
    echo "target=y_fault_location"
    echo "window_length=${WINDOW_LENGTH_FLOAT}"
    echo "slurm_array_job_id=${SLURM_ARRAY_JOB_ID}"
    echo "slurm_array_task_id=${SLURM_ARRAY_TASK_ID}"
    echo "slurm_job_id=${SLURM_JOB_ID}"
    echo "hostname=$(hostname)"
    echo "start_time=$(date --iso-8601=seconds)"
    echo "python=$(which python)"
    echo "conda_environment=${CONDA_DEFAULT_ENV:-unknown}"
} > "$EXPERIMENT_OUTPUT_DIR/run_environment.txt"

if git -C "$PROJECT_DIR" rev-parse HEAD >/dev/null 2>&1; then
    git -C "$PROJECT_DIR" rev-parse HEAD \
        > "$EXPERIMENT_OUTPUT_DIR/git_commit.txt"

    git -C "$PROJECT_DIR" status --short \
        > "$EXPERIMENT_OUTPUT_DIR/git_status.txt"
fi

# ============================================================
# Prepare node-local window directory
# ============================================================

if [[ -z "${TMPDIR:-}" ]]; then
    echo "ERROR: TMPDIR is not set."
    exit 1
fi

JOB_TMP_DIR="$TMPDIR/$SLURM_JOB_ID"

WINDOWS_TMP_PATH="$JOB_TMP_DIR/windows_tmp"

rm -rf "$WINDOWS_TMP_PATH"
mkdir -p "$WINDOWS_TMP_PATH"

echo "Copying ${TOPOLOGY} windows to:"
echo "$WINDOWS_TMP_PATH"

COPY_START="$(date +%s)"

for EXTENSION in raw parquet json; do

    FILE_PATTERN="*${TOPOLOGY}_W${WINDOW_LENGTH_TAG}_*.${EXTENSION}"

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
    echo "ERROR: No window files were copied."
    exit 1
fi

COPY_DURATION="$(( $(date +%s) - COPY_START ))"

echo "Staged files: ${STAGED_FILE_COUNT}"
echo "Copy duration: ${COPY_DURATION} seconds"

# ============================================================
# Run supervisor repository GRU
# ============================================================

cd "$PROJECT_DIR" || exit 1

echo
echo "Starting native supervisor-repository GRU experiment."
echo

python src/dl_psp/models/run_dl_experiment.py \
    dataset="$TOPOLOGY" \
    dataset.dataset_directory="$DATASET_DIR" \
    model.model_name=gru_regressor \
    window_extraction.window_length="$WINDOW_LENGTH_FLOAT" \
    window_extraction.windows_local_dir="$WINDOWS_TMP_PATH" \
    training.target_label=y_fault_location \
    training.n_splits=5 \
    training.val_size=0.15 \
    training.split_seed=42 \
    training.batch_size=256 \
    training.epochs=500 \
    training.patience=15 \
    training.tune_lr_wd=true \
    training.tune_cache_dir="$TUNING_DIR" \
    training.ckpt_dir="$CHECKPOINT_DIR" \
    tracking.use_wandb=false \
    tracking.project="chapter4_${EXPERIMENT_ID}" \
    +training.out_dir="$EXPERIMENT_OUTPUT_DIR"

echo
echo "============================================================"
echo "Experiment completed: ${EXPERIMENT_ID}"
echo "End: $(date)"
echo
echo "Output directory:"
echo "$EXPERIMENT_OUTPUT_DIR"
echo "============================================================"
echo

find "$EXPERIMENT_OUTPUT_DIR" \
    -type f \
    -printf '%P\t%s bytes\n' |
sort

# ============================================================
# Cleanup
# ============================================================

echo
echo "Cleaning temporary directory:"
echo "$JOB_TMP_DIR"

rm -rf "$JOB_TMP_DIR"

echo "Cleanup completed."
