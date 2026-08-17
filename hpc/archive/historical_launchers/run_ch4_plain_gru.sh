#!/bin/bash -l
#SBATCH --job-name=ch4_plain_gru
#SBATCH --ntasks=1
#SBATCH --gres=gpu:v100:1
#SBATCH --partition=v100
#SBATCH --time=2:00:00
#SBATCH --array=0-1
#SBATCH --output=./hpc/hpc_logs/%x-%A_%a-on-%N.out

set -euo pipefail

# ============================================================
# Environment
# ============================================================

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate Masters_thesis_env_py312

DATASET_DIR="/home/vault/iwi5/iwi5305h"

THESIS_DIR="/home/hpc/iwi5/iwi5305h/Masters_thesis_PR_LABS"

PROJECT_DIR="$THESIS_DIR/third_party/dl_fault_repo"

SOURCE_WINDOWS_DIR="$DATASET_DIR/windows_tmp"

WINDOW_LENGTH_TAG="0p060"
WINDOW_LENGTH_FLOAT="0.060"

export PYTHONPATH="${PROJECT_DIR}/src:${THESIS_DIR}:${PYTHONPATH:-}"

export WANDB_MODE="disabled"

mkdir -p "$THESIS_DIR/hpc/hpc_logs"

cd "$THESIS_DIR" || exit 1

# ============================================================
# Select one topology per array task
# ============================================================

case "$SLURM_ARRAY_TASK_ID" in

    0)
        EXPERIMENT_ID="G90"
        TOPOLOGY="hv_double_line_90kv"
        WINDOW_MODE="single_fault_start"

        EXPECTED_ROWS="9022"
        EXPECTED_EVENTS="9022"
        ;;

    1)
        EXPERIMENT_ID="G110"
        TOPOLOGY="hv_double_line_110kv"
        WINDOW_MODE="all_fault_start"

        EXPECTED_ROWS="3648"
        EXPECTED_EVENTS="912"
        ;;

    *)
        echo "Unknown array index: $SLURM_ARRAY_TASK_ID"
        exit 1
        ;;

esac

echo "============================================================"
echo "Experiment:   $EXPERIMENT_ID"
echo "Topology:     $TOPOLOGY"
echo "Window mode:  $WINDOW_MODE"
echo "Expected rows:$EXPECTED_ROWS"
echo "Expected events:$EXPECTED_EVENTS"
echo "Host:         $(hostname)"
echo "Job:          $SLURM_JOB_ID"
echo "Start:        $(date)"
echo "============================================================"

# ============================================================
# Output locations
# ============================================================

RUN_ROOT="$THESIS_DIR/outputs/chapter4/gru_baselines/${SLURM_ARRAY_JOB_ID}"

EXPERIMENT_ROOT="$RUN_ROOT/$EXPERIMENT_ID"

CHECKPOINT_DIR="$EXPERIMENT_ROOT/checkpoints"

mkdir -p \
    "$EXPERIMENT_ROOT" \
    "$CHECKPOINT_DIR"

# ============================================================
# Copy this topology's windows to node-local storage
# ============================================================

if [[ -z "${TMPDIR:-}" ]]; then
    echo "ERROR: TMPDIR is not set."
    exit 1
fi

JOB_TMP_DIR="$TMPDIR/$SLURM_JOB_ID"

WINDOWS_TMP_PATH="$JOB_TMP_DIR/windows_tmp"

rm -rf "$WINDOWS_TMP_PATH"
mkdir -p "$WINDOWS_TMP_PATH"

echo "Copying windows to:"
echo "$WINDOWS_TMP_PATH"

for EXTENSION in raw parquet json; do

    FILE_PATTERN="*${TOPOLOGY}_W${WINDOW_LENGTH_TAG}_*.${EXTENSION}"

    echo "Searching for:"
    echo "$FILE_PATTERN"

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
    else
        echo "WARNING: No ${EXTENSION} files found."
    fi

done

STAGED_COUNT="$(
    find "$WINDOWS_TMP_PATH" \
        -maxdepth 1 \
        -type f |
    wc -l
)"

if [[ "$STAGED_COUNT" -eq 0 ]]; then
    echo "ERROR: No window files were staged."
    exit 1
fi

echo "Staged files: $STAGED_COUNT"

# ============================================================
# Train the independent plain GRU
# ============================================================

python KOL/run_kol_experiment.py \
    dataset="$TOPOLOGY" \
    dataset.dataset_directory="$DATASET_DIR" \
    window_extraction.window_length="$WINDOW_LENGTH_FLOAT" \
    window_extraction.windows_local_dir="$WINDOWS_TMP_PATH" \
    model.model_name=gru_regressor \
    model.hidden_size=128 \
    model.num_layers=2 \
    model.dropout=0.1 \
    model.bidirectional=false \
    training.target_label=y_fault_location \
    training.feature_groups_include='[lines,loads,winds,extgrid]' \
    training.materialize_feature_filters=false \
    training.n_splits=5 \
    training.val_size=0.15 \
    training.split_seed=42 \
    training.seeds='[42]' \
    training.batch_size=256 \
    training.learning_rate=0.0001 \
    training.weight_decay=0.0001 \
    training.epochs=500 \
    training.patience=15 \
    training.tune_lr_wd=false \
    training.num_workers=4 \
    training.prefetch_factor=2 \
    training.pin_memory=true \
    training.ckpt_dir="$CHECKPOINT_DIR" \
    tracking.use_wandb=false \
    +training.use_operator_features=false \
    +training.apply_window_filter_without_operator=true \
    +training.kol_window_mode="$WINDOW_MODE" \
    +training.cv_mode=stratified_location \
    +training.cv_stratify_col=y_fault_location \
    +training.line_filter=null \
    +training.input_representation=waveform \
    +training.kol_prediction_mode=plain_gru \
    +training.out_dir="$EXPERIMENT_ROOT"

echo
echo "============================================================"
echo "$EXPERIMENT_ID completed"
echo "Output root:"
echo "$EXPERIMENT_ROOT"
echo "End: $(date)"
echo "============================================================"

find "$EXPERIMENT_ROOT" \
    -maxdepth 4 \
    -type f \
    -printf '%P\t%s bytes\n' |
sort

rm -rf "$JOB_TMP_DIR"
