#!/bin/bash -l
#SBATCH --job-name=ch4_double_ended
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:v100:1
#SBATCH --partition=v100
#SBATCH --time=08:00:00
#SBATCH --output=./hpc/hpc_logs/ch4_double_ended-%j-on-%N.out
#SBATCH --error=./hpc/hpc_logs/ch4_double_ended-%j-on-%N.err

set -euo pipefail

# =============================================================================
# CHAPTER 4 — FINAL DOUBLE-ENDED HYBRID EXPERIMENTS
#
#   C90-2E
#   C110-2E
#   L90-2E
#   L110-2E
#
# Smoke-test override:
#   TRAINING_EPOCHS=1 TRAINING_PATIENCE=1 bash run_chapter4_double_ended.sh
#
# Production defaults:
#   epochs=150
#   patience=20
# =============================================================================

TRAINING_EPOCHS="${TRAINING_EPOCHS:-150}"
TRAINING_PATIENCE="${TRAINING_PATIENCE:-20}"
PREDICTION_MODE="${PREDICTION_MODE:-threeph_add_ground_mul}"

# -----------------------------------------------------------------------------
# Environment
# -----------------------------------------------------------------------------

source \
  /home/hpc/iwi5/iwi5305h/miniconda3/etc/profile.d/conda.sh

conda activate Masters_thesis_env_py312

export THESIS_DIR="${THESIS_DIR:-/home/hpc/iwi5/iwi5305h/Masters_thesis_PR_LABS}"
export THIRD_PARTY_DIR="$THESIS_DIR/third_party/dl_fault_repo"

export SOURCE_WINDOWS_DIR="${SOURCE_WINDOWS_DIR:-/home/vault/iwi5/iwi5305h/windows_tmp}"

export TWO_ENDED_INPUT_BASE="$THESIS_DIR/outputs/chapter4/model_inputs/two_ended_posseq"

export TWO_ENDED_ENV="$TWO_ENDED_INPUT_BASE/LATEST_TWO_ENDED_INPUTS.env"

export WINDOW_TAG="0p060"
export STEP_TAG="0p005"

export HYDRA_FULL_ERROR=1
export WANDB_MODE=disabled

export PYTHONPATH="$THIRD_PARTY_DIR/src:$THESIS_DIR:${PYTHONPATH:-}"

: "${SLURM_JOB_ID:?SLURM_JOB_ID is missing. Run inside salloc or submit with sbatch.}"
: "${TMPDIR:?TMPDIR is missing. Run inside salloc or submit with sbatch.}"

cd "$THESIS_DIR"

# -----------------------------------------------------------------------------
# Load prepared two-ended model inputs
# -----------------------------------------------------------------------------

if [ ! -s "$TWO_ENDED_ENV" ]; then
    echo "ERROR: Missing latest two-ended environment file:"
    echo "$TWO_ENDED_ENV"
    exit 1
fi

set -a

# shellcheck disable=SC1090
source "$TWO_ENDED_ENV"

set +a

: "${TWO_ENDED_INPUT_DIR:?TWO_ENDED_INPUT_DIR is missing}"
: "${P90_2E_PRIOR:?P90_2E_PRIOR is missing}"
: "${P90_2E_PRIOR_COL:?P90_2E_PRIOR_COL is missing}"
: "${P110_2E_PRIOR:?P110_2E_PRIOR is missing}"
: "${P110_2E_PRIOR_COL:?P110_2E_PRIOR_COL is missing}"

EXPECTED_PRIOR_COL="d_two_ended_posseq_plus_input_pct"

if [ "$P90_2E_PRIOR_COL" != "$EXPECTED_PRIOR_COL" ]; then
    echo "ERROR: Unexpected 90 kV prior column:"
    echo "$P90_2E_PRIOR_COL"
    exit 1
fi

if [ "$P110_2E_PRIOR_COL" != "$EXPECTED_PRIOR_COL" ]; then
    echo "ERROR: Unexpected 110 kV prior column:"
    echo "$P110_2E_PRIOR_COL"
    exit 1
fi

for PRIOR_PATH in \
    "$P90_2E_PRIOR" \
    "$P110_2E_PRIOR"
do
    if [ ! -s "$PRIOR_PATH" ]; then
        echo "ERROR: Prior file is missing or empty:"
        echo "$PRIOR_PATH"
        exit 1
    fi
done

if ! grep -q \
    'threeph_add_ground_mul' \
    KOL/models/kol_residual_models.py
then
    echo "ERROR: threeph_add_ground_mul is not present in current source."
    exit 1
fi

echo
echo "============================================================"
echo "FINAL DOUBLE-ENDED INPUT SELECTION"
echo "============================================================"
echo "Input directory: $TWO_ENDED_INPUT_DIR"
echo
echo "90 kV prior:     $P90_2E_PRIOR"
echo "90 kV column:    $P90_2E_PRIOR_COL"
echo
echo "110 kV prior:    $P110_2E_PRIOR"
echo "110 kV column:   $P110_2E_PRIOR_COL"
echo
echo "Prediction mode: $PREDICTION_MODE"
echo "Epochs:          $TRAINING_EPOCHS"
echo "Patience:        $TRAINING_PATIENCE"
echo "============================================================"

# -----------------------------------------------------------------------------
# Validate exact frozen cohorts and prepared prior values
# -----------------------------------------------------------------------------

export P90_2E_PRIOR
export P110_2E_PRIOR

python - <<'PY'
import os

import numpy as np
import pandas as pd


PRIOR_COLUMN = "d_two_ended_posseq_plus_input_pct"


def audit(
    path,
    *,
    expected_rows,
    expected_events,
    expected_windows=None,
):
    frame = pd.read_csv(path)

    required = {
        "sample_id",
        "window_idx",
        PRIOR_COLUMN,
    }

    missing = sorted(required - set(frame.columns))

    if missing:
        raise KeyError(f"{path}: missing columns {missing}")

    if len(frame) != expected_rows:
        raise ValueError(
            f"{path}: expected {expected_rows} rows, "
            f"found {len(frame)}"
        )

    event_count = frame["sample_id"].nunique(dropna=False)

    if event_count != expected_events:
        raise ValueError(
            f"{path}: expected {expected_events} events, "
            f"found {event_count}"
        )

    if frame.duplicated(["sample_id", "window_idx"]).any():
        raise ValueError(
            f"{path}: duplicate (sample_id, window_idx) keys"
        )

    prior = pd.to_numeric(
        frame[PRIOR_COLUMN],
        errors="coerce",
    ).to_numpy(dtype=float)

    if not np.isfinite(prior).all():
        raise ValueError(
            f"{path}: prepared prior contains non-finite values"
        )

    if prior.min() < 0.0 or prior.max() > 100.0:
        raise ValueError(
            f"{path}: prepared prior lies outside [0,100]"
        )

    if expected_windows is not None:
        observed_windows = set(
            pd.to_numeric(
                frame["window_idx"],
                errors="raise",
            ).astype(int)
        )

        if observed_windows != set(expected_windows):
            raise ValueError(
                f"{path}: expected window indices "
                f"{sorted(expected_windows)}, found "
                f"{sorted(observed_windows)}"
            )

        rows_per_event = frame.groupby("sample_id").size()

        if rows_per_event.min() != 4 or rows_per_event.max() != 4:
            raise ValueError(
                f"{path}: expected four rows per event; "
                f"observed min={rows_per_event.min()}, "
                f"max={rows_per_event.max()}"
            )

    print()
    print(path)
    print(f"  rows:        {len(frame)}")
    print(f"  events:      {event_count}")
    print(f"  prior range: [{prior.min():.6f}, {prior.max():.6f}]")


audit(
    os.environ["P90_2E_PRIOR"],
    expected_rows=9022,
    expected_events=9022,
)

audit(
    os.environ["P110_2E_PRIOR"],
    expected_rows=3648,
    expected_events=912,
    expected_windows={8, 9, 10, 11},
)

print()
print("FINAL TWO-ENDED PRIOR AUDIT PASSED")
PY

# -----------------------------------------------------------------------------
# Unique output and temporary paths
# -----------------------------------------------------------------------------

RUN_TIMESTAMP="$(date +%Y%m%d_%H%M%S)"

export RUN_ROOT="$THESIS_DIR/outputs/chapter4/hybrid_double_ended/${SLURM_JOB_ID}_${RUN_TIMESTAMP}"

export LOG_DIR="$RUN_ROOT/logs"
export STATUS_CSV="$RUN_ROOT/experiment_status.csv"

export JOB_TMP_DIR="$TMPDIR/$SLURM_JOB_ID"
export WINDOWS_TMP_PATH="$JOB_TMP_DIR/windows_tmp_chapter4"

if [ -e "$RUN_ROOT" ]; then
    echo "ERROR: Refusing to overwrite existing run directory:"
    echo "$RUN_ROOT"
    exit 1
fi

mkdir -p \
    "$RUN_ROOT" \
    "$LOG_DIR" \
    "$WINDOWS_TMP_PATH"

printf \
'experiment_id,status,return_code,start_time,end_time,topology,prior_path,prior_column,operator_features,model_mode,prediction_mode,output_dir,log_file\n' \
> "$STATUS_CSV"

cat > "$RUN_ROOT/run_environment.txt" <<EOF
slurm_job_id=$SLURM_JOB_ID
hostname=$(hostname)
start_time=$(date --iso-8601=seconds)
conda_environment=${CONDA_DEFAULT_ENV:-unknown}
python=$(which python)
thesis_dir=$THESIS_DIR
third_party_dir=$THIRD_PARTY_DIR
source_windows_dir=$SOURCE_WINDOWS_DIR
windows_tmp_path=$WINDOWS_TMP_PATH
two_ended_input_dir=$TWO_ENDED_INPUT_DIR
p90_prior=$P90_2E_PRIOR
p90_prior_column=$P90_2E_PRIOR_COL
p110_prior=$P110_2E_PRIOR
p110_prior_column=$P110_2E_PRIOR_COL
operator_features=[]
prediction_mode=$PREDICTION_MODE
training_epochs=$TRAINING_EPOCHS
training_patience=$TRAINING_PATIENCE
EOF

if git -C "$THESIS_DIR" rev-parse HEAD >/dev/null 2>&1; then
    git -C "$THESIS_DIR" rev-parse HEAD \
        > "$RUN_ROOT/git_commit.txt"

    git -C "$THESIS_DIR" status --short \
        > "$RUN_ROOT/git_status.txt"
fi

echo
echo "============================================================"
echo "CHAPTER 4 DOUBLE-ENDED EXPERIMENTS"
echo "============================================================"
echo "Slurm job:       $SLURM_JOB_ID"
echo "Hostname:        $(hostname)"
echo "Run root:        $RUN_ROOT"
echo "Temporary path:  $WINDOWS_TMP_PATH"
echo "Prediction mode: $PREDICTION_MODE"
echo "Epochs:          $TRAINING_EPOCHS"
echo "Patience:        $TRAINING_PATIENCE"
echo "Start:           $(date)"
echo "============================================================"

nvidia-smi

# -----------------------------------------------------------------------------
# Stage waveform datasets
# -----------------------------------------------------------------------------

stage_topology() {
    local topology="$1"

    local marker="$WINDOWS_TMP_PATH/.staged_${topology}"
    local pattern="*${topology}_W${WINDOW_TAG}_S${STEP_TAG}*"

    if [ -f "$marker" ]; then
        echo "Dataset already staged: $topology"
        return
    fi

    local count

    count="$(
        find "$SOURCE_WINDOWS_DIR" \
            -maxdepth 1 \
            -type f \
            -name "$pattern" \
            | wc -l
    )"

    echo
    echo "Staging topology: $topology"
    echo "Pattern:          $pattern"
    echo "Matching files:   $count"

    if [ "$count" -eq 0 ]; then
        echo "ERROR: No source waveform files found for $topology"
        exit 1
    fi

    find "$SOURCE_WINDOWS_DIR" \
        -maxdepth 1 \
        -type f \
        -name "$pattern" \
        -exec cp -t "$WINDOWS_TMP_PATH" {} +

    local raw_file="$WINDOWS_TMP_PATH/X_${topology}_W${WINDOW_TAG}_S${STEP_TAG}.raw"

    if [ ! -f "$raw_file" ]; then
        echo "ERROR: Expected raw waveform was not staged:"
        echo "$raw_file"
        exit 1
    fi

    touch "$marker"

    ls -lh "$raw_file"
}

stage_topology hv_double_line_90kv
stage_topology hv_double_line_110kv

df -h "$JOB_TMP_DIR" || true

# -----------------------------------------------------------------------------
# Compile active source files
# -----------------------------------------------------------------------------

python -m py_compile \
    KOL/run_kol_experiment.py \
    KOL/models/kol_residual_models.py \
    KOL/training/kol_residual_train.py \
    KOL/training/kol_fold_runner.py \
    KOL/training/kol_experiment.py \
    KOL/common/operator_features.py \
    KOL/datasets/kol_data_preparation.py \
    third_party/dl_fault_repo/src/dl_psp/utils/run_utils.py

echo
echo "Python compile check passed."

# -----------------------------------------------------------------------------
# Sequential runner
# -----------------------------------------------------------------------------

OPERATOR_FEATURE_COLS='[]'
FAIL_COUNT=0

run_experiment() {
    local experiment_id="$1"
    local topology="$2"
    local prior_path="$3"
    local prior_column="$4"
    local model_mode="$5"
    local hidden_size="$6"
    local num_layers="$7"
    local dropout="$8"
    local window_mode="$9"
    local expected_waveform_features="${10}"

    local experiment_output="$RUN_ROOT/$experiment_id"
    local checkpoint_dir="$experiment_output/checkpoints"
    local log_file="$LOG_DIR/${experiment_id}.log"

    if [ -e "$experiment_output" ]; then
        echo "ERROR: Refusing to overwrite:"
        echo "$experiment_output"
        FAIL_COUNT=$((FAIL_COUNT + 1))
        return
    fi

    mkdir -p "$checkpoint_dir"

    local start_time
    local end_time
    local return_code
    local status

    start_time="$(date --iso-8601=seconds)"

    local command=(
        python
        -u
        KOL/run_kol_experiment.py

        "dataset=$topology"

        model.model_name=gru_regressor
        "model.hidden_size=$hidden_size"
        "model.num_layers=$num_layers"
        "model.dropout=$dropout"
        model.bidirectional=false

        training.target_label=y_fault_location
        'training.feature_groups_include=[lines,loads,winds,extgrid]'

        training.batch_size=64
        "training.epochs=$TRAINING_EPOCHS"
        "training.patience=$TRAINING_PATIENCE"

        training.tune_lr_wd=false
        training.learning_rate=0.0003
        training.weight_decay=0.0001

        'training.seeds=[42]'
        training.n_splits=5
        training.split_seed=42

        "training.ckpt_dir=$checkpoint_dir"
        "+training.out_dir=$experiment_output"

        +training.use_operator_features=true
        "+training.operator_features_path=$prior_path"
        "+training.operator_prior_col=$prior_column"
        "+training.operator_feature_cols=$OPERATOR_FEATURE_COLS"

        "+training.kol_model_mode=$model_mode"
        "+training.kol_prediction_mode=$PREDICTION_MODE"
        "+training.kol_window_mode=$window_mode"

        +training.cv_mode=stratified_location
        +training.cv_stratify_col=y_fault_location

        +training.input_representation=waveform

        +training.case_emb_dim=8
        +training.fusion_head_hidden_size=64
        +training.bounded_residual_max=1.0
        +training.gate_init_bias=-3.0

        window_extraction.window_length=0.060
        window_extraction.step_length_seconds=0.005
        "window_extraction.windows_local_dir=$WINDOWS_TMP_PATH"

        "tracking.project=chapter4_${experiment_id}"
    )

    echo
    echo "======================================================================"
    echo "STARTING $experiment_id"
    echo "======================================================================"
    echo "Topology:                   $topology"
    echo "Prior path:                 $prior_path"
    echo "Prior column:               $prior_column"
    echo "Operator features:          []"
    echo "Model mode:                 $model_mode"
    echo "Prediction mode:            $PREDICTION_MODE"
    echo "Window mode:                $window_mode"
    echo "Expected waveform features: $expected_waveform_features"
    echo "Output directory:           $experiment_output"
    echo "Log file:                   $log_file"
    echo "======================================================================"

    printf 'COMMAND:'
    printf ' %q' "${command[@]}"
    printf '\n'

    set +e

    "${command[@]}" \
        2>&1 \
        | tee "$log_file"

    return_code="${PIPESTATUS[0]}"

    set -e

    if [ "$return_code" -eq 0 ]; then
        status="COMPLETED"
    else
        status="FAILED"
        FAIL_COUNT=$((FAIL_COUNT + 1))
    fi

    end_time="$(date --iso-8601=seconds)"

    printf \
        '%s,%s,%s,%s,%s,%s,%s,%s,"%s",%s,%s,%s,%s\n' \
        "$experiment_id" \
        "$status" \
        "$return_code" \
        "$start_time" \
        "$end_time" \
        "$topology" \
        "$prior_path" \
        "$prior_column" \
        "$OPERATOR_FEATURE_COLS" \
        "$model_mode" \
        "$PREDICTION_MODE" \
        "$experiment_output" \
        "$log_file" \
        >> "$STATUS_CSV"

    echo
    echo "$experiment_id finished with status: $status"

    grep -E \
        "Feature filter|Selected raw feature indices|KOL window mode|KOL model mode|KOL prediction mode|Selected operator feature columns|Setup:|split sizes|Prior-only comparison|Final model evaluation|CV aggregate metrics|test/mae|test/rmse|test/prior_mae|test/prior_rmse|test/improvement_rate|test/worsened_rate|test/effective_correction" \
        "$log_file" \
        || true
}

# -----------------------------------------------------------------------------
# Correction models
# -----------------------------------------------------------------------------

run_experiment \
    C90-2E \
    hv_double_line_90kv \
    "$P90_2E_PRIOR" \
    "$P90_2E_PRIOR_COL" \
    legacy_residual \
    384 \
    3 \
    0.2 \
    single_fault_start \
    48

run_experiment \
    C110-2E \
    hv_double_line_110kv \
    "$P110_2E_PRIOR" \
    "$P110_2E_PRIOR_COL" \
    legacy_residual \
    384 \
    3 \
    0.2 \
    all_fault_start \
    18

# -----------------------------------------------------------------------------
# Bounded residual-fusion models
# -----------------------------------------------------------------------------

run_experiment \
    L90-2E \
    hv_double_line_90kv \
    "$P90_2E_PRIOR" \
    "$P90_2E_PRIOR_COL" \
    bounded_residual_fusion \
    64 \
    1 \
    0.0 \
    single_fault_start \
    48

run_experiment \
    L110-2E \
    hv_double_line_110kv \
    "$P110_2E_PRIOR" \
    "$P110_2E_PRIOR_COL" \
    bounded_residual_fusion \
    64 \
    1 \
    0.0 \
    all_fault_start \
    18

# -----------------------------------------------------------------------------
# Final summary
# -----------------------------------------------------------------------------

echo
echo "============================================================"
echo "DOUBLE-ENDED RUN FINISHED"
echo "============================================================"
echo "End time: $(date)"
echo "Run root: $RUN_ROOT"
echo "Status:   $STATUS_CSV"
echo "Failures: $FAIL_COUNT"
echo "============================================================"

if command -v column >/dev/null 2>&1; then
    column -s, -t "$STATUS_CSV"
else
    cat "$STATUS_CSV"
fi

if [ "$FAIL_COUNT" -gt 0 ]; then
    exit 1
fi

