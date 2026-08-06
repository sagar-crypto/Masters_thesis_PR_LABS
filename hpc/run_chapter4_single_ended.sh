#!/bin/bash -l
#SBATCH --job-name=ch4_single_ended
#SBATCH --ntasks=1
#SBATCH --gres=gpu:v100:1
#SBATCH --partition=v100
#SBATCH --time=08:00:00
#SBATCH --output=./hpc/hpc_logs/ch4_single_ended-%j-on-%N.out
#SBATCH --error=./hpc/hpc_logs/ch4_single_ended-%j-on-%N.err

set -euo pipefail

# =============================================================================
# CHAPTER 4 — SINGLE-ENDED CORRECTION AND COMBINATION EXPERIMENTS
#
# Runs sequentially on one allocated GPU:
#
#   C90-1E
#   C110-1E
#   L90-1E
#   L110-1E
#
# Final model paths:
#
#   Correction:
#       kol_model_mode=legacy_residual
#       KOLGRUCaseResidualRegressor
#
#   Combination learning:
#       kol_model_mode=bounded_residual_fusion
#       KOLGRUBoundedResidualFusionRegressor
#
#   Prediction rule:
#       threeph_add_ground_mul
#
# This is a plain Bash script. Run it inside an salloc session.
# =============================================================================


# -----------------------------------------------------------------------------
# 1. Environment
# -----------------------------------------------------------------------------

source \
  /home/hpc/iwi5/iwi5305h/miniconda3/etc/profile.d/conda.sh

conda activate Masters_thesis_env_py312

export THESIS_DIR="${THESIS_DIR:-/home/hpc/iwi5/iwi5305h/Masters_thesis_PR_LABS}"

export THIRD_PARTY_DIR="$THESIS_DIR/third_party/dl_fault_repo"

export SOURCE_WINDOWS_DIR="${SOURCE_WINDOWS_DIR:-/home/vault/iwi5/iwi5305h/windows_tmp}"

export SINGLE_ENDED_INPUT_BASE="$THESIS_DIR/outputs/chapter4/model_inputs/caseaware_bestmae"

export SINGLE_ENDED_ENV="$SINGLE_ENDED_INPUT_BASE/LATEST_CASEAWARE_SINGLE_ENDED_INPUTS.env"

export WINDOW_TAG="0p060"
export STEP_TAG="0p005"

export HYDRA_FULL_ERROR=1
export WANDB_MODE=disabled

export PYTHONPATH="$THIRD_PARTY_DIR/src:$THESIS_DIR:${PYTHONPATH:-}"


# -----------------------------------------------------------------------------
# 2. Confirm that this runs inside salloc
# -----------------------------------------------------------------------------

: "${SLURM_JOB_ID:?SLURM_JOB_ID is missing. Run this script inside salloc.}"

: "${TMPDIR:?TMPDIR is missing. Run this script inside salloc.}"


# -----------------------------------------------------------------------------
# 3. Load the newly generated prior paths and feature lists
# -----------------------------------------------------------------------------

if [ ! -f "$SINGLE_ENDED_ENV" ]; then
    echo "ERROR: The latest single-ended input environment does not exist:"
    echo "$SINGLE_ENDED_ENV"
    echo
    echo "Run the custom CSV preparation script first."
    exit 1
fi

# Export every variable defined by the environment file so that the Python
# audit below can also read them through os.environ.
set -a

# shellcheck disable=SC1090
source "$SINGLE_ENDED_ENV"

set +a
# Compatibility alias for the new case-aware environment.
export SINGLE_ENDED_INPUT_DIR="${CASEAWARE_INPUT_DIR:?CASEAWARE_INPUT_DIR is missing from the environment file}"

# =============================================================================
# FINAL ACTIVE SINGLE-ENDED PRIOR SELECTION
#
# 90 kV:
#   fault-case-dependent mapping
#
# 110 kV:
#   fault-line-and-fault-case-dependent mapping
# =============================================================================

: "${P90_1E_CASE_PRIOR:?P90_1E_CASE_PRIOR is missing}"
: "${P110_1E_LINE_CASE_PRIOR:?P110_1E_LINE_CASE_PRIOR is missing}"

export P90_1E_PRIOR="$P90_1E_CASE_PRIOR"
export P90_1E_PRIOR_COL="d_90kv_case_bestmae_input_pct"

export P110_1E_PRIOR="$P110_1E_LINE_CASE_PRIOR"
export P110_1E_PRIOR_COL="d_110kv_line_case_bestmae_input_pct"

export ACTIVE_90_MAPPING_MODE="case"
export ACTIVE_110_MAPPING_MODE="line_case"

echo "============================================================"
echo "Final active mapping selection"
echo "============================================================"
echo "90 kV mapping:  $ACTIVE_90_MAPPING_MODE"
echo "90 kV prior:    $P90_1E_PRIOR"
echo "90 kV column:   $P90_1E_PRIOR_COL"
echo
echo "110 kV mapping: $ACTIVE_110_MAPPING_MODE"
echo "110 kV prior:   $P110_1E_PRIOR"
echo "110 kV column:  $P110_1E_PRIOR_COL"
echo "============================================================"

# =============================================================================
# Validate the final mixed mapping selection
# =============================================================================

EXPECTED_P90_PRIOR_COL="d_90kv_case_bestmae_input_pct"
EXPECTED_P110_PRIOR_COL="d_110kv_line_case_bestmae_input_pct"

if [ "$P90_1E_PRIOR_COL" != "$EXPECTED_P90_PRIOR_COL" ]; then
    echo "ERROR: Unexpected active 90 kV prior column."
    echo "Expected: $EXPECTED_P90_PRIOR_COL"
    echo "Found:    $P90_1E_PRIOR_COL"
    exit 1
fi

if [ "$P110_1E_PRIOR_COL" != "$EXPECTED_P110_PRIOR_COL" ]; then
    echo "ERROR: Unexpected active 110 kV prior column."
    echo "Expected: $EXPECTED_P110_PRIOR_COL"
    echo "Found:    $P110_1E_PRIOR_COL"
    exit 1
fi

case "$P90_1E_PRIOR" in
    *"/kol_operator_features_hv_double_line_90kv_case_bestmae_"*)
        ;;
    *)
        echo "ERROR: The active 90 kV file is not the case-aware prior:"
        echo "$P90_1E_PRIOR"
        exit 1
        ;;
esac

case "$P110_1E_PRIOR" in
    *"/kol_operator_features_hv_double_line_110kv_line_case_bestmae_"*)
        ;;
    *)
        echo "ERROR: The active 110 kV file is not the line-case-aware prior:"
        echo "$P110_1E_PRIOR"
        exit 1
        ;;
esac

for ACTIVE_PRIOR in \
    "$P90_1E_PRIOR" \
    "$P110_1E_PRIOR"
do
    if [ ! -s "$ACTIVE_PRIOR" ]; then
        echo "ERROR: Active prior file is missing or empty:"
        echo "$ACTIVE_PRIOR"
        exit 1
    fi
done

echo
echo "============================================================"
echo "FINAL MIXED PRIOR VALIDATION PASSED"
echo "============================================================"
echo "90 kV:  case mapping"
echo "Column: $P90_1E_PRIOR_COL"
echo "File:   $P90_1E_PRIOR"
echo
echo "110 kV: line-case mapping"
echo "Column: $P110_1E_PRIOR_COL"
echo "File:   $P110_1E_PRIOR"
echo "============================================================"

echo "============================================================"
echo "Loaded single-ended model inputs"
echo "============================================================"
echo
echo "Input directory:"
echo "$SINGLE_ENDED_INPUT_DIR"
echo
echo "90 kV prior:"
echo "$P90_1E_PRIOR"
echo
echo "90 kV prior column:"
echo "$P90_1E_PRIOR_COL"
echo
echo "90 kV operator features:"
echo "$P90_1E_OPERATOR_FEATURE_COLS"
echo
echo "110 kV prior:"
echo "$P110_1E_PRIOR"
echo
echo "110 kV prior column:"
echo "$P110_1E_PRIOR_COL"
echo
echo "110 kV operator features:"
echo "$P110_1E_OPERATOR_FEATURE_COLS"
echo


for REQUIRED_PRIOR in \
    "$P90_1E_PRIOR" \
    "$P110_1E_PRIOR"
do
    if [ ! -s "$REQUIRED_PRIOR" ]; then
        echo "ERROR: Required prior file is missing or empty:"
        echo "$REQUIRED_PRIOR"
        exit 1
    fi
done


# -----------------------------------------------------------------------------
# 4. Unique run paths
# -----------------------------------------------------------------------------

RUN_TIMESTAMP="$(date +%Y%m%d_%H%M%S)"

export RUN_ROOT="$THESIS_DIR/outputs/chapter4/hybrid_single_ended/${SLURM_JOB_ID}_${RUN_TIMESTAMP}"

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
'experiment_id,status,return_code,start_time,end_time,topology,prior_path,prior_column,operator_features,model_mode,output_dir,log_file\n' \
> "$STATUS_CSV"


cd "$THESIS_DIR"


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
single_ended_input_dir=$SINGLE_ENDED_INPUT_DIR
p90_prior=$P90_1E_PRIOR
p90_prior_column=$P90_1E_PRIOR_COL
p90_operator_features=$P90_1E_OPERATOR_FEATURE_COLS
p110_prior=$P110_1E_PRIOR
p110_prior_column=$P110_1E_PRIOR_COL
p110_operator_features=$P110_1E_OPERATOR_FEATURE_COLS
EOF


if git -C "$THESIS_DIR" rev-parse HEAD >/dev/null 2>&1; then
    git -C "$THESIS_DIR" rev-parse HEAD \
        > "$RUN_ROOT/git_commit.txt"

    git -C "$THESIS_DIR" status --short \
        > "$RUN_ROOT/git_status.txt"
fi


echo "============================================================"
echo "Chapter 4 single-ended experiments"
echo "============================================================"
echo "Slurm job:     $SLURM_JOB_ID"
echo "Hostname:      $(hostname)"
echo "Run root:      $RUN_ROOT"
echo "Temporary dir: $WINDOWS_TMP_PATH"
echo "Start time:    $(date)"
echo "============================================================"

nvidia-smi


# -----------------------------------------------------------------------------
# 5. Stage both waveform datasets once
# -----------------------------------------------------------------------------

stage_topology() {
    local topology="$1"

    local marker="$WINDOWS_TMP_PATH/.staged_${topology}"

    local pattern="*${topology}_W${WINDOW_TAG}_S${STEP_TAG}*"

    if [ -f "$marker" ]; then
        echo
        echo "Dataset already staged:"
        echo "$topology"
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
    echo "Staging topology:"
    echo "$topology"
    echo
    echo "Search pattern:"
    echo "$pattern"
    echo
    echo "Matching files:"
    echo "$count"

    if [ "$count" -eq 0 ]; then
        echo "ERROR: No source window files were found for:"
        echo "$topology"
        exit 1
    fi

    find "$SOURCE_WINDOWS_DIR" \
        -maxdepth 1 \
        -type f \
        -name "$pattern" \
        -exec cp -t "$WINDOWS_TMP_PATH" {} +

    local raw_file="$WINDOWS_TMP_PATH/X_${topology}_W${WINDOW_TAG}_S${STEP_TAG}.raw"

    if [ ! -f "$raw_file" ]; then
        echo "ERROR: Expected raw waveform file was not staged:"
        echo "$raw_file"
        exit 1
    fi

    touch "$marker"

    echo
    echo "Staged raw waveform:"
    ls -lh "$raw_file"
}


stage_topology hv_double_line_90kv

stage_topology hv_double_line_110kv


echo
echo "All staged window files:"

find "$WINDOWS_TMP_PATH" \
    -maxdepth 1 \
    -type f \
    -printf '%f\t%s bytes\n' \
    | sort


df -h "$JOB_TMP_DIR" || true


# -----------------------------------------------------------------------------
# 6. Compile-check active repository files
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
# 9. Sequential experiment runner
# -----------------------------------------------------------------------------

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

    local operator_feature_cols="${10}"

    local expected_waveform_features="${11}"


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
        training.epochs=150
        training.patience=20

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
        "+training.operator_feature_cols=$operator_feature_cols"

        "+training.kol_model_mode=$model_mode"

        +training.kol_prediction_mode=threeph_add_ground_mul

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
    echo "Topology:                  $topology"
    echo "Prior path:                $prior_path"
    echo "Prior column:              $prior_column"
    echo "Operator features:         $operator_feature_cols"
    echo "Model mode:                $model_mode"
    echo "Prediction mode:           threeph_add_ground_mul"
    echo "Window mode:               $window_mode"
    echo "Expected waveform features: $expected_waveform_features"
    echo "Output directory:          $experiment_output"
    echo "Log file:                  $log_file"
    echo "======================================================================"


    printf 'COMMAND:'

    printf ' %q' \
        "${command[@]}"

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
        '%s,%s,%s,%s,%s,%s,%s,%s,"%s",%s,%s,%s\n' \
        "$experiment_id" \
        "$status" \
        "$return_code" \
        "$start_time" \
        "$end_time" \
        "$topology" \
        "$prior_path" \
        "$prior_column" \
        "$operator_feature_cols" \
        "$model_mode" \
        "$experiment_output" \
        "$log_file" \
        >> "$STATUS_CSV"


    echo
    echo "$experiment_id finished with status:"
    echo "$status"


    grep -E \
        "Feature filter|Selected raw feature indices|Applied custom filtering|KOL window mode|KOL model mode|KOL prediction mode|Selected operator feature columns|Setup:|split sizes|Bounded residual|Prior correction statistics|Final model evaluation|CV aggregate metrics|test/mae|test/rmse|test/prior_mae|test/prior_rmse|test/direct_gru_mae|test/improvement_rate|test/worsened_rate|test/effective_correction" \
        "$log_file" \
        || true
}


# =============================================================================
# 10. Run correction models
# =============================================================================

run_experiment \
    C90-1E \
    hv_double_line_90kv \
    "$P90_1E_PRIOR" \
    "$P90_1E_PRIOR_COL" \
    legacy_residual \
    384 \
    3 \
    0.2 \
    single_fault_start \
    "$P90_1E_OPERATOR_FEATURE_COLS" \
    48


run_experiment \
    C110-1E \
    hv_double_line_110kv \
    "$P110_1E_PRIOR" \
    "$P110_1E_PRIOR_COL" \
    legacy_residual \
    384 \
    3 \
    0.2 \
    all_fault_start \
    "$P110_1E_OPERATOR_FEATURE_COLS" \
    18


# =============================================================================
# 11. Run bounded-residual combination-learning models
# =============================================================================

run_experiment \
    L90-1E \
    hv_double_line_90kv \
    "$P90_1E_PRIOR" \
    "$P90_1E_PRIOR_COL" \
    bounded_residual_fusion \
    64 \
    1 \
    0.0 \
    single_fault_start \
    "$P90_1E_OPERATOR_FEATURE_COLS" \
    48


run_experiment \
    L110-1E \
    hv_double_line_110kv \
    "$P110_1E_PRIOR" \
    "$P110_1E_PRIOR_COL" \
    bounded_residual_fusion \
    64 \
    1 \
    0.0 \
    all_fault_start \
    "$P110_1E_OPERATOR_FEATURE_COLS" \
    18


# -----------------------------------------------------------------------------
# 12. Final summary
# -----------------------------------------------------------------------------

echo
echo "============================================================"
echo "SINGLE-ENDED RUN FINISHED"
echo "============================================================"
echo "End time:  $(date)"
echo "Run root:  $RUN_ROOT"
echo "Status:    $STATUS_CSV"
echo "Failures:  $FAIL_COUNT"
echo "============================================================"


if command -v column >/dev/null 2>&1; then
    column \
        -s, \
        -t \
        "$STATUS_CSV"
else
    cat "$STATUS_CSV"
fi


echo
echo "Node-local waveforms were left in place:"
echo "$WINDOWS_TMP_PATH"
echo
echo "This allows the double-ended script to reuse them in the same allocation."


if [ "$FAIL_COUNT" -gt 0 ]; then
    exit 1
fi


exit 0
