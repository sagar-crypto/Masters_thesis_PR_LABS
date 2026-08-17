#!/bin/bash -l
#SBATCH --job-name=ch4_2e_raw_prior
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:v100:1
#SBATCH --partition=v100
#SBATCH --time=12:00:00
#SBATCH --output=./hpc/hpc_logs/tmp90_afs_hybrids-%j-on-%N.out
#SBATCH --error=./hpc/hpc_logs/tmp90_afs_hybrids-%j-on-%N.err

set -euo pipefail


TRAINING_EPOCHS="${TRAINING_EPOCHS:-150}"
TRAINING_PATIENCE="${TRAINING_PATIENCE:-20}"

PREDICTION_MODE="${PREDICTION_MODE:-threeph_add_ground_mul}"


source \
  /home/hpc/iwi5/iwi5305h/miniconda3/etc/profile.d/conda.sh

conda activate Masters_thesis_env_py312


export THESIS_DIR="${THESIS_DIR:-/home/hpc/iwi5/iwi5305h/Masters_thesis_PR_LABS}"

export THIRD_PARTY_DIR=\
"$THESIS_DIR/third_party/dl_fault_repo"

export SOURCE_WINDOWS_DIR="${SOURCE_WINDOWS_DIR:-/home/vault/iwi5/iwi5305h/windows_tmp}"

export TEMP_BASE=\
"$THESIS_DIR/outputs/chapter4/temp_90kv_afs_check"

export INPUT_ENV=\
"$TEMP_BASE/LATEST_INPUTS.env"

export WINDOW_TAG="0p060"
export STEP_TAG="0p005"

export HYDRA_FULL_ERROR=1
export WANDB_MODE=disabled

export PYTHONPATH=\
"$THIRD_PARTY_DIR/src:$THESIS_DIR:${PYTHONPATH:-}"


: "${SLURM_JOB_ID:?Submit this script with sbatch.}"
: "${TMPDIR:?TMPDIR is missing.}"


cd "$THESIS_DIR"

mkdir -p hpc/hpc_logs


if [ ! -s "$INPUT_ENV" ]; then
    echo "ERROR: Missing temporary model-input environment:"
    echo "$INPUT_ENV"

    echo
    echo "Run this first:"
    echo "bash hpc/archive/development_checks/temp_02_build_90kv_all_fault_start_inputs.sh"

    exit 1
fi


set -a

# shellcheck disable=SC1090
source "$INPUT_ENV"

set +a


: "${TEMP_90_AFS_INPUT_DIR:?Missing TEMP_90_AFS_INPUT_DIR}"

: "${P90_1E_AFS_PRIOR:?Missing P90_1E_AFS_PRIOR}"
: "${P90_1E_AFS_PRIOR_COL:?Missing P90_1E_AFS_PRIOR_COL}"
: "${P90_1E_AFS_FEATURE_COLS:?Missing P90_1E_AFS_FEATURE_COLS}"

: "${P90_2E_AFS_PRIOR:?Missing P90_2E_AFS_PRIOR}"
: "${P90_2E_AFS_PRIOR_COL:?Missing P90_2E_AFS_PRIOR_COL}"
: "${P90_2E_AFS_FEATURE_COLS:?Missing P90_2E_AFS_FEATURE_COLS}"


for path in \
    "$P90_1E_AFS_PRIOR" \
    "$P90_2E_AFS_PRIOR"
do
    if [ ! -s "$path" ]; then
        echo "ERROR: Missing model-input CSV:"
        echo "$path"
        exit 1
    fi
done


export \
  P90_1E_AFS_PRIOR \
  P90_1E_AFS_PRIOR_COL \
  P90_2E_AFS_PRIOR \
  P90_2E_AFS_PRIOR_COL


python - <<'PY'
import os

import numpy as np
import pandas as pd


specs = [
    (
        os.environ[
            "P90_1E_AFS_PRIOR"
        ],
        os.environ[
            "P90_1E_AFS_PRIOR_COL"
        ],
    ),
    (
        os.environ[
            "P90_2E_AFS_PRIOR"
        ],
        os.environ[
            "P90_2E_AFS_PRIOR_COL"
        ],
    ),
]


reference_keys = None


for (
    path,
    prior_column,
) in specs:
    frame = pd.read_csv(
        path
    )

    required = {
        "sample_id",
        "window_idx",
        prior_column,
    }

    missing = sorted(
        required
        - set(frame.columns)
    )

    if missing:
        raise KeyError(
            f"{path}: missing "
            f"columns {missing}"
        )

    if frame.duplicated(
        [
            "sample_id",
            "window_idx",
        ]
    ).any():
        raise ValueError(
            f"{path}: duplicate keys"
        )

    keys = frame[
        [
            "sample_id",
            "window_idx",
        ]
    ].astype(str)

    if reference_keys is None:
        reference_keys = keys

    elif not keys.equals(
        reference_keys
    ):
        raise ValueError(
            "1E and 2E model-input "
            "keys do not match"
        )

    event_sizes = (
        frame.groupby(
            "sample_id"
        )
        .size()
    )

    if event_sizes.size != 9022:
        raise ValueError(
            f"{path}: expected "
            f"9022 events, found "
            f"{event_sizes.size}"
        )

    if int(event_sizes.min()) < 1:
        raise ValueError(
            f"{path}: at least one event "
            "has no retained window"
        )

    prior = pd.to_numeric(
        frame[
            prior_column
        ],
        errors="coerce",
    ).to_numpy(dtype=float)

    if not np.isfinite(
        prior
    ).all():
        raise ValueError(
            f"{path}: prior contains "
            "non-finite values"
        )

    if (
        prior.min() < 0.0
        or prior.max() > 100.0
    ):
        raise ValueError(
            f"{path}: prior lies "
            "outside [0,100]"
        )

    print()
    print(path)
    print(
        "  rows:",
        len(frame),
    )
    print(
        "  events:",
        event_sizes.size,
    )
    print(
        "  windows/event:",
        f"min={int(event_sizes.min())}, "
        f"max={int(event_sizes.max())}, "
        f"mean={float(event_sizes.mean()):.3f}",
    )
    print(
        "  prior range:",
        float(
            prior.min()
        ),
        float(
            prior.max()
        ),
    )


print()
print(
    "TEMPORARY 90 kV "
    "ALL-FAULT-START INPUT AUDIT PASSED"
)
PY


if ! grep -q \
    'threeph_add_ground_mul' \
    KOL/models/kol_residual_models.py
then
    echo "ERROR: threeph_add_ground_mul is missing from the active source."
    exit 1
fi


python -m py_compile \
    KOL/run_kol_experiment.py \
    KOL/models/kol_residual_models.py \
    KOL/training/kol_residual_train.py \
    KOL/training/kol_fold_runner.py \
    KOL/training/kol_experiment.py \
    KOL/common/operator_features.py \
    KOL/datasets/kol_data_preparation.py \
    third_party/dl_fault_repo/src/dl_psp/utils/run_utils.py


RUN_TIMESTAMP="$(
    date +%Y%m%d_%H%M%S
)"


export RUN_ROOT=\
"$TEMP_BASE/hybrid_runs/${SLURM_JOB_ID}_${RUN_TIMESTAMP}"

export LOG_DIR=\
"$RUN_ROOT/logs"

export STATUS_CSV=\
"$RUN_ROOT/experiment_status.csv"

export JOB_TMP_DIR=\
"$TMPDIR/$SLURM_JOB_ID"

export WINDOWS_TMP_PATH=\
"$JOB_TMP_DIR/windows_tmp_chapter4"


mkdir -p \
  "$RUN_ROOT" \
  "$LOG_DIR" \
  "$WINDOWS_TMP_PATH"


printf \
'experiment_id,status,return_code,start_time,end_time,prior_path,prior_column,operator_features,model_mode,window_mode,output_dir,log_file\n' \
> "$STATUS_CSV"


cat > "$RUN_ROOT/run_environment.txt" <<EOF
slurm_job_id=$SLURM_JOB_ID
hostname=$(hostname)
start_time=$(date --iso-8601=seconds)

variant=temporary_90kv_all_fault_start_check
input_dir=$TEMP_90_AFS_INPUT_DIR

p90_1e_prior=$P90_1E_AFS_PRIOR
p90_1e_prior_column=$P90_1E_AFS_PRIOR_COL
p90_1e_features=$P90_1E_AFS_FEATURE_COLS

p90_2e_prior=$P90_2E_AFS_PRIOR
p90_2e_prior_column=$P90_2E_AFS_PRIOR_COL
p90_2e_features=$P90_2E_AFS_FEATURE_COLS

window_mode=all_fault_start
prediction_mode=$PREDICTION_MODE
epochs=$TRAINING_EPOCHS
patience=$TRAINING_PATIENCE
EOF


stage_topology() {
    local topology=\
"hv_double_line_90kv"

    local marker=\
"$WINDOWS_TMP_PATH/.staged_${topology}"

    local pattern=\
"*${topology}_W${WINDOW_TAG}_S${STEP_TAG}*"

    if [ -f "$marker" ]; then
        echo "Waveform data already staged."
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
    echo "Staging 90 kV waveform files"
    echo "Pattern: $pattern"
    echo "Count:   $count"

    if [ "$count" -eq 0 ]; then
        echo "ERROR: No 90 kV waveform files matched:"
        echo "$pattern"
        exit 1
    fi

    find "$SOURCE_WINDOWS_DIR" \
        -maxdepth 1 \
        -type f \
        -name "$pattern" \
        -exec cp \
          -t "$WINDOWS_TMP_PATH" \
          {} +

    local raw_file=\
"$WINDOWS_TMP_PATH/X_${topology}_W${WINDOW_TAG}_S${STEP_TAG}.raw"

    if [ ! -f "$raw_file" ]; then
        echo "ERROR: Staged raw waveform is missing:"
        echo "$raw_file"
        exit 1
    fi

    touch "$marker"

    ls -lh "$raw_file"
}


stage_topology

nvidia-smi


FAIL_COUNT=0


run_experiment() {
    local experiment_id="$1"
    local prior_path="$2"
    local prior_column="$3"
    local operator_feature_cols="$4"
    local model_mode="$5"
    local hidden_size="$6"
    local num_layers="$7"
    local dropout="$8"

    local experiment_output=\
"$RUN_ROOT/$experiment_id"

    local checkpoint_dir=\
"$experiment_output/checkpoints"

    local log_file=\
"$LOG_DIR/${experiment_id}.log"

    local start_time
    local end_time
    local return_code
    local status

    mkdir -p \
      "$checkpoint_dir"

    start_time="$(
        date --iso-8601=seconds
    )"

    local command=(
        python
        -u
        KOL/run_kol_experiment.py

        dataset=hv_double_line_90kv

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
        "+training.operator_feature_cols=$operator_feature_cols"

        "+training.kol_model_mode=$model_mode"
        "+training.kol_prediction_mode=$PREDICTION_MODE"
        +training.kol_window_mode=all_fault_start

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

    echo "Prior:"
    echo "$prior_path"

    echo
    echo "Prior column:"
    echo "$prior_column"

    echo
    echo "Operator features:"
    echo "$operator_feature_cols"

    echo
    echo "Model mode:"
    echo "$model_mode"

    echo
    echo "Window mode:"
    echo "all_fault_start"

    echo
    echo "Output:"
    echo "$experiment_output"

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

        FAIL_COUNT=$(
            (
                FAIL_COUNT + 1
            )
        )
    fi

    end_time="$(
        date --iso-8601=seconds
    )"

    printf \
        '%s,%s,%s,%s,%s,%s,%s,"%s",%s,%s,%s,%s\n' \
        "$experiment_id" \
        "$status" \
        "$return_code" \
        "$start_time" \
        "$end_time" \
        "$prior_path" \
        "$prior_column" \
        "$operator_feature_cols" \
        "$model_mode" \
        "all_fault_start" \
        "$experiment_output" \
        "$log_file" \
        >> "$STATUS_CSV"

    echo
    echo "$experiment_id finished with status:"
    echo "$status"

    grep -E \
        "KOL window mode|KOL model mode|KOL prediction mode|Selected operator feature columns|split sizes|Prior-only comparison|Final model evaluation|CV aggregate metrics|test/mae|test/rmse|test/prior_mae|test/prior_rmse|test/improvement_rate|test/worsened_rate" \
        "$log_file" \
        || true
}


# ------------------------------------------------------------------
# One-ended, case-best-MAE temporary prior
# ------------------------------------------------------------------

run_experiment \
    C90-1E-AFS-TMP \
    "$P90_1E_AFS_PRIOR" \
    "$P90_1E_AFS_PRIOR_COL" \
    "$P90_1E_AFS_FEATURE_COLS" \
    legacy_residual \
    384 \
    3 \
    0.2


run_experiment \
    L90-1E-AFS-TMP \
    "$P90_1E_AFS_PRIOR" \
    "$P90_1E_AFS_PRIOR_COL" \
    "$P90_1E_AFS_FEATURE_COLS" \
    bounded_residual_fusion \
    64 \
    1 \
    0.0


# ------------------------------------------------------------------
# Synchronized two-ended positive-sequence temporary prior
# ------------------------------------------------------------------

run_experiment \
    C90-2E-AFS-TMP \
    "$P90_2E_AFS_PRIOR" \
    "$P90_2E_AFS_PRIOR_COL" \
    "$P90_2E_AFS_FEATURE_COLS" \
    legacy_residual \
    384 \
    3 \
    0.2


run_experiment \
    L90-2E-AFS-TMP \
    "$P90_2E_AFS_PRIOR" \
    "$P90_2E_AFS_PRIOR_COL" \
    "$P90_2E_AFS_FEATURE_COLS" \
    bounded_residual_fusion \
    64 \
    1 \
    0.0


echo
echo "============================================================"
echo "TEMPORARY 90 kV ALL-FAULT-START RUN FINISHED"
echo "============================================================"

echo "Run root:"
echo "$RUN_ROOT"

echo
echo "Status:"
echo "$STATUS_CSV"

echo
echo "Failures:"
echo "$FAIL_COUNT"

echo "============================================================"


if command -v column >/dev/null 2>&1
then
    column \
      -s, \
      -t \
      "$STATUS_CSV"
else
    cat "$STATUS_CSV"
fi


if [ "$FAIL_COUNT" -gt 0 ]; then
    exit 1
fi
