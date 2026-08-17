#!/bin/bash -l
#SBATCH --job-name=ch4_2e_raw_prior
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:v100:1
#SBATCH --partition=v100
#SBATCH --time=08:00:00
#SBATCH --output=./hpc/hpc_logs/ch4_2e_raw_prior-%j-on-%N.out
#SBATCH --error=./hpc/hpc_logs/ch4_2e_raw_prior-%j-on-%N.err

set -euo pipefail


# =============================================================================
# CHAPTER 4 — RAW/BASE TWO-ENDED PRIOR ABLATION
#
# Experiments:
#
#   C90-2E-RAW
#   C110-2E-RAW
#   L90-2E-RAW
#   L110-2E-RAW
#
# Scientific change:
#
#   Existing final runs:
#       d_two_ended_posseq_plus_input_pct
#       finite values clipped to [0, 100]
#       invalid values replaced by 50
#
#   This ablation:
#       d_two_ended_posseq_plus_pct
#       finite values remain completely unclipped
#       only non-finite values receive a 50 pp fallback
#
# No architecture, CV, window, seed, or training hyperparameter is changed.
#
# Smoke test:
#
#   sbatch \
#     --export=ALL,TRAINING_EPOCHS=1,TRAINING_PATIENCE=1 \
#     hpc/archive/historical_launchers/run_chapter4_double_ended_raw_prior.sh
#
# Production:
#
#   sbatch hpc/archive/historical_launchers/run_chapter4_double_ended_raw_prior.sh
# =============================================================================


# -----------------------------------------------------------------------------
# Training settings
# -----------------------------------------------------------------------------

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

export THIRD_PARTY_DIR=\
"$THESIS_DIR/third_party/dl_fault_repo"

export SOURCE_WINDOWS_DIR="${SOURCE_WINDOWS_DIR:-/home/vault/iwi5/iwi5305h/windows_tmp}"


export WINDOW_TAG="0p060"
export STEP_TAG="0p005"

export HYDRA_FULL_ERROR=1
export WANDB_MODE=disabled

export PYTHONPATH=\
"$THIRD_PARTY_DIR/src:$THESIS_DIR:${PYTHONPATH:-}"


: "${SLURM_JOB_ID:?SLURM_JOB_ID is missing. Submit using sbatch or run inside a Slurm allocation.}"

: "${TMPDIR:?TMPDIR is missing. Submit using sbatch or run inside a Slurm allocation.}"


cd "$THESIS_DIR"


# -----------------------------------------------------------------------------
# Exact raw/base two-ended CSV files
# -----------------------------------------------------------------------------

export RAW_EXPORT_RUN_DIR=\
"$THESIS_DIR/outputs/chapter4/fresh_prior_exports/"\
"12098749_20260717_230756"


export P90_2E_BASE_PRIOR=\
"$RAW_EXPORT_RUN_DIR/hv_double_line_90kv/"\
"kol_operator_features_hv_double_line_90kv_"\
"all_lines_both_single_fault_start.csv"


export P110_2E_BASE_PRIOR=\
"$RAW_EXPORT_RUN_DIR/hv_double_line_110kv/"\
"kol_operator_features_hv_double_line_110kv_"\
"all_lines_both_all_fault_start.csv"


export RAW_PRIOR_COLUMN=\
"d_two_ended_posseq_plus_pct"


# The same training-column name is used for both topologies.
#
# It contains:
#
#   - every finite raw value without clipping;
#   - 50 pp only where the raw estimate is non-finite.
#
export TRAINING_PRIOR_COLUMN=\
"d_two_ended_posseq_plus_raw_finite_pct"


for BASE_PRIOR_PATH in \
    "$P90_2E_BASE_PRIOR" \
    "$P110_2E_BASE_PRIOR"
do
    if [ ! -s "$BASE_PRIOR_PATH" ]; then
        echo "ERROR: Raw/base two-ended CSV is missing or empty:"
        echo "$BASE_PRIOR_PATH"
        exit 1
    fi
done


# -----------------------------------------------------------------------------
# Unique output and temporary paths
# -----------------------------------------------------------------------------

RUN_TIMESTAMP="$(date +%Y%m%d_%H%M%S)"


export RUN_ROOT=\
"$THESIS_DIR/outputs/chapter4/"\
"hybrid_double_ended_raw_prior/"\
"${SLURM_JOB_ID}_${RUN_TIMESTAMP}"


export LOG_DIR="$RUN_ROOT/logs"
export INPUT_VIEW_DIR="$RUN_ROOT/input_views"
export STATUS_CSV="$RUN_ROOT/experiment_status.csv"


export JOB_TMP_DIR="$TMPDIR/$SLURM_JOB_ID"

export WINDOWS_TMP_PATH=\
"$JOB_TMP_DIR/windows_tmp_chapter4"


if [ -e "$RUN_ROOT" ]; then
    echo "ERROR: Refusing to overwrite existing run directory:"
    echo "$RUN_ROOT"
    exit 1
fi


mkdir -p \
    "$RUN_ROOT" \
    "$LOG_DIR" \
    "$INPUT_VIEW_DIR" \
    "$WINDOWS_TMP_PATH"


# -----------------------------------------------------------------------------
# Compact training views
# -----------------------------------------------------------------------------

export P90_2E_PRIOR=\
"$INPUT_VIEW_DIR/"\
"kol_operator_features_hv_double_line_90kv_"\
"two_ended_raw_finite_single_fault_start_model_input.csv"


export P110_2E_PRIOR=\
"$INPUT_VIEW_DIR/"\
"kol_operator_features_hv_double_line_110kv_"\
"two_ended_raw_finite_all_fault_start_model_input.csv"


export P90_2E_PRIOR_COL="$TRAINING_PRIOR_COLUMN"
export P110_2E_PRIOR_COL="$TRAINING_PRIOR_COLUMN"


# -----------------------------------------------------------------------------
# Validate base CSVs and create raw-finite training views
# -----------------------------------------------------------------------------

python - <<'PY'
from __future__ import annotations

import json
import os

from pathlib import Path

import numpy as np
import pandas as pd


RAW_COLUMN = os.environ[
    "RAW_PRIOR_COLUMN"
]

TRAIN_COLUMN = os.environ[
    "TRAINING_PRIOR_COLUMN"
]


AUDIT_OUTPUT = (
    Path(os.environ["RUN_ROOT"])
    / "raw_prior_input_audit.json"
)


def canonicalize_sample_ids(
    values: pd.Series,
) -> pd.Series:
    def convert(value):
        if pd.isna(value):
            return "<missing>"

        text = str(value).strip()

        try:
            numeric = float(text)

            if (
                np.isfinite(numeric)
                and numeric.is_integer()
            ):
                return str(int(numeric))

        except Exception:
            pass

        return text

    return values.map(convert)


def prepare_training_view(
    *,
    base_path: Path,
    output_path: Path,
    topology: str,
    expected_rows: int,
    expected_events: int,
    expected_rows_per_event: int,
    expected_invalid: int,
    expected_below_zero: int,
    expected_above_hundred: int,
    expected_windows: set[int] | None,
) -> dict:
    print()
    print("=" * 80)
    print(f"RAW PRIOR AUDIT: {topology}")
    print("=" * 80)
    print("Base CSV:", base_path)

    frame = pd.read_csv(
        base_path
    )

    required_columns = {
        "sample_id",
        "window_idx",
        RAW_COLUMN,
    }

    missing_columns = sorted(
        required_columns
        - set(frame.columns)
    )

    if missing_columns:
        raise KeyError(
            f"{base_path}: missing columns "
            f"{missing_columns}"
        )

    if len(frame) != expected_rows:
        raise ValueError(
            f"{base_path}: expected "
            f"{expected_rows} rows, found "
            f"{len(frame)}"
        )

    frame["sample_id"] = (
        canonicalize_sample_ids(
            frame["sample_id"]
        )
    )

    event_count = frame[
        "sample_id"
    ].nunique(dropna=False)

    if event_count != expected_events:
        raise ValueError(
            f"{base_path}: expected "
            f"{expected_events} events, found "
            f"{event_count}"
        )

    if frame.duplicated(
        [
            "sample_id",
            "window_idx",
        ]
    ).any():
        raise ValueError(
            f"{base_path}: duplicate "
            "(sample_id, window_idx) keys"
        )

    rows_per_event = (
        frame.groupby(
            "sample_id",
            dropna=False,
        )
        .size()
    )

    if (
        rows_per_event.min()
        != expected_rows_per_event
        or rows_per_event.max()
        != expected_rows_per_event
    ):
        raise ValueError(
            f"{base_path}: expected "
            f"{expected_rows_per_event} rows "
            "per event; observed "
            f"min={rows_per_event.min()}, "
            f"max={rows_per_event.max()}"
        )

    if expected_windows is not None:
        observed_windows = set(
            pd.to_numeric(
                frame["window_idx"],
                errors="raise",
            ).astype(int)
        )

        if observed_windows != expected_windows:
            raise ValueError(
                f"{base_path}: expected windows "
                f"{sorted(expected_windows)}, "
                f"found "
                f"{sorted(observed_windows)}"
            )

    raw_prior = pd.to_numeric(
        frame[RAW_COLUMN],
        errors="coerce",
    )

    finite_mask = np.isfinite(
        raw_prior
    )

    invalid_count = int(
        (~finite_mask).sum()
    )

    below_zero_count = int(
        (
            raw_prior[finite_mask]
            < 0.0
        ).sum()
    )

    above_hundred_count = int(
        (
            raw_prior[finite_mask]
            > 100.0
        ).sum()
    )

    if invalid_count != expected_invalid:
        raise ValueError(
            f"{base_path}: expected "
            f"{expected_invalid} non-finite "
            f"values, found {invalid_count}"
        )

    if (
        below_zero_count
        != expected_below_zero
    ):
        raise ValueError(
            f"{base_path}: expected "
            f"{expected_below_zero} finite "
            "values below zero, found "
            f"{below_zero_count}"
        )

    if (
        above_hundred_count
        != expected_above_hundred
    ):
        raise ValueError(
            f"{base_path}: expected "
            f"{expected_above_hundred} finite "
            "values above 100, found "
            f"{above_hundred_count}"
        )

    # -------------------------------------------------------------
    # Preserve all finite raw values exactly.
    #
    # Replace only non-finite values with the neutral 50 pp fallback.
    #
    # No finite value is clipped.
    # -------------------------------------------------------------

    training_prior = (
        raw_prior.where(
            finite_mask,
            50.0,
        )
    )

    if not np.isfinite(
        training_prior
    ).all():
        raise ValueError(
            f"{base_path}: training prior "
            "still contains non-finite values"
        )

    finite_difference = (
        training_prior[finite_mask]
        - raw_prior[finite_mask]
    ).abs()

    maximum_finite_difference = (
        float(
            finite_difference.max()
        )
        if not finite_difference.empty
        else 0.0
    )

    if maximum_finite_difference != 0.0:
        raise ValueError(
            f"{base_path}: finite raw prior "
            "values were unexpectedly changed"
        )

    training_view = frame[
        [
            "sample_id",
            "window_idx",
        ]
    ].copy()

    training_view[
        TRAIN_COLUMN
    ] = training_prior.to_numpy(
        dtype=float
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    training_view.to_csv(
        output_path,
        index=False,
    )

    # Read it back and verify what training will receive.
    saved = pd.read_csv(
        output_path
    )

    saved_prior = pd.to_numeric(
        saved[TRAIN_COLUMN],
        errors="coerce",
    ).to_numpy(dtype=float)

    if len(saved) != expected_rows:
        raise ValueError(
            f"{output_path}: saved training "
            "view has the wrong row count"
        )

    if not np.isfinite(
        saved_prior
    ).all():
        raise ValueError(
            f"{output_path}: saved training "
            "view contains non-finite values"
        )

    print(f"Rows:                    {len(frame)}")
    print(f"Events:                  {event_count}")
    print(
        "Rows per event:          "
        f"{rows_per_event.min()}"
    )
    print(
        "Finite raw estimates:    "
        f"{int(finite_mask.sum())}"
    )
    print(
        "Non-finite raw values:   "
        f"{invalid_count}"
    )
    print(
        "Finite values below 0:   "
        f"{below_zero_count}"
    )
    print(
        "Finite values above 100: "
        f"{above_hundred_count}"
    )
    print(
        "Finite values clipped:   0"
    )
    print(
        "Fallback replacements:   "
        f"{invalid_count}"
    )
    print(
        "Maximum finite change:   "
        f"{maximum_finite_difference:.12g}"
    )
    print(
        "Raw finite range:        "
        f"[{raw_prior[finite_mask].min():.6f}, "
        f"{raw_prior[finite_mask].max():.6f}]"
    )
    print(
        "Training prior range:    "
        f"[{saved_prior.min():.6f}, "
        f"{saved_prior.max():.6f}]"
    )
    print("Training CSV:", output_path)

    return {
        "topology": topology,
        "base_csv": str(
            base_path
        ),
        "training_csv": str(
            output_path
        ),
        "raw_column": RAW_COLUMN,
        "training_column": (
            TRAIN_COLUMN
        ),
        "rows": int(
            len(frame)
        ),
        "events": int(
            event_count
        ),
        "rows_per_event": int(
            expected_rows_per_event
        ),
        "finite_raw_count": int(
            finite_mask.sum()
        ),
        "invalid_raw_count": int(
            invalid_count
        ),
        "below_zero_count": int(
            below_zero_count
        ),
        "above_hundred_count": int(
            above_hundred_count
        ),
        "finite_clipping_count": 0,
        "fallback_count": int(
            invalid_count
        ),
        "fallback_value_pp": 50.0,
        "maximum_finite_change_pp": (
            maximum_finite_difference
        ),
        "raw_finite_min_pp": float(
            raw_prior[
                finite_mask
            ].min()
        ),
        "raw_finite_max_pp": float(
            raw_prior[
                finite_mask
            ].max()
        ),
        "training_min_pp": float(
            saved_prior.min()
        ),
        "training_max_pp": float(
            saved_prior.max()
        ),
    }


audit_rows = []


audit_rows.append(
    prepare_training_view(
        base_path=Path(
            os.environ[
                "P90_2E_BASE_PRIOR"
            ]
        ),
        output_path=Path(
            os.environ[
                "P90_2E_PRIOR"
            ]
        ),
        topology=(
            "hv_double_line_90kv"
        ),
        expected_rows=9022,
        expected_events=9022,
        expected_rows_per_event=1,
        expected_invalid=8,
        expected_below_zero=45,
        expected_above_hundred=48,
        expected_windows=None,
    )
)


audit_rows.append(
    prepare_training_view(
        base_path=Path(
            os.environ[
                "P110_2E_BASE_PRIOR"
            ]
        ),
        output_path=Path(
            os.environ[
                "P110_2E_PRIOR"
            ]
        ),
        topology=(
            "hv_double_line_110kv"
        ),
        expected_rows=3648,
        expected_events=912,
        expected_rows_per_event=4,
        expected_invalid=0,
        expected_below_zero=100,
        expected_above_hundred=110,
        expected_windows={
            8,
            9,
            10,
            11,
        },
    )
)


payload = {
    "status": "PASS",
    "experiment_variant": (
        "raw_finite_two_ended_prior"
    ),
    "policy": (
        "Keep every finite raw synchronized "
        "two-ended estimate without clipping. "
        "Replace only non-finite estimates "
        "with 50 percentage points."
    ),
    "inputs": audit_rows,
}


with AUDIT_OUTPUT.open(
    "w",
    encoding="utf-8",
) as file:
    json.dump(
        payload,
        file,
        indent=2,
    )


print()
print("=" * 80)
print("RAW TWO-ENDED INPUT AUDIT PASSED")
print("=" * 80)
print("Audit record:", AUDIT_OUTPUT)
PY


# -----------------------------------------------------------------------------
# Source-code checks
# -----------------------------------------------------------------------------

if ! grep -q \
    'threeph_add_ground_mul' \
    KOL/models/kol_residual_models.py
then
    echo "ERROR: threeph_add_ground_mul is not present in current source."
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


echo
echo "Python compile check passed."


# -----------------------------------------------------------------------------
# Status and provenance files
# -----------------------------------------------------------------------------

printf \
'experiment_id,status,return_code,start_time,end_time,topology,prior_path,prior_column,operator_features,model_mode,prediction_mode,window_mode,output_dir,log_file\n' \
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

experiment_variant=raw_finite_two_ended_prior

raw_export_run_dir=$RAW_EXPORT_RUN_DIR
raw_prior_column=$RAW_PRIOR_COLUMN
training_prior_column=$TRAINING_PRIOR_COLUMN

p90_base_prior=$P90_2E_BASE_PRIOR
p90_training_prior=$P90_2E_PRIOR
p90_invalid_policy=fallback_50_pp
p90_finite_clipping=none

p110_base_prior=$P110_2E_BASE_PRIOR
p110_training_prior=$P110_2E_PRIOR
p110_invalid_policy=fail_if_nonfinite
p110_finite_clipping=none

operator_features=[]
prediction_mode=$PREDICTION_MODE

training_epochs=$TRAINING_EPOCHS
training_patience=$TRAINING_PATIENCE
learning_rate=0.0003
weight_decay=0.0001
n_splits=5
split_seed=42
training_seed=42
EOF


if git -C "$THESIS_DIR" rev-parse HEAD >/dev/null 2>&1
then
    git -C "$THESIS_DIR" rev-parse HEAD \
        > "$RUN_ROOT/git_commit.txt"

    git -C "$THESIS_DIR" status --short \
        > "$RUN_ROOT/git_status.txt"
fi


echo
echo "============================================================"
echo "CHAPTER 4 RAW TWO-ENDED PRIOR EXPERIMENTS"
echo "============================================================"

echo "Slurm job:       $SLURM_JOB_ID"
echo "Hostname:        $(hostname)"
echo "Run root:        $RUN_ROOT"
echo "Temporary path:  $WINDOWS_TMP_PATH"

echo
echo "90 kV base CSV:"
echo "$P90_2E_BASE_PRIOR"

echo
echo "90 kV training CSV:"
echo "$P90_2E_PRIOR"

echo
echo "110 kV base CSV:"
echo "$P110_2E_BASE_PRIOR"

echo
echo "110 kV training CSV:"
echo "$P110_2E_PRIOR"

echo
echo "Training prior column:"
echo "$TRAINING_PRIOR_COLUMN"

echo
echo "Finite-value clipping: none"
echo "90 kV invalid fallback: 50 pp"
echo "110 kV invalid fallback: none"

echo
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

    local marker=\
"$WINDOWS_TMP_PATH/.staged_${topology}"

    local pattern=\
"*${topology}_W${WINDOW_TAG}_S${STEP_TAG}*"

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

    local raw_file=\
"$WINDOWS_TMP_PATH/X_${topology}_W${WINDOW_TAG}_S${STEP_TAG}.raw"

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
# Sequential experiment runner
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

    local experiment_output=\
"$RUN_ROOT/$experiment_id"

    local checkpoint_dir=\
"$experiment_output/checkpoints"

    local log_file=\
"$LOG_DIR/${experiment_id}.log"

    if [ -e "$experiment_output" ]; then
        echo "ERROR: Refusing to overwrite:"
        echo "$experiment_output"

        FAIL_COUNT=$(
            (
                FAIL_COUNT + 1
            )
        )

        return
    fi

    mkdir -p "$checkpoint_dir"

    local start_time
    local end_time
    local return_code
    local status

    start_time="$(
        date --iso-8601=seconds
    )"

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
        '%s,%s,%s,%s,%s,%s,%s,%s,"%s",%s,%s,%s,%s,%s\n' \
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
        "$window_mode" \
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
# Direct residual-correction models
# -----------------------------------------------------------------------------

run_experiment \
    C90-2E-RAW \
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
    C110-2E-RAW \
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
    L90-2E-RAW \
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
    L110-2E-RAW \
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
echo "RAW TWO-ENDED RUN FINISHED"
echo "============================================================"

echo "End time: $(date)"
echo "Run root: $RUN_ROOT"
echo "Status:   $STATUS_CSV"
echo "Failures: $FAIL_COUNT"

echo "============================================================"


if command -v column >/dev/null 2>&1
then
    column -s, -t "$STATUS_CSV"
else
    cat "$STATUS_CSV"
fi


if [ "$FAIL_COUNT" -gt 0 ]; then
    exit 1
fi
