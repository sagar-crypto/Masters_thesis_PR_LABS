#!/bin/bash -l
#SBATCH --job-name=ch4_fresh_priors
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=./hpc/hpc_logs/tmp90_afs_physics-%j-on-%N.out
#SBATCH --error=./hpc/hpc_logs/tmp90_afs_physics-%j-on-%N.err

set -euo pipefail


# =============================================================================
# TEMPORARY 90 kV ALL-FAULT-START PHYSICS EXPORT
#
# Exports:
#
#   default
#   opposite
#   both
#   two_ended_posseq
#
# The normal repository exporter is used for:
#
#   default
#   opposite
#   both
#
# The synchronized two-ended rows are generated through a temporary inline
# Python block. No repository source file is modified.
# =============================================================================


# -----------------------------------------------------------------------------
# Environment
# -----------------------------------------------------------------------------

source \
  /home/hpc/iwi5/iwi5305h/miniconda3/etc/profile.d/conda.sh

conda activate Masters_thesis_env_py312


export THESIS_DIR="${THESIS_DIR:-/home/hpc/iwi5/iwi5305h/Masters_thesis_PR_LABS}"

export THIRD_PARTY_DIR=\
"$THESIS_DIR/third_party/dl_fault_repo"

export WINDOWS_DIR="${WINDOWS_DIR:-/home/vault/iwi5/iwi5305h/windows_tmp}"

export VAULT_ROOT="/home/vault/iwi5/iwi5305h"

export PYTHONPATH=\
"$THIRD_PARTY_DIR/src:$THESIS_DIR:${PYTHONPATH:-}"

export HYDRA_FULL_ERROR=1
export WANDB_MODE=disabled


: "${SLURM_JOB_ID:?Submit this script using sbatch.}"


cd "$THESIS_DIR"

mkdir -p hpc/hpc_logs


# -----------------------------------------------------------------------------
# Temporary output paths
# -----------------------------------------------------------------------------

RUN_TAG="${SLURM_JOB_ID}_$(date +%Y%m%d_%H%M%S)"

export TEMP_BASE=\
"$THESIS_DIR/outputs/chapter4/temp_90kv_afs_check"

export RAW_DIR=\
"$TEMP_BASE/raw_exports/$RUN_TAG"

export HYDRA_DIR=\
"$RAW_DIR/hydra"

export RAW_POINTER=\
"$TEMP_BASE/LATEST_RAW_DIR.txt"


mkdir -p \
  "$RAW_DIR" \
  "$HYDRA_DIR"


export TOPOLOGY="hv_double_line_90kv"
export WINDOW_MODE="all_fault_start"
export WINDOW_LENGTH="0.060"
export STEP_LENGTH="0.005"


echo
echo "============================================================"
echo "TEMPORARY 90 kV ALL-FAULT-START PHYSICS EXPORT"
echo "============================================================"

echo "Job ID:          $SLURM_JOB_ID"
echo "Topology:        $TOPOLOGY"
echo "Window mode:     $WINDOW_MODE"
echo "Waveform source: $WINDOWS_DIR"
echo "Temporary output:"
echo "$RAW_DIR"

echo "============================================================"


# -----------------------------------------------------------------------------
# Normal one-ended/both exporter
# -----------------------------------------------------------------------------

run_standard_export() {
    local side_mode="$1"

    local standardized_name=\
"kol_operator_features_${TOPOLOGY}_all_lines_${side_mode}_${WINDOW_MODE}.csv"

    local destination_path=\
"$RAW_DIR/$standardized_name"

    local mode_hydra_dir=\
"$HYDRA_DIR/$side_mode"

    local export_marker=\
"$RAW_DIR/.started_${side_mode}"

    mkdir -p "$mode_hydra_dir"

    rm -f "$export_marker"

    touch "$export_marker"

    # Shared filesystems may have coarse timestamp resolution.
    sleep 1

    echo
    echo "============================================================"
    echo "EXPORTING MODE: $side_mode"
    echo "============================================================"

    python -u KOL/run_kol_physics_baseline.py \
        dataset="$TOPOLOGY" \
        +training.operator_side_mode="$side_mode" \
        +training.operator_window_mode="$WINDOW_MODE" \
        'training.feature_groups_include=[lines,loads,winds,extgrid]' \
        window_extraction.window_length="$WINDOW_LENGTH" \
        window_extraction.step_length_seconds="$STEP_LENGTH" \
        window_extraction.windows_local_dir="$WINDOWS_DIR" \
        hydra.run.dir="$mode_hydra_dir"

    # -------------------------------------------------------------
    # Discover the filename written by the active local exporter.
    #
    # Your branch currently produces names such as:
    #
    # kol_operator_features_hv_double_line_90kv_
    # default_all_fault_start_mod_takagi_tf_only.csv
    # -------------------------------------------------------------

    local source_path

    source_path="$(
        find "$VAULT_ROOT" \
            -maxdepth 1 \
            -type f \
            -name \
"kol_operator_features_${TOPOLOGY}_*${side_mode}_${WINDOW_MODE}*.csv" \
            -newer "$export_marker" \
            -printf '%T@ %p\n' \
        | sort -nr \
        | head -n 1 \
        | cut -d' ' -f2-
    )"

    if [ -z "$source_path" ]; then
        echo
        echo "ERROR: Could not discover the newly generated export."
        echo
        echo "Search pattern:"
        echo \
"kol_operator_features_${TOPOLOGY}_*${side_mode}_${WINDOW_MODE}*.csv"

        echo
        echo "Existing matching files:"

        find "$VAULT_ROOT" \
            -maxdepth 1 \
            -type f \
            -name \
"kol_operator_features_${TOPOLOGY}_*${side_mode}_${WINDOW_MODE}*.csv" \
            -printf '%TY-%Tm-%Td %TH:%TM:%TS  %p\n' \
        | sort \
        || true

        exit 1
    fi

    if [ ! -s "$source_path" ]; then
        echo "ERROR: Discovered source export is empty:"
        echo "$source_path"
        exit 1
    fi

    echo
    echo "Discovered source:"
    echo "$source_path"

    cp \
      "$source_path" \
      "$destination_path"

    if [ ! -s "$destination_path" ]; then
        echo "ERROR: Standardized copy was not created:"
        echo "$destination_path"
        exit 1
    fi

    echo
    echo "Standardized temporary export:"
    echo "$destination_path"

    ls -lh "$destination_path"
}


run_standard_export default
run_standard_export opposite
run_standard_export both


# -----------------------------------------------------------------------------
# Temporary synchronized two-ended positive-sequence export
# -----------------------------------------------------------------------------
#
# The active local generic exporter does not dispatch the
# "two_ended_posseq" mode. We therefore call the already existing
# two-ended row builder directly.
#
# No source file in the repository is changed.
# -----------------------------------------------------------------------------

export TWO_ENDED_DESTINATION=\
"$RAW_DIR/kol_operator_features_${TOPOLOGY}_all_lines_two_ended_posseq_${WINDOW_MODE}.csv"


echo
echo "============================================================"
echo "EXPORTING MODE: two_ended_posseq"
echo "Temporary direct row-builder invocation"
echo "============================================================"


python -u <<'PY'
from __future__ import annotations

import os

from collections import Counter
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd

from hydra import compose, initialize_config_dir

from KOL.common.operator_data_prep import (
    apply_operator_window_selection,
    load_and_filter_operator_data,
)

try:
    from KOL.common.operator_row_builders import (
        build_two_ended_posseq_operator_row,
    )
except ImportError as error:
    raise ImportError(
        "The repository contains no importable "
        "build_two_ended_posseq_operator_row function. "
        "The temporary direct export cannot continue."
    ) from error


thesis_dir = Path(
    os.environ["THESIS_DIR"]
)

config_dir = (
    thesis_dir
    / "third_party"
    / "dl_fault_repo"
    / "config"
)

destination = Path(
    os.environ[
        "TWO_ENDED_DESTINATION"
    ]
)


overrides = [
    "dataset=hv_double_line_90kv",
    "+training.operator_side_mode=two_ended_posseq",
    "+training.operator_window_mode=all_fault_start",
    "training.feature_groups_include=[lines,loads,winds,extgrid]",
    (
        "window_extraction.window_length="
        + os.environ["WINDOW_LENGTH"]
    ),
    (
        "window_extraction.step_length_seconds="
        + os.environ["STEP_LENGTH"]
    ),
    (
        "window_extraction.windows_local_dir="
        + os.environ["WINDOWS_DIR"]
    ),
]


with initialize_config_dir(
    version_base=None,
    config_dir=str(config_dir),
):
    config = compose(
        config_name="main-config.yaml",
        overrides=overrides,
    )


(
    frame,
    X_eval,
    metadata,
    sampling_frequency,
    nominal_frequency,
    target_column,
    topology,
) = load_and_filter_operator_data(
    config
)


(
    frame,
    X_eval,
    selected_window_mode,
) = apply_operator_window_selection(
    df=frame,
    X_eval=X_eval,
    config=config,
    fs=sampling_frequency,
    f_nom=nominal_frequency,
)


if selected_window_mode != (
    "all_fault_start"
):
    raise RuntimeError(
        "Unexpected window mode: "
        f"{selected_window_mode}"
    )


feature_names = list(
    metadata["feature_names"]
)


rows = []
reason_counts: Counter = Counter()


for row_index in range(
    len(frame)
):
    row = cast(
        pd.Series,
        frame.iloc[row_index],
    )

    waveform = np.asarray(
        X_eval[row_index],
        dtype=np.float32,
    )

    output_row, reason = (
        build_two_ended_posseq_operator_row(
            row=row,
            x_raw_full=waveform,
            feature_names=feature_names,
            topology=topology,
            fs=sampling_frequency,
            f_nom=nominal_frequency,
            y_col=target_column,
        )
    )

    if output_row is None:
        reason_counts[
            str(reason)
        ] += 1
        continue

    rows.append(
        output_row
    )


output = pd.DataFrame(
    rows
)


print()
print(
    "===== Temporary synchronized "
    "two-ended export ====="
)

print(
    "Selected input rows:",
    len(frame),
)

print(
    "Rows exported:",
    len(output),
)


if reason_counts:
    print()
    print("Skipped rows:")

    for (
        reason,
        count,
    ) in reason_counts.most_common():
        print(
            f"{reason}: {count}"
        )


if output.empty:
    raise RuntimeError(
        "No synchronized two-ended "
        "operator rows were produced."
    )


required_columns = {
    "sample_id",
    "window_idx",
    "y_fault_location",
    "y_fault_line",
    "case",
    "d_two_ended_posseq_plus_pct",
}


missing = sorted(
    required_columns
    - set(output.columns)
)


if missing:
    raise KeyError(
        "Temporary synchronized "
        "two-ended export is missing "
        f"columns: {missing}"
    )


destination.parent.mkdir(
    parents=True,
    exist_ok=True,
)


output.to_csv(
    destination,
    index=False,
)


print()
print(
    "Saved synchronized two-ended "
    "temporary export:"
)

print(destination)


print()
print("Preview:")

preview_columns = [
    "sample_id",
    "window_idx",
    "y_fault_line",
    "y_fault_location",
    "case",
    "d_two_ended_posseq_plus_pct",
    "two_ended_posseq_plus_reason",
]

preview_columns = [
    column
    for column
    in preview_columns
    if column in output.columns
]

print(
    output[
        preview_columns
    ]
    .head(10)
    .to_string(index=False)
)
PY


if [ ! -s "$TWO_ENDED_DESTINATION" ]; then
    echo "ERROR: Two-ended temporary export is missing:"
    echo "$TWO_ENDED_DESTINATION"
    exit 1
fi


ls -lh "$TWO_ENDED_DESTINATION"


# -----------------------------------------------------------------------------
# Validate all four exports
# -----------------------------------------------------------------------------

export RAW_DIR


python - <<'PY'
from __future__ import annotations

import json
import os

from pathlib import Path

import numpy as np
import pandas as pd


raw_dir = Path(
    os.environ["RAW_DIR"]
)


expected_files = {
    "default": (
        raw_dir
        / (
            "kol_operator_features_"
            "hv_double_line_90kv_"
            "all_lines_default_"
            "all_fault_start.csv"
        )
    ),

    "opposite": (
        raw_dir
        / (
            "kol_operator_features_"
            "hv_double_line_90kv_"
            "all_lines_opposite_"
            "all_fault_start.csv"
        )
    ),

    "both": (
        raw_dir
        / (
            "kol_operator_features_"
            "hv_double_line_90kv_"
            "all_lines_both_"
            "all_fault_start.csv"
        )
    ),

    "two_ended_posseq": (
        raw_dir
        / (
            "kol_operator_features_"
            "hv_double_line_90kv_"
            "all_lines_two_ended_posseq_"
            "all_fault_start.csv"
        )
    ),
}


for mode, path in (
    expected_files.items()
):
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing {mode} export: "
            f"{path}"
        )


def canonical_sample_id(
    value,
):
    if pd.isna(value):
        return "<missing>"

    text = str(value).strip()

    try:
        number = float(text)

        if (
            np.isfinite(number)
            and number.is_integer()
        ):
            return str(
                int(number)
            )

    except Exception:
        pass

    return text


def canonicalize(
    frame: pd.DataFrame,
    path: Path,
) -> pd.DataFrame:
    required = {
        "sample_id",
        "window_idx",
        "y_fault_location",
        "y_fault_line",
        "case",
    }

    missing = sorted(
        required
        - set(frame.columns)
    )

    if missing:
        raise KeyError(
            f"{path}: missing columns "
            f"{missing}"
        )

    output = frame.copy()

    output["sample_id"] = (
        output["sample_id"]
        .map(
            canonical_sample_id
        )
    )

    output["window_idx"] = (
        pd.to_numeric(
            output["window_idx"],
            errors="raise",
        )
        .astype(int)
    )

    output = (
        output.sort_values(
            [
                "sample_id",
                "window_idx",
            ],
            kind="stable",
        )
        .reset_index(drop=True)
    )

    if output.duplicated(
        [
            "sample_id",
            "window_idx",
        ]
    ).any():
        raise ValueError(
            f"{path}: duplicate "
            "(sample_id, window_idx) keys"
        )

    return output


reference_keys = None
summary_rows = []


for mode, path in (
    expected_files.items()
):
    frame = canonicalize(
        pd.read_csv(path),
        path,
    )

    keys = frame[
        [
            "sample_id",
            "window_idx",
        ]
    ]

    if reference_keys is None:
        reference_keys = keys

    elif not keys.equals(
        reference_keys
    ):
        raise RuntimeError(
            "Export key mismatch for "
            f"mode {mode}"
        )

    event_sizes = (
        frame.groupby(
            "sample_id",
            dropna=False,
        )
        .size()
    )

    event_count = int(
        event_sizes.size
    )

    if event_count != 9022:
        raise RuntimeError(
            f"{path.name}: expected "
            f"9022 events, found "
            f"{event_count}"
        )

    if int(
        event_sizes.min()
    ) < 2:
        raise RuntimeError(
            f"{path.name}: expected "
            "multiple fault-start windows "
            "per event; minimum observed "
            f"was {int(event_sizes.min())}"
        )

    count_distribution = (
        event_sizes
        .value_counts()
        .sort_index()
    )

    summary_rows.append(
        {
            "mode": mode,
            "file": path.name,
            "rows": int(
                len(frame)
            ),
            "events": (
                event_count
            ),
            "minimum_windows_per_event": int(
                event_sizes.min()
            ),
            "maximum_windows_per_event": int(
                event_sizes.max()
            ),
            "window_count_distribution": {
                str(
                    int(window_count)
                ): int(event_total)
                for (
                    window_count,
                    event_total,
                ) in count_distribution.items()
            },
            "window_indices": sorted(
                frame[
                    "window_idx"
                ]
                .unique()
                .tolist()
            ),
            "columns": int(
                len(frame.columns)
            ),
        }
    )


row_counts = {
    record["rows"]
    for record in summary_rows
}


if len(row_counts) != 1:
    raise RuntimeError(
        "The four exports have "
        "different row counts"
    )


payload = {
    "status": "PASS",
    "topology": (
        "hv_double_line_90kv"
    ),
    "window_mode": (
        "all_fault_start"
    ),
    "rows": int(
        next(
            iter(row_counts)
        )
    ),
    "events": 9022,
    "exports": summary_rows,
}


audit_path = (
    raw_dir
    / "raw_export_audit.json"
)


audit_path.write_text(
    json.dumps(
        payload,
        indent=2,
    ),
    encoding="utf-8",
)


print()
print("=" * 80)
print(
    "90 kV ALL-FAULT-START "
    "PHYSICS EXPORT PASSED"
)
print("=" * 80)

print(
    json.dumps(
        payload,
        indent=2,
    )
)

print()
print("Audit:")
print(audit_path)
PY


# -----------------------------------------------------------------------------
# Save latest successful raw-export pointer
# -----------------------------------------------------------------------------

mkdir -p "$TEMP_BASE"


pointer_tmp=\
"$RAW_POINTER.tmp"


printf '%s\n' \
  "$RAW_DIR" \
  > "$pointer_tmp"


mv \
  "$pointer_tmp" \
  "$RAW_POINTER"


echo
echo "============================================================"
echo "TEMPORARY PHYSICS EXPORT COMPLETE"
echo "============================================================"

echo "Raw export directory:"
echo "$RAW_DIR"

echo
echo "Latest successful pointer:"
echo "$RAW_POINTER"

echo
echo "Next step:"
echo "bash temp_02_build_90kv_all_fault_start_inputs.sh"

echo "============================================================"
