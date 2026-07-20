#!/bin/bash -l
#SBATCH --job-name=ch4_fresh_priors
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=./hpc/hpc_logs/%x/%x-%j-on-%N.out

set -euo pipefail

# =============================================================================
# BUILD CUSTOM BEST-MAE SINGLE-ENDED PRIORS
#
# Generates for each topology:
#
#   1. case-aware:
#      fault case -> lowest-MAE eligible analytical operator
#
#   2. line-and-case-aware:
#      faulted line + fault case -> lowest-MAE eligible operator
#
# The final model-input CSV contains no target, case, line or selected-operator
# label. Those details are saved only in separate audit/diagnostic files.
#
# IMPORTANT:
# Operator selection is performed using the complete frozen cohort and is
# therefore explicitly documented as full-cohort outcome selection.
# =============================================================================

source \
  /home/hpc/iwi5/iwi5305h/miniconda3/etc/profile.d/conda.sh

conda activate Masters_thesis_env_py312

export THESIS_DIR="/home/hpc/iwi5/iwi5305h/Masters_thesis_PR_LABS"

export FRESH_PRIOR_ROOT="$THESIS_DIR/outputs/chapter4/fresh_prior_exports"

export LATEST_FRESH_PRIOR_POINTER="$FRESH_PRIOR_ROOT/LATEST_FRESH_PRIOR_RUN.txt"

export PRIOR_RUN_DIR="${PRIOR_RUN_DIR:-}"

# case or line_case
export DEFAULT_MAPPING_MODE="${DEFAULT_MAPPING_MODE:-line_case}"

if [ "$DEFAULT_MAPPING_MODE" != "case" ] \
   && [ "$DEFAULT_MAPPING_MODE" != "line_case" ]
then
    echo "ERROR: DEFAULT_MAPPING_MODE must be case or line_case"
    exit 1
fi

if [ -z "$PRIOR_RUN_DIR" ]; then
    if [ ! -f "$LATEST_FRESH_PRIOR_POINTER" ]; then
        echo "ERROR: Latest fresh-prior pointer is missing:"
        echo "$LATEST_FRESH_PRIOR_POINTER"
        exit 1
    fi

    PRIOR_RUN_DIR="$(
      head -n 1 "$LATEST_FRESH_PRIOR_POINTER"
    )"

    export PRIOR_RUN_DIR
fi

if [ ! -d "$PRIOR_RUN_DIR" ]; then
    echo "ERROR: Fresh prior run directory does not exist:"
    echo "$PRIOR_RUN_DIR"
    exit 1
fi

RUN_TIMESTAMP="$(date +%Y%m%d_%H%M%S)"

export OUTPUT_BASE="$THESIS_DIR/outputs/chapter4/model_inputs/caseaware_bestmae"

export OUTPUT_DIR="$OUTPUT_BASE/$RUN_TIMESTAMP"

if [ -e "$OUTPUT_DIR" ]; then
    echo "ERROR: Refusing to overwrite:"
    echo "$OUTPUT_DIR"
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

echo "============================================================"
echo "Building custom case-aware single-ended priors"
echo "============================================================"
echo "Fresh prior run:     $PRIOR_RUN_DIR"
echo "Default mapping:     $DEFAULT_MAPPING_MODE"
echo "Output directory:    $OUTPUT_DIR"
echo "============================================================"

echo
echo "Fresh 90 kV exports:"

find \
  "$PRIOR_RUN_DIR/raw_exports/hv_double_line_90kv" \
  -maxdepth 1 \
  -type f \
  -name "*.csv" \
  -printf "%f\n" \
  | sort

echo
echo "Fresh 110 kV exports:"

find \
  "$PRIOR_RUN_DIR/raw_exports/hv_double_line_110kv" \
  -maxdepth 1 \
  -type f \
  -name "*.csv" \
  -printf "%f\n" \
  | sort


python - <<'PY'
from __future__ import annotations

import hashlib
import json
import os
import shlex
from pathlib import Path

import numpy as np
import pandas as pd


prior_run_dir = Path(
    os.environ["PRIOR_RUN_DIR"]
)

output_dir = Path(
    os.environ["OUTPUT_DIR"]
)

output_base = Path(
    os.environ["OUTPUT_BASE"]
)

default_mapping_mode = os.environ[
    "DEFAULT_MAPPING_MODE"
]


# =============================================================================
# Frozen specifications
# =============================================================================

SPECS = {
    "90kV": {
        "topology": "hv_double_line_90kv",
        "source_dir": (
            prior_run_dir
            / "raw_exports"
            / "hv_double_line_90kv"
        ),
        "expected_rows": 9022,
        "expected_events": 9022,
        "windows_per_event": 1,
        "window_indices": None,
        "window_mode": "single_fault_start",
    },
    "110kV": {
        "topology": "hv_double_line_110kv",
        "source_dir": (
            prior_run_dir
            / "raw_exports"
            / "hv_double_line_110kv"
        ),
        "expected_rows": 3648,
        "expected_events": 912,
        "windows_per_event": 4,
        "window_indices": {
            8,
            9,
            10,
            11,
        },
        "window_mode": "all_fault_start",
    },
}


TARGET_ALIASES = [
    "y_fault_location",
    "y_true",
    "fault_location",
]

CASE_ALIASES = [
    "case",
    "fault_case",
    "sc_type",
    "y_fault_case",
    "fault_type",
]

LINE_ALIASES = [
    "y_fault_line",
    "fault_line",
    "line",
    "line_name",
]


# Candidate columns must represent distance estimates in percentage points.
EXCLUDED_DISTANCE_TOKENS = {
    "error",
    "residual",
    "target",
    "true",
    "reason",
    "valid",
    "fallback",
    "flag",
    "diff",
    "ratio",
    "weight",
    "confidence",
    "score",
    "improvement",

    # Exclude explicitly synchronized/fused two-ended operators from 1E.
    "two_ended",
    "twoended",
    "posseq_plus",
    "both_mean",
    "both_weighted",
    "fusion",
}


FEATURES_110 = [
    "d_both_diff_real_pct",

    "ratio_V0_V1_local",
    "ratio_V2_V1_local",
    "ratio_I0_I1_local",
    "ratio_I2_I1_local",
    "abs_Z0_app_local",
    "abs_Z2_app_local",

    "ratio_V0_V1_remote",
    "ratio_V2_V1_remote",
    "ratio_I0_I1_remote",
    "ratio_I2_I1_remote",
    "abs_Z0_app_remote",
    "abs_Z2_app_remote",
]


FEATURE_ALIASES_90 = {
    "ratio_V0_V1": [
        "ratio_V0_V1",
        "ratio_V0_V1_local",
    ],
    "ratio_V2_V1": [
        "ratio_V2_V1",
        "ratio_V2_V1_local",
    ],
    "ratio_I0_I1": [
        "ratio_I0_I1",
        "ratio_I0_I1_local",
    ],
    "ratio_I2_I1": [
        "ratio_I2_I1",
        "ratio_I2_I1_local",
    ],
    "abs_Z0_app": [
        "abs_Z0_app",
        "abs_Z0_app_local",
    ],
    "abs_Z2_app": [
        "abs_Z2_app",
        "abs_Z2_app_local",
    ],
}


# =============================================================================
# Helpers
# =============================================================================

def first_existing(
    dataframe: pd.DataFrame,
    aliases: list[str],
    label: str,
) -> str:
    for column in aliases:
        if column in dataframe.columns:
            return column

    raise RuntimeError(
        f"Could not find {label}. Tried:\n"
        + "\n".join(
            f" - {column}"
            for column in aliases
        )
    )


def canonicalize(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    output = dataframe.copy()

    output["sample_id"] = (
        output["sample_id"]
        .astype(str)
        .str.strip()
    )

    output["window_idx"] = pd.to_numeric(
        output["window_idx"],
        errors="raise",
    ).astype(int)

    return (
        output.sort_values(
            [
                "sample_id",
                "window_idx",
            ],
            kind="stable",
        )
        .reset_index(drop=True)
    )


def validate_cohort(
    dataframe: pd.DataFrame,
    spec: dict,
    label: str,
) -> pd.DataFrame:
    required = {
        "sample_id",
        "window_idx",
    }

    missing = sorted(
        required.difference(
            dataframe.columns
        )
    )

    if missing:
        raise RuntimeError(
            f"{label}: missing key columns {missing}"
        )

    output = canonicalize(
        dataframe
    )

    if len(output) != spec[
        "expected_rows"
    ]:
        raise RuntimeError(
            f"{label}: expected "
            f"{spec['expected_rows']} rows, "
            f"found {len(output)}"
        )

    duplicate_count = int(
        output.duplicated(
            [
                "sample_id",
                "window_idx",
            ]
        ).sum()
    )

    if duplicate_count:
        raise RuntimeError(
            f"{label}: found "
            f"{duplicate_count} duplicate keys"
        )

    event_counts = (
        output.groupby("sample_id")
        .size()
    )

    if len(event_counts) != spec[
        "expected_events"
    ]:
        raise RuntimeError(
            f"{label}: expected "
            f"{spec['expected_events']} events, "
            f"found {len(event_counts)}"
        )

    observed_windows = sorted(
        event_counts.unique().tolist()
    )

    if observed_windows != [
        spec["windows_per_event"]
    ]:
        raise RuntimeError(
            f"{label}: expected "
            f"{spec['windows_per_event']} "
            f"windows/event, found "
            f"{observed_windows}"
        )

    if spec["window_indices"] is not None:
        observed_indices = set(
            output["window_idx"].unique()
        )

        if observed_indices != spec[
            "window_indices"
        ]:
            raise RuntimeError(
                f"{label}: expected window_idx "
                f"{sorted(spec['window_indices'])}, "
                f"found {sorted(observed_indices)}"
            )

    return output


def target_to_pp(
    values: pd.Series,
) -> np.ndarray:
    numeric = pd.to_numeric(
        values,
        errors="coerce",
    ).to_numpy(
        dtype=np.float64
    )

    finite = numeric[
        np.isfinite(numeric)
    ]

    if len(finite) == 0:
        raise RuntimeError(
            "No finite targets were found."
        )

    if np.max(
        np.abs(finite)
    ) <= 1.5:
        numeric = numeric * 100.0

    return numeric


def transform_prior(
    raw_values: np.ndarray,
) -> tuple[
    np.ndarray,
    np.ndarray,
]:
    raw_values = np.asarray(
        raw_values,
        dtype=np.float64,
    )

    finite = np.isfinite(
        raw_values
    )

    model_values = np.full(
        len(raw_values),
        50.0,
        dtype=np.float64,
    )

    model_values[finite] = np.clip(
        raw_values[finite],
        0.0,
        100.0,
    )

    fallback = ~finite

    return (
        model_values,
        fallback,
    )


def metric_record(
    target_pp: np.ndarray,
    raw_values: np.ndarray,
) -> dict:
    model_values, fallback = transform_prior(
        raw_values
    )

    errors = np.abs(
        model_values - target_pp
    )

    return {
        "mae_pp": float(
            np.mean(errors)
        ),
        "rmse_pp": float(
            np.sqrt(
                np.mean(
                    (
                        model_values
                        - target_pp
                    )
                    ** 2
                )
            )
        ),
        "coverage": float(
            np.mean(
                np.isfinite(
                    raw_values
                )
            )
        ),
        "fallback_count": int(
            fallback.sum()
        ),
        "model_values": model_values,
        "fallback": fallback,
    }


def candidate_distance_column(
    column: str,
    source_path: Path,
) -> bool:
    name = column.lower()

    if not name.startswith("d_"):
        return False

    if "pct" not in name:
        return False

    if any(
        token in name
        for token in EXCLUDED_DISTANCE_TOKENS
    ):
        return False

    # In a both-side export d_phys_real_pct may be the fused result rather
    # than a true one-terminal estimate.
    if (
        name == "d_phys_real_pct"
        and "_both_" in source_path.name.lower()
    ):
        return False

    return True


def hydra_list(
    columns: list[str],
) -> str:
    return (
        "["
        + ",".join(columns)
        + "]"
    )


def vector_hash(
    values: np.ndarray,
) -> str:
    normalized = np.nan_to_num(
        np.asarray(
            values,
            dtype=np.float64,
        ),
        nan=1e30,
        posinf=1e31,
        neginf=-1e31,
    )

    return hashlib.sha256(
        normalized.tobytes()
    ).hexdigest()


# =============================================================================
# Process one topology
# =============================================================================

def process_topology(
    topology_label: str,
    spec: dict,
) -> dict:
    source_paths = sorted(
        spec["source_dir"].glob(
            "*.csv"
        )
    )

    if not source_paths:
        raise RuntimeError(
            f"{topology_label}: no fresh CSV files found in:\n"
            f"{spec['source_dir']}"
        )

    valid_sources = []

    for path in source_paths:
        try:
            dataframe = pd.read_csv(
                path
            )

            dataframe = validate_cohort(
                dataframe,
                spec,
                path.name,
            )

            valid_sources.append(
                (
                    path,
                    dataframe,
                )
            )

        except Exception as error:
            print(
                f"Rejected {path.name}: {error}"
            )

    if not valid_sources:
        raise RuntimeError(
            f"{topology_label}: no valid fresh "
            "source files passed the cohort audit"
        )

    # Choose a metadata-rich source containing target, case and line.
    reference_candidates = []

    for path, dataframe in valid_sources:
        has_target = any(
            column in dataframe.columns
            for column in TARGET_ALIASES
        )

        has_case = any(
            column in dataframe.columns
            for column in CASE_ALIASES
        )

        has_line = any(
            column in dataframe.columns
            for column in LINE_ALIASES
        )

        if has_target and has_case and has_line:
            reference_candidates.append(
                (
                    len(dataframe.columns),
                    path,
                    dataframe,
                )
            )

    if not reference_candidates:
        raise RuntimeError(
            f"{topology_label}: no source contains "
            "target, fault case and fault line metadata"
        )

    (
        _,
        reference_path,
        reference,
    ) = max(
        reference_candidates,
        key=lambda row: row[0],
    )

    target_column = first_existing(
        reference,
        TARGET_ALIASES,
        "target column",
    )

    case_column = first_existing(
        reference,
        CASE_ALIASES,
        "fault-case column",
    )

    line_column = first_existing(
        reference,
        LINE_ALIASES,
        "fault-line column",
    )

    target_pp = target_to_pp(
        reference[target_column]
    )

    reference_keys = reference[
        [
            "sample_id",
            "window_idx",
        ]
    ].copy()

    candidates = {}
    seen_vectors = {}

    for path, dataframe in valid_sources:
        keys = dataframe[
            [
                "sample_id",
                "window_idx",
            ]
        ]

        if not keys.equals(
            reference_keys
        ):
            print(
                f"Rejected candidate file due to key mismatch: "
                f"{path.name}"
            )
            continue

        for column in dataframe.columns:
            if not candidate_distance_column(
                column,
                path,
            ):
                continue

            raw_values = pd.to_numeric(
                dataframe[column],
                errors="coerce",
            ).to_numpy(
                dtype=np.float64
            )

            fingerprint = vector_hash(
                raw_values
            )

            # Avoid ranking identical copies multiple times.
            if fingerprint in seen_vectors:
                continue

            label = (
                f"{path.name}::{column}"
            )

            candidates[label] = {
                "label": label,
                "path": str(path),
                "source_file": path.name,
                "column": column,
                "raw_values": raw_values,
            }

            seen_vectors[fingerprint] = label

    if not candidates:
        raise RuntimeError(
            f"{topology_label}: no eligible "
            "single-ended distance columns were found"
        )

    # Resolve approved operator features.
    if topology_label == "90kV":
        feature_output = {}

        for output_name, aliases in (
            FEATURE_ALIASES_90.items()
        ):
            resolved = None

            for _, dataframe in valid_sources:
                for alias in aliases:
                    if alias in dataframe.columns:
                        values = pd.to_numeric(
                            dataframe[alias],
                            errors="coerce",
                        ).to_numpy(
                            dtype=np.float64
                        )

                        if np.isfinite(
                            values
                        ).all():
                            resolved = values
                            break

                if resolved is not None:
                    break

            if resolved is None:
                raise RuntimeError(
                    f"90kV: could not resolve "
                    f"operator feature {output_name}"
                )

            feature_output[
                output_name
            ] = resolved

        operator_feature_columns = list(
            feature_output.keys()
        )

    else:
        feature_output = {}

        for column in FEATURES_110:
            resolved = None

            for _, dataframe in valid_sources:
                if column in dataframe.columns:
                    values = pd.to_numeric(
                        dataframe[column],
                        errors="coerce",
                    ).to_numpy(
                        dtype=np.float64
                    )

                    if np.isfinite(
                        values
                    ).all():
                        resolved = values
                        break

            if resolved is None:
                raise RuntimeError(
                    f"110kV: required operator "
                    f"feature is missing or non-finite: "
                    f"{column}"
                )

            feature_output[
                column
            ] = resolved

        operator_feature_columns = (
            FEATURES_110
        )

    result = {
        "reference_path": str(
            reference_path
        ),
        "target_column": (
            target_column
        ),
        "case_column": (
            case_column
        ),
        "line_column": (
            line_column
        ),
        "candidate_count": int(
            len(candidates)
        ),
        "mappings": {},
    }

    for mapping_mode in [
        "case",
        "line_case",
    ]:
        if mapping_mode == "case":
            group_columns = [
                case_column,
            ]
        else:
            group_columns = [
                line_column,
                case_column,
            ]

        selected_model_values = np.full(
            len(reference),
            np.nan,
            dtype=np.float64,
        )

        selected_raw_values = np.full(
            len(reference),
            np.nan,
            dtype=np.float64,
        )

        selected_fallback = np.zeros(
            len(reference),
            dtype=np.int8,
        )

        selected_labels = np.empty(
            len(reference),
            dtype=object,
        )

        mapping_rows = []

        grouped_indices = (
            reference.groupby(
                group_columns,
                dropna=False,
                sort=True,
            )
            .indices
        )

        for group_key, indices in (
            grouped_indices.items()
        ):
            indices = np.asarray(
                indices,
                dtype=int,
            )

            if not isinstance(
                group_key,
                tuple,
            ):
                group_key = (
                    group_key,
                )

            target_group = target_pp[
                indices
            ]

            rankings = []

            for label, candidate in (
                candidates.items()
            ):
                raw_group = candidate[
                    "raw_values"
                ][indices]

                metrics = metric_record(
                    target_group,
                    raw_group,
                )

                rankings.append(
                    {
                        "candidate": label,
                        "source_file": (
                            candidate[
                                "source_file"
                            ]
                        ),
                        "column": (
                            candidate[
                                "column"
                            ]
                        ),
                        "mae_pp": (
                            metrics[
                                "mae_pp"
                            ]
                        ),
                        "rmse_pp": (
                            metrics[
                                "rmse_pp"
                            ]
                        ),
                        "coverage": (
                            metrics[
                                "coverage"
                            ]
                        ),
                        "fallback_count": (
                            metrics[
                                "fallback_count"
                            ]
                        ),
                    }
                )

            ranking_dataframe = (
                pd.DataFrame(
                    rankings
                )
                .sort_values(
                    [
                        "mae_pp",
                        "fallback_count",
                        "candidate",
                    ],
                    kind="stable",
                )
                .reset_index(drop=True)
            )

            best = ranking_dataframe.iloc[
                0
            ]

            best_label = best[
                "candidate"
            ]

            best_candidate = candidates[
                best_label
            ]

            best_raw = best_candidate[
                "raw_values"
            ][indices]

            (
                best_model,
                best_fallback,
            ) = transform_prior(
                best_raw
            )

            selected_raw_values[
                indices
            ] = best_raw

            selected_model_values[
                indices
            ] = best_model

            selected_fallback[
                indices
            ] = best_fallback.astype(
                np.int8
            )

            selected_labels[
                indices
            ] = best_label

            identity = {
                column: value
                for column, value in zip(
                    group_columns,
                    group_key,
                )
            }

            mapping_rows.append(
                {
                    **identity,
                    "rows": int(
                        len(indices)
                    ),
                    "selected_candidate": (
                        best_label
                    ),
                    "selected_source_file": (
                        best[
                            "source_file"
                        ]
                    ),
                    "selected_column": (
                        best[
                            "column"
                        ]
                    ),
                    "selected_mae_pp": float(
                        best[
                            "mae_pp"
                        ]
                    ),
                    "selected_rmse_pp": float(
                        best[
                            "rmse_pp"
                        ]
                    ),
                    "coverage": float(
                        best[
                            "coverage"
                        ]
                    ),
                    "fallback_count": int(
                        best[
                            "fallback_count"
                        ]
                    ),
                }
            )

        if not np.isfinite(
            selected_model_values
        ).all():
            raise RuntimeError(
                f"{topology_label}/{mapping_mode}: "
                "selected prior contains non-finite values"
            )

        overall_errors = np.abs(
            selected_model_values
            - target_pp
        )

        overall_mae = float(
            np.mean(
                overall_errors
            )
        )

        overall_rmse = float(
            np.sqrt(
                np.mean(
                    (
                        selected_model_values
                        - target_pp
                    )
                    ** 2
                )
            )
        )

        if topology_label == "90kV":
            prior_column = (
                f"d_90kv_{mapping_mode}_"
                "bestmae_input_pct"
            )
        else:
            prior_column = (
                f"d_110kv_{mapping_mode}_"
                "bestmae_input_pct"
            )

        model_input = reference[
            [
                "sample_id",
                "window_idx",
            ]
        ].copy()

        model_input[
            prior_column
        ] = selected_model_values.astype(
            np.float32
        )

        for column, values in (
            feature_output.items()
        ):
            model_input[column] = (
                values.astype(
                    np.float32
                )
            )

        if topology_label == "90kV":
            output_filename = (
                "kol_operator_features_"
                "hv_double_line_90kv_"
                f"{mapping_mode}_bestmae_"
                "fullcohort_single_fault_start_"
                "model_input.csv"
            )
        else:
            output_filename = (
                "kol_operator_features_"
                "hv_double_line_110kv_"
                f"{mapping_mode}_bestmae_"
                "fullcohort_all_fault_start_"
                "model_input.csv"
            )

        model_input_path = (
            output_dir
            / output_filename
        )

        model_input.to_csv(
            model_input_path,
            index=False,
        )

        mapping_dataframe = pd.DataFrame(
            mapping_rows
        )

        mapping_path = (
            output_dir
            / (
                f"{topology_label}_"
                f"{mapping_mode}_"
                "operator_mapping.csv"
            )
        )

        mapping_dataframe.to_csv(
            mapping_path,
            index=False,
        )

        diagnostic = reference[
            [
                "sample_id",
                "window_idx",
                line_column,
                case_column,
                target_column,
            ]
        ].copy()

        diagnostic[
            "y_true_pp"
        ] = target_pp

        diagnostic[
            "selected_candidate"
        ] = selected_labels

        diagnostic[
            "selected_raw_prior_pp"
        ] = selected_raw_values

        diagnostic[
            prior_column
        ] = selected_model_values

        diagnostic[
            "fallback_flag"
        ] = selected_fallback

        diagnostic[
            "absolute_error_pp"
        ] = overall_errors

        diagnostic_path = (
            output_dir
            / (
                f"{topology_label}_"
                f"{mapping_mode}_"
                "selection_diagnostics.csv.gz"
            )
        )

        diagnostic.to_csv(
            diagnostic_path,
            index=False,
            compression="gzip",
        )

        result[
            "mappings"
        ][mapping_mode] = {
            "model_input_path": str(
                model_input_path
            ),
            "prior_column": (
                prior_column
            ),
            "operator_feature_columns": (
                operator_feature_columns
            ),
            "mapping_path": str(
                mapping_path
            ),
            "diagnostic_path": str(
                diagnostic_path
            ),
            "rows": int(
                len(model_input)
            ),
            "events": int(
                model_input[
                    "sample_id"
                ].nunique()
            ),
            "duplicate_keys": int(
                model_input.duplicated(
                    [
                        "sample_id",
                        "window_idx",
                    ]
                ).sum()
            ),
            "overall_mae_pp": (
                overall_mae
            ),
            "overall_rmse_pp": (
                overall_rmse
            ),
            "fallback_count": int(
                selected_fallback.sum()
            ),
            "selection_scope": (
                "complete frozen cohort"
            ),
            "selection_status": (
                "full-cohort outcome-selected "
                "fault-type-conditioned prior"
            ),
        }

    return result


# =============================================================================
# Run both topologies
# =============================================================================

audit = {}

for topology_label, spec in (
    SPECS.items()
):
    print()
    print(
        "=" * 90
    )
    print(
        f"Processing {topology_label}"
    )
    print(
        "=" * 90
    )

    audit[
        topology_label
    ] = process_topology(
        topology_label,
        spec,
    )


# =============================================================================
# Save audit and environment files
# =============================================================================

audit_path = (
    output_dir
    / "caseaware_bestmae_audit.json"
)

audit_path.write_text(
    json.dumps(
        audit,
        indent=2,
    )
)


summary_rows = []

for topology_label, topology_audit in (
    audit.items()
):
    for mapping_mode, mapping in (
        topology_audit[
            "mappings"
        ].items()
    ):
        summary_rows.append(
            {
                "topology": (
                    topology_label
                ),
                "mapping_mode": (
                    mapping_mode
                ),
                "rows": (
                    mapping[
                        "rows"
                    ]
                ),
                "events": (
                    mapping[
                        "events"
                    ]
                ),
                "prior_column": (
                    mapping[
                        "prior_column"
                    ]
                ),
                "overall_mae_pp": (
                    mapping[
                        "overall_mae_pp"
                    ]
                ),
                "overall_rmse_pp": (
                    mapping[
                        "overall_rmse_pp"
                    ]
                ),
                "fallback_count": (
                    mapping[
                        "fallback_count"
                    ]
                ),
                "model_input_path": (
                    mapping[
                        "model_input_path"
                    ]
                ),
            }
        )


summary_dataframe = pd.DataFrame(
    summary_rows
)

summary_path = (
    output_dir
    / "caseaware_bestmae_summary.csv"
)

summary_dataframe.to_csv(
    summary_path,
    index=False,
)


selected_90 = audit[
    "90kV"
][
    "mappings"
][
    default_mapping_mode
]

selected_110 = audit[
    "110kV"
][
    "mappings"
][
    default_mapping_mode
]


environment_lines = [
    (
        "CASEAWARE_INPUT_DIR="
        + shlex.quote(
            str(output_dir)
        )
    ),
    (
        "CASEAWARE_MAPPING_MODE="
        + shlex.quote(
            default_mapping_mode
        )
    ),

    # Active paths consumed by the training script.
    (
        "P90_1E_PRIOR="
        + shlex.quote(
            selected_90[
                "model_input_path"
            ]
        )
    ),
    (
        "P90_1E_PRIOR_COL="
        + selected_90[
            "prior_column"
        ]
    ),
    (
        "P90_1E_OPERATOR_FEATURE_COLS="
        + shlex.quote(
            hydra_list(
                selected_90[
                    "operator_feature_columns"
                ]
            )
        )
    ),
    (
        "P110_1E_PRIOR="
        + shlex.quote(
            selected_110[
                "model_input_path"
            ]
        )
    ),
    (
        "P110_1E_PRIOR_COL="
        + selected_110[
            "prior_column"
        ]
    ),
    (
        "P110_1E_OPERATOR_FEATURE_COLS="
        + shlex.quote(
            hydra_list(
                selected_110[
                    "operator_feature_columns"
                ]
            )
        )
    ),

    # Explicit paths to both generated alternatives.
    (
        "P90_1E_CASE_PRIOR="
        + shlex.quote(
            audit[
                "90kV"
            ][
                "mappings"
            ][
                "case"
            ][
                "model_input_path"
            ]
        )
    ),
    (
        "P90_1E_LINE_CASE_PRIOR="
        + shlex.quote(
            audit[
                "90kV"
            ][
                "mappings"
            ][
                "line_case"
            ][
                "model_input_path"
            ]
        )
    ),
    (
        "P110_1E_CASE_PRIOR="
        + shlex.quote(
            audit[
                "110kV"
            ][
                "mappings"
            ][
                "case"
            ][
                "model_input_path"
            ]
        )
    ),
    (
        "P110_1E_LINE_CASE_PRIOR="
        + shlex.quote(
            audit[
                "110kV"
            ][
                "mappings"
            ][
                "line_case"
            ][
                "model_input_path"
            ]
        )
    ),
]

environment_text = (
    "\n".join(
        environment_lines
    )
    + "\n"
)


run_env_path = (
    output_dir
    / "caseaware_single_ended_inputs.env"
)

latest_env_path = (
    output_base
    / "LATEST_CASEAWARE_SINGLE_ENDED_INPUTS.env"
)

run_env_path.write_text(
    environment_text
)

latest_env_tmp = (
    output_base
    / ".LATEST_CASEAWARE_SINGLE_ENDED_INPUTS.env.tmp"
)

latest_env_tmp.write_text(
    environment_text
)

latest_env_tmp.replace(
    latest_env_path
)


latest_path = (
    output_base
    / "LATEST_CASEAWARE_SINGLE_ENDED_INPUT_DIR.txt"
)

latest_path_tmp = (
    output_base
    / ".LATEST_CASEAWARE_SINGLE_ENDED_INPUT_DIR.txt.tmp"
)

latest_path_tmp.write_text(
    str(output_dir)
    + "\n"
)

latest_path_tmp.replace(
    latest_path
)


print()
print(
    "=" * 100
)
print(
    "CUSTOM CASE-AWARE PRIOR GENERATION COMPLETED"
)
print(
    "=" * 100
)

print(
    summary_dataframe.to_string(
        index=False
    )
)

print()
print(
    "Active mapping mode:"
)
print(
    default_mapping_mode
)

print()
print(
    "Latest environment file:"
)
print(
    latest_env_path
)

print()
print(
    environment_text
)
PY


echo
echo "Generated files:"

find "$OUTPUT_DIR" \
  -maxdepth 1 \
  -type f \
  -printf "%f\t%s bytes\n" \
  | sort

echo
echo "Latest training environment:"

cat \
  "$OUTPUT_BASE/LATEST_CASEAWARE_SINGLE_ENDED_INPUTS.env"

echo
echo "Custom case-aware CSV generation completed."

