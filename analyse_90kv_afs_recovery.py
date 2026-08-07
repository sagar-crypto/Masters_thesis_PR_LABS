from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


AFS_EXPERIMENTS = [
    "C90-1E-AFS-TMP",
    "L90-1E-AFS-TMP",
    "C90-2E-AFS-TMP",
    "L90-2E-AFS-TMP",
]

SFS_MAP = {
    "C90-1E-AFS-TMP": "C90-1E",
    "L90-1E-AFS-TMP": "L90-1E",
    "C90-2E-AFS-TMP": "C90-2E",
    "L90-2E-AFS-TMP": "L90-2E",
}

EXPECTED_ROWS = 40564
EXPECTED_EVENTS = 9022
EXPECTED_WINDOW_INDICES = {
    8,
    9,
    10,
    11,
    12,
}

TOL = 1e-10


def canonical_id(
    value: Any,
) -> str:
    if value is None or (
        isinstance(
            value,
            float,
        )
        and np.isnan(
            value
        )
    ):
        return "<missing>"

    text = str(
        value
    ).strip()

    try:
        number = float(
            text
        )

        if (
            np.isfinite(
                number
            )
            and number.is_integer()
        ):
            return str(
                int(
                    number
                )
            )

    except Exception:
        pass

    return text


def read_table(
    path: Path,
) -> pd.DataFrame:
    suffixes = "".join(
        path.suffixes
    ).lower()

    if suffixes.endswith(
        ".parquet"
    ):
        return pd.read_parquet(
            path
        )

    if suffixes.endswith(
        ".csv.gz"
    ):
        return pd.read_csv(
            path,
            compression="gzip",
        )

    if suffixes.endswith(
        ".csv"
    ):
        return pd.read_csv(
            path
        )

    raise ValueError(
        f"Unsupported table type: {path}"
    )


def write_csv(
    frame: pd.DataFrame,
    path: Path,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    frame.to_csv(
        path,
        index=False,
    )


def to_pp(
    values: (
        pd.Series
        | np.ndarray
    ),
) -> np.ndarray:
    array = pd.to_numeric(
        pd.Series(
            values
        ),
        errors="coerce",
    ).to_numpy(
        dtype=float
    )

    finite = array[
        np.isfinite(
            array
        )
    ]

    if (
        finite.size
        and float(
            np.nanmax(
                np.abs(
                    finite
                )
            )
        ) <= 1.5
    ):
        array = (
            array
            * 100.0
        )

    return array


def find_prediction_files(
    experiment_root: Path,
) -> list[Path]:
    groups: dict[
        Path,
        list[Path],
    ] = {}

    patterns = (
        "**/preds/fold*.parquet",
        "**/preds/fold*.csv.gz",
        "**/preds/fold*.csv",
    )

    for pattern in patterns:
        for path in experiment_root.glob(
            pattern
        ):
            groups.setdefault(
                path.parent,
                [],
            ).append(
                path
            )

    valid = []

    for (
        parent,
        files,
    ) in groups.items():
        unique_folds = set()

        for path in files:
            match = re.search(
                r"fold(\d+)",
                path.name,
            )

            if match:
                unique_folds.add(
                    int(
                        match.group(
                            1
                        )
                    )
                )

        if unique_folds == {
            0,
            1,
            2,
            3,
            4,
        }:
            valid.append(
                (
                    parent.stat().st_mtime,
                    parent,
                    sorted(
                        files
                    ),
                )
            )

    if not valid:
        raise FileNotFoundError(
            "No complete five-fold prediction "
            f"set under {experiment_root}"
        )

    valid.sort(
        key=lambda item: item[0]
    )

    return valid[-1][2]


def load_prediction_pool(
    experiment_root: Path,
    experiment: str,
) -> pd.DataFrame:
    files = find_prediction_files(
        experiment_root
    )

    frames = []

    for path in files:
        frame = read_table(
            path
        )

        match = re.search(
            r"fold(\d+)",
            path.name,
        )

        fold = (
            int(
                match.group(
                    1
                )
            )
            if match
            else None
        )

        if (
            "fold"
            not in frame.columns
            and fold is not None
        ):
            frame["fold"] = fold

        frame[
            "source_prediction_file"
        ] = str(
            path
        )

        frames.append(
            frame
        )

    pooled = pd.concat(
        frames,
        ignore_index=True,
    )

    pooled[
        "experiment"
    ] = experiment

    required = {
        "sample_id",
        "window_idx",
        "fold",
        "y_true",
        "y_pred",
    }

    missing = sorted(
        required
        - set(
            pooled.columns
        )
    )

    if missing:
        raise KeyError(
            f"{experiment}: recovered predictions "
            f"missing {missing}"
        )

    pooled[
        "sample_id"
    ] = pooled[
        "sample_id"
    ].map(
        canonical_id
    )

    pooled[
        "window_idx"
    ] = pd.to_numeric(
        pooled[
            "window_idx"
        ],
        errors="raise",
    ).astype(
        int
    )

    pooled[
        "fold"
    ] = pd.to_numeric(
        pooled[
            "fold"
        ],
        errors="raise",
    ).astype(
        int
    )

    pooled[
        "y_true_pp"
    ] = to_pp(
        pooled[
            "y_true"
        ]
    )

    pooled[
        "y_pred_pp"
    ] = to_pp(
        pooled[
            "y_pred"
        ]
    )

    prior_col = next(
        (
            column
            for column in [
                "d_prior",
                "d_phys_prior",
                "d_phys_real_pct",
                "prior",
            ]
            if column
            in pooled.columns
        ),
        None,
    )

    if prior_col is None:
        raise KeyError(
            f"{experiment}: no physical-prior "
            "column in recovered predictions"
        )

    pooled[
        "prior_pp"
    ] = to_pp(
        pooled[
            prior_col
        ]
    )

    if (
        "y_fault_location"
        not in pooled.columns
    ):
        pooled[
            "y_fault_location"
        ] = pooled[
            "y_true_pp"
        ]

    else:
        location = to_pp(
            pooled[
                "y_fault_location"
            ]
        )

        if np.nanmax(
            np.abs(
                location
            )
        ) > 1000:
            location = pd.to_numeric(
                pooled[
                    "y_fault_location"
                ],
                errors="coerce",
            ).to_numpy(
                dtype=float
            )

        pooled[
            "y_fault_location"
        ] = location

    if pooled.duplicated(
        [
            "sample_id",
            "window_idx",
        ]
    ).any():
        duplicates = int(
            pooled.duplicated(
                [
                    "sample_id",
                    "window_idx",
                ]
            ).sum()
        )

        raise RuntimeError(
            f"{experiment}: {duplicates} "
            "duplicate event/window rows"
        )

    return pooled


def merge_prior_audit(
    window: pd.DataFrame,
    row_audit_path: Path,
    prior_view: str,
) -> pd.DataFrame:
    output = window.copy()

    if not row_audit_path.exists():
        for column in [
            "prior_raw_pp",
            "prior_fallback_flag",
            "prior_clipped_low_flag",
            "prior_clipped_high_flag",
        ]:
            output[
                column
            ] = np.nan

        return output

    audit = pd.read_csv(
        row_audit_path
    )

    audit[
        "sample_id"
    ] = audit[
        "sample_id"
    ].map(
        canonical_id
    )

    audit[
        "window_idx"
    ] = pd.to_numeric(
        audit[
            "window_idx"
        ],
        errors="raise",
    ).astype(
        int
    )

    if prior_view == "1E":
        rename = {
            "selected_1e_raw_pct": (
                "prior_raw_pp"
            ),
            "selected_1e_input_pct": (
                "audit_prior_pp"
            ),
            "selected_1e_fallback": (
                "prior_fallback_flag"
            ),
            "selected_1e_candidate": (
                "selected_prior_candidate"
            ),
        }

    else:
        rename = {
            "raw_2e_pct": (
                "prior_raw_pp"
            ),
            "input_2e_pct": (
                "audit_prior_pp"
            ),
            "input_2e_fallback": (
                "prior_fallback_flag"
            ),
        }

    keep = [
        "sample_id",
        "window_idx",
    ] + [
        column
        for column
        in rename
        if column
        in audit.columns
    ]

    audit = audit[
        keep
    ].rename(
        columns=rename
    )

    output = output.merge(
        audit,
        on=[
            "sample_id",
            "window_idx",
        ],
        how="left",
        validate="one_to_one",
    )

    raw = pd.to_numeric(
        output.get(
            "prior_raw_pp"
        ),
        errors="coerce",
    )

    output[
        "prior_clipped_low_flag"
    ] = (
        (
            raw < 0
        )
        & raw.notna()
    ).astype(
        "Int64"
    )

    output[
        "prior_clipped_high_flag"
    ] = (
        (
            raw > 100
        )
        & raw.notna()
    ).astype(
        "Int64"
    )

    if (
        "prior_fallback_flag"
        in output.columns
    ):
        output[
            "prior_fallback_flag"
        ] = pd.to_numeric(
            output[
                "prior_fallback_flag"
            ],
            errors="coerce",
        ).astype(
            "Int64"
        )

    else:
        output[
            "prior_fallback_flag"
        ] = pd.Series(
            pd.NA,
            index=output.index,
            dtype="Int64",
        )

    if (
        "audit_prior_pp"
        in output.columns
    ):
        diff = np.nanmax(
            np.abs(
                output[
                    "prior_pp"
                ]
                - output[
                    "audit_prior_pp"
                ]
            )
        )

        if (
            diff > 1e-4
        ):
            raise RuntimeError(
                "Recovered prior differs from "
                "model-input audit by "
                f"{diff} pp"
            )

    return output


def validate_afs_window_table(
    frame: pd.DataFrame,
    experiment: str,
) -> dict[str, Any]:
    if len(
        frame
    ) != EXPECTED_ROWS:
        raise RuntimeError(
            f"{experiment}: expected "
            f"{EXPECTED_ROWS} rows, "
            f"found {len(frame)}"
        )

    if frame[
        "sample_id"
    ].nunique() != EXPECTED_EVENTS:
        raise RuntimeError(
            f"{experiment}: expected "
            f"{EXPECTED_EVENTS} events, "
            "found "
            f"{frame['sample_id'].nunique()}"
        )

    sizes = frame.groupby(
        "sample_id"
    ).size()

    if (
        int(
            sizes.min()
        ) != 4
        or int(
            sizes.max()
        ) != 5
    ):
        raise RuntimeError(
            f"{experiment}: expected "
            "4--5 windows/event, observed "
            f"{sizes.min()}--{sizes.max()}"
        )

    mean_windows = float(
        sizes.mean()
    )

    if not math.isclose(
        mean_windows,
        EXPECTED_ROWS
        / EXPECTED_EVENTS,
        rel_tol=0,
        abs_tol=1e-12,
    ):
        raise RuntimeError(
            f"{experiment}: unexpected "
            f"mean windows/event {mean_windows}"
        )

    observed_windows = set(
        frame[
            "window_idx"
        ].unique().tolist()
    )

    if (
        observed_windows
        != EXPECTED_WINDOW_INDICES
    ):
        raise RuntimeError(
            f"{experiment}: expected window "
            "indices "
            f"{sorted(EXPECTED_WINDOW_INDICES)}, "
            "found "
            f"{sorted(observed_windows)}"
        )

    fold_counts = frame.groupby(
        "sample_id"
    )[
        "fold"
    ].nunique()

    if int(
        fold_counts.max()
    ) != 1:
        raise RuntimeError(
            f"{experiment}: at least one "
            "event crosses outer folds"
        )

    return {
        "experiment": experiment,
        "rows": int(
            len(
                frame
            )
        ),
        "events": int(
            frame[
                "sample_id"
            ].nunique()
        ),
        "min_windows_per_event": int(
            sizes.min()
        ),
        "max_windows_per_event": int(
            sizes.max()
        ),
        "mean_windows_per_event": (
            mean_windows
        ),
        "window_indices": sorted(
            observed_windows
        ),
        "fold_event_leakage_count": int(
            (
                fold_counts > 1
            ).sum()
        ),
    }


def unique_value(
    series: pd.Series,
    *,
    label: str,
    tolerance: float | None = None,
) -> Any:
    nonmissing = series.dropna()

    if nonmissing.empty:
        return np.nan

    if tolerance is not None:
        values = pd.to_numeric(
            nonmissing,
            errors="coerce",
        ).to_numpy(
            dtype=float
        )

        if (
            np.nanmax(
                values
            )
            - np.nanmin(
                values
            )
            > tolerance
        ):
            raise RuntimeError(
                f"Non-constant {label} "
                "within event"
            )

        return float(
            np.nanmean(
                values
            )
        )

    values = nonmissing.astype(
        str
    ).unique()

    if len(
        values
    ) != 1:
        raise RuntimeError(
            f"Non-constant {label} "
            f"within event: {values[:5]}"
        )

    return nonmissing.iloc[0]


def aggregate_events(
    window: pd.DataFrame,
    experiment: str,
) -> pd.DataFrame:
    rows = []

    for (
        sample_id,
        group,
    ) in window.groupby(
        "sample_id",
        sort=False,
    ):
        pred = group[
            "y_pred_pp"
        ].to_numpy(
            dtype=float
        )

        prior = group[
            "prior_pp"
        ].to_numpy(
            dtype=float
        )

        y_true = unique_value(
            group[
                "y_true_pp"
            ],
            label="target",
            tolerance=1e-8,
        )

        row = {
            "experiment": experiment,
            "sample_id": sample_id,
            "fold": int(
                unique_value(
                    group[
                        "fold"
                    ],
                    label="fold",
                    tolerance=0,
                )
            ),
            "y_true_pp": (
                y_true
            ),
            "y_pred_pp": float(
                np.mean(
                    pred
                )
            ),
            "prior_pp": float(
                np.mean(
                    prior
                )
            ),
            "n_windows": int(
                len(
                    group
                )
            ),
            "within_event_prediction_std_pp": float(
                np.std(
                    pred,
                    ddof=0,
                )
            ),
            "within_event_prediction_range_pp": float(
                np.max(
                    pred
                )
                - np.min(
                    pred
                )
            ),
            "within_event_prior_std_pp": float(
                np.std(
                    prior,
                    ddof=0,
                )
            ),
            "within_event_prior_range_pp": float(
                np.max(
                    prior
                )
                - np.min(
                    prior
                )
            ),
            "mean_window_model_ae_pp": float(
                np.mean(
                    np.abs(
                        pred
                        - y_true
                    )
                )
            ),
            "mean_window_prior_ae_pp": float(
                np.mean(
                    np.abs(
                        prior
                        - y_true
                    )
                )
            ),
        }

        row[
            "event_model_ae_pp"
        ] = abs(
            row[
                "y_pred_pp"
            ]
            - y_true
        )

        row[
            "event_prior_ae_pp"
        ] = abs(
            row[
                "prior_pp"
            ]
            - y_true
        )

        row[
            "averaging_helped_model"
        ] = (
            row[
                "event_model_ae_pp"
            ]
            < row[
                "mean_window_model_ae_pp"
            ]
            - TOL
        )

        row[
            "averaging_helped_prior"
        ] = (
            row[
                "event_prior_ae_pp"
            ]
            < row[
                "mean_window_prior_ae_pp"
            ]
            - TOL
        )

        for column in [
            "case",
            "y_fault_line",
            "event_type",
            "status",
        ]:
            if (
                column
                in group.columns
            ):
                row[
                    column
                ] = unique_value(
                    group[
                        column
                    ],
                    label=column,
                )

        if (
            "y_fault_location"
            in group.columns
        ):
            row[
                "y_fault_location"
            ] = unique_value(
                group[
                    "y_fault_location"
                ],
                label=(
                    "y_fault_location"
                ),
                tolerance=1e-8,
            )

        else:
            row[
                "y_fault_location"
            ] = y_true

        for flag in [
            "prior_fallback_flag",
            "prior_clipped_low_flag",
            "prior_clipped_high_flag",
        ]:
            if (
                flag
                in group.columns
            ):
                numeric = pd.to_numeric(
                    group[
                        flag
                    ],
                    errors="coerce",
                )

                row[
                    flag
                    + "_count"
                ] = int(
                    numeric.fillna(
                        0
                    ).sum()
                )

        rows.append(
            row
        )

    output = pd.DataFrame(
        rows
    )

    if len(
        output
    ) != EXPECTED_EVENTS:
        raise RuntimeError(
            f"{experiment}: event aggregation "
            f"produced {len(output)} rows"
        )

    return output


def metric_dict(
    y_true: Iterable[float],
    y_pred: Iterable[float],
) -> dict[str, float | int]:
    y_true_np = np.asarray(
        list(
            y_true
        ),
        dtype=float,
    )

    y_pred_np = np.asarray(
        list(
            y_pred
        ),
        dtype=float,
    )

    mask = (
        np.isfinite(
            y_true_np
        )
        & np.isfinite(
            y_pred_np
        )
    )

    y_true_np = y_true_np[
        mask
    ]

    y_pred_np = y_pred_np[
        mask
    ]

    if not len(
        y_true_np
    ):
        empty = {
            key: np.nan
            for key in [
                "mae_pp",
                "rmse_pp",
                "median_ae_pp",
                "bias_pp",
                "p90_ae_pp",
                "p95_ae_pp",
                "p99_ae_pp",
                "max_ae_pp",
                "cvar95_ae_pp",
                "exceed_5_rate",
                "exceed_10_rate",
                "exceed_20_rate",
                "exceed_30_rate",
            ]
        }

        empty["n"] = 0

        return empty

    error = (
        y_pred_np
        - y_true_np
    )

    ae = np.abs(
        error
    )

    threshold = float(
        np.quantile(
            ae,
            0.95,
        )
    )

    tail = ae[
        ae >= threshold
    ]

    return {
        "n": int(
            len(
                ae
            )
        ),
        "mae_pp": float(
            np.mean(
                ae
            )
        ),
        "rmse_pp": float(
            np.sqrt(
                np.mean(
                    error ** 2
                )
            )
        ),
        "median_ae_pp": float(
            np.median(
                ae
            )
        ),
        "bias_pp": float(
            np.mean(
                error
            )
        ),
        "p90_ae_pp": float(
            np.quantile(
                ae,
                0.90,
            )
        ),
        "p95_ae_pp": float(
            np.quantile(
                ae,
                0.95,
            )
        ),
        "p99_ae_pp": float(
            np.quantile(
                ae,
                0.99,
            )
        ),
        "max_ae_pp": float(
            np.max(
                ae
            )
        ),
        "cvar95_ae_pp": float(
            np.mean(
                tail
            )
        ),
        "exceed_5_rate": float(
            np.mean(
                ae > 5.0
            )
        ),
        "exceed_10_rate": float(
            np.mean(
                ae > 10.0
            )
        ),
        "exceed_20_rate": float(
            np.mean(
                ae > 20.0
            )
        ),
        "exceed_30_rate": float(
            np.mean(
                ae > 30.0
            )
        ),
    }


def bootstrap_mean_ci(
    values: np.ndarray,
    *,
    n_resamples: int = 10000,
    seed: int = 42,
) -> tuple[float, float]:
    values = np.asarray(
        values,
        dtype=float,
    )

    values = values[
        np.isfinite(
            values
        )
    ]

    if values.size == 0:
        return (
            np.nan,
            np.nan,
        )

    rng = np.random.default_rng(
        seed
    )

    means = np.empty(
        n_resamples,
        dtype=float,
    )

    batch = 100
    offset = 0

    while offset < n_resamples:
        count = min(
            batch,
            n_resamples
            - offset,
        )

        indices = rng.integers(
            0,
            len(
                values
            ),
            size=(
                count,
                len(
                    values
                ),
            ),
        )

        means[
            offset:
            offset + count
        ] = values[
            indices
        ].mean(
            axis=1
        )

        offset += count

    return (
        float(
            np.quantile(
                means,
                0.025,
            )
        ),
        float(
            np.quantile(
                means,
                0.975,
            )
        ),
    )


def predictor_metrics_rows(
    event: pd.DataFrame,
    experiment: str,
    *,
    scope: str = "overall",
    subgroup: str = "all",
) -> list[dict[str, Any]]:
    rows = []

    for (
        predictor,
        column,
    ) in [
        (
            "model",
            "y_pred_pp",
        ),
        (
            "prior",
            "prior_pp",
        ),
    ]:
        rows.append(
            {
                "experiment": (
                    experiment
                ),
                "scope": scope,
                "subgroup": subgroup,
                "predictor": (
                    predictor
                ),
                **metric_dict(
                    event[
                        "y_true_pp"
                    ],
                    event[
                        column
                    ],
                ),
            }
        )

    return rows


def paired_prior_row(
    event: pd.DataFrame,
    experiment: str,
    *,
    scope: str = "overall",
    subgroup: str = "all",
) -> dict[str, Any]:
    prior_ae = np.abs(
        event[
            "prior_pp"
        ].to_numpy(
            dtype=float
        )
        - event[
            "y_true_pp"
        ].to_numpy(
            dtype=float
        )
    )

    model_ae = np.abs(
        event[
            "y_pred_pp"
        ].to_numpy(
            dtype=float
        )
        - event[
            "y_true_pp"
        ].to_numpy(
            dtype=float
        )
    )

    reduction = (
        prior_ae
        - model_ae
    )

    (
        low,
        high,
    ) = bootstrap_mean_ci(
        reduction,
        n_resamples=10000,
        seed=42,
    )

    fold_reductions = []

    for _fold, group in event.groupby(
        "fold"
    ):
        fold_reductions.append(
            float(
                group[
                    "event_prior_ae_pp"
                ].mean()
                - group[
                    "event_model_ae_pp"
                ].mean()
            )
        )

    prior_mae = float(
        np.mean(
            prior_ae
        )
    )

    model_mae = float(
        np.mean(
            model_ae
        )
    )

    return {
        "experiment": experiment,
        "scope": scope,
        "subgroup": subgroup,
        "n_events": int(
            len(
                event
            )
        ),
        "prior_mae_pp": (
            prior_mae
        ),
        "model_mae_pp": (
            model_mae
        ),
        "absolute_mae_reduction_pp": (
            prior_mae
            - model_mae
        ),
        "relative_mae_reduction": (
            (
                prior_mae
                - model_mae
            )
            / prior_mae
            if prior_mae
            else np.nan
        ),
        "improved_rate": float(
            np.mean(
                model_ae
                < prior_ae
                - TOL
            )
        ),
        "worsened_rate": float(
            np.mean(
                model_ae
                > prior_ae
                + TOL
            )
        ),
        "unchanged_rate": float(
            np.mean(
                np.abs(
                    model_ae
                    - prior_ae
                )
                <= TOL
            )
        ),
        "improved_fold_count": int(
            np.sum(
                np.asarray(
                    fold_reductions
                )
                > 0
            )
        ),
        "bootstrap_reduction_ci_low_pp": (
            low
        ),
        "bootstrap_reduction_ci_high_pp": (
            high
        ),
    }


def add_group_metrics(
    event_tables: dict[
        str,
        pd.DataFrame,
    ],
    group_column: str,
    scope_name: str,
) -> pd.DataFrame:
    rows = []

    for (
        experiment,
        event,
    ) in event_tables.items():
        for (
            group_value,
            group,
        ) in event.groupby(
            group_column,
            dropna=False,
        ):
            base = paired_prior_row(
                group,
                experiment,
                scope=scope_name,
                subgroup=str(
                    group_value
                ),
            )

            model = metric_dict(
                group[
                    "y_true_pp"
                ],
                group[
                    "y_pred_pp"
                ],
            )

            prior = metric_dict(
                group[
                    "y_true_pp"
                ],
                group[
                    "prior_pp"
                ],
            )

            rows.append(
                {
                    **base,
                    "group_column": (
                        group_column
                    ),
                    "group_value": (
                        group_value
                    ),
                    **{
                        f"model_{key}": value
                        for (
                            key,
                            value,
                        ) in model.items()
                    },
                    **{
                        f"prior_{key}": value
                        for (
                            key,
                            value,
                        ) in prior.items()
                    },
                }
            )

    return pd.DataFrame(
        rows
    )


def normalize_sfs_event(
    frame: pd.DataFrame,
    experiment: str,
) -> pd.DataFrame:
    source = frame.copy()

    if (
        "experiment"
        in source.columns
    ):
        source = source[
            source[
                "experiment"
            ].astype(
                str
            ) == experiment
        ].copy()

    elif (
        "experiment_id"
        in source.columns
    ):
        source = source[
            source[
                "experiment_id"
            ].astype(
                str
            ) == experiment
        ].copy()

    aliases = {
        "sample_id": [
            "sample_id",
            "event_id",
        ],
        "fold": [
            "fold",
            "outer_fold",
        ],
        "y_true_pp": [
            "y_true_pp",
            "y_true_pct",
            "y_true",
            "target_pp",
        ],
        "y_pred_pp": [
            "y_pred_pp",
            "y_pred_pct",
            "y_pred",
            "model_prediction_pp",
        ],
        "prior_pp": [
            "prior_pp",
            "y_prior_pct",
            "d_prior_pp",
            "d_prior",
            "physical_prior_pp",
        ],
        "case": [
            "case",
            "fault_case",
        ],
        "y_fault_line": [
            "y_fault_line",
            "line",
        ],
        "y_fault_location": [
            "y_fault_location",
            "location_pct",
            "y_true_pct",
            "y_true_pp",
            "y_true",
        ],
    }

    output = pd.DataFrame(
        index=source.index
    )

    for (
        target,
        choices,
    ) in aliases.items():
        column = next(
            (
                choice
                for choice
                in choices
                if choice
                in source.columns
            ),
            None,
        )

        if column is not None:
            output[
                target
            ] = source[
                column
            ]

    required = {
        "sample_id",
        "fold",
        "y_true_pp",
        "y_pred_pp",
        "prior_pp",
    }

    missing = sorted(
        required
        - set(
            output.columns
        )
    )

    if missing:
        raise KeyError(
            f"SFS {experiment}: missing columns "
            f"{missing}; available="
            f"{list(source.columns)}"
        )

    output[
        "sample_id"
    ] = output[
        "sample_id"
    ].map(
        canonical_id
    )

    output[
        "fold"
    ] = pd.to_numeric(
        output[
            "fold"
        ],
        errors="raise",
    ).astype(
        int
    )

    for column in [
        "y_true_pp",
        "y_pred_pp",
        "prior_pp",
        "y_fault_location",
    ]:
        if (
            column
            in output.columns
        ):
            output[
                column
            ] = to_pp(
                output[
                    column
                ]
            )

    output[
        "experiment"
    ] = experiment

    if output[
        "sample_id"
    ].duplicated().any():
        output[
            "window_idx"
        ] = np.arange(
            len(
                output
            )
        )

        output = aggregate_events(
            output,
            experiment,
        )

    return output.reset_index(
        drop=True
    )


def load_sfs_events(
    thesis_dir: Path,
) -> tuple[
    dict[
        str,
        pd.DataFrame,
    ],
    Path,
]:
    final_dir = (
        thesis_dir
        / "outputs"
        / "chapter4"
        / "posthoc_final"
    )

    candidates = [
        (
            final_dir
            / "posthoc_unified_event_predictions.parquet"
        ),
        (
            final_dir
            / "posthoc_unified_event_predictions.csv"
        ),
    ]

    source_path = next(
        (
            path
            for path
            in candidates
            if path.exists()
        ),
        None,
    )

    if source_path is None:
        raise FileNotFoundError(
            "Final SFS unified event predictions "
            "not found under "
            "outputs/chapter4/posthoc_final"
        )

    source = read_table(
        source_path
    )

    tables = {}

    for sfs_experiment in SFS_MAP.values():
        tables[
            sfs_experiment
        ] = normalize_sfs_event(
            source,
            sfs_experiment,
        )

    return (
        tables,
        source_path,
    )


def _plain_config(
    value: Any,
) -> Any:
    try:
        from omegaconf import (
            OmegaConf,
        )

        if OmegaConf.is_config(
            value
        ):
            return OmegaConf.to_container(
                value,
                resolve=False,
            )

    except Exception:
        pass

    if isinstance(
        value,
        dict,
    ):
        return {
            str(
                key
            ): _plain_config(
                item
            )
            for (
                key,
                item,
            ) in value.items()
        }

    if (
        hasattr(
            value,
            "__dict__",
        )
        and not isinstance(
            value,
            (
                str,
                bytes,
            ),
        )
    ):
        try:
            return {
                str(
                    key
                ): _plain_config(
                    item
                )
                for (
                    key,
                    item,
                ) in vars(
                    value
                ).items()
            }

        except Exception:
            return value

    if isinstance(
        value,
        (
            list,
            tuple,
        ),
    ):
        return [
            _plain_config(
                item
            )
            for item
            in value
        ]

    return value


def _nested_get(
    mapping: Any,
    path: str,
    default: Any = None,
) -> Any:
    current = mapping

    for token in path.split(
        "."
    ):
        if (
            isinstance(
                current,
                dict,
            )
            and token in current
        ):
            current = current[
                token
            ]

        else:
            return default

    return current


def checkpoint_signature(
    experiment_dir: Path,
) -> dict[str, Any]:
    checkpoints = sorted(
        experiment_dir.glob(
            "checkpoints/*fold0*seed42.pt"
        )
    )

    if not checkpoints:
        checkpoints = sorted(
            experiment_dir.glob(
                "checkpoints/*.pt"
            )
        )

    if not checkpoints:
        return {
            "status": "UNRESOLVED",
            "reason": (
                "checkpoint_not_found"
            ),
        }

    checkpoint_path = checkpoints[0]

    try:
        import torch

        try:
            payload = torch.load(
                checkpoint_path,
                map_location="cpu",
                weights_only=False,
            )

        except TypeError:
            payload = torch.load(
                checkpoint_path,
                map_location="cpu",
            )

    except Exception as error:
        return {
            "status": "UNRESOLVED",
            "reason": (
                "checkpoint_load_failed:"
                f"{type(error).__name__}:"
                f"{error}"
            ),
            "checkpoint": str(
                checkpoint_path
            ),
        }

    config = _plain_config(
        payload.get(
            "config",
            payload.get(
                "cfg",
                {},
            ),
        )
    )

    meta = _plain_config(
        payload.get(
            "meta",
            {},
        )
    )

    return {
        "status": "RESOLVED",
        "checkpoint": str(
            checkpoint_path
        ),
        "topology": _nested_get(
            config,
            "dataset.topology",
            meta.get(
                "topology"
            ),
        ),
        "target_label": _nested_get(
            config,
            "training.target_label",
            meta.get(
                "target_label"
            ),
        ),
        "hidden_size": _nested_get(
            config,
            "model.hidden_size",
        ),
        "num_layers": _nested_get(
            config,
            "model.num_layers",
        ),
        "dropout": _nested_get(
            config,
            "model.dropout",
        ),
        "bidirectional": _nested_get(
            config,
            "model.bidirectional",
        ),
        "kol_model_mode": _nested_get(
            config,
            "training.kol_model_mode",
        ),
        "kol_prediction_mode": _nested_get(
            config,
            "training.kol_prediction_mode",
        ),
        "input_representation": _nested_get(
            config,
            "training.input_representation",
        ),
        "n_splits": _nested_get(
            config,
            "training.n_splits",
        ),
        "split_seed": _nested_get(
            config,
            "training.split_seed",
        ),
        "seeds": _nested_get(
            config,
            "training.seeds",
        ),
        "operator_prior_col": _nested_get(
            config,
            "training.operator_prior_col",
        ),
        "operator_feature_cols": _nested_get(
            config,
            "training.operator_feature_cols",
        ),
        "window_length": _nested_get(
            config,
            "window_extraction.window_length",
        ),
        "step_length": _nested_get(
            config,
            "window_extraction.step_length_seconds",
        ),
    }


def compare_checkpoint_signatures(
    sfs: dict[str, Any],
    afs: dict[str, Any],
) -> dict[str, Any]:
    if (
        sfs.get(
            "status"
        ) != "RESOLVED"
        or afs.get(
            "status"
        ) != "RESOLVED"
    ):
        return {
            "model_configuration_status": (
                "UNRESOLVED"
            ),
            "seed_status": (
                "UNRESOLVED"
            ),
            "operator_feature_column_status": (
                "UNRESOLVED"
            ),
            "configuration_mismatches": json.dumps(
                {
                    "sfs": sfs.get(
                        "reason"
                    ),
                    "afs": afs.get(
                        "reason"
                    ),
                }
            ),
        }

    model_fields = [
        "topology",
        "target_label",
        "hidden_size",
        "num_layers",
        "dropout",
        "bidirectional",
        "kol_model_mode",
        "kol_prediction_mode",
        "input_representation",
        "n_splits",
        "split_seed",
        "window_length",
        "step_length",
    ]

    mismatches = {}
    unresolved_fields = []

    for field in model_fields:
        left = sfs.get(
            field
        )

        right = afs.get(
            field
        )

        if (
            left is None
            or right is None
        ):
            unresolved_fields.append(
                field
            )

        elif str(
            left
        ) != str(
            right
        ):
            mismatches[
                field
            ] = {
                "sfs": left,
                "afs": right,
            }

    if mismatches:
        model_status = "FAIL"

    elif unresolved_fields:
        model_status = "PARTIAL"

    else:
        model_status = "PASS"

    seed_status = (
        "PASS"
        if (
            str(
                sfs.get(
                    "seeds"
                )
            )
            == str(
                afs.get(
                    "seeds"
                )
            )
            and str(
                sfs.get(
                    "split_seed"
                )
            )
            == str(
                afs.get(
                    "split_seed"
                )
            )
        )
        else "FAIL"
    )

    def norm_cols(
        value: Any,
    ) -> list[str] | None:
        if value is None:
            return None

        if isinstance(
            value,
            str,
        ):
            return [
                token.strip()
                for token
                in value.strip(
                    "[]"
                ).split(
                    ","
                )
                if token.strip()
            ]

        if isinstance(
            value,
            (
                list,
                tuple,
            ),
        ):
            return [
                str(
                    token
                )
                for token
                in value
            ]

        return [
            str(
                value
            )
        ]

    sfs_cols = norm_cols(
        sfs.get(
            "operator_feature_cols"
        )
    )

    afs_cols = norm_cols(
        afs.get(
            "operator_feature_cols"
        )
    )

    if (
        sfs_cols is None
        or afs_cols is None
    ):
        feature_status = (
            "UNRESOLVED"
        )

    else:
        feature_status = (
            "PASS"
            if sfs_cols
            == afs_cols
            else "FAIL"
        )

    return {
        "model_configuration_status": (
            model_status
        ),
        "seed_status": (
            seed_status
        ),
        "operator_feature_column_status": (
            feature_status
        ),
        "configuration_mismatches": json.dumps(
            {
                "mismatches": (
                    mismatches
                ),
                "unresolved_fields": (
                    unresolved_fields
                ),
            },
            sort_keys=True,
        ),
    }


def compare_sfs_afs(
    afs_events: dict[
        str,
        pd.DataFrame,
    ],
    sfs_events: dict[
        str,
        pd.DataFrame,
    ],
    config_audits: dict[
        str,
        dict[str, Any],
    ],
) -> pd.DataFrame:
    rows = []

    for (
        afs_experiment,
        sfs_experiment,
    ) in SFS_MAP.items():
        afs = afs_events[
            afs_experiment
        ].copy()

        sfs = sfs_events[
            sfs_experiment
        ].copy()

        merged = sfs.merge(
            afs,
            on=[
                "sample_id"
            ],
            suffixes=(
                "_sfs",
                "_afs",
            ),
            validate="one_to_one",
        )

        ids_identical = (
            len(
                merged
            )
            == len(
                sfs
            )
            == len(
                afs
            )
            == EXPECTED_EVENTS
        )

        targets_identical = bool(
            np.allclose(
                merged[
                    "y_true_pp_sfs"
                ],
                merged[
                    "y_true_pp_afs"
                ],
                atol=1e-8,
                rtol=0,
            )
        )

        folds_identical = bool(
            (
                merged[
                    "fold_sfs"
                ].to_numpy()
                == merged[
                    "fold_afs"
                ].to_numpy()
            ).all()
        )

        priors_identical = bool(
            np.allclose(
                merged[
                    "prior_pp_sfs"
                ],
                merged[
                    "prior_pp_afs"
                ],
                atol=1e-8,
                rtol=0,
            )
        )

        config_audit = config_audits.get(
            afs_experiment,
            {},
        )

        def comparison_row(
            group: pd.DataFrame,
            scope: str,
            subgroup: str,
            predictor: str,
        ) -> dict[str, Any]:
            if predictor == "model":
                sfs_pred = group[
                    "y_pred_pp_sfs"
                ].to_numpy(
                    dtype=float
                )

                afs_pred = group[
                    "y_pred_pp_afs"
                ].to_numpy(
                    dtype=float
                )

            else:
                sfs_pred = group[
                    "prior_pp_sfs"
                ].to_numpy(
                    dtype=float
                )

                afs_pred = group[
                    "prior_pp_afs"
                ].to_numpy(
                    dtype=float
                )

            truth = group[
                "y_true_pp_sfs"
            ].to_numpy(
                dtype=float
            )

            sfs_ae = np.abs(
                sfs_pred
                - truth
            )

            afs_ae = np.abs(
                afs_pred
                - truth
            )

            diff = (
                afs_ae
                - sfs_ae
            )

            (
                low,
                high,
            ) = bootstrap_mean_ci(
                diff,
                n_resamples=10000,
                seed=42,
            )

            sfs_metrics = metric_dict(
                truth,
                sfs_pred,
            )

            afs_metrics = metric_dict(
                truth,
                afs_pred,
            )

            return {
                "afs_experiment": (
                    afs_experiment
                ),
                "sfs_experiment": (
                    sfs_experiment
                ),
                "scope": scope,
                "subgroup": subgroup,
                "predictor": (
                    predictor
                ),
                "n_events": int(
                    len(
                        group
                    )
                ),
                "event_ids_identical": (
                    ids_identical
                ),
                "targets_identical": (
                    targets_identical
                ),
                "fold_assignments_identical": (
                    folds_identical
                ),
                "event_averaged_priors_identical": (
                    priors_identical
                ),
                "model_configuration_status": (
                    config_audit.get(
                        "model_configuration_status",
                        "UNRESOLVED",
                    )
                ),
                "seed_status": (
                    config_audit.get(
                        "seed_status",
                        "UNRESOLVED",
                    )
                ),
                "operator_feature_column_status": (
                    config_audit.get(
                        "operator_feature_column_status",
                        "UNRESOLVED",
                    )
                ),
                "configuration_mismatches": (
                    config_audit.get(
                        "configuration_mismatches",
                        "{}",
                    )
                ),
                "operator_feature_value_status": (
                    "NOT_EXPECTED_ROW_IDENTICAL_"
                    "AFTER_EVENT_AVERAGING"
                    if "1E"
                    in afs_experiment
                    else (
                        "NO_SCALAR_OPERATOR_FEATURES"
                    )
                ),
                "sfs_mae_pp": (
                    sfs_metrics[
                        "mae_pp"
                    ]
                ),
                "sfs_rmse_pp": (
                    sfs_metrics[
                        "rmse_pp"
                    ]
                ),
                "sfs_p95_ae_pp": (
                    sfs_metrics[
                        "p95_ae_pp"
                    ]
                ),
                "sfs_p99_ae_pp": (
                    sfs_metrics[
                        "p99_ae_pp"
                    ]
                ),
                "afs_mae_pp": (
                    afs_metrics[
                        "mae_pp"
                    ]
                ),
                "afs_rmse_pp": (
                    afs_metrics[
                        "rmse_pp"
                    ]
                ),
                "afs_p95_ae_pp": (
                    afs_metrics[
                        "p95_ae_pp"
                    ]
                ),
                "afs_p99_ae_pp": (
                    afs_metrics[
                        "p99_ae_pp"
                    ]
                ),
                "afs_minus_sfs_mae_pp": float(
                    np.mean(
                        diff
                    )
                ),
                "bootstrap_difference_ci_low_pp": (
                    low
                ),
                "bootstrap_difference_ci_high_pp": (
                    high
                ),
                "afs_improved_rate": float(
                    np.mean(
                        afs_ae
                        < sfs_ae
                        - TOL
                    )
                ),
                "afs_worsened_rate": float(
                    np.mean(
                        afs_ae
                        > sfs_ae
                        + TOL
                    )
                ),
                "afs_unchanged_rate": float(
                    np.mean(
                        np.abs(
                            afs_ae
                            - sfs_ae
                        )
                        <= TOL
                    )
                ),
            }

        rows.append(
            comparison_row(
                merged,
                "overall",
                "all",
                "model",
            )
        )

        rows.append(
            comparison_row(
                merged,
                "overall",
                "all",
                "prior",
            )
        )

        for (
            fold,
            group,
        ) in merged.groupby(
            "fold_sfs"
        ):
            rows.append(
                comparison_row(
                    group,
                    "fold",
                    str(
                        int(
                            fold
                        )
                    ),
                    "model",
                )
            )

        for group_column in [
            "case_sfs",
            "y_fault_line_sfs",
        ]:
            if (
                group_column
                in merged.columns
            ):
                scope = (
                    "fault_case"
                    if group_column.startswith(
                        "case"
                    )
                    else "protected_line"
                )

                for (
                    group_value,
                    group,
                ) in merged.groupby(
                    group_column,
                    dropna=False,
                ):
                    rows.append(
                        comparison_row(
                            group,
                            scope,
                            str(
                                group_value
                            ),
                            "model",
                        )
                    )

    return pd.DataFrame(
        rows
    )


def physical_prior_table(
    thesis_dir: Path,
    afs_input_dir: Path,
    row_audit_path: Path,
) -> pd.DataFrame:
    temp_base = (
        thesis_dir
        / "outputs"
        / "chapter4"
        / "temp_90kv_afs_check"
    )

    pointer = (
        temp_base
        / "LATEST_RAW_DIR.txt"
    )

    if not pointer.exists():
        return pd.DataFrame(
            [
                {
                    "prior_name": (
                        "UNAVAILABLE"
                    ),
                    "notes": (
                        f"Missing {pointer}"
                    ),
                }
            ]
        )

    raw_dir = Path(
        pointer.read_text(
            encoding="utf-8"
        ).splitlines()[0].strip()
    )

    default_candidates = list(
        raw_dir.glob(
            "*default*all_fault_start*.csv"
        )
    )

    both_candidates = list(
        raw_dir.glob(
            "*both*all_fault_start*.csv"
        )
    )

    one_input_candidates = list(
        afs_input_dir.glob(
            "*case_bestmae*model_input.csv"
        )
    )

    two_input_candidates = list(
        afs_input_dir.glob(
            "*two_ended_posseq*model_input.csv"
        )
    )

    if (
        not default_candidates
        or not both_candidates
        or not one_input_candidates
        or not two_input_candidates
    ):
        return pd.DataFrame(
            [
                {
                    "prior_name": (
                        "UNAVAILABLE"
                    ),
                    "notes": (
                        "Could not resolve all "
                        "four prior files"
                    ),
                }
            ]
        )

    audit = (
        pd.read_csv(
            row_audit_path
        )
        if row_audit_path.exists()
        else None
    )

    specifications = [
        (
            "standard_one_ended_reference",
            default_candidates[0],
            "d_phys_real_pct",
            None,
        ),
        (
            "exact_one_ended_model_input",
            one_input_candidates[0],
            "d_phys_real_pct",
            "selected_1e_fallback",
        ),
        (
            "standard_synchronised_two_ended_reference",
            both_candidates[0],
            "d_two_ended_posseq_plus_pct",
            None,
        ),
        (
            "exact_synchronised_two_ended_model_input",
            two_input_candidates[0],
            "d_phys_real_pct",
            "input_2e_fallback",
        ),
    ]

    rows = []

    for (
        name,
        path,
        prediction_col,
        fallback_col,
    ) in specifications:
        frame = pd.read_csv(
            path
        )

        frame[
            "sample_id"
        ] = frame[
            "sample_id"
        ].map(
            canonical_id
        )

        frame[
            "window_idx"
        ] = pd.to_numeric(
            frame[
                "window_idx"
            ],
            errors="raise",
        ).astype(
            int
        )

        target_col = next(
            (
                column
                for column in [
                    "y_fault_location",
                    "y_true",
                ]
                if column
                in frame.columns
            ),
            None,
        )

        if (
            target_col is None
            and audit is not None
        ):
            merge = audit[
                [
                    "sample_id",
                    "window_idx",
                    "y_fault_location",
                ]
            ].copy()

            merge[
                "sample_id"
            ] = merge[
                "sample_id"
            ].map(
                canonical_id
            )

            frame = frame.merge(
                merge,
                on=[
                    "sample_id",
                    "window_idx",
                ],
                how="left",
            )

            target_col = (
                "y_fault_location"
            )

        pred = pd.to_numeric(
            frame[
                prediction_col
            ],
            errors="coerce",
        )

        finite_mask = np.isfinite(
            pred
        )

        frame[
            "prediction_pp"
        ] = pred

        event = (
            frame.loc[
                finite_mask
            ]
            .groupby(
                "sample_id",
                as_index=False,
            )
            .agg(
                prediction_pp=(
                    "prediction_pp",
                    "mean",
                ),
                y_true_pp=(
                    target_col,
                    "mean",
                ),
                n_windows=(
                    "window_idx",
                    "size",
                ),
            )
        )

        event[
            "y_true_pp"
        ] = to_pp(
            event[
                "y_true_pp"
            ]
        )

        fallback_count = 0

        if (
            fallback_col
            and audit is not None
            and fallback_col
            in audit.columns
        ):
            fallback_count = int(
                pd.to_numeric(
                    audit[
                        fallback_col
                    ],
                    errors="coerce",
                )
                .fillna(
                    0
                )
                .sum()
            )

        metrics = metric_dict(
            event[
                "y_true_pp"
            ],
            event[
                "prediction_pp"
            ],
        )

        sizes = frame.groupby(
            "sample_id"
        ).size()

        rows.append(
            {
                "prior_name": name,
                "source_file": str(
                    path
                ),
                "prediction_column": (
                    prediction_col
                ),
                "rows": int(
                    len(
                        frame
                    )
                ),
                "events": int(
                    frame[
                        "sample_id"
                    ].nunique()
                ),
                "minimum_windows_per_event": int(
                    sizes.min()
                ),
                "maximum_windows_per_event": int(
                    sizes.max()
                ),
                "mean_windows_per_event": float(
                    sizes.mean()
                ),
                "finite_count": int(
                    finite_mask.sum()
                ),
                "invalid_count": int(
                    (
                        ~finite_mask
                    ).sum()
                ),
                "fallback_count": (
                    fallback_count
                ),
                **metrics,
            }
        )

    return pd.DataFrame(
        rows
    )


def markdown_table(
    frame: pd.DataFrame,
    columns: list[str],
    digits: int = 4,
) -> str:
    if frame.empty:
        return "_No rows available._"

    display = frame[
        columns
    ].copy()

    for column in display.columns:
        if pd.api.types.is_float_dtype(
            display[
                column
            ]
        ):
            display[
                column
            ] = display[
                column
            ].map(
                lambda value: (
                    f"{value:.{digits}f}"
                    if pd.notna(
                        value
                    )
                    else ""
                )
            )

    headers = list(
        display.columns
    )

    lines = [
        "| "
        + " | ".join(
            headers
        )
        + " |",
        "| "
        + " | ".join(
            [
                "---"
            ]
            * len(
                headers
            )
        )
        + " |",
    ]

    for _index, row in display.iterrows():
        lines.append(
            "| "
            + " | ".join(
                str(
                    row[
                        column
                    ]
                )
                for column
                in headers
            )
            + " |"
        )

    return "\n".join(
        lines
    )


def sha256(
    path: Path,
) -> str:
    digest = hashlib.sha256()

    with path.open(
        "rb"
    ) as handle:
        for chunk in iter(
            lambda: handle.read(
                1024
                * 1024
            ),
            b"",
        ):
            digest.update(
                chunk
            )

    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--thesis-dir",
        default=(
            "/home/hpc/iwi5/iwi5305h/"
            "Masters_thesis_PR_LABS"
        ),
    )

    parser.add_argument(
        "--afs-source-run",
        default=(
            "/home/hpc/iwi5/iwi5305h/"
            "Masters_thesis_PR_LABS/"
            "outputs/chapter4/"
            "temp_90kv_afs_check/"
            "hybrid_runs/"
            "1764759_20260729_150855"
        ),
    )

    parser.add_argument(
        "--recovery-root",
        required=True,
    )

    parser.add_argument(
        "--afs-input-dir",
        default=(
            "/home/hpc/iwi5/iwi5305h/"
            "Masters_thesis_PR_LABS/"
            "outputs/chapter4/"
            "temp_90kv_afs_check/"
            "model_inputs/"
            "20260729_122907"
        ),
    )

    parser.add_argument(
        "--output-dir",
        required=True,
    )

    args = parser.parse_args()

    thesis_dir = Path(
        args.thesis_dir
    )

    source_run = Path(
        args.afs_source_run
    )

    recovery_root = Path(
        args.recovery_root
    )

    afs_input_dir = Path(
        args.afs_input_dir
    )

    output_dir = Path(
        args.output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    row_audit_path = (
        afs_input_dir
        / "temporary_prior_row_audit.csv"
    )

    event_tables: dict[
        str,
        pd.DataFrame,
    ] = {}

    window_tables: dict[
        str,
        pd.DataFrame,
    ] = {}

    audit_rows = []

    for experiment in AFS_EXPERIMENTS:
        prediction_root = (
            recovery_root
            / experiment
        )

        window = load_prediction_pool(
            prediction_root,
            experiment,
        )

        prior_view = (
            "1E"
            if "1E"
            in experiment
            else "2E"
        )

        window = merge_prior_audit(
            window,
            row_audit_path,
            prior_view,
        )

        audit_rows.append(
            validate_afs_window_table(
                window,
                experiment,
            )
        )

        event = aggregate_events(
            window,
            experiment,
        )

        window_tables[
            experiment
        ] = window

        event_tables[
            experiment
        ] = event

        write_csv(
            window,
            output_dir
            / f"{experiment}_window_oof.csv",
        )

        write_csv(
            event,
            output_dir
            / f"{experiment}_event_oof.csv",
        )

    combined_window = pd.concat(
        window_tables.values(),
        ignore_index=True,
    )

    combined_event = pd.concat(
        event_tables.values(),
        ignore_index=True,
    )

    write_csv(
        combined_window,
        output_dir
        / "90kv_afs_pooled_window_oof.csv",
    )

    write_csv(
        combined_event,
        output_dir
        / "90kv_afs_pooled_event_oof.csv",
    )

    event_metric_rows = []
    paired_rows = []
    tail_rows = []
    temporal_rows = []

    for (
        experiment,
        event,
    ) in event_tables.items():
        event_metric_rows.extend(
            predictor_metrics_rows(
                event,
                experiment,
            )
        )

        for (
            fold,
            fold_frame,
        ) in event.groupby(
            "fold"
        ):
            event_metric_rows.extend(
                predictor_metrics_rows(
                    fold_frame,
                    experiment,
                    scope="fold",
                    subgroup=str(
                        int(
                            fold
                        )
                    ),
                )
            )

        paired_rows.append(
            paired_prior_row(
                event,
                experiment,
            )
        )

        for (
            fold,
            fold_frame,
        ) in event.groupby(
            "fold"
        ):
            paired_rows.append(
                paired_prior_row(
                    fold_frame,
                    experiment,
                    scope="fold",
                    subgroup=str(
                        int(
                            fold
                        )
                    ),
                )
            )

        window = window_tables[
            experiment
        ]

        window_model = metric_dict(
            window[
                "y_true_pp"
            ],
            window[
                "y_pred_pp"
            ],
        )

        event_model = metric_dict(
            event[
                "y_true_pp"
            ],
            event[
                "y_pred_pp"
            ],
        )

        window_prior = metric_dict(
            window[
                "y_true_pp"
            ],
            window[
                "prior_pp"
            ],
        )

        event_prior = metric_dict(
            event[
                "y_true_pp"
            ],
            event[
                "prior_pp"
            ],
        )

        tail_rows.append(
            {
                "experiment": (
                    experiment
                ),
                "window_model_mae_pp": (
                    window_model[
                        "mae_pp"
                    ]
                ),
                "event_model_mae_pp": (
                    event_model[
                        "mae_pp"
                    ]
                ),
                "model_aggregation_gain_pp": (
                    window_model[
                        "mae_pp"
                    ]
                    - event_model[
                        "mae_pp"
                    ]
                ),
                "window_prior_mae_pp": (
                    window_prior[
                        "mae_pp"
                    ]
                ),
                "event_prior_mae_pp": (
                    event_prior[
                        "mae_pp"
                    ]
                ),
                "prior_aggregation_gain_pp": (
                    window_prior[
                        "mae_pp"
                    ]
                    - event_prior[
                        "mae_pp"
                    ]
                ),
                "events_helped_by_model_averaging_rate": float(
                    event[
                        "averaging_helped_model"
                    ].mean()
                ),
                "events_helped_by_prior_averaging_rate": float(
                    event[
                        "averaging_helped_prior"
                    ].mean()
                ),
                "mean_within_event_prediction_std_pp": float(
                    event[
                        "within_event_prediction_std_pp"
                    ].mean()
                ),
                "mean_within_event_prediction_range_pp": float(
                    event[
                        "within_event_prediction_range_pp"
                    ].mean()
                ),
                "p95_within_event_prediction_range_pp": float(
                    event[
                        "within_event_prediction_range_pp"
                    ].quantile(
                        0.95
                    )
                ),
                "event_model_p95_ae_pp": (
                    event_model[
                        "p95_ae_pp"
                    ]
                ),
                "event_model_p99_ae_pp": (
                    event_model[
                        "p99_ae_pp"
                    ]
                ),
                "event_model_cvar95_ae_pp": (
                    event_model[
                        "cvar95_ae_pp"
                    ]
                ),
            }
        )

        for (
            window_idx,
            group,
        ) in window.groupby(
            "window_idx"
        ):
            for (
                predictor,
                pred_col,
            ) in [
                (
                    "model",
                    "y_pred_pp",
                ),
                (
                    "prior",
                    "prior_pp",
                ),
            ]:
                temporal_rows.append(
                    {
                        "experiment": (
                            experiment
                        ),
                        "temporal_scope": (
                            "window_idx"
                        ),
                        "window_idx": int(
                            window_idx
                        ),
                        "predictor": (
                            predictor
                        ),
                        **metric_dict(
                            group[
                                "y_true_pp"
                            ],
                            group[
                                pred_col
                            ],
                        ),
                    }
                )

        temporal_rows.append(
            {
                "experiment": (
                    experiment
                ),
                "temporal_scope": (
                    "event_aggregation"
                ),
                "window_idx": np.nan,
                "predictor": (
                    "model"
                ),
                "n": int(
                    len(
                        event
                    )
                ),
                "mae_pp": (
                    event_model[
                        "mae_pp"
                    ]
                ),
                "aggregation_gain_pp": (
                    window_model[
                        "mae_pp"
                    ]
                    - event_model[
                        "mae_pp"
                    ]
                ),
                "events_helped_rate": float(
                    event[
                        "averaging_helped_model"
                    ].mean()
                ),
                "mean_within_event_std_pp": float(
                    event[
                        "within_event_prediction_std_pp"
                    ].mean()
                ),
                "mean_within_event_range_pp": float(
                    event[
                        "within_event_prediction_range_pp"
                    ].mean()
                ),
                "p95_within_event_range_pp": float(
                    event[
                        "within_event_prediction_range_pp"
                    ].quantile(
                        0.95
                    )
                ),
            }
        )

    event_metrics = pd.DataFrame(
        event_metric_rows
    )

    paired_metrics = pd.DataFrame(
        paired_rows
    )

    tail_metrics = pd.DataFrame(
        tail_rows
    )

    temporal_metrics = pd.DataFrame(
        temporal_rows
    )

    case_metrics = add_group_metrics(
        event_tables,
        "case",
        "fault_case",
    )

    line_metrics = add_group_metrics(
        event_tables,
        "y_fault_line",
        "protected_line",
    )

    for event in event_tables.values():
        event[
            "location_bin"
        ] = pd.cut(
            event[
                "y_fault_location"
            ].clip(
                0,
                100,
            ),
            bins=[
                -1e-9,
                20,
                40,
                60,
                80,
                100 + 1e-9,
            ],
            labels=[
                "0-20",
                "20-40",
                "40-60",
                "60-80",
                "80-100",
            ],
            include_lowest=True,
            right=True,
        ).astype(
            str
        )

    location_metrics = add_group_metrics(
        event_tables,
        "location_bin",
        "fault_location_bin",
    )

    (
        sfs_events,
        sfs_source_path,
    ) = load_sfs_events(
        thesis_dir
    )

    single_root = (
        thesis_dir
        / "outputs"
        / "chapter4"
        / "hybrid_single_ended"
        / "1751690_20260718_001736"
    )

    double_root = (
        thesis_dir
        / "outputs"
        / "chapter4"
        / "hybrid_double_ended"
        / "1751934_20260718_115538"
    )

    sfs_roots = {
        "C90-1E": (
            single_root
            / "C90-1E"
        ),
        "L90-1E": (
            single_root
            / "L90-1E"
        ),
        "C90-2E": (
            double_root
            / "C90-2E"
        ),
        "L90-2E": (
            double_root
            / "L90-2E"
        ),
    }

    config_audits = {}
    config_audit_rows = []

    for (
        afs_experiment,
        sfs_experiment,
    ) in SFS_MAP.items():
        sfs_signature = checkpoint_signature(
            sfs_roots[
                sfs_experiment
            ]
        )

        afs_signature = checkpoint_signature(
            source_run
            / afs_experiment
        )

        comparison = compare_checkpoint_signatures(
            sfs_signature,
            afs_signature,
        )

        config_audits[
            afs_experiment
        ] = comparison

        config_audit_rows.append(
            {
                "afs_experiment": (
                    afs_experiment
                ),
                "sfs_experiment": (
                    sfs_experiment
                ),
                "sfs_checkpoint": (
                    sfs_signature.get(
                        "checkpoint"
                    )
                ),
                "afs_checkpoint": (
                    afs_signature.get(
                        "checkpoint"
                    )
                ),
                **comparison,
            }
        )

    config_audit_df = pd.DataFrame(
        config_audit_rows
    )

    sfs_vs_afs = compare_sfs_afs(
        event_tables,
        sfs_events,
        config_audits,
    )

    physical_priors = physical_prior_table(
        thesis_dir,
        afs_input_dir,
        row_audit_path,
    )

    required_outputs = {
        "90kv_afs_event_metrics.csv": (
            event_metrics
        ),
        "90kv_afs_paired_metrics.csv": (
            paired_metrics
        ),
        "90kv_afs_tail_metrics.csv": (
            tail_metrics
        ),
        "90kv_afs_case_metrics.csv": (
            case_metrics
        ),
        "90kv_afs_line_metrics.csv": (
            line_metrics
        ),
        "90kv_afs_location_metrics.csv": (
            location_metrics
        ),
        "90kv_afs_temporal_metrics.csv": (
            temporal_metrics
        ),
        "90kv_sfs_vs_afs_comparison.csv": (
            sfs_vs_afs
        ),
        "90kv_afs_physical_prior_metrics.csv": (
            physical_priors
        ),
        "90kv_afs_recovery_integrity_audit.csv": (
            pd.DataFrame(
                audit_rows
            )
        ),
        "90kv_sfs_vs_afs_configuration_audit.csv": (
            config_audit_df
        ),
    }

    for (
        filename,
        frame,
    ) in required_outputs.items():
        write_csv(
            frame,
            output_dir
            / filename,
        )

    overall_event = event_metrics[
        (
            event_metrics[
                "scope"
            ] == "overall"
        )
        & (
            event_metrics[
                "predictor"
            ] == "model"
        )
    ]

    overall_paired = paired_metrics[
        paired_metrics[
            "scope"
        ] == "overall"
    ]

    overall_comparison = sfs_vs_afs[
        (
            sfs_vs_afs[
                "scope"
            ] == "overall"
        )
        & (
            sfs_vs_afs[
                "predictor"
            ] == "model"
        )
    ]

    comparison_audit_pass = bool(
        overall_comparison[
            "event_ids_identical"
        ].all()
        and overall_comparison[
            "targets_identical"
        ].all()
        and overall_comparison[
            "fold_assignments_identical"
        ].all()
    )

    prior_identity_pass = bool(
        overall_comparison[
            "event_averaged_priors_identical"
        ].all()
    )

    metrics_lookup = (
        overall_event.set_index(
            "experiment"
        )[
            "mae_pp"
        ].to_dict()
    )

    direct_1e_better = (
        metrics_lookup.get(
            "C90-1E-AFS-TMP",
            np.inf,
        )
        < metrics_lookup.get(
            "L90-1E-AFS-TMP",
            np.inf,
        )
    )

    bounded_2e_safer = (
        metrics_lookup.get(
            "L90-2E-AFS-TMP",
            np.inf,
        )
        <= metrics_lookup.get(
            "C90-2E-AFS-TMP",
            np.inf,
        )
    )

    two_paired = (
        overall_paired.set_index(
            "experiment"
        )[
            "absolute_mae_reduction_pp"
        ].to_dict()
    )

    replacement_suitable = (
        comparison_audit_pass
        and prior_identity_pass
        and bool(
            (
                overall_comparison[
                    "afs_minus_sfs_mae_pp"
                ]
                <= 0
            ).all()
        )
    )

    # The temporary one-ended prior was selected using
    # the complete AFS cohort and the true target.
    replacement_suitable = False

    if (
        direct_1e_better
        and bounded_2e_safer
    ):
        adaptation_conclusion = (
            "The AFS sensitivity analysis supports the existing "
            "interpretation: direct correction is stronger for the "
            "one-ended prior, while bounded adaptation is at least "
            "as safe for the two-ended prior."
        )

    elif (
        direct_1e_better
        and not bounded_2e_safer
    ):
        adaptation_conclusion = (
            "The AFS sensitivity analysis preserves the one-ended "
            "advantage of direct correction, but does not reproduce "
            "the expected two-ended safety advantage of bounded "
            "adaptation. The existing conclusion is therefore only "
            "partially supported by this temporary protocol."
        )

    else:
        adaptation_conclusion = (
            "The AFS sensitivity analysis does not cleanly reproduce "
            "the existing direct-versus-bounded pattern. It should be "
            "treated as a sensitivity result rather than a replacement "
            "conclusion."
        )

    report = f"""# 90 kV all-fault-start recovery and analysis

## Recovery status

Checkpoint-only test inference was used. The recovery consumed the original five checkpoints and the saved test-group assignments for each experiment. No model was retrained and the original checkpoints were not overwritten.

{markdown_table(pd.DataFrame(audit_rows), ['experiment', 'rows', 'events', 'min_windows_per_event', 'max_windows_per_event', 'mean_windows_per_event', 'fold_event_leakage_count'])}

## Pooled event-level headline metrics

These values are calculated directly from the pooled 9,022-event OOF files. They are not averages of fold metrics.

{markdown_table(overall_event, ['experiment', 'mae_pp', 'rmse_pp', 'median_ae_pp', 'bias_pp', 'p95_ae_pp', 'p99_ae_pp', 'cvar95_ae_pp'])}

## Paired model-versus-prior analysis

{markdown_table(overall_paired, ['experiment', 'prior_mae_pp', 'model_mae_pp', 'absolute_mae_reduction_pp', 'relative_mae_reduction', 'improved_rate', 'worsened_rate', 'improved_fold_count', 'bootstrap_reduction_ci_low_pp', 'bootstrap_reduction_ci_high_pp'])}

## Single-fault-start versus all-fault-start

The SFS source was `{sfs_source_path}`.

{markdown_table(overall_comparison, ['afs_experiment', 'sfs_experiment', 'sfs_mae_pp', 'afs_mae_pp', 'afs_minus_sfs_mae_pp', 'bootstrap_difference_ci_low_pp', 'bootstrap_difference_ci_high_pp', 'afs_improved_rate', 'afs_worsened_rate', 'event_ids_identical', 'targets_identical', 'fold_assignments_identical', 'event_averaged_priors_identical', 'model_configuration_status', 'seed_status', 'operator_feature_column_status'])}

## All-fault-start physical-prior table

{markdown_table(physical_priors, ['prior_name', 'rows', 'events', 'mean_windows_per_event', 'finite_count', 'invalid_count', 'fallback_count', 'mae_pp', 'rmse_pp', 'p95_ae_pp', 'p99_ae_pp'])}

## Protocol decision

**The all-fault-start results should not replace the single-fault-start results as the main Chapter 4 protocol.** The temporary one-ended AFS prior was selected by lowest MAE using the complete AFS cohort and its true targets. This makes it an exploratory, target-informed sensitivity prior rather than a clean deployment-style baseline. In addition, the SFS and AFS event-averaged priors are not required to be identical; the comparison CSV records the observed identity result. The AFS results remain useful as a temporal-aggregation sensitivity check.

Comparison audit for event IDs, targets and folds: **{'PASS' if comparison_audit_pass else 'FAIL'}**. Event-averaged prior identity across all pairs: **{'PASS' if prior_identity_pass else 'FAIL'}**. Replacement-suitability decision: **{'YES' if replacement_suitable else 'NO'}**.

## Direct versus bounded adaptation

{adaptation_conclusion}

For the two-ended AFS experiments, the pooled event-level MAE reductions versus the matched prior were `{two_paired.get('C90-2E-AFS-TMP', np.nan):.4f}` pp for direct correction and `{two_paired.get('L90-2E-AFS-TMP', np.nan):.4f}` pp for bounded adaptation. These values should be interpreted together with their bootstrap intervals in `90kv_afs_paired_metrics.csv`.

## Output files

- `90kv_afs_event_metrics.csv`
- `90kv_afs_paired_metrics.csv`
- `90kv_afs_tail_metrics.csv`
- `90kv_afs_case_metrics.csv`
- `90kv_afs_line_metrics.csv`
- `90kv_afs_location_metrics.csv`
- `90kv_afs_temporal_metrics.csv`
- `90kv_sfs_vs_afs_comparison.csv`
- `90kv_afs_physical_prior_metrics.csv`
- four experiment-level window OOF CSVs
- four experiment-level event OOF CSVs
- pooled window and event OOF CSVs
"""

    report_path = (
        output_dir
        / "90kv_afs_final_report.md"
    )

    report_path.write_text(
        report,
        encoding="utf-8",
    )

    manifest = []

    for path in sorted(
        output_dir.iterdir()
    ):
        if path.is_file():
            manifest.append(
                {
                    "file": (
                        path.name
                    ),
                    "bytes": (
                        path.stat().st_size
                    ),
                    "sha256": (
                        sha256(
                            path
                        )
                    ),
                }
            )

    (
        output_dir
        / "90kv_afs_output_manifest.json"
    ).write_text(
        json.dumps(
            {
                "status": "PASS",
                "source_run": str(
                    source_run
                ),
                "recovery_root": str(
                    recovery_root
                ),
                "afs_input_dir": str(
                    afs_input_dir
                ),
                "sfs_source": str(
                    sfs_source_path
                ),
                "files": manifest,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        "Analysis completed:",
        output_dir,
    )

    print(
        "Report:",
        report_path,
    )

    print(
        overall_event.to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()

