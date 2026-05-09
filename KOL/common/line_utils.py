from __future__ import annotations

import pandas as pd

from KOL.common.constants import HV110KV_LINE_PARAMS
from typing import cast


def get_line_params_for_row(
    row: pd.Series,
    topology: str,
) -> tuple[float, float, float, float, float]:
    fault_line = str(row["y_fault_line"]).strip()

    if topology == "hv_double_line_110kv":
        if fault_line not in HV110KV_LINE_PARAMS:
            raise ValueError(f"Unknown 110kV y_fault_line value: {fault_line}")

        params = HV110KV_LINE_PARAMS[fault_line]
        missing_keys = [k for k, v in params.items() if v is None]
        if missing_keys:
            raise ValueError(f"Missing constant parameters for {fault_line}: {missing_keys}")

        return (
            float(params["r1"]),
            float(params["x1"]),
            float(params["r0"]),
            float(params["x0"]),
            float(params["length"]),
        )

    line_name = fault_line.lower()
    mapping = {
        "line_1_2_a": (
            "line_1_2_a_rline",
            "line_1_2_a_xline",
            "line_1_2_a_rline0",
            "line_1_2_a_xline0",
            "line_1_2_a_length",
        ),
        "line_1_2_b": (
            "line_1_2_b_rline",
            "line_1_2_b_xline",
            "line_1_2_b_rline0",
            "line_1_2_b_xline0",
            "line_1_2_b_length",
        ),
        "line_2_3_a": (
            "line_2_3_a_rline",
            "line_2_3_a_xline",
            "line_2_3_a_rline0",
            "line_2_3_a_xline0",
            "line_2_3_a_length",
        ),
        "line_2_3_b": (
            "line_2_3_b_rline",
            "line_2_3_b_xline",
            "line_2_3_b_rline0",
            "line_2_3_b_xline0",
            "line_2_3_b_length",
        ),
    }

    if line_name not in mapping:
        raise ValueError(f"Unknown y_fault_line value: {row['y_fault_line']}")

    r1_col, x1_col, r0_col, x0_col, L_col = mapping[line_name]
    return (
        float(cast(float, row[r1_col])),
        float(cast(float, row[x1_col])),
        float(cast(float, row[r0_col])),
        float(cast(float, row[x0_col])),
        float(cast(float, row[L_col])),
    )

def load_full_labels_csv(full_labels_path: str) -> pd.DataFrame:
    return pd.read_csv(full_labels_path, sep=";")


def attach_line_parameter_metadata(
    labels_df_used: pd.DataFrame,
    full_labels_path: str | None,
    topology: str,
) -> pd.DataFrame:
    df = labels_df_used.copy()

    if topology == "hv_double_line_110kv":
        return df

    if full_labels_path is None:
        raise ValueError("full_labels_path must be provided for this topology")

    labels_full = load_full_labels_csv(full_labels_path)

    if "sample_id" not in df.columns:
        raise ValueError("Processed labels do not contain 'sample_id'.")
    if "rep_id" not in labels_full.columns:
        raise ValueError("Full labels.csv does not contain 'rep_id'.")

    needed_cols = [
        "rep_id",
        "line_1_2_a_length", "line_1_2_a_xline", "line_1_2_a_rline", "line_1_2_a_xline0", "line_1_2_a_rline0",
        "line_1_2_b_length", "line_1_2_b_xline", "line_1_2_b_rline", "line_1_2_b_xline0", "line_1_2_b_rline0",
        "line_2_3_a_length", "line_2_3_a_xline", "line_2_3_a_rline", "line_2_3_a_xline0", "line_2_3_a_rline0",
        "line_2_3_b_length", "line_2_3_b_xline", "line_2_3_b_rline", "line_2_3_b_xline0", "line_2_3_b_rline0",
    ]

    missing = [c for c in needed_cols if c not in labels_full.columns]
    if missing:
        raise ValueError(f"Missing columns in full labels.csv: {missing}")

    labels_params = labels_full[needed_cols].drop_duplicates(subset=["rep_id"])

    out = df.merge(
        labels_params,
        left_on="sample_id",
        right_on="rep_id",
        how="left",
    )

    missing_rows = int(out["line_1_2_a_length"].isna().sum())
    if missing_rows > 0:
        raise ValueError(
            f"{missing_rows} rows could not be matched from processed labels to full labels "
            f"using sample_id -> rep_id"
        )

    return out


def get_default_side_tokens(fault_line: str) -> tuple[str, str]:
    line_map = {
        "Line_1_2_a": ("Bus_1", "Line_01_02A"),
        "Line_1_2_b": ("Bus_1", "Line_01_02B"),
        "Line_2_3_a": ("Bus_2", "Line_02_03A"),
        "Line_2_3_b": ("Bus_2", "Line_02_03B"),
        "MainLn1-2A": ("Bus_1", "MainLn1-2A"),
        "MainLn1-2B": ("Bus_1", "MainLn1-2B"),
        "MainLn2-3A": ("Bus_2", "MainLn2-3A"),
        "MainLn2-3B": ("Bus_2", "MainLn2-3B"),
    }
    if fault_line not in line_map:
        raise ValueError(f"Unknown fault_line: {fault_line}")
    return line_map[fault_line]
