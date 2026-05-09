from __future__ import annotations

import numpy as np
import pandas as pd

from KOL.common.constants import CASE_TO_IDX


def derive_fault_case_from_processed_labels(row: pd.Series) -> str:
    a = int(row["y_phase_A"])
    b = int(row["y_phase_B"])
    c = int(row["y_phase_C"])
    grounded = int(row["y_is_grounded"])

    phases = []
    if a == 1:
        phases.append("a")
    if b == 1:
        phases.append("b")
    if c == 1:
        phases.append("c")

    if len(phases) == 3:
        return "3ph"

    if len(phases) == 1:
        if grounded == 1:
            return f"slg_{phases[0]}"
        return "invalid"

    if len(phases) == 2:
        pair = "".join(phases)
        if pair == "ac":
            pair = "ca"
        if grounded == 1:
            return f"llg_{pair}"
        return f"ll_{pair}"

    return "invalid"


def formula_name_for_case(case: str) -> str:
    mapping = {
        "3ph": "positive-sequence: Z = V1 / I1",
        "slg_a": "single-line-ground A: Z = Va / (Ia + k0*I0)",
        "slg_b": "single-line-ground B: Z = Vb / (Ib + k0*I0)",
        "slg_c": "single-line-ground C: Z = Vc / (Ic + k0*I0)",
        "ll_ab": "line-line AB: Z = (Va - Vb) / (Ia - Ib)",
        "ll_bc": "line-line BC: Z = (Vb - Vc) / (Ib - Ic)",
        "ll_ca": "line-line CA: Z = (Vc - Va) / (Ic - Ia)",
        "llg_ab": "double-line-ground AB: Z = (Va - Vb) / ((Ia - Ib) + k0*I0)",
        "llg_bc": "double-line-ground BC: Z = (Vb - Vc) / ((Ib - Ic) + k0*I0)",
        "llg_ca": "double-line-ground CA: Z = (Vc - Va) / ((Ic - Ia) + k0*I0)",
    }
    return mapping.get(case, "unknown")


def build_case_index(labels_df_used: pd.DataFrame) -> np.ndarray:
    if "case" in labels_df_used.columns:
        case_series = labels_df_used["case"].astype(str)
    else:
        case_series = labels_df_used.apply(derive_fault_case_from_processed_labels, axis=1)

    case_idx = case_series.map(CASE_TO_IDX)
    if case_idx.isna().any():
        bad = case_series[case_idx.isna()].unique().tolist()
        raise ValueError(f"Unknown case labels encountered: {bad}")

    return case_idx.astype(np.int64).to_numpy()


def idx_to_case_mapping() -> dict[int, str]:
    return {v: k for k, v in CASE_TO_IDX.items()}
