from __future__ import annotations
from typing import cast
from collections import Counter

import numpy as np
import pandas as pd
import hydra

from dl_psp.data.data_utils import load_windowed_dataset
from dl_psp.data.features import maybe_filter_features
from dl_psp.data.filters import build_valid_row_indices_hv_double_line_90kv
from psp_helper.config import MainConfig


# ----------------------------
# Physics helpers
# ----------------------------

def dft_phasor_1cycle(x: np.ndarray) -> complex:
    n = len(x)
    if n <= 0:
        return 0j
    k = np.arange(n, dtype=np.float64)
    w = np.exp(-1j * 2.0 * np.pi * k / n)
    return (2.0 / n) * np.sum(x.astype(np.float64) * w)


def symm_pos_seq(a: complex, b: complex, c: complex) -> complex:
    alpha = np.exp(1j * 2.0 * np.pi / 3.0)
    return (a + alpha * b + (alpha ** 2) * c) / 3.0


def k0_from_line(r1: float, x1: float, r0: float, x0: float) -> complex:
    z1 = complex(r1, x1)
    z0 = complex(r0, x0)
    if abs(z1) < 1e-12:
        return 0j
    return (z0 - z1) / z1


def compute_classical_distance(
    z_app: complex,
    r1: float,
    x1: float,
    line_len_km: float,
    mode: str = "abs",
) -> float:
    z1 = complex(r1, x1)
    if abs(z1) < 1e-12 or not np.isfinite(line_len_km):
        return np.nan

    if mode == "abs":
        d = (abs(z_app) / abs(z1)) * float(line_len_km)
    elif mode == "real":
        d = np.real(z_app / z1) * float(line_len_km)
    else:
        raise ValueError(f"Unknown distance mode: {mode}")

    return float(np.clip(d, 0.0, float(line_len_km)))


def compute_zapp_from_window(
    x_raw: np.ndarray,
    fs: float,
    f_nom: float,
    r1: float,
    x1: float,
    r0: float,
    x0: float,
    case: str,
    dt_start: float,
) -> tuple[complex, str, str]:
    if x_raw.shape[1] < 6:
        return np.nan + 1j * np.nan, "invalid", "too_few_channels"

    spc = int(np.rint(fs / f_nom))
    if spc <= 1:
        return np.nan + 1j * np.nan, "invalid", "invalid_spc"

    onset_idx = onset_idx_from_dt_start(dt_start, fs)

    pre_start = onset_idx - spc
    pre_end = onset_idx
    post_start = onset_idx
    post_end = onset_idx + spc

    if pre_start < 0:
        return np.nan + 1j * np.nan, case, "pre_window_out_of_bounds"
    if post_end > x_raw.shape[0]:
        return np.nan + 1j * np.nan, case, "post_window_out_of_bounds"

    Va_po = dft_phasor_1cycle(x_raw[post_start:post_end, 0])
    Vb_po = dft_phasor_1cycle(x_raw[post_start:post_end, 1])
    Vc_po = dft_phasor_1cycle(x_raw[post_start:post_end, 2])
    Ia_po = dft_phasor_1cycle(x_raw[post_start:post_end, 3])
    Ib_po = dft_phasor_1cycle(x_raw[post_start:post_end, 4])
    Ic_po = dft_phasor_1cycle(x_raw[post_start:post_end, 5])

    eps = 1e-9
    i0_po = (Ia_po + Ib_po + Ic_po) / 3.0
    k0 = k0_from_line(r1, x1, r0, x0)

    if case == "3ph":
        v1_po = symm_pos_seq(Va_po, Vb_po, Vc_po)
        i1_po = symm_pos_seq(Ia_po, Ib_po, Ic_po)
        z_po = v1_po / (i1_po + eps)
    elif case == "slg_a":
        z_po = Va_po / (Ia_po + k0 * i0_po + eps)
    elif case == "slg_b":
        z_po = Vb_po / (Ib_po + k0 * i0_po + eps)
    elif case == "slg_c":
        z_po = Vc_po / (Ic_po + k0 * i0_po + eps)
    elif case == "ll_ab":
        z_po = (Va_po - Vb_po) / ((Ia_po - Ib_po) + eps)
    elif case == "ll_bc":
        z_po = (Vb_po - Vc_po) / ((Ib_po - Ic_po) + eps)
    elif case == "ll_ca":
        z_po = (Vc_po - Va_po) / ((Ic_po - Ia_po) + eps)
    elif case == "llg_ab":
        z_po = (Va_po - Vb_po) / ((Ia_po - Ib_po) + k0 * i0_po + eps)
    elif case == "llg_bc":
        z_po = (Vb_po - Vc_po) / ((Ib_po - Ic_po) + k0 * i0_po + eps)
    elif case == "llg_ca":
        z_po = (Vc_po - Va_po) / ((Ic_po - Ia_po) + k0 * i0_po + eps)
    else:
        return np.nan + 1j * np.nan, case, "unknown_case"

    if not (np.isfinite(np.real(z_po)) and np.isfinite(np.imag(z_po))):
        return np.nan + 1j * np.nan, case, "zapp_not_finite"

    return z_po, case, "ok"


# ----------------------------
# Column / metadata helpers
# ----------------------------

def get_line_params_for_row(row: pd.Series) -> tuple[float, float, float, float, float]:
    line_name = str(row["y_fault_line"]).strip().lower()

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
        float(row[r1_col]),
        float(row[x1_col]),
        float(row[r0_col]),
        float(row[x0_col]),
        float(row[L_col]),
    )


def load_full_labels_csv(full_labels_path: str) -> pd.DataFrame:
    return pd.read_csv(full_labels_path, sep=";")


def attach_line_parameter_metadata(
    labels_df_used: pd.DataFrame,
    full_labels_path: str,
) -> pd.DataFrame:
    labels_full = load_full_labels_csv(full_labels_path)
    df = labels_df_used.copy()

    if "sample_id" not in df.columns:
        raise ValueError("Processed labels do not contain 'sample_id'.")
    if "rep_id" not in labels_full.columns:
        raise ValueError("Full labels.csv does not contain 'rep_id'.")

    needed_cols = [
        "rep_id",
        "line_1_2_a_length",
        "line_1_2_a_xline",
        "line_1_2_a_rline",
        "line_1_2_a_xline0",
        "line_1_2_a_rline0",
        "line_1_2_b_length",
        "line_1_2_b_xline",
        "line_1_2_b_rline",
        "line_1_2_b_xline0",
        "line_1_2_b_rline0",
        "line_2_3_a_length",
        "line_2_3_a_xline",
        "line_2_3_a_rline",
        "line_2_3_a_xline0",
        "line_2_3_a_rline0",
        "line_2_3_b_length",
        "line_2_3_b_xline",
        "line_2_3_b_rline",
        "line_2_3_b_xline0",
        "line_2_3_b_rline0",
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
    print(f"Merged dataframe shape: {out.shape}")
    print(f"Rows with missing line parameters after merge: {missing_rows}")

    if missing_rows > 0:
        raise ValueError(
            f"{missing_rows} rows could not be matched from processed labels to full labels "
            f"using sample_id -> rep_id"
        )

    return out


def extract_line_vi_channels(
    x_raw: np.ndarray,
    feature_names: list[str],
    fault_line: str,
) -> np.ndarray:
    line_map = {
        "Line_1_2_a": ("Bus_1", "Line_01_02A"),
        "Line_1_2_b": ("Bus_1", "Line_01_02B"),
        "Line_2_3_a": ("Bus_2", "Line_02_03A"),
        "Line_2_3_b": ("Bus_2", "Line_02_03B"),
    }

    if fault_line not in line_map:
        raise ValueError(f"Unknown fault_line: {fault_line}")

    bus_token, line_token = line_map[fault_line]
    name_to_idx = {name: i for i, name in enumerate(feature_names)}

    i1 = name_to_idx[f"{bus_token}_{line_token}_cur_L1_A"]
    i2 = name_to_idx[f"{bus_token}_{line_token}_cur_L2_A"]
    i3 = name_to_idx[f"{bus_token}_{line_token}_cur_L3_A"]
    v1 = name_to_idx[f"{bus_token}_{line_token}_vol_L1_V"]
    v2 = name_to_idx[f"{bus_token}_{line_token}_vol_L2_V"]
    v3 = name_to_idx[f"{bus_token}_{line_token}_vol_L3_V"]

    return np.stack(
        [
            x_raw[:, v1],
            x_raw[:, v2],
            x_raw[:, v3],
            x_raw[:, i1],
            x_raw[:, i2],
            x_raw[:, i3],
        ],
        axis=1,
    )


def onset_idx_from_dt_start(dt_start: float, fs: float) -> int:
    return int(np.rint((-float(dt_start)) * float(fs)))


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


def select_one_window_per_sample(
    df: pd.DataFrame,
    X_eval: np.ndarray,
    fs: float,
    f_nom: float,
) -> tuple[pd.DataFrame, np.ndarray]:
    if "sample_id" not in df.columns:
        raise ValueError("df must contain 'sample_id'")
    if "dt_start" not in df.columns:
        raise ValueError("df must contain 'dt_start'")
    if "status" not in df.columns:
        raise ValueError("df must contain 'status'")

    work = df.copy().reset_index(drop=True)
    work["_row_idx"] = np.arange(len(work))

    spc = int(np.rint(fs / f_nom))
    T = X_eval.shape[1]

    work = work.loc[
        work["status"].astype(str).str.lower() == "fault_start"
    ].copy()

    work["_onset_idx"] = np.rint((-work["dt_start"].astype(float)) * fs).astype(int)

    work["_valid_timing"] = (
        (work["_onset_idx"] >= spc) &
        (work["_onset_idx"] + spc <= T)
    )

    work = work.loc[work["_valid_timing"]].copy()

    target_idx = T // 2
    work["_timing_score"] = np.abs(work["_onset_idx"] - target_idx)

    work = work.sort_values(
        ["sample_id", "_timing_score", "window_idx"],
        ascending=[True, True, True],
    ).reset_index(drop=True)

    selected = work.groupby("sample_id", as_index=False).first()

    row_idx = selected["_row_idx"].to_numpy(dtype=int)
    X_sel = X_eval[row_idx]

    selected = selected.drop(
        columns=["_row_idx", "_onset_idx", "_valid_timing", "_timing_score"],
        errors="ignore",
    )

    return selected.reset_index(drop=True), X_sel


# ----------------------------
# Audit helpers
# ----------------------------

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


def get_line_vi_channel_names(
    feature_names: list[str],
    fault_line: str,
) -> list[str]:
    line_map = {
        "Line_1_2_a": ("Bus_1", "Line_01_02A"),
        "Line_1_2_b": ("Bus_1", "Line_01_02B"),
        "Line_2_3_a": ("Bus_2", "Line_02_03A"),
        "Line_2_3_b": ("Bus_2", "Line_02_03B"),
    }

    if fault_line not in line_map:
        raise ValueError(f"Unknown fault_line: {fault_line}")

    bus_token, line_token = line_map[fault_line]

    return [
        f"{bus_token}_{line_token}_vol_L1_V",
        f"{bus_token}_{line_token}_vol_L2_V",
        f"{bus_token}_{line_token}_vol_L3_V",
        f"{bus_token}_{line_token}_cur_L1_A",
        f"{bus_token}_{line_token}_cur_L2_A",
        f"{bus_token}_{line_token}_cur_L3_A",
    ]


def audit_case_and_formula_mapping(
    df: pd.DataFrame,
    feature_names: list[str],
    max_print: int | None = 50,
) -> pd.DataFrame:
    rows = []

    for i in range(len(df)):
        row = cast(pd.Series, df.iloc[i])

        case = derive_fault_case_from_processed_labels(row)
        formula = formula_name_for_case(case)

        try:
            channel_names = get_line_vi_channel_names(
                feature_names=feature_names,
                fault_line=str(row["y_fault_line"]),
            )
            channel_ok = True
            channel_error = ""
        except Exception as e:
            channel_names = []
            channel_ok = False
            channel_error = str(e)

        a = int(row["y_phase_A"])
        b = int(row["y_phase_B"])
        c = int(row["y_phase_C"])
        grounded = int(row["y_is_grounded"])

        active = []
        if a == 1:
            active.append("a")
        if b == 1:
            active.append("b")
        if c == 1:
            active.append("c")

        if len(active) == 3:
            expected_case = "3ph"
        elif len(active) == 1 and grounded == 1:
            expected_case = f"slg_{active[0]}"
        elif len(active) == 2 and grounded == 0:
            pair = "".join(active)
            if pair == "ac":
                pair = "ca"
            expected_case = f"ll_{pair}"
        elif len(active) == 2 and grounded == 1:
            pair = "".join(active)
            if pair == "ac":
                pair = "ca"
            expected_case = f"llg_{pair}"
        else:
            expected_case = "invalid"

        case_ok = (case == expected_case)

        rows.append(
            {
                "row_idx": i,
                "sample_id": row["sample_id"],
                "status": row["status"],
                "fault_line": row["y_fault_line"],
                "fault_location_pct": float(row["y_fault_location"]),
                "dt_start": float(row["dt_start"]),
                "y_phase_A": a,
                "y_phase_B": b,
                "y_phase_C": c,
                "y_is_grounded": grounded,
                "derived_case": case,
                "expected_case": expected_case,
                "case_ok": case_ok,
                "formula": formula,
                "channel_ok": channel_ok,
                "channel_error": channel_error,
                "channels": " | ".join(channel_names),
            }
        )

    audit_df = pd.DataFrame(rows)

    print("\n===== CASE / FORMULA AUDIT =====")
    print(f"Total rows audited: {len(audit_df)}")
    print(f"Rows with case mismatch: {(~audit_df['case_ok']).sum()}")
    print(f"Rows with channel mapping error: {(~audit_df['channel_ok']).sum()}")

    if max_print is not None:
        print("\nFirst audited rows:")
        print(audit_df.head(max_print).to_string(index=False))

    return audit_df


# ----------------------------
# Main
# ----------------------------

@hydra.main(
    version_base=None,
    config_path="../third_party/dl_fault_repo/config",
    config_name="main-config.yaml",
)
def main(config: MainConfig) -> None:
    print("Loading processed windows...")
    X, labels_df, meta = load_windowed_dataset(config)

    include_groups = config.training.feature_groups_include
    materialize = config.training.materialize_feature_filters
    X_used, _feature_indices_for_ds = maybe_filter_features(
        X=X,
        meta=meta,
        include_groups=include_groups,
        materialize=materialize,
    )

    valid_row_idx = build_valid_row_indices_hv_double_line_90kv(
        labels_df, "y_fault_location"
    )
    if valid_row_idx is None:
        labels_df_used = labels_df.reset_index(drop=True)
        X_valid = X_used
    else:
        labels_df_used = labels_df.iloc[valid_row_idx].reset_index(drop=True)
        X_valid = X_used[valid_row_idx]

    print(f"Loaded X_valid shape: {X_valid.shape}")
    print(f"labels_df_used shape: {labels_df_used.shape}")

    line_filter = None
    loc_min = None
    loc_max = None
    f_nom = 50.0
    y_col = "y_fault_location"

    full_labels_path = "/home/vault/iwi5/iwi5305h/new_dataset_90kv/labels.csv"
    df = attach_line_parameter_metadata(
        labels_df_used=labels_df_used,
        full_labels_path=full_labels_path,
    )
    X_eval = X_valid

    print(f"Merged dataframe shape: {df.shape}")
    print("Unique y_fault_line values:")
    print(df["y_fault_line"].value_counts(dropna=False).to_string())

    if line_filter is not None:
        mask = df["y_fault_line"].astype(str) == str(line_filter)
        df = df.loc[mask].reset_index(drop=True)
        X_eval = X_eval[mask.to_numpy()]
        print(f"After line filter '{line_filter}': {len(df)} samples")

    y_pct = df[y_col].astype(float).to_numpy()

    if loc_min is not None:
        mask = y_pct >= loc_min
        df = df.loc[mask].reset_index(drop=True)
        X_eval = X_eval[mask]
        y_pct = y_pct[mask]

    if loc_max is not None:
        mask = y_pct <= loc_max
        df = df.loc[mask].reset_index(drop=True)
        X_eval = X_eval[mask]
        y_pct = y_pct[mask]

    T_full = X_eval.shape[1]
    window_s = float(config.window_extraction.window_length)
    fs = T_full / window_s
    print(f"Inferred fs: {fs:.3f} Hz from T={T_full}, window={window_s:.6f}s")

    df, X_eval = select_one_window_per_sample(
        df=df,
        X_eval=X_eval,
        fs=fs,
        f_nom=f_nom,
    )
    print(f"Subset size after selecting one window per sample_id: {len(df)}")
    print("Unique sample_id count after selection:", df["sample_id"].nunique())
    print("Status distribution after selection:")
    print(df["status"].value_counts(dropna=False).to_string())

    spc = int(np.rint(fs / f_nom))
    onset_idx = np.rint((-df["dt_start"].astype(float)) * fs).astype(int)

    print("\ndt_start stats after selection:")
    print(df["dt_start"].describe().to_string())

    print("\nOnset index stats after selection:")
    print(pd.Series(onset_idx).describe().to_string())

    invalid_timing = ((onset_idx < spc) | (onset_idx + spc > X_eval.shape[1])).sum()
    print(f"\nRows violating timing after selection: {int(invalid_timing)}")

    feature_names = meta["feature_names"]

    audit_df = audit_case_and_formula_mapping(
        df=df,
        feature_names=feature_names,
        max_print=40,
    )

    audit_path = "/home/vault/iwi5/iwi5305h/run_kol_formula_audit.csv"
    audit_df.to_csv(audit_path, index=False)
    print(f"\nSaved audit CSV to: {audit_path}")

    rows = []
    reason_counts = Counter()

    for i in range(len(df)):
        row = cast(pd.Series, df.iloc[i])
        x_raw_full = np.asarray(X_eval[i], dtype=np.float32)

        case = derive_fault_case_from_processed_labels(row)
        if case == "invalid":
            reason_counts["invalid_case_from_processed_labels"] += 1
            continue

        x_vi = extract_line_vi_channels(
            x_raw=x_raw_full,
            feature_names=feature_names,
            fault_line=str(row["y_fault_line"]),
        )

        r1, x1, r0, x0, L_km = get_line_params_for_row(row)

        z_app, case, reason = compute_zapp_from_window(
            x_raw=x_vi,
            fs=fs,
            f_nom=float(f_nom),
            r1=r1,
            x1=x1,
            r0=r0,
            x0=x0,
            case=case,
            dt_start=float(row["dt_start"]),
        )

        if reason != "ok":
            reason_counts[reason] += 1
            continue

        z1 = complex(r1, x1)

        ratio_real = float(np.real(z_app / (z1 + 1e-12)))
        ratio_abs = float(abs(z_app) / (abs(z1) + 1e-12))

        d_phys_raw_real_km = float(ratio_real * L_km)
        d_phys_clipped_real_km = float(np.clip(d_phys_raw_real_km, 0.0, L_km))
        d_phys_real_pct = float(100.0 * d_phys_clipped_real_km / L_km)

        d_phys_raw_abs_km = float(ratio_abs * L_km)
        d_phys_clipped_abs_km = float(np.clip(d_phys_raw_abs_km, 0.0, L_km))
        d_phys_abs_pct = float(100.0 * d_phys_clipped_abs_km / L_km)

        rows.append(
            {
                "sample_id": row["sample_id"],
                "window_idx": row["window_idx"],
                "status": row["status"],
                "dt_start": float(row["dt_start"]),
                "y_fault_line": row["y_fault_line"],
                "y_fault_location": float(row[y_col]),
                "case": case,
                "r1": r1,
                "x1": x1,
                "r0": r0,
                "x0": x0,
                "line_len_km": L_km,
                "z_app_real": float(np.real(z_app)),
                "z_app_imag": float(np.imag(z_app)),
                "ratio_real": ratio_real,
                "ratio_abs": ratio_abs,
                "d_phys_raw_real_km": d_phys_raw_real_km,
                "d_phys_clipped_real_km": d_phys_clipped_real_km,
                "d_phys_real_pct": d_phys_real_pct,
                "d_phys_raw_abs_km": d_phys_raw_abs_km,
                "d_phys_clipped_abs_km": d_phys_clipped_abs_km,
                "d_phys_abs_pct": d_phys_abs_pct,
            }
        )

    feat_df = pd.DataFrame(rows)

    print("\n===== Operator feature export =====")
    print(f"Rows exported: {len(feat_df)}")

    if reason_counts:
        print("\nSkipped rows:")
        for k, v in reason_counts.most_common():
            print(f"{k}: {v}")

    if len(feat_df) == 0:
        raise RuntimeError("No operator features were produced.")

    print("\nOperator feature preview:")
    print(
        feat_df[
            [
                "sample_id",
                "window_idx",
                "y_fault_line",
                "y_fault_location",
                "case",
                "d_phys_real_pct",
                "d_phys_abs_pct",
                "ratio_real",
                "ratio_abs",
            ]
        ].head(10).to_string(index=False)
    )

    out_path = "/home/vault/iwi5/iwi5305h/kol_operator_features.csv"
    feat_df.to_csv(out_path, index=False)
    print(f"\nSaved operator features to: {out_path}")


if __name__ == "__main__":
    main()
