import os
import sys
from pathlib import Path

# Add project root (one level up from this script) to Python’s import path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
import numpy as np
import pandas as pd
from config import MAIN_DATA_PATH, CACHE_FILE_PATH

# -------------------------
# CONFIG
# -------------------------
LABELS_CSV = f"{MAIN_DATA_PATH}/labels.csv"
NPY_DIR    = CACHE_FILE_PATH  # folder with replica_*.npy
OUT_CSV    = "/home/vault/iwi5/iwi5305h/event_features_quick.csv"
OUT_X_NPY = "/home/vault/iwi5/iwi5305h/X_faultwin.npy"
OUT_Y_NPY = "/home/vault/iwi5/iwi5305h/y_faultwin.npy"

PRE_MS  = 40.0
POST_MS = 40.0
EPS_I   = 1e-9


# Process only these rep_ids for now
REP_IDS = [1, 7]
N_EVENTS = 200

F_NOM = 50.0
WINDOW_CYCLES = 1

# Channel mapping for your old .npy (based on your first-row dump pattern)
# You MUST confirm these indices match the signals you want.
TIME_COL = 0

# Cub_1\Bus1Line_1_2_a_ai_exp_ct_vt
IA_COL = 7
IB_COL = 8
IC_COL = 9

VA_COL = 10
VB_COL = 11
VC_COL = 12



def estimate_fs(time_arr: np.ndarray) -> float:
    dt = np.diff(time_arr)
    dt = dt[np.isfinite(dt)]
    dt = dt[dt > 0]
    if len(dt) == 0:
        raise ValueError("Cannot estimate fs from time array.")
    return 1.0 / float(np.median(dt))


def dft_phasor_1cycle(x: np.ndarray) -> complex:
    N = len(x)
    n = np.arange(N, dtype=np.float64)
    w = np.exp(-1j * 2.0 * np.pi * n / N)
    return (2.0 / N) * np.sum(x.astype(np.float64) * w)


def compute_classical_distance(Z_app: complex, r_line: float, x_line: float, L: float) -> float:
    z_line = complex(r_line, x_line)
    if abs(z_line) < 1e-12:
        return np.nan
    d = (abs(Z_app) / abs(z_line)) * L
    return float(np.clip(d, 0.0, L))


def process_one_event(row: pd.Series, npy_path: str, line_prefix: str = "line_1_2_a"):
    data = np.load(npy_path)  # shape (T, C)
    t = data[:, TIME_COL].astype(np.float64)

    fs = estimate_fs(t)

    t_fault = float(row["t_evnt_start"])
    fault_idx = int(np.argmin(np.abs(t - t_fault)))

    pre_samp  = int(round((PRE_MS / 1000.0) * fs))
    post_samp = int(round((POST_MS / 1000.0) * fs))

    start = fault_idx - pre_samp
    end   = fault_idx + post_samp  # exclusive

    if start < 0 or end > len(t) or end <= start:
        return None

    # Windowed raw signals (model input)
    Va_win = data[start:end, VA_COL].astype(np.float32)
    Vb_win = data[start:end, VB_COL].astype(np.float32)
    Vc_win = data[start:end, VC_COL].astype(np.float32)

    Ia_win = data[start:end, IA_COL].astype(np.float32)
    Ib_win = data[start:end, IB_COL].astype(np.float32)
    Ic_win = data[start:end, IC_COL].astype(np.float32)

    # (T, 6) in a fixed order: Vabc then Iabc
    X_win = np.stack([Va_win, Vb_win, Vc_win, Ia_win, Ib_win, Ic_win], axis=-1)


    # Target
    y = float(row["sc_location"])  # in percent

    # Optional: keep some KO features too (still useful for debugging / hybrid models)
    # You can compute a 1-cycle phasor pre/post later if you want.

    # Line params (for meta/debug; not required for the waveform tensor)
    L = float(row[f"{line_prefix}_length"])
    r_line = float(row[f"{line_prefix}_rline"])
    x_line = float(row[f"{line_prefix}_xline"])

    meta = {
        "rep_id": int(row["rep_id"]),
        "fs_est": float(fs),
        "t_fault": float(t_fault),
        "fault_idx": int(fault_idx),
        "start_idx": int(start),
        "end_idx": int(end),
        "win_len_samples": int(end - start),
        "pre_ms": float(PRE_MS),
        "post_ms": float(POST_MS),
        "L_km": float(L),
        "r_line": float(r_line),
        "x_line": float(x_line),
        "sc_location_pct": float(y),
        "n_channels": int(X_win.shape[1]),
    }

    return X_win, y, meta



def main():
    labels = pd.read_csv(LABELS_CSV, sep=";").copy()
    labels = labels.sort_values("rep_id").head(N_EVENTS)

    X_list = []
    y_list = []
    meta_rows = []

    for _, row in labels.iterrows():
        rep_id = int(row["rep_id"])
        npy_path = os.path.join(NPY_DIR, f"replica_{rep_id}.npy")
        if not os.path.exists(npy_path):
            print(f"Missing: {npy_path}")
            continue

        out = process_one_event(row, npy_path)
        if out is None:
            continue

        X_win, y, meta = out
        X_list.append(X_win)
        y_list.append(y)
        meta_rows.append(meta)

    if len(X_list) == 0:
        print("No events processed. Check indices, fault times, and bounds.")
        return

    # Ensure all windows have identical length (they should, if fs is consistent)
    lens = [x.shape[0] for x in X_list]
    if len(set(lens)) != 1:
        # If fs differs slightly, window length may differ by 1-2 samples.
        # We'll trim to the minimum length to stack safely.
        min_len = min(lens)
        X_list = [x[:min_len, :] for x in X_list]
        for m in meta_rows:
            m["win_len_samples"] = min_len

    X = np.stack(X_list, axis=0).astype(np.float32)  # (N, T, C)
    y = np.array(y_list, dtype=np.float32)           # (N,)

    np.save(OUT_X_NPY, X)
    np.save(OUT_Y_NPY, y)

    df_meta = pd.DataFrame(meta_rows)
    df_meta.to_csv(OUT_CSV, index=False)

    print(f"Saved X: {OUT_X_NPY}  shape={X.shape} dtype={X.dtype}")
    print(f"Saved y: {OUT_Y_NPY}  shape={y.shape} dtype={y.dtype}")
    print(f"Saved meta: {OUT_CSV} rows={len(df_meta)}")



if __name__ == "__main__":
    main()
