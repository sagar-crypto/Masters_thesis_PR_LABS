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

# Process only these rep_ids for now
REP_IDS = [1, 7]
N_EVENTS = 200

F_NOM = 50.0
WINDOW_CYCLES = 1

# Channel mapping for your old .npy (based on your first-row dump pattern)
# You MUST confirm these indices match the signals you want.
TIME_COL = 0

# If your .npy layout is [t, Va, Vb, Vc, Ia, Ib, Ic, ...]
IA_COL = 7   # Sekundärstrom L1 in A (Bus1Line_1_2_a)
VA_COL = 10  # Sekundärspannung L1 in V (Bus1Line_1_2_a)


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
    samples_per_cycle = int(round(fs / F_NOM))
    win_len = WINDOW_CYCLES * samples_per_cycle

    t_fault = float(row["t_evnt_start"])
    fault_idx = int(np.argmin(np.abs(t - t_fault)))

    start = fault_idx
    end = start + win_len
    if end > len(t):
        return None

    Va = data[start:end, VA_COL].astype(np.float64)
    Ia = data[start:end, IA_COL].astype(np.float64)

    V_ph = dft_phasor_1cycle(Va)
    I_ph = dft_phasor_1cycle(Ia)

    Z_app = V_ph / (I_ph + 1e-9)

    L = float(row[f"{line_prefix}_length"])
    r_line = float(row[f"{line_prefix}_rline"])
    x_line = float(row[f"{line_prefix}_xline"])

    d_classic = compute_classical_distance(Z_app, r_line, x_line, L)
    classic_pct = 100.0 * d_classic / L

    return {
        "rep_id": int(row["rep_id"]),
        "fs_est": fs,
        "t_fault": t_fault,
        "L_km": L,
        "Z_real": float(np.real(Z_app)),
        "Z_imag": float(np.imag(Z_app)),
        "d_classic_km": float(d_classic),
        "classic_pct": float(classic_pct),
        "sc_location_pct": float(row["sc_location"]),  # target (%)
    }


def main():
    labels = pd.read_csv(LABELS_CSV, sep=";").copy()
    labels = labels.sort_values("rep_id").head(N_EVENTS)

    rows = []
    for _, row in labels.iterrows():
        rep_id = int(row["rep_id"])
        npy_path = os.path.join(NPY_DIR, f"replica_{rep_id}.npy")
        if not os.path.exists(npy_path):
            print(f"Missing: {npy_path}")
            continue

        out = process_one_event(row, npy_path)
        if out is not None:
            rows.append(out)

    df_out = pd.DataFrame(rows)
    df_out.to_csv(OUT_CSV, index=False)
    print(df_out)

    # quick “value” number for supervisor (on 2 samples):
    if len(df_out) > 0:
        mae = np.mean(np.abs(df_out["classic_pct"] - df_out["sc_location_pct"]))
        print(f"Classical MAE on {len(df_out)} samples: {mae:.2f} % of line length")


if __name__ == "__main__":
    main()
