import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# ✅ Correct import path + correct constant names
from preprocessing.build_events_features import (
    OUT_X_NPY, OUT_Y_NPY, OUT_CSV,
    dft_phasor_1cycle, compute_classical_distance,
    F_NOM, WINDOW_CYCLES, PRE_MS
)

# ----------------------------
# Waveform-based correction model (Δpct)
# ----------------------------
class WaveDeltaCNN(nn.Module):
    def __init__(self, n_channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(n_channels, 16, kernel_size=9, padding=4),
            nn.ReLU(),
            nn.MaxPool1d(2),  # 512 -> 256

            nn.Conv1d(16, 32, kernel_size=9, padding=4),
            nn.ReLU(),
            nn.MaxPool1d(2),  # 256 -> 128

            nn.Conv1d(32, 64, kernel_size=9, padding=4),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),  # -> (B, 64, 1)
        )
        self.head = nn.Sequential(
            nn.Linear(64 + 1, 32),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(32, 1),
        )

    def forward(self, x, classic_pct):
        x = x.transpose(1, 2)
        z = self.net(x).squeeze(-1)            # (B,64)
        z = torch.cat([z, classic_pct.unsqueeze(1)], dim=1)  # (B,65)
        return self.head(z).squeeze(-1)


def compute_classic_pct_from_window(
    X_win: np.ndarray,
    fs_est: float,
    r_line: float,
    x_line: float,
    L_km: float
) -> float:
    """
    Compute classical distance percentage using a 1-cycle phasor just BEFORE fault inside the window.

    X_win: (T, C), channels assumed [Va, Ia]
    Fault location in window is approx at sample pre_samples.
    """
    pre_samples = int(round((PRE_MS / 1000.0) * fs_est))
    samples_per_cycle = int(round(fs_est / F_NOM))
    win_len = WINDOW_CYCLES * samples_per_cycle

    pre_end = pre_samples
    pre_start = pre_end - win_len

    if pre_start < 0 or pre_end > X_win.shape[0]:
        return np.nan

    Va_pre = X_win[pre_start:pre_end, 0]  # Va
    Ia_pre = X_win[pre_start:pre_end, 3]  # Ia (since channels are Vabc Iabc)

    V_ph = dft_phasor_1cycle(Va_pre)
    I_ph = dft_phasor_1cycle(Ia_pre)
    Z_app = V_ph / (I_ph + 1e-9)

    d_km = compute_classical_distance(Z_app, r_line, x_line, L_km)
    if not np.isfinite(d_km) or L_km <= 1e-12:
        return np.nan

    return float(100.0 * d_km / L_km)


def normalize_waveforms_train_stats(X_train: np.ndarray, X_test: np.ndarray):
    """
    Channel-wise z-normalization using training statistics.
    X_* shapes: (N, T, C)
    """
    mu = X_train.mean(axis=(0, 1), keepdims=True)
    sigma = X_train.std(axis=(0, 1), keepdims=True) + 1e-6
    return (X_train - mu) / sigma, (X_test - mu) / sigma


def plot_one(i, X, meta):
    import matplotlib.pyplot as plt
    fs = float(meta.loc[i, "fs_est"])
    pre_samples = int(round((PRE_MS/1000.0) * fs))

    plt.figure()
    plt.plot(X[i, :, 0], label="Va")
    plt.plot(X[i, :, 1], label="Ia")
    plt.axvline(pre_samples, linestyle="--", label="fault (expected)")
    plt.title(f"Sample {i} | rep_id={int(meta.loc[i,'rep_id'])} | fs={fs:.2f}")
    plt.legend()
    plt.show()



def main():
    # ----------------------------
    # Load tensors + metadata
    # ----------------------------
    X = np.load(OUT_X_NPY).astype(np.float32)  # (N, T, C)
    y = np.load(OUT_Y_NPY).astype(np.float32)  # (N,)
    meta = pd.read_csv(OUT_CSV)

    assert len(X) == len(y) == len(meta), "Mismatch between X, y, and meta lengths."
    print(f"Loaded X: {X.shape}  y: {y.shape}  meta: {meta.shape}")

    plot_one(0, X, meta)

    # ----------------------------
    # Classical baseline (classic_pct) computed from the window
    # ----------------------------
    classic_pct = []
    for i in range(len(X)):
        fs_est = float(meta.loc[i, "fs_est"])
        r_line = float(meta.loc[i, "r_line"])
        x_line = float(meta.loc[i, "x_line"])
        L_km = float(meta.loc[i, "L_km"])

        cp = compute_classic_pct_from_window(X[i], fs_est, r_line, x_line, L_km)
        classic_pct.append(cp)

    classic_pct = np.array(classic_pct, dtype=np.float32)

    # Drop invalid
    ok = np.isfinite(classic_pct)
    X = X[ok]
    y = y[ok]
    classic_pct = classic_pct[ok]
    meta = meta.loc[ok].reset_index(drop=True)

    mae_classic = mean_absolute_error(y, classic_pct)
    print(f"Classical MAE (from window KO): {mae_classic:.2f} %")

    # ----------------------------
    # Train/test split (IMPORTANT: split using waveform X, not scalar feats)
    # ----------------------------
    X_train, X_test, y_train, y_test, c_train, c_test = train_test_split(
        X, y, classic_pct, test_size=0.2, random_state=42
    )

    # Normalize waveform channels using training stats
    X_train, X_test = normalize_waveforms_train_stats(X_train, X_test)

    # Torch datasets
    train_ds = TensorDataset(
        torch.from_numpy(X_train),
        torch.from_numpy(y_train),
        torch.from_numpy(c_train)
    )
    test_ds = TensorDataset(
        torch.from_numpy(X_test),
        torch.from_numpy(y_test),
        torch.from_numpy(c_test)
    )

    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
    test_loader  = DataLoader(test_ds, batch_size=128, shuffle=False)

    # ----------------------------
    # Model predicts Δpct, final prediction = classic_pct + Δpct
    # ----------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = WaveDeltaCNN(n_channels=X.shape[2]).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.SmoothL1Loss(beta=1.0)

    mae_kol = None
    best_mae = float("inf")
    best_state = None
    best_epoch = None

    # ----------------------------
    # Train
    # ----------------------------
    for epoch in range(1, 51):
        model.train()
        for xb, yb, cb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            cb = cb.to(device)

            opt.zero_grad()
            delta = model(xb, cb)
            y_hat = cb + delta
            loss = loss_fn(y_hat, yb)
            loss.backward()
            opt.step()

        if epoch in {1, 5, 10, 20, 30, 50}:
            model.eval()
            preds, trues = [], []
            with torch.no_grad():
                for xb, yb, cb in test_loader:
                    xb = xb.to(device)
                    yb = yb.to(device)
                    cb = cb.to(device)

                    delta = model(xb, cb)
                    y_hat = cb + delta
                    preds.append(y_hat.cpu().numpy())
                    trues.append(yb.cpu().numpy())

            preds = np.concatenate(preds)
            trues = np.concatenate(trues)
            mae_kol = mean_absolute_error(trues, preds)
            if mae_kol < best_mae:
                best_mae = mae_kol
                best_epoch = epoch
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

            print(f"Best KOL MAE: {best_mae:.2f} % at epoch {best_epoch}")
            print(f"Epoch {epoch:2d} | KOL Test MAE: {mae_kol:.2f} %")

    print(f"\nFinal Classical MAE: {mae_classic:.2f} %")
    print(f"Final KOL MAE:       {mae_kol:.2f} %" if mae_kol is not None else "Final KOL MAE: n/a")


if __name__ == "__main__":
    main()
