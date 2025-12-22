import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

CSV_PATH = "/home/vault/iwi5/iwi5305h/event_features_quick.csv" 

# ----------------------------
# Simple correction model
# ----------------------------
class DeltaModel(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)  # Δpct


def main():
    df = pd.read_csv(CSV_PATH)

    # --- Ensure we have classic_pct ---
    if "classic_pct" not in df.columns:
        if "d_classic" in df.columns and "L_km" in df.columns:
            df["classic_pct"] = 100.0 * df["d_classic"] / df["L_km"]
        elif "d_classic_km" in df.columns and "L_km" in df.columns:
            df["classic_pct"] = 100.0 * df["d_classic_km"] / df["L_km"]
        else:
            raise ValueError("Need classic_pct OR (d_classic and L_km) in CSV.")

    # Target
    y = df["sc_location_pct"].to_numpy(dtype=np.float32)

    # Features for the NN (keep minimal, stable)
    feature_cols = ["classic_pct", "Z_real", "Z_imag"]
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in CSV: {missing}")

    X = df[feature_cols].to_numpy(dtype=np.float32)

    # Baseline MAE (classical only)
    mae_classic = mean_absolute_error(y, df["classic_pct"].to_numpy(dtype=np.float32))
    print(f"Classical MAE: {mae_classic:.2f} %")

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Scale inputs (important for NN stability)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_test  = scaler.transform(X_test).astype(np.float32)

    # Torch datasets
    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    test_ds  = TensorDataset(torch.from_numpy(X_test),  torch.from_numpy(y_test))

    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)
    test_loader  = DataLoader(test_ds, batch_size=512, shuffle=False)

    # Model predicts Δpct, final prediction = classic_pct + Δpct
    model = DeltaModel(in_dim=X.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.L1Loss()  # MAE

    # We need classic_pct too for reconstruction of final y_hat
    # Easiest: include classic_pct as feature 0, and scale it; but for adding back,
    # we want the ORIGINAL (unscaled) classic_pct from test set.
    classic_pct_test = df.loc[df.index.isin(df.index[-len(X_test):]), "classic_pct"].to_numpy(dtype=np.float32)
    # The line above is not correct because train_test_split shuffles indices.
    # So we rebuild classic_pct arrays using the split by storing them alongside X.
    # Quick fix: re-split with classic_pct carried.

    # --- redo split with classic_pct carried ---
    classic_pct = df["classic_pct"].to_numpy(dtype=np.float32)
    X_train, X_test, y_train, y_test, c_train, c_test = train_test_split(
        X, y, classic_pct, test_size=0.2, random_state=42
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_test  = scaler.transform(X_test).astype(np.float32)

    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train), torch.from_numpy(c_train))
    test_ds  = TensorDataset(torch.from_numpy(X_test),  torch.from_numpy(y_test),  torch.from_numpy(c_test))

    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)
    test_loader  = DataLoader(test_ds, batch_size=512, shuffle=False)

    # Train a few epochs
    for epoch in range(1, 31):
        model.train()
        for xb, yb, cb in train_loader:
            opt.zero_grad()
            delta = model(xb)           # Δpct
            y_hat = cb + delta          # classic_pct + Δpct
            loss = loss_fn(y_hat, yb)
            loss.backward()
            opt.step()

        if epoch in {1, 5, 10, 20, 30}:
            model.eval()
            preds, trues = [], []
            with torch.no_grad():
                for xb, yb, cb in test_loader:
                    delta = model(xb)
                    y_hat = cb + delta
                    preds.append(y_hat.numpy())
                    trues.append(yb.numpy())
            preds = np.concatenate(preds)
            trues = np.concatenate(trues)
            mae_kol = mean_absolute_error(trues, preds)
            print(f"Epoch {epoch:2d} | KOL Test MAE: {mae_kol:.2f} %")

    # Final report
    print(f"\nFinal Classical MAE: {mae_classic:.2f} %")
    print(f"Final KOL MAE:       {mae_kol:.2f} %")


if __name__ == "__main__":
    main()
