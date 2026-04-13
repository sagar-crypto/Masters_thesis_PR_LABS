from __future__ import annotations

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from KOL.common.constants import ALL_CASE_TO_IDX
from KOL.common.signal_ops import dft_phasor_1cycle_torch, classical_alpha_torch
from KOL.common.physics_core import k0_from_line_torch, compute_classical_distance_real_torch
from psp_helper.utils.logging import get_logger

logger = get_logger(__name__)


def compute_distance_with_learned_params(
    batch: dict,
    delta_alpha_re: torch.Tensor,
    delta_alpha_im: torch.Tensor,
    delta_k0_re: torch.Tensor,
    delta_k0_im: torch.Tensor,
    fs: float,
    f_nom: float = 50.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    x_seq = batch["x_seq"]
    case_idx = batch["case_idx"]
    r1 = batch["r1"]
    x1 = batch["x1"]
    r0 = batch["r0"]
    x0 = batch["x0"]
    line_len_km = batch["line_len_km"]
    dt_start = batch["dt_start"]

    B, T, F = x_seq.shape
    if F != 6:
        raise ValueError(f"Expected 6 VI channels, got {F}")

    spc = int(np.rint(fs / f_nom))
    onset_idx = torch.round((-dt_start) * fs).long()

    d_pred_list, d_classical_list = [], []
    alpha_learned_list, k0_learned_list = [], []

    eps = 1e-9
    alpha_classical = classical_alpha_torch(x_seq.device)
    idx_to_case = {v: k for k, v in ALL_CASE_TO_IDX.items()}

    for b in range(B):
        t0 = int(onset_idx[b].item())
        post_start, post_end = t0, t0 + spc

        if post_start < 0 or post_end > T:
            zero_f = torch.tensor(0.0, dtype=torch.float32, device=x_seq.device)
            zero_c = torch.tensor(0.0 + 0.0j, dtype=torch.complex64, device=x_seq.device)
            d_pred_list.append(zero_f)
            d_classical_list.append(zero_f)
            alpha_learned_list.append(alpha_classical)
            k0_learned_list.append(zero_c)
            continue

        win = x_seq[b, post_start:post_end, :]
        Va = dft_phasor_1cycle_torch(win[:, 0])
        Vb = dft_phasor_1cycle_torch(win[:, 1])
        Vc = dft_phasor_1cycle_torch(win[:, 2])
        Ia = dft_phasor_1cycle_torch(win[:, 3])
        Ib = dft_phasor_1cycle_torch(win[:, 4])
        Ic = dft_phasor_1cycle_torch(win[:, 5])

        I0 = (Ia + Ib + Ic) / 3.0

        k0_classical = k0_from_line_torch(
            r1[b].unsqueeze(0),
            x1[b].unsqueeze(0),
            r0[b].unsqueeze(0),
            x0[b].unsqueeze(0),
        ).squeeze(0)

        alpha_learned = alpha_classical + torch.complex(delta_alpha_re[b], delta_alpha_im[b])
        k0_learned = k0_classical + torch.complex(delta_k0_re[b], delta_k0_im[b])

        case_name = idx_to_case[int(case_idx[b].item())]

        if case_name == "3ph":
            v1_c = (Va + alpha_classical * Vb + (alpha_classical ** 2) * Vc) / 3.0
            i1_c = (Ia + alpha_classical * Ib + (alpha_classical ** 2) * Ic) / 3.0
            z_app_classical = v1_c / (i1_c + eps)

            v1_l = (Va + alpha_learned * Vb + (alpha_learned ** 2) * Vc) / 3.0
            i1_l = (Ia + alpha_learned * Ib + (alpha_learned ** 2) * Ic) / 3.0
            z_app_learned = v1_l / (i1_l + eps)

        elif case_name == "slg_a":
            z_app_classical = Va / (Ia + k0_classical * I0 + eps)
            z_app_learned = Va / (Ia + k0_learned * I0 + eps)
        elif case_name == "slg_b":
            z_app_classical = Vb / (Ib + k0_classical * I0 + eps)
            z_app_learned = Vb / (Ib + k0_learned * I0 + eps)
        elif case_name == "slg_c":
            z_app_classical = Vc / (Ic + k0_classical * I0 + eps)
            z_app_learned = Vc / (Ic + k0_learned * I0 + eps)
        elif case_name == "ll_ab":
            z_app_classical = (Va - Vb) / ((Ia - Ib) + eps)
            z_app_learned = z_app_classical
        elif case_name == "ll_bc":
            z_app_classical = (Vb - Vc) / ((Ib - Ic) + eps)
            z_app_learned = z_app_classical
        elif case_name == "ll_ca":
            z_app_classical = (Vc - Va) / ((Ic - Ia) + eps)
            z_app_learned = z_app_classical
        elif case_name == "llg_ab":
            z_app_classical = (Va - Vb) / ((Ia - Ib) + eps)
            z_app_learned = z_app_classical
        elif case_name == "llg_bc":
            z_app_classical = (Vb - Vc) / ((Ib - Ic) + eps)
            z_app_learned = z_app_classical
        elif case_name == "llg_ca":
            z_app_classical = (Vc - Va) / ((Ic - Ia) + eps)
            z_app_learned = z_app_classical
        else:
            raise ValueError(f"Unsupported case: {case_name}")

        d_km_classical = compute_classical_distance_real_torch(z_app_classical, r1[b], x1[b], line_len_km[b])
        d_km_learned = compute_classical_distance_real_torch(z_app_learned, r1[b], x1[b], line_len_km[b])

        d_classical_norm = d_km_classical / (line_len_km[b] + eps)
        d_pred_norm = d_km_learned / (line_len_km[b] + eps)

        d_pred_list.append(d_pred_norm)
        d_classical_list.append(d_classical_norm)
        alpha_learned_list.append(alpha_learned)
        k0_learned_list.append(k0_learned)

    return (
        torch.stack(d_pred_list, dim=0),
        torch.stack(d_classical_list, dim=0),
        torch.stack(alpha_learned_list, dim=0),
        torch.stack(k0_learned_list, dim=0),
    )


def train_k0_model(
    model,
    train_loader,
    val_loader,
    optimizer,
    device,
    fs: float,
    epochs: int = 50,
    patience: int = 10,
    use_scheduler: bool = True,
    scheduler_factor: float = 0.5,
    scheduler_patience: int = 3,
    scheduler_min_lr: float = 1e-6,
):
    criterion = nn.MSELoss()
    best_val = float("inf")
    best_state = None
    bad_epochs = 0

    scheduler = None
    if use_scheduler:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=float(scheduler_factor),
            patience=int(scheduler_patience),
            min_lr=float(scheduler_min_lr),
        )

    for epoch in range(epochs):
        model.train()
        train_losses = []

        for batch in train_loader:
            batch = move_batch_to_device(batch, device)
            optimizer.zero_grad()

            delta_alpha_re, delta_alpha_im, delta_k0_re, delta_k0_im = model(
                batch["x_seq"], batch["case_idx"]
            )

            d_pred, _, _, _ = compute_distance_with_learned_params(
                batch=batch,
                delta_alpha_re=delta_alpha_re,
                delta_alpha_im=delta_alpha_im,
                delta_k0_re=delta_k0_re,
                delta_k0_im=delta_k0_im,
                fs=fs,
                f_nom=50.0,
            )

            loss = criterion(d_pred, batch["y"])
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                batch = move_batch_to_device(batch, device)
                delta_alpha_re, delta_alpha_im, delta_k0_re, delta_k0_im = model(
                    batch["x_seq"], batch["case_idx"]
                )

                d_pred, _, _, _ = compute_distance_with_learned_params(
                    batch=batch,
                    delta_alpha_re=delta_alpha_re,
                    delta_alpha_im=delta_alpha_im,
                    delta_k0_re=delta_k0_re,
                    delta_k0_im=delta_k0_im,
                    fs=fs,
                    f_nom=50.0,
                )

                val_losses.append(criterion(d_pred, batch["y"]).item())

        mean_train = float(np.mean(train_losses)) if train_losses else float("nan")
        mean_val = float(np.mean(val_losses)) if val_losses else float("inf")
        current_lr = float(optimizer.param_groups[0]["lr"])

        logger.info(
            "epoch %d | train_loss=%.6f | val_loss=%.6f | lr=%.8f",
            epoch + 1, mean_train, mean_val, current_lr
        )

        if scheduler is not None:
            scheduler.step(mean_val)

        min_delta = 1e-4
        if mean_val < (best_val - min_delta):
            best_val = mean_val
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                logger.info(
                    "Early stopping at epoch %d | best_val=%.6f | current_val=%.6f",
                    epoch + 1, best_val, mean_val
                )
                break

    if best_state is not None:
        model.load_state_dict(best_state)


def compute_casewise_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    case_idx: np.ndarray,
    y_classical: np.ndarray | None = None,
) -> pd.DataFrame:
    rows = []
    idx_to_case = {v: k for k, v in ALL_CASE_TO_IDX.items()}

    for cid in sorted(np.unique(case_idx)):
        mask = case_idx == cid
        if int(mask.sum()) == 0:
            continue

        yt = y_true[mask]
        yp = y_pred[mask]
        mae = float(np.mean(np.abs(yp - yt)))
        mse = float(np.mean((yp - yt) ** 2))
        rmse = float(np.sqrt(mse))

        row = {
            "case_idx": int(cid),
            "case": idx_to_case.get(int(cid), f"unknown_{cid}"),
            "n": int(mask.sum()),
            "mae_learned": mae,
            "rmse_learned": rmse,
            "mse_learned": mse,
        }

        if y_classical is not None:
            yc = y_classical[mask]
            mae_c = float(np.mean(np.abs(yc - yt)))
            mse_c = float(np.mean((yc - yt) ** 2))
            rmse_c = float(np.sqrt(mse_c))
            row["mae_classical"] = mae_c
            row["rmse_classical"] = rmse_c
            row["mse_classical"] = mse_c
            row["mae_gain_vs_classical"] = mae_c - mae

        rows.append(row)

    return pd.DataFrame(rows).sort_values("case").reset_index(drop=True)


def evaluate_operator_model(
    model,
    test_loader,
    device,
    fs: float,
):
    model.eval()
    y_true, y_pred, y_classical, case_idx_all = [], [], [], []
    delta_alpha_re_all, delta_alpha_im_all, delta_k0_re_all, delta_k0_im_all = [], [], [], []
    alpha_learned_re, alpha_learned_im, k0_learned_re, k0_learned_im = [], [], [], []

    with torch.no_grad():
        for batch in test_loader:
            batch = move_batch_to_device(batch, device)

            delta_alpha_re, delta_alpha_im, delta_k0_re, delta_k0_im = model(
                batch["x_seq"], batch["case_idx"]
            )

            d_pred, d_classical, alpha_learned, k0_learned = compute_distance_with_learned_params(
                batch=batch,
                delta_alpha_re=delta_alpha_re,
                delta_alpha_im=delta_alpha_im,
                delta_k0_re=delta_k0_re,
                delta_k0_im=delta_k0_im,
                fs=fs,
                f_nom=50.0,
            )

            y_true.append(batch["y"].cpu().numpy())
            y_pred.append(d_pred.cpu().numpy())
            y_classical.append(d_classical.cpu().numpy())
            case_idx_all.append(batch["case_idx"].cpu().numpy())

            delta_alpha_re_all.append(delta_alpha_re.cpu().numpy())
            delta_alpha_im_all.append(delta_alpha_im.cpu().numpy())
            delta_k0_re_all.append(delta_k0_re.cpu().numpy())
            delta_k0_im_all.append(delta_k0_im.cpu().numpy())

            alpha_learned_re.append(torch.real(alpha_learned).cpu().numpy())
            alpha_learned_im.append(torch.imag(alpha_learned).cpu().numpy())
            k0_learned_re.append(torch.real(k0_learned).cpu().numpy())
            k0_learned_im.append(torch.imag(k0_learned).cpu().numpy())

    y_true = np.concatenate(y_true).astype(np.float64)
    y_pred = np.concatenate(y_pred).astype(np.float64)
    y_classical = np.concatenate(y_classical).astype(np.float64)
    case_idx_all = np.concatenate(case_idx_all).astype(np.int64)

    mse = float(np.mean((y_pred - y_true) ** 2))
    mae = float(np.mean(np.abs(y_pred - y_true)))
    rmse = float(np.sqrt(mse))

    classical_mse = float(np.mean((y_classical - y_true) ** 2))
    classical_mae = float(np.mean(np.abs(y_classical - y_true)))
    classical_rmse = float(np.sqrt(classical_mse))

    case_metrics_df = compute_casewise_regression_metrics(
        y_true=y_true,
        y_pred=y_pred,
        case_idx=case_idx_all,
        y_classical=y_classical,
    )

    logger.info("Per-case test metrics:")
    logger.info("\n%s", case_metrics_df.to_string(index=False))

    aux = {
        "case_idx": case_idx_all,
        "y_classical": y_classical,
        "abs_error_classical": np.abs(y_classical - y_true),
        "abs_error_learned": np.abs(y_pred - y_true),
        "delta_alpha_re": np.concatenate(delta_alpha_re_all).astype(np.float64),
        "delta_alpha_im": np.concatenate(delta_alpha_im_all).astype(np.float64),
        "delta_k0_re": np.concatenate(delta_k0_re_all).astype(np.float64),
        "delta_k0_im": np.concatenate(delta_k0_im_all).astype(np.float64),
        "alpha_learned_re": np.concatenate(alpha_learned_re).astype(np.float64),
        "alpha_learned_im": np.concatenate(alpha_learned_im).astype(np.float64),
        "k0_learned_re": np.concatenate(k0_learned_re).astype(np.float64),
        "k0_learned_im": np.concatenate(k0_learned_im).astype(np.float64),
    }

    logger.info(
        "Classical vs learned | classical_mae=%.6f | learned_mae=%.6f | gain=%.6f",
        classical_mae, mae, classical_mae - mae,
    )

    return {
        "loss": mse,
        "mae": mae,
        "rmse": rmse,
        "classical_loss": classical_mse,
        "classical_mae": classical_mae,
        "classical_rmse": classical_rmse,
    }, y_true, y_pred, aux, case_metrics_df


def save_k0_predictions_csv(
    out_dir: str,
    fold_idx: int,
    idx_test: np.ndarray,
    labels_df: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    aux: dict,
) -> str:
    os.makedirs(out_dir, exist_ok=True)

    df_meta = labels_df.iloc[idx_test].reset_index(drop=True).copy()
    df_meta["y_true"] = y_true
    df_meta["y_pred"] = y_pred
    df_meta["abs_error"] = np.abs(y_pred - y_true)

    idx_to_case = {v: k for k, v in ALL_CASE_TO_IDX.items()}
    if "case_idx" in aux:
        df_meta["case_idx"] = aux["case_idx"]
        df_meta["case"] = [idx_to_case.get(int(i), f"unknown_{i}") for i in aux["case_idx"]]

    for k, v in aux.items():
        df_meta[k] = v

    pred_path = os.path.join(out_dir, f"fold{fold_idx}_learned_k0_predictions.csv")
    df_meta.to_csv(pred_path, index=False)
    return pred_path


def move_batch_to_device(batch: dict, device: torch.device) -> dict:
    out = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device)
        else:
            out[k] = v
    return out
