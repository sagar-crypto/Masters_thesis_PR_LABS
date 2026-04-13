from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from KOL.models.kol_residual_models import apply_kol_prediction_rule


def train_kol_case_k0(
    model,
    train_loader,
    val_loader,
    optimizer,
    device,
    logger,
    prediction_mode: str = "ground_only_mul",
    epochs: int = 20,
    patience: int = 15,
):
    criterion = nn.MSELoss()
    best_val = float("inf")
    best_state = None
    bad_epochs = 0

    for epoch in range(epochs):
        model.train()
        train_losses = []

        for x_seq, d_prior, c_idx, op_feat, y in train_loader:
            x_seq = x_seq.to(device)
            d_prior = d_prior.to(device).float()
            c_idx = c_idx.to(device).long()
            op_feat = op_feat.to(device).float()
            y = y.to(device).float()

            optimizer.zero_grad()
            residual = model(x_seq, c_idx, d_prior, op_feat)
            pred = apply_kol_prediction_rule(d_prior, c_idx, residual, prediction_mode)

            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses = []
        residual_vals = []

        with torch.no_grad():
            for x_seq, d_prior, c_idx, op_feat, y in val_loader:
                x_seq = x_seq.to(device)
                d_prior = d_prior.to(device).float()
                c_idx = c_idx.to(device).long()
                op_feat = op_feat.to(device).float()
                y = y.to(device).float()

                residual = model(x_seq, c_idx, d_prior, op_feat)
                pred = apply_kol_prediction_rule(d_prior, c_idx, residual, prediction_mode)

                val_losses.append(criterion(pred, y).item())
                residual_vals.append(residual.cpu().numpy())

        mean_train = float(np.mean(train_losses)) if train_losses else float("nan")
        mean_val = float(np.mean(val_losses)) if val_losses else float("inf")
        mean_residual = float(np.mean(np.concatenate(residual_vals))) if residual_vals else float("nan")

        logger.info(
            "epoch %d | train_loss=%.6f | val_loss=%.6f | mean_residual=%.6f",
            epoch + 1, mean_train, mean_val, mean_residual
        )

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


def evaluate_kol_case_k0(model, test_loader, device, logger, prediction_mode: str = "ground_only_mul"):
    model.eval()
    y_true, y_pred, residual_all, dprior_all, case_all = [], [], [], [], []

    with torch.no_grad():
        for x_seq, d_prior, c_idx, op_feat, y in test_loader:
            x_seq = x_seq.to(device)
            d_prior = d_prior.to(device).float()
            c_idx = c_idx.to(device).long()
            op_feat = op_feat.to(device).float()

            residual = model(x_seq, c_idx, d_prior, op_feat)
            pred = apply_kol_prediction_rule(d_prior, c_idx, residual, prediction_mode)

            y_pred.append(pred.cpu().numpy())
            y_true.append(y.numpy())
            residual_all.append(residual.cpu().numpy())
            dprior_all.append(d_prior.cpu().numpy())
            case_all.append(c_idx.cpu().numpy())

    y_true = np.concatenate(y_true).astype(np.float64)
    y_pred = np.concatenate(y_pred).astype(np.float64)
    residual_all = np.concatenate(residual_all).astype(np.float64)
    dprior_all = np.concatenate(dprior_all).astype(np.float64)
    case_all = np.concatenate(case_all).astype(np.int64)

    mse = float(np.mean((y_pred - y_true) ** 2))
    mae = float(np.mean(np.abs(y_pred - y_true)))
    rmse = float(np.sqrt(mse))
    prior_mae = float(np.mean(np.abs(dprior_all - y_true)))
    prior_rmse = float(np.sqrt(np.mean((dprior_all - y_true) ** 2)))

    logger.info(
        "Residual stats | mean=%.6f | std=%.6f | min=%.6f | max=%.6f",
        float(residual_all.mean()),
        float(residual_all.std()),
        float(residual_all.min()),
        float(residual_all.max()),
    )
    logger.info(
        "Prior-only comparison | mae=%.6f | rmse=%.6f || model | mae=%.6f | rmse=%.6f",
        prior_mae, prior_rmse, mae, rmse
    )

    return {
        "loss": mse,
        "mae": mae,
        "rmse": rmse,
        "prior_mae": prior_mae,
        "prior_rmse": prior_rmse,
    }, y_true, y_pred, residual_all, dprior_all, case_all


def predict_on_kol_case_k0(model, loader, device, prediction_mode: str = "ground_only_mul"):
    model.eval()
    y_true, y_pred, residual_all, dprior_all, case_all = [], [], [], [], []

    with torch.no_grad():
        for x_seq, d_prior, c_idx, op_feat, y in loader:
            x_seq = x_seq.to(device)
            d_prior = d_prior.to(device).float()
            c_idx = c_idx.to(device).long()
            op_feat = op_feat.to(device).float()

            residual = model(x_seq, c_idx, d_prior, op_feat)
            pred = apply_kol_prediction_rule(d_prior, c_idx, residual, prediction_mode)

            y_pred.append(pred.cpu().numpy())
            y_true.append(y.numpy())
            residual_all.append(residual.cpu().numpy())
            dprior_all.append(d_prior.cpu().numpy())
            case_all.append(c_idx.cpu().numpy())

    return (
        np.concatenate(y_true).astype(np.float64),
        np.concatenate(y_pred).astype(np.float64),
        np.concatenate(residual_all).astype(np.float64),
        np.concatenate(dprior_all).astype(np.float64),
        np.concatenate(case_all).astype(np.int64),
    )
