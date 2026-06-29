from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from KOL.models.kol_residual_models import apply_kol_prediction_rule, apply_kol_prediction_rule_unclipped


def _compute_clip_stats(pred_unclipped: torch.Tensor) -> dict[str, float]:
    return {
        "clip_low_frac": float((pred_unclipped < 0.0).float().mean().item()),
        "clip_high_frac": float((pred_unclipped > 1.0).float().mean().item()),
        "pred_unclipped_min": float(pred_unclipped.min().item()),
        "pred_unclipped_max": float(pred_unclipped.max().item()),
        "pred_unclipped_mean": float(pred_unclipped.mean().item()),
    }


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
    use_unclipped_loss = False
    best_val = float("inf")
    best_state = None
    bad_epochs = 0

    for epoch in range(epochs):
        model.train()
        train_losses = []

        for batch in train_loader:
            batch_dict = _move_kol_batch_to_device(batch, device)

            d_prior = batch_dict["d_prior"]
            c_idx = batch_dict["c_idx"]
            y = batch_dict["y"]

            optimizer.zero_grad()

            residual = _call_kol_model_from_prepared_batch(model, batch_dict)

            if use_unclipped_loss:
                pred_for_loss = apply_kol_prediction_rule_unclipped(
                    d_phys_prior=d_prior,
                    case_idx=c_idx,
                    residual=residual,
                    mode=prediction_mode,
                )
            else:
                pred_for_loss = apply_kol_prediction_rule(
                    d_phys_prior=d_prior,
                    case_idx=c_idx,
                    residual=residual,
                    mode=prediction_mode,
                )

            loss = criterion(pred_for_loss, y)
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

        model.eval()
        val_losses = []
        residual_vals = []

        with torch.no_grad():
            for batch in val_loader:
                batch_dict = _move_kol_batch_to_device(batch, device)

                d_prior = batch_dict["d_prior"]
                c_idx = batch_dict["c_idx"]
                y = batch_dict["y"]

                residual = _call_kol_model_from_prepared_batch(model, batch_dict)

                if use_unclipped_loss:
                    pred_for_loss = apply_kol_prediction_rule_unclipped(
                    d_phys_prior=d_prior,
                    case_idx=c_idx,
                    residual=residual,
                    mode=prediction_mode,
                )
                else:
                    pred_for_loss = apply_kol_prediction_rule(
                        d_phys_prior=d_prior,
                        case_idx=c_idx,
                        residual=residual,
                        mode=prediction_mode,
                    )

                val_losses.append(criterion(pred_for_loss, y).item())
                residual_vals.append(residual.detach().cpu().numpy())

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
    clip_stats_batches = []

    with torch.no_grad():
        for batch in test_loader:
            batch_dict = _move_kol_batch_to_device(batch, device)

            d_prior = batch_dict["d_prior"]
            c_idx = batch_dict["c_idx"]
            y = batch_dict["y"]

            residual = _call_kol_model_from_prepared_batch(model, batch_dict)
            pred_unclipped = apply_kol_prediction_rule_unclipped(
                d_prior,
                c_idx,
                residual,
                prediction_mode,
            )

            clip_stats_batches.append(_compute_clip_stats(pred_unclipped))

            pred = apply_kol_prediction_rule(
                d_prior,
                c_idx,
                residual,
                prediction_mode,
            )

            y_pred.append(pred.detach().cpu().numpy())
            y_true.append(y.detach().cpu().numpy())
            residual_all.append(residual.detach().cpu().numpy())
            dprior_all.append(d_prior.detach().cpu().numpy())
            case_all.append(c_idx.detach().cpu().numpy())

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
    if clip_stats_batches:
        clip_stats = {
            "clip_low_frac": float(np.mean([s["clip_low_frac"] for s in clip_stats_batches])),
            "clip_high_frac": float(np.mean([s["clip_high_frac"] for s in clip_stats_batches])),
            "pred_unclipped_min": float(np.min([s["pred_unclipped_min"] for s in clip_stats_batches])),
            "pred_unclipped_max": float(np.max([s["pred_unclipped_max"] for s in clip_stats_batches])),
            "pred_unclipped_mean": float(np.mean([s["pred_unclipped_mean"] for s in clip_stats_batches])),
        }
    else:
        clip_stats = {
            "clip_low_frac": 0.0,
            "clip_high_frac": 0.0,
            "pred_unclipped_min": 0.0,
            "pred_unclipped_max": 0.0,
            "pred_unclipped_mean": 0.0,
        }

    metrics = {
        "loss": mse,
        "mae": mae,
        "rmse": rmse,
        "prior_mae": prior_mae,
        "prior_rmse": prior_rmse,
    }

    metrics.update(clip_stats)

    return metrics, y_true, y_pred, residual_all, dprior_all, case_all


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

def _move_kol_batch_to_device(batch, device):
    """
    Supports both batch formats.

    Single-input KOL batch:
        x_seq, d_prior, c_idx, op_feat, y

    Dual-input KOL batch:
        x_waveform, x_phasor, d_prior, c_idx, op_feat, y
    """
    if len(batch) == 5:
        x_seq, d_prior, c_idx, op_feat, y = batch

        x_seq = x_seq.to(device).float()
        d_prior = d_prior.to(device).float()
        c_idx = c_idx.to(device).long()
        op_feat = op_feat.to(device).float()
        y = y.to(device).float()

        if op_feat.ndim == 1:
            op_feat = op_feat.unsqueeze(-1)

        return {
            "is_dual": False,
            "x_seq": x_seq,
            "x_waveform": None,
            "x_phasor": None,
            "d_prior": d_prior,
            "c_idx": c_idx,
            "op_feat": op_feat,
            "y": y,
        }

    if len(batch) == 6:
        x_waveform, x_phasor, d_prior, c_idx, op_feat, y = batch

        x_waveform = x_waveform.to(device).float()
        x_phasor = x_phasor.to(device).float()
        d_prior = d_prior.to(device).float()
        c_idx = c_idx.to(device).long()
        op_feat = op_feat.to(device).float()
        y = y.to(device).float()

        if op_feat.ndim == 1:
            op_feat = op_feat.unsqueeze(-1)

        return {
            "is_dual": True,
            "x_seq": None,
            "x_waveform": x_waveform,
            "x_phasor": x_phasor,
            "d_prior": d_prior,
            "c_idx": c_idx,
            "op_feat": op_feat,
            "y": y,
        }

    raise ValueError(f"Unexpected KOL batch length: {len(batch)}")


def _call_kol_model_from_prepared_batch(model, prepared_batch):
    """
    Calls either the old single-GRU model or the new dual-GRU model.
    """
    if prepared_batch["is_dual"]:
        return model(
            prepared_batch["x_waveform"],
            prepared_batch["x_phasor"],
            prepared_batch["c_idx"],
            prepared_batch["d_prior"],
            prepared_batch["op_feat"],
        )

    return model(
        prepared_batch["x_seq"],
        prepared_batch["c_idx"],
        prepared_batch["d_prior"],
        prepared_batch["op_feat"],
    )
