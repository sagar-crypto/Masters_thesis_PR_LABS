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


def train_learned_fusion(
    model,
    train_loader,
    val_loader,
    optimizer,
    device,
    logger,
    model_mode: str,
    epochs: int = 20,
    patience: int = 15,
):
    """Train GRU-only, convex fusion, or bounded residual fusion."""

    model_mode = str(
        model_mode
    ).lower().strip()

    supported_modes = {
        "gru_only",
        "learned_fusion",
        "bounded_residual_fusion",
    }

    if model_mode not in supported_modes:
        raise ValueError(
            f"Unsupported model_mode="
            f"'{model_mode}'. "
            f"Supported modes: "
            f"{sorted(supported_modes)}"
        )

    if model_mode == "bounded_residual_fusion":
        criterion = nn.SmoothL1Loss(
            beta=0.05
        )
    else:
        criterion = nn.MSELoss()

    best_val_score = float("inf")
    best_state = None
    bad_epochs = 0

    for epoch in range(int(epochs)):
        model.train()

        train_losses = []
        train_residuals = []
        train_gates = []

        for batch in train_loader:
            batch_dict = (
                _move_kol_batch_to_device(
                    batch,
                    device,
                )
            )

            if batch_dict["is_dual"]:
                raise NotImplementedError(
                    "GRU fusion modes support "
                    "waveform input only."
                )

            x_seq = batch_dict["x_seq"]
            d_prior = batch_dict["d_prior"]
            case_idx = batch_dict["c_idx"]
            op_feat = batch_dict["op_feat"]
            y = batch_dict["y"]

            optimizer.zero_grad()

            if (
                model_mode
                == "bounded_residual_fusion"
            ):
                (
                    d_kol,
                    d_kol_unclipped,
                    residual,
                    gate,
                ) = model(
                    x_seq,
                    d_prior,
                    case_idx,
                    op_feat
                )

                main_loss = criterion(
                    d_kol_unclipped,
                    y,
                )

                # Small regularization only.
                residual_penalty = (
                    residual.abs().mean()
                )

                loss = (
                    main_loss
                    + 0.001
                    * residual_penalty
                )

                train_residuals.append(
                    residual.detach()
                    .cpu()
                    .numpy()
                )

                train_gates.append(
                    gate.detach()
                    .cpu()
                    .numpy()
                )

            else:
                (
                    d_kol,
                    d_gru,
                    alpha,
                ) = model(
                    x_seq,
                    d_prior,
                )

                if model_mode == "gru_only":
                    loss = criterion(
                        d_gru,
                        y,
                    )

                elif model_mode == "learned_fusion":
                    fusion_loss = criterion(
                        d_kol,
                        y,
                    )

                    direct_gru_loss = criterion(
                        d_gru,
                        y,
                    )

                    loss = (
                        fusion_loss
                        + direct_gru_loss
                    )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=5.0,
            )

            optimizer.step()

            train_losses.append(
                float(loss.item())
            )

        model.eval()

        val_losses = []
        val_abs_errors = []
        val_residuals = []
        val_gates = []
        val_aux_values = []

        with torch.no_grad():
            for batch in val_loader:
                batch_dict = (
                    _move_kol_batch_to_device(
                        batch,
                        device,
                    )
                )

                if batch_dict["is_dual"]:
                    raise NotImplementedError(
                        "GRU fusion modes support "
                        "waveform input only."
                    )

                x_seq = batch_dict["x_seq"]
                d_prior = batch_dict["d_prior"]
                case_idx = batch_dict["c_idx"]
                op_feat = batch_dict["op_feat"]
                y = batch_dict["y"]

                if (
                    model_mode
                    == "bounded_residual_fusion"
                ):
                    (
                        d_kol,
                        d_kol_unclipped,
                        residual,
                        gate,
                    ) = model(
                        x_seq,
                        d_prior,
                        case_idx,
                        op_feat,
                    )

                    val_loss = criterion(
                        d_kol_unclipped,
                        y,
                    )

                    prediction_for_metrics = (
                        d_kol
                    )

                    val_residuals.append(
                        residual.detach()
                        .cpu()
                        .numpy()
                    )

                    val_gates.append(
                        gate.detach()
                        .cpu()
                        .numpy()
                    )

                else:
                    (
                        d_kol,
                        d_gru,
                        alpha,
                    ) = model(
                        x_seq,
                        d_prior,
                    )

                    if model_mode == "gru_only":
                        prediction_for_metrics = (
                            d_gru
                        )
                    else:
                        prediction_for_metrics = (
                            d_kol
                        )

                    val_loss = criterion(
                        prediction_for_metrics,
                        y,
                    )

                    val_aux_values.append(
                        d_gru.detach()
                        .cpu()
                        .numpy()
                    )

                    val_gates.append(
                        alpha.detach()
                        .cpu()
                        .numpy()
                    )

                val_losses.append(
                    float(val_loss.item())
                )

                val_abs_errors.append(
                    torch.abs(
                        prediction_for_metrics
                        - y
                    )
                    .detach()
                    .cpu()
                    .numpy()
                )

        mean_train_loss = (
            float(
                np.mean(
                    train_losses
                )
            )
            if train_losses
            else float("nan")
        )

        mean_val_loss = (
            float(
                np.mean(
                    val_losses
                )
            )
            if val_losses
            else float("inf")
        )

        mean_val_mae = (
            float(
                np.mean(
                    np.concatenate(
                        val_abs_errors
                    )
                )
            )
            if val_abs_errors
            else float("inf")
        )

        if (
            model_mode
            == "bounded_residual_fusion"
        ):
            mean_gate = (
                float(
                    np.mean(
                        np.concatenate(
                            val_gates
                        )
                    )
                )
                if val_gates
                else float("nan")
            )

            mean_residual = (
                float(
                    np.mean(
                        np.concatenate(
                            val_residuals
                        )
                    )
                )
                if val_residuals
                else float("nan")
            )

            mean_abs_residual = (
                float(
                    np.mean(
                        np.abs(
                            np.concatenate(
                                val_residuals
                            )
                        )
                    )
                )
                if val_residuals
                else float("nan")
            )

            logger.info(
                "epoch %d | mode=%s | "
                "train_loss=%.6f | "
                "val_loss=%.6f | "
                "val_mae=%.6f | "
                "mean_gate=%.6f | "
                "mean_residual=%.6f | "
                "mean_abs_residual=%.6f",
                epoch + 1,
                model_mode,
                mean_train_loss,
                mean_val_loss,
                mean_val_mae,
                mean_gate,
                mean_residual,
                mean_abs_residual,
            )

            # Select the checkpoint using the thesis metric.
            checkpoint_score = (
                mean_val_mae
            )

        else:
            mean_alpha = (
                float(
                    np.mean(
                        np.concatenate(
                            val_gates
                        )
                    )
                )
                if val_gates
                else float("nan")
            )

            mean_d_gru = (
                float(
                    np.mean(
                        np.concatenate(
                            val_aux_values
                        )
                    )
                )
                if val_aux_values
                else float("nan")
            )

            logger.info(
                "epoch %d | mode=%s | "
                "train_loss=%.6f | "
                "val_loss=%.6f | "
                "val_mae=%.6f | "
                "mean_alpha=%.6f | "
                "mean_d_gru=%.6f",
                epoch + 1,
                model_mode,
                mean_train_loss,
                mean_val_loss,
                mean_val_mae,
                mean_alpha,
                mean_d_gru,
            )

            checkpoint_score = (
                mean_val_loss
            )

        min_delta = 1e-4

        if checkpoint_score < (
            best_val_score
            - min_delta
        ):
            best_val_score = (
                checkpoint_score
            )

            best_state = {
                key: value.detach()
                .cpu()
                .clone()
                for key, value
                in model.state_dict().items()
            }

            bad_epochs = 0

        else:
            bad_epochs += 1

            if bad_epochs >= int(patience):
                logger.info(
                    "Early stopping at epoch %d | "
                    "best_val_score=%.6f | "
                    "current_val_score=%.6f",
                    epoch + 1,
                    best_val_score,
                    checkpoint_score,
                )
                break

    if best_state is not None:
        model.load_state_dict(
            best_state
        )

    return best_val_score


def evaluate_learned_fusion(
    model,
    test_loader,
    device,
    logger,
    model_mode: str,
):
    """Evaluate direct GRU, convex fusion, or bounded residual fusion."""

    model_mode = str(
        model_mode
    ).lower().strip()

    supported_modes = {
        "gru_only",
        "learned_fusion",
        "bounded_residual_fusion",
    }

    if model_mode not in supported_modes:
        raise ValueError(
            f"Unsupported model_mode="
            f"'{model_mode}'. "
            f"Supported modes: "
            f"{sorted(supported_modes)}"
        )

    model.eval()

    y_true_all = []
    y_pred_all = []
    d_prior_all = []
    auxiliary_all = []
    alpha_all = []
    case_all = []

    with torch.no_grad():
        for batch in test_loader:
            batch_dict = (
                _move_kol_batch_to_device(
                    batch,
                    device,
                )
            )

            if batch_dict["is_dual"]:
                raise NotImplementedError(
                    "GRU fusion modes support "
                    "waveform input only."
                )

            x_seq = batch_dict["x_seq"]
            d_prior = batch_dict["d_prior"]
            case_idx = batch_dict["c_idx"]
            op_feat = batch_dict["op_feat"]
            y = batch_dict["y"]

            if (
                model_mode
                == "bounded_residual_fusion"
            ):
                (
                    d_kol,
                    d_kol_unclipped,
                    residual,
                    gate,
                ) = model(
                    x_seq,
                    d_prior,
                    case_idx,
                    op_feat
                )

                y_pred = d_kol
                auxiliary = residual
                alpha = gate

            else:
                (
                    d_kol,
                    d_gru,
                    alpha,
                ) = model(
                    x_seq,
                    d_prior,
                )

                if model_mode == "gru_only":
                    y_pred = d_gru
                else:
                    y_pred = d_kol

                auxiliary = d_gru

            y_true_all.append(
                y.detach()
                .cpu()
                .numpy()
            )

            y_pred_all.append(
                y_pred.detach()
                .cpu()
                .numpy()
            )

            d_prior_all.append(
                d_prior.detach()
                .cpu()
                .numpy()
            )

            auxiliary_all.append(
                auxiliary.detach()
                .cpu()
                .numpy()
            )

            alpha_all.append(
                alpha.detach()
                .cpu()
                .numpy()
            )

            case_all.append(
                case_idx.detach()
                .cpu()
                .numpy()
            )

    y_true_np = np.concatenate(
        y_true_all
    ).astype(np.float64)

    y_pred_np = np.concatenate(
        y_pred_all
    ).astype(np.float64)

    d_prior_np = np.concatenate(
        d_prior_all
    ).astype(np.float64)

    auxiliary_np = np.concatenate(
        auxiliary_all
    ).astype(np.float64)

    alpha_np = np.concatenate(
        alpha_all
    ).astype(np.float64)

    case_np = np.concatenate(
        case_all
    ).astype(np.int64)

    squared_error = (
        y_pred_np
        - y_true_np
    ) ** 2

    abs_error = np.abs(
        y_pred_np
        - y_true_np
    )

    prior_abs_error = np.abs(
        d_prior_np
        - y_true_np
    )

    mse = float(
        np.mean(
            squared_error
        )
    )

    mae = float(
        np.mean(
            abs_error
        )
    )

    rmse = float(
        np.sqrt(
            mse
        )
    )

    prior_mae = float(
        np.mean(
            prior_abs_error
        )
    )

    prior_rmse = float(
        np.sqrt(
            np.mean(
                (
                    d_prior_np
                    - y_true_np
                ) ** 2
            )
        )
    )

    alpha_mean = float(
        np.mean(
            alpha_np
        )
    )

    alpha_std = float(
        np.std(
            alpha_np
        )
    )

    alpha_min = float(
        np.min(
            alpha_np
        )
    )

    alpha_max = float(
        np.max(
            alpha_np
        )
    )

    if (
        model_mode
        == "bounded_residual_fusion"
    ):
        effective_correction = (
            y_pred_np
            - d_prior_np
        )

        improvement_rate = float(
            np.mean(
                abs_error
                < prior_abs_error
            )
        )

        worsened_rate = float(
            np.mean(
                abs_error
                > prior_abs_error
            )
        )

        metrics = {
            "loss": mse,
            "mae": mae,
            "rmse": rmse,
            "prior_mae": prior_mae,
            "prior_rmse": prior_rmse,
            "residual_mean": float(
                np.mean(
                    auxiliary_np
                )
            ),
            "residual_abs_mean": float(
                np.mean(
                    np.abs(
                        auxiliary_np
                    )
                )
            ),
            "residual_std": float(
                np.std(
                    auxiliary_np
                )
            ),
            "alpha_mean": alpha_mean,
            "alpha_std": alpha_std,
            "alpha_min": alpha_min,
            "alpha_max": alpha_max,
            "improvement_rate": (
                improvement_rate
            ),
            "worsened_rate": (
                worsened_rate
            ),
            "effective_correction_mean": float(
                np.mean(
                    effective_correction
                )
            ),
            "effective_correction_abs_mean": float(
                np.mean(
                    np.abs(
                        effective_correction
                    )
                )
            ),
            "effective_correction_std": float(
                np.std(
                    effective_correction
                )
            ),
        }

        logger.info(
            "Final model evaluation | "
            "mode=%s | mae=%.6f | "
            "rmse=%.6f | "
            "prior_mae=%.6f | "
            "prior_rmse=%.6f",
            model_mode,
            mae,
            rmse,
            prior_mae,
            prior_rmse,
        )

        logger.info(
            "Residual/gate statistics | "
            "residual_mean=%.6f | "
            "residual_abs_mean=%.6f | "
            "residual_std=%.6f | "
            "gate_mean=%.6f | "
            "gate_std=%.6f",
            metrics["residual_mean"],
            metrics["residual_abs_mean"],
            metrics["residual_std"],
            alpha_mean,
            alpha_std,
        )

        logger.info(
            "Prior correction statistics | "
            "improvement_rate=%.4f | "
            "worsened_rate=%.4f | "
            "effective_correction_mean=%.6f | "
            "effective_correction_abs_mean=%.6f",
            improvement_rate,
            worsened_rate,
            metrics[
                "effective_correction_mean"
            ],
            metrics[
                "effective_correction_abs_mean"
            ],
        )

    else:
        direct_gru_mae = float(
            np.mean(
                np.abs(
                    auxiliary_np
                    - y_true_np
                )
            )
        )

        direct_gru_rmse = float(
            np.sqrt(
                np.mean(
                    (
                        auxiliary_np
                        - y_true_np
                    ) ** 2
                )
            )
        )

        metrics = {
            "loss": mse,
            "mae": mae,
            "rmse": rmse,
            "prior_mae": prior_mae,
            "prior_rmse": prior_rmse,
            "direct_gru_mae": (
                direct_gru_mae
            ),
            "direct_gru_rmse": (
                direct_gru_rmse
            ),
            "alpha_mean": alpha_mean,
            "alpha_std": alpha_std,
            "alpha_min": alpha_min,
            "alpha_max": alpha_max,
        }

        logger.info(
            "Final model evaluation | "
            "mode=%s | mae=%.6f | "
            "rmse=%.6f | "
            "prior_mae=%.6f | "
            "prior_rmse=%.6f | "
            "direct_gru_mae=%.6f | "
            "direct_gru_rmse=%.6f",
            model_mode,
            mae,
            rmse,
            prior_mae,
            prior_rmse,
            direct_gru_mae,
            direct_gru_rmse,
        )

    return (
        metrics,
        y_true_np,
        y_pred_np,
        d_prior_np,
        auxiliary_np,
        alpha_np,
        case_np,
    )
