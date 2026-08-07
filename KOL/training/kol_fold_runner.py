from __future__ import annotations

import os
from itertools import islice
from typing import Any

import numpy as np
import torch

import dl_psp.data.labels as L
from dl_psp.models.model_utils import create_model_from_name
from dl_psp.utils.eval_utils import evaluate, predict_on_loader
from dl_psp.utils.run_utils import (
    save_checkpoint,
    save_fold_predictions,
    save_fold_splits,
    set_seed,
)
from dl_psp.utils.subgroup_metrics import maybe_run_subgroup_analysis
from dl_psp.utils.train_utils import make_loaders, train_best_on_val

from KOL.common.constants import CASE_TO_IDX
from KOL.common.cv_utils import (
    split_train_val_from_train_pool,
    validate_checkpoint_metadata,
)
from KOL.datasets.kol_datasets import make_kol_loaders, make_kol_dual_loaders
from KOL.models.kol_residual_models import (
    KOLGRUCaseResidualRegressor,
    KOLDualGRUCaseResidualRegressor,
    KOLGRULearnedFusionRegressor,
    KOLGRUBoundedResidualFusionRegressor,
)
from KOL.training.kol_residual_train import (
    evaluate_kol_case_k0,
    evaluate_learned_fusion,
    train_kol_case_k0,
    train_learned_fusion,
)


class _LimitedLoader:
    def __init__(self, loader, maximum):
        self.loader = loader
        self.maximum = int(maximum)
        if self.maximum < 0:
            raise ValueError("batch limits must be non-negative")

    def __iter__(self):
        return islice(iter(self.loader), self.maximum)

    def __len__(self):
        return min(len(self.loader), self.maximum)

    def __getattr__(self, name):
        return getattr(self.loader, name)


def _limit_loader(loader, maximum):
    return loader if maximum is None else _LimitedLoader(loader, maximum)


def _build_checkpoint_path(
    *,
    ckpt_dir: str,
    config,
    target_label: str,
    model_name: str,
    window_ms: int,
    fold_idx: int,
    seed: int,
) -> str:
    os.makedirs(ckpt_dir, exist_ok=True)

    return os.path.join(
        ckpt_dir,
        f"{config.dataset.topology}__{target_label}__{model_name}__W{window_ms}ms__fold{fold_idx}__seed{seed}.pt",
    )


def _save_outputs_for_fold(
    *,
    config,
    model,
    model_name: str,
    task_type: str,
    target_label: str,
    include_groups,
    feature_indices_for_ds,
    class_to_idx,
    test_metrics: dict[str, Any],
    X_used_filtered: np.ndarray,
    fold_idx: int,
    seed: int,
    ckpt_path: str,
    run_out_dir: str,
    idx_train: np.ndarray,
    idx_val: np.ndarray,
    idx_test: np.ndarray,
    idx_test_predictions: np.ndarray | None,
    labels_df_used,
    y_true_np: np.ndarray,
    y_pred_np: np.ndarray,
    y_score_np: np.ndarray,
    d_phys_prior,
    dprior_np,
    residual_np,
    d_gru_np,
    alpha_np,
    case_np,
    logger,
) -> None:
    save_checkpoint(
        path=ckpt_path,
        model=model,
        config=config,
        meta={
            "topology": str(config.dataset.topology),
            "target_label": target_label,
            "model_name": model_name,
            "window_length_s": float(config.window_extraction.window_length),
            "fold_idx": int(fold_idx),
            "seed": int(seed),
        },
        task_type=task_type,
        target_label=target_label,
        include_groups=list(include_groups) if include_groups is not None else [],
        feature_indices_for_ds=feature_indices_for_ds,
        class_to_idx=class_to_idx if task_type == "multiclass" else None,
        test_metrics=test_metrics,
        X_used_shape=tuple(X_used_filtered.shape),
        fold_idx=int(fold_idx),
        seed=int(seed),
    )
    logger.info("Saved checkpoint: %s", ckpt_path)

    try:
        split_path = save_fold_splits(
            out_dir=run_out_dir,
            fold_idx=fold_idx,
            idx_train=idx_train,
            idx_val=idx_val,
            idx_test=idx_test,
            labels_df=labels_df_used,
            group_col=L.SAMPLE_ID,
        )
        logger.info("Saved fold splits: %s", split_path)
    except Exception as e:
        logger.warning("Failed to save fold splits: %s", e)

    try:
        meta_cols = [
            L.EVENT_TYPE,
            L.STATUS,
            L.Y_FAULT_LINE,
            L.SAMPLE_ID,
        ]

        if "window_idx" in labels_df_used.columns:
            meta_cols.append("window_idx")

        if "case" in labels_df_used.columns:
            meta_cols.append("case")

        logger.info(
            "DEBUG about to save predictions | fold=%d | run_out_dir=%s | len(test_idx)=%d | y_true_shape=%s | y_pred_shape=%s | y_score_shape=%s",
            fold_idx,
            run_out_dir,
            len(idx_test),
            np.asarray(y_true_np).shape,
            np.asarray(y_pred_np).shape,
            np.asarray(y_score_np).shape,
        )

        extra_cols = None

        if d_phys_prior is not None:
            extra_cols = {
                "d_prior": dprior_np,
                "case_idx": case_np,
            }

            if residual_np is not None:
                extra_cols["residual"] = residual_np

            if d_gru_np is not None:
                extra_cols["d_gru"] = d_gru_np

            if alpha_np is not None:
                extra_cols["alpha"] = alpha_np

        pred_path = save_fold_predictions(
            out_dir=run_out_dir,
            fold_idx=fold_idx,
            idx_test=idx_test if idx_test_predictions is None else idx_test_predictions,
            labels_df=labels_df_used,
            y_true=y_true_np,
            y_pred=y_pred_np,
            y_score=y_score_np,
            task_type=task_type,
            meta_cols=meta_cols,
            extra_cols=extra_cols,
        )
        logger.info("Saved fold predictions: %s", pred_path)

    except Exception as e:
        logger.warning("Failed to save fold predictions: %s", e)

    try:
        fold_analysis_dir = os.path.join(run_out_dir, f"fold{fold_idx}_subgroups")
        os.makedirs(fold_analysis_dir, exist_ok=True)

        maybe_run_subgroup_analysis(
            task_type=task_type,
            labels_df=labels_df_used,
            valid_row_idx=np.arange(len(labels_df_used)),
            idx_test=idx_test,
            y_true=y_true_np,
            y_pred=y_pred_np,
            out_dir=fold_analysis_dir,
            group_cols=["event_type", "status", "y_fault_line", "case"],
            logger=logger,
            min_support=5,
            regression_add_true_bins=True,
            regression_true_bins_decimals=3,
        )

        logger.info(
            "Saved subgroup analysis for fold %d to: %s",
            fold_idx,
            fold_analysis_dir,
        )

    except Exception as e:
        logger.warning("Subgroup analysis failed: %s", e)


def run_one_fold(
    *,
    fold_idx: int,
    n_splits: int,
    train_pool_idx,
    test_idx,
    config,
    X_used_filtered: np.ndarray,
    y_all: np.ndarray,
    labels_df_used,
    groups_used,
    feature_indices_for_ds,
    task_type: str,
    criterion,
    primary_name: str,
    higher_is_better: bool,
    d_phys_prior,
    case_idx,
    op_features,
    kol_prediction_mode: str,
    kol_model_mode: str,
    model_name: str,
    F_eff: int,
    flat_dim: int,
    out_dim: int,
    best_lr: float,
    best_wd: float,
    seed: int,
    device,
    run_out_dir: str,
    ckpt_dir: str,
    class_to_idx,
    target_label: str,
    include_groups,
    window_ms: int,
    valid_row_idx,
    eval_only: bool,
    resave_eval_only: bool,
    logger,
    X_phasor_all: np.ndarray | None = None,
    n_phasor_features: int | None = None,
) -> dict[str, Any]:
    train_pool_idx = np.asarray(train_pool_idx, dtype=int)
    test_idx = np.asarray(test_idx, dtype=int)

    set_seed(int(seed))

    idx_train, idx_val = split_train_val_from_train_pool(
        groups_used=groups_used,
        train_pool_idx=train_pool_idx,
        val_size=float(config.training.val_size),
        split_seed=int(config.training.split_seed) + int(fold_idx),
    )

    max_train_batches = getattr(config.training, "max_train_batches", None)
    max_val_batches = getattr(config.training, "max_val_batches", None)
    max_test_batches = getattr(config.training, "max_test_batches", None)

    logger.info(
        "[fold %d/%d] split sizes: train=%d | val=%d | test=%d",
        fold_idx + 1,
        n_splits,
        len(idx_train),
        len(idx_val),
        len(test_idx),
    )

    logger.info(
        "=== fold %d/%d | seed %d ===",
        fold_idx + 1,
        n_splits,
        int(seed),
    )

    ckpt_path = _build_checkpoint_path(
        ckpt_dir=ckpt_dir,
        config=config,
        target_label=target_label,
        model_name=model_name,
        window_ms=window_ms,
        fold_idx=fold_idx,
        seed=seed,
    )

    dprior_np = None
    residual_np = None
    d_gru_np = None
    alpha_np = None
    case_np = None

    kol_model_mode = str(kol_model_mode).lower().strip()

    if d_phys_prior is None:
        train_loader, val_loader, test_loader = make_loaders(
            X_used=X_used_filtered,
            y_all=y_all,
            task_type=task_type,
            feature_indices_for_ds=feature_indices_for_ds,
            idx_train=idx_train,
            idx_val=idx_val,
            idx_test=test_idx,
            batch_size=int(config.training.batch_size),
            device=device,
            num_workers=int(config.training.num_workers),
            pin_memory=bool(config.training.pin_memory),
            prefetch_factor=int(config.training.prefetch_factor),
            row_indices=valid_row_idx,
        )

        model = create_model_from_name(
            config,
            n_features=int(F_eff),
            flattened_dim=int(flat_dim),
            out_dim=int(out_dim),
        ).to(device)

        train_loader = _limit_loader(train_loader, max_train_batches)
        val_loader = _limit_loader(val_loader, max_val_batches)
        test_loader = _limit_loader(test_loader, max_test_batches)

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(best_lr),
            weight_decay=float(best_wd),
        )

        if eval_only:
            if not os.path.exists(ckpt_path):
                raise FileNotFoundError(
                    f"eval_only=true but checkpoint not found: {ckpt_path}"
                )

            ckpt = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(ckpt["model_state_dict"])
            validate_checkpoint_metadata(
                ckpt=ckpt,
                expected_topology=str(config.dataset.topology),
                expected_target=target_label,
                expected_model=model_name.replace("kol_", ""),
                expected_window_s=float(config.window_extraction.window_length),
                logger=logger,
            )
            logger.info("Loaded checkpoint: %s", ckpt_path)

        else:
            train_best_on_val(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                optimizer=optimizer,
                criterion=criterion,
                device=device,
                task_type=task_type,
                epochs=int(config.training.epochs),
                primary_name=primary_name,
                higher_is_better=higher_is_better,
                binary_threshold=float(config.training.binary_threshold),
                patience=int(getattr(config.training, "patience", 15)),
            )

        test_metrics = evaluate(
            model,
            test_loader,
            device,
            task_type,
            binary_threshold=float(config.training.binary_threshold),
        )

        y_true_np, y_pred_np, y_score_np = predict_on_loader(
            model=model,
            loader=test_loader,
            device=device,
            task_type=task_type,
            binary_threshold=float(config.training.binary_threshold),
        )

    else:
        if task_type != "regression":
            raise NotImplementedError(
                "First KOL residual version currently supports regression only."
            )

        if kol_model_mode in {
            "gru_only",
            "learned_fusion",
            "bounded_residual_fusion",
        }:
            if X_phasor_all is not None:
                raise NotImplementedError(
                    "The GRU fusion pipelines support "
                    "training.input_representation=waveform only."
                )
            op_features_for_loader = (
                op_features
                if kol_model_mode == "bounded_residual_fusion"
                else None
            )

            train_loader, val_loader, test_loader = make_kol_loaders(
                    X_used=X_used_filtered,
                    y_all=y_all,
                    d_phys_prior=d_phys_prior,
                    case_idx=case_idx,
                    op_features=op_features_for_loader,
                    idx_train=idx_train,
                    idx_val=idx_val,
                    idx_test=test_idx,
                    feature_indices_for_ds=feature_indices_for_ds,
                    batch_size=int(config.training.batch_size),
                    num_workers=int(config.training.num_workers),
                    pin_memory=bool(config.training.pin_memory),
                )

        elif X_phasor_all is not None:
            train_loader, val_loader, test_loader = make_kol_dual_loaders(
                X_waveform=X_used_filtered,
                X_phasor=X_phasor_all,
                y_all=y_all,
                d_phys_prior=d_phys_prior,
                case_idx=case_idx,
                op_features=op_features,
                idx_train=idx_train,
                idx_val=idx_val,
                idx_test=test_idx,
                batch_size=int(config.training.batch_size),
                num_workers=int(config.training.num_workers),
                pin_memory=bool(config.training.pin_memory),
            )

        else:
            train_loader, val_loader, test_loader = make_kol_loaders(
                X_used=X_used_filtered,
                y_all=y_all,
                d_phys_prior=d_phys_prior,
                case_idx=case_idx,
                op_features=op_features,
                idx_train=idx_train,
                idx_val=idx_val,
                idx_test=test_idx,
                feature_indices_for_ds=feature_indices_for_ds,
                batch_size=int(config.training.batch_size),
                num_workers=int(config.training.num_workers),
                pin_memory=bool(config.training.pin_memory),
            )

        logger.info("op_features is None: %s", op_features is None)
        train_loader = _limit_loader(train_loader, max_train_batches)
        val_loader = _limit_loader(val_loader, max_val_batches)
        test_loader = _limit_loader(test_loader, max_test_batches)
        if op_features is not None:
            logger.info("op_features shape: %s", op_features.shape)
            logger.info("n_op_features passed to model: %d", int(op_features.shape[1]))

        if kol_model_mode in {
            "gru_only",
            "learned_fusion",
        }:
            model = KOLGRULearnedFusionRegressor(
                n_features=int(F_eff),
                hidden_size=int(
                    getattr(
                        config.model,
                        "hidden_size",
                        128,
                    )
                ),
                num_layers=int(
                    getattr(
                        config.model,
                        "num_layers",
                        2,
                    )
                ),
                dropout=float(
                    getattr(
                        config.model,
                        "dropout",
                        0.1,
                    )
                ),
                bidirectional=bool(
                    getattr(
                        config.model,
                        "bidirectional",
                        False,
                    )
                ),
            ).to(device)

        elif kol_model_mode == "bounded_residual_fusion":
            model = KOLGRUBoundedResidualFusionRegressor(
                n_features=int(F_eff),
                n_op_features=(
                    0
                    if op_features is None
                    else int(op_features.shape[1])
                ),
                hidden_size=int(
                    getattr(
                        config.model,
                        "hidden_size",
                        64,
                    )
                ),
                num_layers=int(
                    getattr(
                        config.model,
                        "num_layers",
                        1,
                    )
                ),
                dropout=float(
                    getattr(
                        config.model,
                        "dropout",
                        0.0,
                    )
                ),
                bidirectional=bool(
                    getattr(
                        config.model,
                        "bidirectional",
                        False,
                    )
                ),
                n_cases=len(
                    CASE_TO_IDX
                ),
                case_emb_dim=int(
                    getattr(
                        config.training,
                        "case_emb_dim",
                        8,
                    )
                ),
                head_hidden_size=int(
                    getattr(
                        config.training,
                        "fusion_head_hidden_size",
                        64,
                    )
                ),
                residual_max=float(
                    getattr(
                        config.training,
                        "bounded_residual_max",
                        1.0,
                    )
                ),
                gate_init_bias=float(
                    getattr(
                        config.training,
                        "gate_init_bias",
                        -3.0,
                    )
                ),
                prediction_mode=(
                    kol_prediction_mode
                ),
            ).to(device)

        elif X_phasor_all is not None:
            if n_phasor_features is None:
                raise ValueError(
                    "n_phasor_features is required for dual-input model."
                )

            model = KOLDualGRUCaseResidualRegressor(
                n_waveform_features=int(F_eff),
                n_phasor_features=int(n_phasor_features),
                n_op_features=0 if op_features is None else int(op_features.shape[1]),
                hidden_size=int(getattr(config.model, "hidden_size", 128)),
                num_layers=int(getattr(config.model, "num_layers", 2)),
                dropout=float(getattr(config.model, "dropout", 0.1)),
                bidirectional=bool(
                    getattr(config.model, "bidirectional", False)
                ),
                n_cases=len(CASE_TO_IDX),
            ).to(device)

        else:
            model = KOLGRUCaseResidualRegressor(
                n_features=int(F_eff),
                n_op_features=0 if op_features is None else int(op_features.shape[1]),
                hidden_size=int(getattr(config.model, "hidden_size", 128)),
                num_layers=int(getattr(config.model, "num_layers", 2)),
                dropout=float(getattr(config.model, "dropout", 0.1)),
                bidirectional=bool(
                    getattr(config.model, "bidirectional", False)
                ),
                n_cases=len(CASE_TO_IDX),
            ).to(device)

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(best_lr),
            weight_decay=float(best_wd),
        )

        if eval_only:
            if not os.path.exists(ckpt_path):
                raise FileNotFoundError(
                    f"eval_only=true but checkpoint not found: {ckpt_path}"
                )

            ckpt = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(ckpt["model_state_dict"])
            validate_checkpoint_metadata(
                ckpt=ckpt,
                expected_topology=str(config.dataset.topology),
                expected_target=target_label,
                expected_model=model_name,
                expected_window_s=float(config.window_extraction.window_length),
                logger=logger,
            )
            logger.info("Loaded checkpoint: %s", ckpt_path)

        else:
            if kol_model_mode in {"gru_only", "learned_fusion", "bounded_residual_fusion"}:
                train_learned_fusion(
                    model=model,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    optimizer=optimizer,
                    device=device,
                    logger=logger,
                    model_mode=kol_model_mode,
                    epochs=int(config.training.epochs),
                    patience=int(getattr(config.training, "patience", 15)),
                )

            elif kol_prediction_mode == "prior_only":
                logger.info("KOL mode is prior_only: skipping training.")

            else:
                train_kol_case_k0(
                    model=model,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    optimizer=optimizer,
                    device=device,
                    prediction_mode=kol_prediction_mode,
                    epochs=int(config.training.epochs),
                    patience=int(getattr(config.training, "patience", 15)),
                    logger=logger,
                )

        if kol_model_mode == "bounded_residual_fusion":
            (
                test_metrics,
                y_true_np,
                y_pred_np,
                dprior_np,
                residual_np,
                alpha_np,
                case_np,
            ) = evaluate_learned_fusion(
                model=model,
                test_loader=test_loader,
                device=device,
                logger=logger,
                model_mode=kol_model_mode,
            )

            y_score_np = residual_np

        elif kol_model_mode in {
            "gru_only",
            "learned_fusion",
        }:
            (
                test_metrics,
                y_true_np,
                y_pred_np,
                dprior_np,
                d_gru_np,
                alpha_np,
                case_np,
            ) = evaluate_learned_fusion(
                model=model,
                test_loader=test_loader,
                device=device,
                logger=logger,
                model_mode=kol_model_mode,
            )

            if kol_model_mode == "learned_fusion":
                y_score_np = alpha_np
            else:
                y_score_np = d_gru_np

        else:
            (
                test_metrics,
                y_true_np,
                y_pred_np,
                residual_np,
                dprior_np,
                case_np,
            ) = evaluate_kol_case_k0(
                model=model,
                test_loader=test_loader,
                device=device,
                prediction_mode=kol_prediction_mode,
                logger=logger,
            )

            y_score_np = residual_np

    # A capped test loader emits a prefix of the sampler's ordered indices.
    # Keep artifact metadata aligned with that partial prediction vector.
    idx_test_predictions = test_idx[: len(y_true_np)]

    if not eval_only or resave_eval_only:
        _save_outputs_for_fold(
            config=config,
            model=model,
            model_name=model_name,
            task_type=task_type,
            target_label=target_label,
            include_groups=include_groups,
            feature_indices_for_ds=feature_indices_for_ds,
            class_to_idx=class_to_idx,
            test_metrics=test_metrics,
            X_used_filtered=X_used_filtered,
            fold_idx=fold_idx,
            seed=seed,
            ckpt_path=ckpt_path,
            run_out_dir=run_out_dir,
            idx_train=idx_train,
            idx_val=idx_val,
            idx_test=test_idx,
            idx_test_predictions=idx_test_predictions,
            labels_df_used=labels_df_used,
            y_true_np=y_true_np,
            y_pred_np=y_pred_np,
            y_score_np=y_score_np,
            d_phys_prior=d_phys_prior,
            dprior_np=dprior_np,
            residual_np=residual_np,
            d_gru_np=d_gru_np,
            alpha_np=alpha_np,
            case_np=case_np,
            logger=logger,
        )

    logger.info(
        "[fold %d/%d seed %d] test_metrics=%s",
        fold_idx + 1,
        n_splits,
        int(seed),
        test_metrics,
    )

    return {
        "fold": int(fold_idx),
        "n_train": int(len(idx_train)),
        "n_val": int(len(idx_val)),
        "n_test": int(len(idx_test_predictions)),
        **{f"test/{k}": float(v) for k, v in test_metrics.items()},
    }
