from __future__ import annotations

import os
from typing import Any

import numpy as np
import pandas as pd

import dl_psp.data.labels as L
from dl_psp.models.model_utils import get_device
from dl_psp.utils.run_utils import (
    get_env_info,
    infer_input_dims,
    set_torch_perf_flags,
)

from KOL.common.cases import build_case_index
from KOL.common.constants import CASE_TO_IDX
from KOL.common.cv_utils import (
    build_cv_splits_stratified,
    canonicalize_multiclass_encoding,
    select_best_lr_wd,
    setup_wandb_logging,
)
from KOL.common.operator_features import load_operator_inputs_if_enabled
from KOL.common.phasor_representation import apply_input_representation
from KOL.common.windowing import get_kol_mode
from KOL.datasets.kol_data_preparation import load_filtered_training_data
from KOL.training.kol_fold_runner import run_one_fold

from dl_psp.data.targets import extract_target
from dl_psp.data.task_spec import get_task_spec, infer_task_type_from_spec


def _prepare_task(config):
    target_label = str(config.training.target_label)

    if target_label not in L.ALL_TARGETS:
        raise ValueError(f"Unknown target_label='{target_label}'")

    spec = get_task_spec(target_label)
    task_type = infer_task_type_from_spec(spec)

    return target_label, spec, task_type


def _prepare_targets(
    *,
    labels_df_used: pd.DataFrame,
    target_label: str,
    task_type: str,
    logger,
) -> tuple[np.ndarray, Any]:
    y_all, class_to_idx = extract_target(labels_df_used, target_label=target_label)

    if task_type == "multiclass":
        y_all, class_to_idx = canonicalize_multiclass_encoding(y_all, task_type)
        logger.info("After canonicalization: %d classes in mapping", len(class_to_idx))

    if np.issubdtype(y_all.dtype, np.floating):
        num_nans = int(np.isnan(y_all).sum())
        if num_nans > 0:
            logger.warning(
                "Target '%s' contains %d NaN values out of %d samples (%.2f%%).",
                target_label,
                num_nans,
                len(y_all),
                (num_nans / len(y_all)) * 100.0,
            )
            raise ValueError("NaN values found in target labels.")

    return y_all, class_to_idx


def _prepare_kol_inputs(
    *,
    config,
    labels_df_used: pd.DataFrame,
    logger,
):
    d_phys_prior, op_features, operator_feature_cols = load_operator_inputs_if_enabled(
        config=config,
        labels_df_used=labels_df_used,
    )

    case_idx = build_case_index(labels_df_used)

    idx_to_case = {v: k for k, v in CASE_TO_IDX.items()}
    labels_df_used = labels_df_used.copy()
    labels_df_used["case"] = [idx_to_case[int(i)] for i in case_idx]

    kol_prediction_mode = get_kol_mode(config)

    logger.info("KOL prediction mode: %s", kol_prediction_mode)
    logger.info("Selected operator feature columns: %s", operator_feature_cols)

    return labels_df_used, d_phys_prior, op_features, operator_feature_cols, case_idx, kol_prediction_mode


def _make_run_output_dir(
    *,
    config,
    topology: str,
    run_id: str,
) -> tuple[str, str, str, str]:
    top_out_dir = str(getattr(config.training, "out_dir", "outputs"))

    if topology == "hv_double_line_90kv":
        default_line_filter = "Line_1_2_a"
    elif topology == "hv_double_line_110kv":
        default_line_filter = "MainLn1-2A"
    else:
        default_line_filter = "all_lines"

    line_filter = str(getattr(config.training, "line_filter", default_line_filter))
    kol_mode_name = str(getattr(config.training, "kol_prediction_mode", "plain"))
    window_mode_name = str(getattr(config.training, "kol_window_mode", "default"))
    operator_path = str(getattr(config.training, "operator_features_path", "no_operator_file"))

    operator_tag = os.path.splitext(os.path.basename(operator_path))[0]
    operator_tag = operator_tag.replace("/", "_").replace(" ", "_")

    safe_line = line_filter.replace("/", "_").replace(" ", "_")
    safe_kol = kol_mode_name.replace("/", "_").replace(" ", "_")
    safe_window = window_mode_name.replace("/", "_").replace(" ", "_")

    run_out_dir = os.path.join(
        top_out_dir,
        f"{safe_line}__{operator_tag}__{safe_kol}__{safe_window}__run_{run_id}",
    )

    os.makedirs(run_out_dir, exist_ok=True)

    return run_out_dir, line_filter, kol_mode_name, window_mode_name


def _aggregate_and_save_metrics(
    *,
    all_fold_metrics: list[dict[str, Any]],
    run_out_dir: str,
    wb_run,
    logger,
) -> pd.DataFrame:
    metrics_df = pd.DataFrame(all_fold_metrics)

    logger.info("===== CV aggregate metrics =====")
    for col in metrics_df.columns:
        if col.startswith("test/"):
            mean_v = metrics_df[col].mean()
            std_v = metrics_df[col].std(ddof=1) if len(metrics_df) > 1 else 0.0
            logger.info("%s: %.6f ± %.6f", col, mean_v, std_v)

    agg_path = os.path.join(run_out_dir, "cv_metrics_summary.csv")
    metrics_df.to_csv(agg_path, index=False)
    logger.info("Saved aggregate metrics to: %s", agg_path)

    if wb_run is not None:
        for col in metrics_df.columns:
            if col.startswith("test/"):
                wb_run.summary[f"{col}_mean"] = float(metrics_df[col].mean())
                wb_run.summary[f"{col}_std"] = float(
                    metrics_df[col].std(ddof=1) if len(metrics_df) > 1 else 0.0
                )

        wb_run.finish()

    return metrics_df


def run_kol_cv_experiment(*, config, logger) -> pd.DataFrame:
    set_torch_perf_flags()

    device = get_device()
    logger.info("Device: %s", device)

    target_label, spec, task_type = _prepare_task(config)

    prepared = load_filtered_training_data(config=config, logger=logger)

    X_used_filtered = prepared.X_used_filtered
    labels_df_used = prepared.labels_df_used
    meta = prepared.meta
    feature_indices_for_ds = prepared.feature_indices_for_ds
    valid_row_idx = prepared.valid_row_idx
    use_ops = prepared.use_ops
    kol_window_mode = prepared.kol_window_mode

    X_used_filtered, meta, feature_indices_for_ds = apply_input_representation(
        X_used_filtered=X_used_filtered,
        meta=meta,
        feature_indices_for_ds=feature_indices_for_ds,
        config=config,
        logger=logger,
    )

    groups_used = labels_df_used[L.SAMPLE_ID]

    (
        labels_df_used,
        d_phys_prior,
        op_features,
        operator_feature_cols,
        case_idx,
        kol_prediction_mode,
    ) = _prepare_kol_inputs(
        config=config,
        labels_df_used=labels_df_used,
        logger=logger,
    )

    logger.info("KOL window mode: %s", kol_window_mode if use_ops else "n/a")

    y_all, class_to_idx = _prepare_targets(
        labels_df_used=labels_df_used,
        target_label=target_label,
        task_type=task_type,
        logger=logger,
    )

    if task_type == "multiclass":
        if class_to_idx is None:
            raise ValueError("class_to_idx is None for multiclass task.")
        out_dim = len(class_to_idx)
    else:
        out_dim = 1

    T, F_eff, flat_dim = infer_input_dims(X_used_filtered, feature_indices_for_ds)

    model_name = str(config.model.model_name)
    kol_mode = get_kol_mode(config) if d_phys_prior is not None else None

    if d_phys_prior is not None:
        model_name = f"kol_{kol_mode}_{model_name}"

    window_s = float(config.window_extraction.window_length)
    window_ms = int(round(1000.0 * window_s))
    step_s = float(getattr(config.window_extraction, "step_length_seconds", np.nan))

    logger.info(
        "Setup: target=%s | task_type=%s | n=%d | T=%d F_eff=%d out_dim=%d | window=%dms (%.3fs) | step=%.3fs | spec=%s",
        target_label,
        task_type,
        len(y_all),
        T,
        F_eff,
        out_dim,
        window_ms,
        window_s,
        step_s,
        spec.log_msg,
    )

    seeds = list(map(int, config.training.seeds))
    if len(seeds) == 0:
        raise RuntimeError("No training seeds provided.")

    seed = int(seeds[0])
    if len(seeds) > 1:
        logger.info("Multiple seeds configured; using first only for CV: %d", seed)

    env_info = get_env_info()

    wb_run = setup_wandb_logging(
        config,
        target_label,
        task_type,
        labels_df_used,
        groups_used,
        F_eff,
        model_name,
        window_s,
        window_ms,
        seed,
        env_info,
    )

    run_id = wb_run.id if wb_run is not None else env_info["timestamp"]
    topology = str(config.dataset.topology)

    run_out_dir, line_filter, kol_mode_name, window_mode_name = _make_run_output_dir(
        config=config,
        topology=topology,
        run_id=run_id,
    )

    n_splits = int(getattr(config.training, "n_splits", 5))

    groups_np = (
        groups_used.to_numpy()
        if hasattr(groups_used, "to_numpy")
        else np.asarray(groups_used)
    )

    cv_mode = str(getattr(config.training, "cv_mode", "group")).lower().strip()
    cv_stratify_col = str(
        getattr(config.training, "cv_stratify_col", "y_fault_location")
    )

    logger.info(
        "CV setup | mode=%s | stratify_col=%s | n_splits=%d",
        cv_mode,
        cv_stratify_col,
        n_splits,
    )

    splits = build_cv_splits_stratified(
        y_all=y_all,
        groups_np=groups_np,
        task_type=task_type,
        n_splits=n_splits,
        seed=int(config.training.split_seed),
        labels_df=labels_df_used,
        cv_mode=cv_mode,
        stratify_col=cv_stratify_col,
    )

    best_lr, best_wd, eval_only, resave_eval_only = select_best_lr_wd(
        config=config,
        X_used=X_used_filtered,
        feature_indices_for_ds=(
            list(feature_indices_for_ds) if feature_indices_for_ds is not None else None
        ),
        task_type=task_type,
        criterion=spec.criterion,
        primary_name=spec.primary_metric,
        higher_is_better=spec.higher_is_better,
        valid_row_idx=valid_row_idx,
        labels_df_used=labels_df_used,
        y_all=y_all,
        out_dim=out_dim,
        F_eff=F_eff,
        flat_dim=flat_dim,
        seed=seed,
        n_splits=n_splits,
        logger=logger,
    )

    if wb_run is not None:
        wb_run.summary["protocol/n_splits"] = int(n_splits)
        wb_run.config.update(
            {"lr_used": float(best_lr), "weight_decay_used": float(best_wd)},
            allow_val_change=True,
        )
        wb_run.summary["opt/lr_used"] = float(best_lr)
        wb_run.summary["opt/wd_used"] = float(best_wd)

    all_fold_metrics: list[dict[str, Any]] = []
    ckpt_dir = str(getattr(config.training, "ckpt_dir", "outputs/checkpoints"))

    include_groups = config.training.feature_groups_include

    for fold_idx, (train_pool_idx, test_idx) in enumerate(splits, start=0):
        metrics_row = run_one_fold(
            fold_idx=fold_idx,
            n_splits=n_splits,
            train_pool_idx=train_pool_idx,
            test_idx=test_idx,
            config=config,
            X_used_filtered=X_used_filtered,
            y_all=y_all,
            labels_df_used=labels_df_used,
            groups_used=groups_used,
            feature_indices_for_ds=feature_indices_for_ds,
            task_type=task_type,
            criterion=spec.criterion,
            primary_name=spec.primary_metric,
            higher_is_better=spec.higher_is_better,
            d_phys_prior=d_phys_prior,
            case_idx=case_idx,
            op_features=op_features,
            kol_prediction_mode=kol_prediction_mode,
            model_name=model_name,
            F_eff=F_eff,
            flat_dim=flat_dim,
            out_dim=out_dim,
            best_lr=float(best_lr),
            best_wd=float(best_wd),
            seed=seed,
            device=device,
            run_out_dir=run_out_dir,
            ckpt_dir=ckpt_dir,
            class_to_idx=class_to_idx,
            target_label=target_label,
            include_groups=include_groups,
            window_ms=window_ms,
            valid_row_idx=valid_row_idx,
            eval_only=bool(eval_only),
            resave_eval_only=bool(resave_eval_only),
            logger=logger,
        )
        all_fold_metrics.append(metrics_row)

    return _aggregate_and_save_metrics(
        all_fold_metrics=all_fold_metrics,
        run_out_dir=run_out_dir,
        wb_run=wb_run,
        logger=logger,
    )
