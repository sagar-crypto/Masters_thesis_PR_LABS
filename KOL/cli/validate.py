"""Family-aware shallow and deep validation for canonical experiments."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from omegaconf import OmegaConf

from .adapter import enable_private_imports, to_private_config
from .common import compose_config, parser, require_private_dependency, validate_metadata


def experiment_family(cfg) -> str:
    family = str(cfg.model.family)
    return "physics" if family == "physics_only" else ("waveform" if family == "waveform_gru" else "hybrid")


def required_input_paths(cfg) -> dict[str, Path]:
    paths = {
        "data_root": Path(str(cfg.paths.data_root)),
        "waveform_path": Path(str(cfg.dataset.waveform_path)),
    }
    family = experiment_family(cfg)
    if family == "hybrid":
        paths["model_input_path"] = Path(str(cfg.dataset.model_input_path))
    if family == "physics":
        if int(cfg.dataset.voltage_kv) == 110:
            paths["topology_file"] = Path(str(cfg.paths.topology_file))
        else:
            paths["labels_90_file"] = Path(str(cfg.paths.labels_90_file))
    return paths


def check_files(cfg) -> list[str]:
    return [f"missing {name}: {path}" for name, path in required_input_paths(cfg).items() if not path.exists()]


def _report_base(cfg) -> dict[str, Any]:
    physics = experiment_family(cfg) == "physics"
    return {
        "experiment": str(cfg.experiment.id),
        "status": "invalid",
        "family": "physics/operator baseline" if physics else str(cfg.model.family),
        "target": str(cfg.target),
        "cohort_mode": str(cfg.protocol.cohort_mode),
        "expected_rows": int(cfg.protocol.expected_rows),
        "expected_events": int(cfg.protocol.expected_events),
        "observed_rows": None,
        "observed_events": None,
        "window_indices": None if cfg.protocol.window_indices is None else list(cfg.protocol.window_indices),
        "folds": int(cfg.folds),
        "split_mode": None if physics else ("group" if experiment_family(cfg) == "waveform" else "stratified_location"),
        "model_input_path": None if experiment_family(cfg) != "hybrid" else str(cfg.dataset.model_input_path),
        "model_class": None if physics else cfg.model.class_name,
        "hidden_size": None if physics else int(cfg.model.hidden_size),
        "num_layers": None if physics else int(cfg.model.num_layers),
        "dropout": None if physics else float(cfg.model.dropout),
        "bidirectional": None if physics else False,
        "waveform_shape": None,
        "errors": [],
    }


def deep_validate(cfg) -> dict[str, Any]:
    report = _report_base(cfg)
    errors = validate_metadata(cfg) + check_files(cfg)
    if errors:
        report["errors"] = errors
        return report
    try:
        from KOL.common.cv_utils import audit_cohort
        require_private_dependency(cfg)
        enable_private_imports()
        legacy = to_private_config(cfg)
        family = experiment_family(cfg)
        if family == "physics":
            from KOL.common.operator_data_prep import load_and_filter_operator_data, apply_operator_window_selection
            from sklearn.model_selection import GroupKFold
            df, X, _meta, fs, f_nom, *_ = load_and_filter_operator_data(legacy)
            df, X, _ = apply_operator_window_selection(df=df, X_eval=X, config=legacy, fs=fs, f_nom=f_nom)
            splits = list(GroupKFold(n_splits=int(cfg.folds)).split(df, groups=df["sample_id"]))
            prior = None
        else:
            from KOL.datasets.kol_data_preparation import load_filtered_training_data
            from KOL.common.cv_utils import build_cv_splits_stratified
            from KOL.common.operator_features import load_operator_inputs_if_enabled
            import logging
            prepared = load_filtered_training_data(config=legacy, logger=logging.getLogger(__name__))
            df, X = prepared.labels_df_used, prepared.X_used_filtered
            effective_shape = [int(X.shape[1]), int(X.shape[2]) if prepared.feature_indices_for_ds is None else len(prepared.feature_indices_for_ds)]
            prior, _, _ = load_operator_inputs_if_enabled(config=legacy, labels_df_used=df)
            splits = build_cv_splits_stratified(
                y_all=np.asarray(df[str(cfg.target)]), groups_np=df["sample_id"].to_numpy(),
                task_type="regression", n_splits=int(cfg.folds), seed=int(cfg.seed), labels_df=df,
                cv_mode="group" if family == "waveform" else "stratified_location",
                stratify_col="y_fault_location",
            )
        result = audit_cohort(
            df, splits, expected_rows=int(cfg.protocol.expected_rows),
            expected_events=int(cfg.protocol.expected_events), expected_folds=int(cfg.folds),
            expected_windows=cfg.protocol.window_indices, prior_values=prior,
        )
        report.update(status="valid", observed_rows=result.rows, observed_events=result.events,
                      waveform_shape=list(X.shape[1:]) if family == "physics" else effective_shape)
    except Exception as exc:
        report["errors"] = [f"{type(exc).__name__}: {exc}"]
    return report


def main() -> int:
    p = parser("Validate experiment metadata and optionally input files")
    p.add_argument("--check-files", action="store_true")
    p.add_argument("--deep", action="store_true")
    args = p.parse_args()
    cfg = compose_config(args.overrides)
    if args.deep:
        report = deep_validate(cfg)
        print(OmegaConf.to_yaml(OmegaConf.create(report), resolve=True, sort_keys=False).rstrip())
        return int(report["status"] != "valid")
    errors = validate_metadata(cfg)
    if args.check_files:
        errors.extend(check_files(cfg))
    print(f"{cfg.experiment.id}: " + ("VALID" if not errors else "INVALID: " + "; ".join(errors)))
    return int(bool(errors))


if __name__ == "__main__":
    raise SystemExit(main())
