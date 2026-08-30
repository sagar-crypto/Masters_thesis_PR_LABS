"""Build the four canonical, target-free Chapter 4 model-input CSVs."""

from __future__ import annotations

import hashlib
import json
import os
import shlex
import shutil
import tempfile
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from KOL.input_builder_constants import (
    CASE_ALIASES,
    EXCLUDED,
    FEATURES_90,
    FEATURES_110,
    KEYS,
    LINE_ALIASES,
    ROLE_ORDER,
    SPECS,
    TARGET_ALIASES,
)
from KOL.prepare_two_ended_prior_file import bound_prior_values


def normalize_sample_id(value: object) -> str:
    if pd.isna(value):
        return "<missing>"
    text = str(value).strip()
    try:
        number = float(text)
        if np.isfinite(number) and number.is_integer():
            return str(int(number))
    except ValueError:
        pass
    return text


def _first(frame: pd.DataFrame, aliases: list[str], label: str) -> str:
    try:
        return next(column for column in aliases if column in frame.columns)
    except StopIteration as exc:
        raise ValueError(f"Missing {label}; tried {aliases}") from exc


def canonicalize(frame: pd.DataFrame, *, label: str) -> pd.DataFrame:
    missing = sorted(set(KEYS) - set(frame.columns))
    if missing:
        raise ValueError(f"{label}: missing key columns {missing}")
    result = frame.copy()
    result["sample_id"] = result["sample_id"].map(normalize_sample_id)
    window = pd.to_numeric(result["window_idx"], errors="coerce")
    if window.isna().any() or not np.equal(window, np.floor(window)).all():
        raise ValueError(f"{label}: window_idx must contain finite integers")
    result["window_idx"] = window.astype(int)
    result = result.sort_values(KEYS, kind="stable").reset_index(drop=True)
    duplicates = int(result.duplicated(KEYS).sum())
    if duplicates:
        raise ValueError(
            f"{label}: duplicate (sample_id, window_idx) keys: {duplicates}"
        )
    return result


def validate_cohort(frame: pd.DataFrame, *, voltage: str, label: str) -> None:
    spec = SPECS[voltage]
    if len(frame) != spec.rows:
        raise ValueError(f"{label}: expected {spec.rows} rows, found {len(frame)}")
    counts = frame.groupby("sample_id", sort=False).size()
    if len(counts) != spec.events:
        raise ValueError(f"{label}: expected {spec.events} events, found {len(counts)}")
    observed = set(map(int, counts.unique()))
    if observed != spec.event_windows:
        raise ValueError(
            f"{label}: expected windows/event {sorted(spec.event_windows)}, found {sorted(observed)}"
        )
    indices = set(map(int, frame["window_idx"].unique()))
    if indices != spec.window_indices:
        raise ValueError(
            f"{label}: expected window indices {sorted(spec.window_indices)}, found {sorted(indices)}"
        )


def _validate_role(frame: pd.DataFrame, *, role: str, voltage: str, path: Path) -> None:
    if role not in path.name.lower():
        raise ValueError(
            f"{path}: filename does not identify required source role '{role}'"
        )
    if "operator_side_mode" not in frame.columns:
        raise ValueError(f"{path}: missing operator_side_mode role metadata")
    values = set(
        frame["operator_side_mode"].dropna().astype(str).str.lower().str.strip()
    )
    accepted = {role} if role != "opposite" else {"opposite", "remote"}
    if not values or not values <= accepted:
        raise ValueError(
            f"{path}: expected {role!r} operator_side_mode, found {sorted(values)}"
        )
    if voltage not in path.name.lower():
        raise ValueError(f"{path}: filename does not identify {voltage} kV topology")


def _target_pct(series: pd.Series) -> np.ndarray:
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float, copy=True)
    if not np.isfinite(values).all():
        raise ValueError("Target contains non-finite values")
    if np.max(np.abs(values)) <= 1.5:
        values *= 100.0
    return values


def _is_candidate(column: str, role: str) -> bool:
    name = column.lower()
    if (
        not name.startswith("d_")
        or "pct" not in name
        or any(x in name for x in EXCLUDED)
    ):
        return False
    return not (role == "both" and name == "d_phys_real_pct")


def _vector_hash(values: np.ndarray) -> str:
    normalized = np.nan_to_num(
        values.astype(np.float64), nan=1e30, posinf=1e31, neginf=-1e31
    )
    return hashlib.sha256(normalized.tobytes()).hexdigest()


def collect_candidates(frames: dict[str, pd.DataFrame]) -> list[dict[str, object]]:
    candidates: list[dict[str, object]] = []
    seen: set[str] = set()
    for role in sorted(frames, key=ROLE_ORDER.__getitem__):
        frame = frames[role]
        for column in sorted(frame.columns):
            if not _is_candidate(column, role):
                continue
            values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
            fingerprint = _vector_hash(values)
            if fingerprint in seen:
                continue
            seen.add(fingerprint)
            candidates.append({"role": role, "column": column, "values": values})
    if not candidates:
        raise ValueError("No eligible one-ended distance candidates found")
    return candidates


def select_candidates(
    reference: pd.DataFrame, candidates: list[dict[str, object]], group_cols: list[str]
) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame, dict[str, int]]:
    target_col = _first(reference, TARGET_ALIASES, "target column")
    target = _target_pct(reference[target_col])
    selected = np.empty(len(reference), dtype=float)
    selected_name = np.empty(len(reference), dtype=object)
    selected_fallback = np.empty(len(reference), dtype=np.int8)
    mappings: list[dict[str, object]] = []
    totals = {"fallback": 0, "clipped_low": 0, "clipped_high": 0}
    for key, indices in reference.groupby(
        group_cols, dropna=False, sort=True
    ).indices.items():
        idx = np.asarray(indices, dtype=int)
        key_tuple = key if isinstance(key, tuple) else (key,)
        ranked = []
        for candidate in candidates:
            bounded, fallback, counts = bound_prior_values(candidate["values"][idx])
            ranked.append(
                (
                    float(np.mean(np.abs(bounded - target[idx]))),
                    counts["fallback"],
                    ROLE_ORDER[str(candidate["role"])],
                    str(candidate["column"]),
                    candidate,
                    bounded,
                    fallback,
                    counts,
                )
            )
        best = min(ranked, key=lambda row: row[:4])
        mae, _, _, _, candidate, bounded, fallback, counts = best
        selected[idx] = bounded
        label = f"{candidate['role']}::{candidate['column']}"
        selected_name[idx] = label
        selected_fallback[idx] = fallback.astype(np.int8)
        totals = {name: totals[name] + counts[name] for name in totals}
        mappings.append(
            {
                **dict(zip(group_cols, key_tuple)),
                "rows": len(idx),
                "selected_source_role": candidate["role"],
                "selected_column": candidate["column"],
                "selected_mae_pp": mae,
                "fallback_count": counts["fallback"],
            }
        )
    diagnostics = reference[KEYS].copy()
    diagnostics["selected_candidate"] = selected_name
    diagnostics["used_fallback"] = selected_fallback
    diagnostics["target_pct"] = target
    diagnostics["selected_prior_pct"] = selected
    diagnostics["absolute_error_pp"] = np.abs(selected - target)
    return selected, pd.DataFrame(mappings), diagnostics, totals


def _features(
    frames: Iterable[pd.DataFrame], definitions: dict[str, list[str]] | list[str]
) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    items = (
        definitions.items()
        if isinstance(definitions, dict)
        else ((x, [x]) for x in definitions)
    )
    for output, aliases in items:
        found = None
        for frame in frames:
            for alias in aliases:
                if alias in frame:
                    values = pd.to_numeric(frame[alias], errors="coerce").to_numpy(
                        dtype=float
                    )
                    if np.isfinite(values).all():
                        found = values
                        break
            if found is not None:
                break
        if found is None:
            raise ValueError(f"Required feature {output!r} is missing or non-finite")
        result[output] = found
    return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _cohort_statistics(frame: pd.DataFrame) -> dict[str, object]:
    event_sizes = frame.groupby("sample_id", sort=False).size()
    return {
        "rows": len(frame),
        "events": int(frame["sample_id"].nunique()),
        "window_indices": sorted(map(int, frame["window_idx"].unique())),
        "windows_per_event": {
            str(int(size)): int(count)
            for size, count in event_sizes.value_counts().sort_index().items()
        },
    }


def _write_env(path: Path, run_dir: Path, outputs: dict[str, Path]) -> None:
    variables = {
        "ACTIVE_INPUT_DIR": run_dir,
        "P90_1E_AFS_PRIOR": outputs["90_1e"],
        "P90_2E_AFS_PRIOR": outputs["90_2e"],
        "P110_1E_LINE_CASE_PRIOR": outputs["110_1e"],
        "P110_2E_PRIOR": outputs["110_2e"],
    }
    path.write_text(
        "\n".join(
            f"{key}={shlex.quote(str(value))}" for key, value in variables.items()
        )
        + "\n"
    )


def build_inputs(
    *,
    raw_90_default: str | Path,
    raw_90_opposite: str | Path,
    raw_90_both: str | Path,
    raw_110_both: str | Path,
    output_root: str | Path = "outputs/chapter4/model_inputs/unified_active",
    activate: bool = False,
    timestamp: str | None = None,
) -> Path:
    sources = {
        "90_default": Path(raw_90_default).resolve(),
        "90_opposite": Path(raw_90_opposite).resolve(),
        "90_both": Path(raw_90_both).resolve(),
        "110_both": Path(raw_110_both).resolve(),
    }
    for path in sources.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    root = Path(output_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    stamp = timestamp or datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    final_dir = root / stamp
    if final_dir.exists():
        raise FileExistsError(final_dir)
    staging = Path(tempfile.mkdtemp(prefix=f".{stamp}.", dir=root))
    try:
        frames: dict[str, pd.DataFrame] = {}
        for key, path in sources.items():
            voltage, role = key.split("_", 1)
            frame = canonicalize(pd.read_csv(path), label=key)
            _validate_role(frame, role=role, voltage=voltage, path=path)
            validate_cohort(frame, voltage=voltage, label=key)
            frames[key] = frame
        reference_keys = frames["90_both"][KEYS]
        for key in ("90_default", "90_opposite"):
            if not frames[key][KEYS].equals(reference_keys):
                raise ValueError(f"{key}: keys do not match raw_90_both")
        reference90, reference110 = frames["90_both"], frames["110_both"]
        case90 = _first(reference90, CASE_ALIASES, "90 kV fault-case column")
        _first(reference90, LINE_ALIASES, "90 kV fault-line column")
        case110 = _first(reference110, CASE_ALIASES, "110 kV fault-case column")
        line110 = _first(reference110, LINE_ALIASES, "110 kV fault-line column")
        candidates90 = collect_candidates(
            {
                k.split("_", 1)[1]: frames[k]
                for k in ("90_both", "90_default", "90_opposite")
            }
        )
        candidates110 = collect_candidates({"both": reference110})
        prior90, mapping90, diag90, counts90 = select_candidates(
            reference90, candidates90, [case90]
        )
        prior110, mapping110, diag110, counts110 = select_candidates(
            reference110, candidates110, [line110, case110]
        )
        prior90_2e, fallback90_2e, counts90_2e = bound_prior_values(
            reference90["d_two_ended_posseq_plus_pct"]
        )
        prior110_2e, fallback110_2e, counts110_2e = bound_prior_values(
            reference110["d_two_ended_posseq_plus_pct"]
        )
        features90 = _features(
            [frames["90_both"], frames["90_default"], frames["90_opposite"]],
            FEATURES_90,
        )
        features110 = _features([reference110], FEATURES_110)
        artifacts = {
            "90_1e": (
                "90kv_one_ended.csv",
                "d_90kv_afs_case_bestmae_input_pct",
                prior90,
                features90,
            ),
            "90_2e": (
                "90kv_two_ended.csv",
                "d_90kv_afs_two_ended_posseq_input_pct",
                prior90_2e,
                {},
            ),
            "110_1e": (
                "110kv_one_ended.csv",
                "d_110kv_line_case_bestmae_input_pct",
                prior110,
                features110,
            ),
            "110_2e": (
                "110kv_two_ended.csv",
                "d_two_ended_posseq_plus_input_pct",
                prior110_2e,
                {},
            ),
        }
        outputs: dict[str, Path] = {}
        for key, (filename, prior_col, prior, features) in artifacts.items():
            reference = reference90 if key.startswith("90") else reference110
            output = reference[KEYS].copy()
            output[prior_col] = prior.astype(np.float32)
            for name, values in features.items():
                output[name] = values.astype(np.float32)
            expected = KEYS + [prior_col] + list(features)
            if (
                list(output.columns) != expected
                or not np.isfinite(
                    output.drop(columns="sample_id").to_numpy(dtype=float)
                ).all()
                or not output[prior_col].between(0, 100).all()
            ):
                raise RuntimeError(f"Internal validation failed for {key}")
            path = staging / filename
            output.to_csv(path, index=False)
            outputs[key] = path
        audit = staging / "audit"
        audit.mkdir()
        mapping90.to_csv(audit / "90kv_one_ended_mapping.csv", index=False)
        mapping110.to_csv(audit / "110kv_one_ended_mapping.csv", index=False)
        diag90.to_csv(
            audit / "90kv_row_diagnostics.csv.gz", index=False, compression="gzip"
        )
        diag110.to_csv(
            audit / "110kv_row_diagnostics.csv.gz", index=False, compression="gzip"
        )
        disclosure = {
            "warning": "One-ended candidates are selected using target MAE over the full cohort; this is target-informed and not fold-safe.",
            "90kv_grouping": "fault case",
            "110kv_grouping": "fault line and fault case",
        }
        (audit / "TARGET_INFORMED_SELECTION.json").write_text(
            json.dumps(disclosure, indent=2) + "\n"
        )
        manifest = {
            "schema_version": 1,
            "created_utc": datetime.now(UTC).isoformat(),
            "sources": {
                key: {"path": str(path), "sha256": _sha256(path)}
                for key, path in sources.items()
            },
            "cohorts": {
                "90kv": _cohort_statistics(reference90),
                "110kv": _cohort_statistics(reference110),
            },
            "transform_counts": {
                "90kv_1e": counts90,
                "90kv_2e": counts90_2e,
                "110kv_1e": counts110,
                "110kv_2e": counts110_2e,
            },
            "two_ended_fallback_rows": {
                "90kv": int(fallback90_2e.sum()),
                "110kv": int(fallback110_2e.sum()),
            },
            "outputs": {
                key: {
                    "path": path.name,
                    "sha256": _sha256(path),
                    "rows": len(pd.read_csv(path)),
                }
                for key, path in outputs.items()
            },
            "disclosure": disclosure["warning"],
        }
        (staging / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        )
        final_outputs = {key: final_dir / path.name for key, path in outputs.items()}
        _write_env(staging / "ACTIVE_INPUTS.env", final_dir, final_outputs)
        os.replace(staging, final_dir)
        if activate:
            latest = root / "LATEST_ACTIVE_INPUTS.env"
            temp_env = root / f".{latest.name}.{os.getpid()}"
            shutil.copyfile(final_dir / "ACTIVE_INPUTS.env", temp_env)
            os.replace(temp_env, latest)
        return final_dir
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
