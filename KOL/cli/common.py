from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import socket
import subprocess
import sys
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[2]
CONF_ROOT = REPO_ROOT / "conf"


def parser(description: str, *, positional: tuple[str, ...] = ()) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)
    for item in positional:
        p.add_argument(item)
    p.add_argument("overrides", nargs="*", help="Hydra overrides, e.g. experiment=C110-1E")
    return p


def compose_config(overrides: list[str] | None = None) -> DictConfig:
    with initialize_config_dir(version_base="1.3", config_dir=str(CONF_ROOT)):
        cfg = compose(config_name="config", overrides=overrides or [], return_hydra_config=False)
    OmegaConf.resolve(cfg)
    return cfg


def sha256(path: Path) -> str | None:
    if not path.is_file() or path.stat().st_size > 1_000_000_000:
        return None
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def git_value(*args: str) -> str:
    result = subprocess.run(["git", *args], cwd=REPO_ROOT, text=True, capture_output=True, check=False)
    return result.stdout.strip()


def package_versions() -> dict[str, str]:
    names = ("hydra-core", "omegaconf", "numpy", "pandas", "torch", "scikit-learn", "wandb", "networkx", "pyarrow")
    values = {}
    for name in names:
        try:
            values[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            values[name] = "unavailable"
    return values


@contextmanager
def provenance_run(cfg: DictConfig, command: str):
    output = Path(str(cfg.paths.output_root)) / str(cfg.experiment.id) / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output.mkdir(parents=True, exist_ok=False)
    (output / "config.yaml").write_text(OmegaConf.to_yaml(cfg, resolve=True), encoding="utf-8")
    metadata = {
        "status": "running", "started_at": datetime.now(timezone.utc).isoformat(),
        "command": command, "overrides": [part for part in command.split() if "=" in part], "seed": cfg.seed,
        "git_commit": git_value("rev-parse", "HEAD"), "git_dirty": bool(git_value("status", "--porcelain")),
        "python": sys.version, "packages": package_versions(), "hostname": socket.gethostname(),
        "platform": platform.platform(), "slurm": {k: v for k, v in os.environ.items() if k.startswith("SLURM_")},
        "inputs": {}, "split_hashes": [],
    }
    for key in ("waveform_path", "model_input_path"):
        value = Path(str(cfg.dataset[key]))
        metadata["inputs"][key] = {"path": str(value), "sha256": sha256(value)}
    (output / "provenance.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    try:
        yield output
    except Exception as exc:
        metadata.update(status="failed", failure_reason=f"{type(exc).__name__}: {exc}")
        raise
    else:
        metadata["status"] = "complete"
    finally:
        metadata["ended_at"] = datetime.now(timezone.utc).isoformat()
        (output / "provenance.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def require_private_dependency(cfg: DictConfig) -> None:
    root = Path(str(cfg.paths.third_party_root))
    if not root.exists():
        raise RuntimeError("Private dl_fault_repo submodule unavailable; run git submodule update --init --recursive")
    src = str(root / "src")
    if src not in sys.path:
        sys.path.insert(0, src)
    try:
        __import__("dl_psp")
        __import__("psp_helper")
    except ImportError as exc:
        raise RuntimeError("Private dl_psp/psp_helper dependency is unavailable; no substitute is permitted") from exc


def validate_metadata(cfg: DictConfig) -> list[str]:
    errors = []
    if cfg.folds != 5 or cfg.seed != 42 or cfg.group_column != "sample_id":
        errors.append("canonical split metadata differs from five folds, seed 42, grouped sample_id")
    if cfg.protocol.aggregation != "arithmetic_mean":
        errors.append("event aggregation is not arithmetic_mean")
    if "outputs/reproducibility_validation/" not in str(cfg.paths.output_root):
        errors.append("output root is outside the canonical isolated root")
    if cfg.prior.mode == "two_ended" and cfg.prior.operator_features:
        errors.append("two-ended prior must not include auxiliary operator features")
    return errors
