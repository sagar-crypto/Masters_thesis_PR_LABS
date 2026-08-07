import sys
from pathlib import Path
import torch
from .common import compose_config, parser, provenance_run, require_private_dependency
from .adapter import enable_private_imports, to_private_config

def main() -> int:
    p = parser("Read-only checkpoint validation and inference", positional=("checkpoint",))
    p.add_argument("--saved-splits")
    args = p.parse_args(); cfg = compose_config(args.overrides); checkpoint = Path(args.checkpoint)
    before = checkpoint.stat().st_mtime_ns
    with provenance_run(cfg, " ".join(sys.argv)) as out:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        metadata = payload.get("metadata", {}) if isinstance(payload, dict) else {}
        checkpoint_id = metadata.get("experiment_id")
        if checkpoint_id and checkpoint_id != cfg.experiment.id: raise ValueError("checkpoint experiment metadata mismatch")
        if args.saved_splits and not Path(args.saved_splits).exists(): raise FileNotFoundError(args.saved_splits)
        require_private_dependency(cfg); enable_private_imports()
        cfg.execution.evaluate_only = True
        legacy = to_private_config(cfg, output_dir=out)
        legacy.training.ckpt_path = str(checkpoint.resolve())
        legacy.training.eval_only = True
        legacy.training.resave_eval_only = True
        if str(cfg.model.family) == "waveform_gru":
            from dl_psp.models.run_dl_experiment import main as legacy_main
            legacy_main.__wrapped__(legacy)
        else:
            from KOL.run_kol_experiment import run
            run(legacy)
        exports = list(out.rglob("*.csv")) + list(out.rglob("*.parquet"))
        if not exports:
            raise RuntimeError("read-only inference completed without a prediction export")
        (out / "checkpoint_validation.txt").write_text(f"checkpoint loaded read-only; exports={len(exports)}\n")
    if checkpoint.stat().st_mtime_ns != before: raise RuntimeError("checkpoint was modified")
    return 0

if __name__ == "__main__": raise SystemExit(main())
