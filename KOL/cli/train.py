import sys
from pathlib import Path
from .common import compose_config, parser, provenance_run, require_private_dependency
from .adapter import enable_private_imports, to_private_config

def main() -> int:
    p = parser("Train a canonical experiment (full data required)")
    p.add_argument("--fold", type=int, choices=range(5), help="Run only one outer fold")
    p.add_argument("--max-epochs", type=int)
    p.add_argument("--max-train-batches", type=int)
    p.add_argument("--max-val-batches", type=int)
    p.add_argument("--max-test-batches", type=int)
    p.add_argument("--disable-tracking", action="store_true")
    args = p.parse_args(); cfg = compose_config(args.overrides)
    cfg.execution.fold = args.fold
    cfg.execution.max_epochs = args.max_epochs
    cfg.execution.max_train_batches = args.max_train_batches
    cfg.execution.max_val_batches = args.max_val_batches
    cfg.execution.max_test_batches = args.max_test_batches
    cfg.execution.disable_tracking = args.disable_tracking
    with provenance_run(cfg, " ".join(sys.argv)) as out:
        require_private_dependency(cfg)
        enable_private_imports()
        legacy = to_private_config(cfg, output_dir=out)
        family = str(cfg.model.family)
        if family == "physics_only":
            raise ValueError("physics configurations must use `python -m KOL.cli.physics`")
        if family == "waveform_gru":
            from dl_psp.models.run_dl_experiment import main as legacy_main
            legacy_main.__wrapped__(legacy)
        else:
            from KOL.run_kol_experiment import run
            run(legacy)
    return 0

if __name__ == "__main__": raise SystemExit(main())
