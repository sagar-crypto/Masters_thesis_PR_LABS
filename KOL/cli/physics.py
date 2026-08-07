import sys
from .common import compose_config, parser, provenance_run, require_private_dependency
from .adapter import enable_private_imports, to_private_config

def main() -> int:
    args = parser("Evaluate a configured physics baseline").parse_args(); cfg = compose_config(args.overrides)
    if str(cfg.model.family) != "physics_only":
        raise ValueError("physics CLI accepts only P-series configurations")
    with provenance_run(cfg, " ".join(sys.argv)) as out:
        require_private_dependency(cfg)
        enable_private_imports()
        from KOL.run_kol_physics_baseline import run
        run(to_private_config(cfg, output_dir=out))
    return 0

if __name__ == "__main__": raise SystemExit(main())
