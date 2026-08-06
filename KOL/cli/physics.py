import sys
from .common import compose_config, parser, provenance_run, require_private_dependency

def main() -> int:
    args = parser("Evaluate a configured physics baseline").parse_args(); cfg = compose_config(args.overrides)
    with provenance_run(cfg, " ".join(sys.argv)):
        require_private_dependency(cfg)
        from KOL.run_kol_physics_baseline import main as legacy_main
        return legacy_main()

if __name__ == "__main__": raise SystemExit(main())

