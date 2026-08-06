import sys
from .common import compose_config, parser, provenance_run, require_private_dependency

def main() -> int:
    p = parser("Train a canonical experiment (full data required)")
    p.add_argument("--fold", type=int, choices=range(5), help="Run only one outer fold")
    args = p.parse_args(); cfg = compose_config(args.overrides)
    with provenance_run(cfg, " ".join(sys.argv)):
        require_private_dependency(cfg)
        raise RuntimeError("Unified adapter validated provenance, but dataset-specific legacy dispatch requires repository data setup; use documented compatibility entry points")

if __name__ == "__main__": raise SystemExit(main())

