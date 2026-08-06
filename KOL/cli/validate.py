from pathlib import Path
from .common import compose_config, parser, validate_metadata

def main() -> int:
    p = parser("Validate experiment metadata and optionally input files")
    p.add_argument("--check-files", action="store_true")
    args = p.parse_args()
    cfg = compose_config(args.overrides)
    errors = validate_metadata(cfg)
    if args.check_files:
        for key in ("waveform_path", "model_input_path"):
            if not Path(str(cfg.dataset[key])).exists(): errors.append(f"missing {key}: {cfg.dataset[key]}")
    print(f"{cfg.experiment.id}: " + ("VALID" if not errors else "INVALID: " + "; ".join(errors)))
    return int(bool(errors))

if __name__ == "__main__": raise SystemExit(main())

