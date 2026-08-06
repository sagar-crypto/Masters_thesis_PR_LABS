import sys
from pathlib import Path
import torch
from .common import compose_config, parser, provenance_run

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
        (out / "checkpoint_validation.txt").write_text("checkpoint loaded read-only; inference requires configured external data\n")
    if checkpoint.stat().st_mtime_ns != before: raise RuntimeError("checkpoint was modified")
    return 0

if __name__ == "__main__": raise SystemExit(main())

