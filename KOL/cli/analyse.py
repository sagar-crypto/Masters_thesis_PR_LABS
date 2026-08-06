import sys
from pathlib import Path
import numpy as np
import pandas as pd
from .common import compose_config, parser, provenance_run

def aggregate_events(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.groupby("sample_id", as_index=False).agg(y_true=("y_true", "first"), y_pred=("y_pred", "mean"), prior_pp=("prior_pp", "mean"))

def assert_matched_prior(frame: pd.DataFrame) -> None:
    if "model_prior_pp" not in frame or not np.array_equal(frame["prior_pp"].to_numpy(), frame["model_prior_pp"].to_numpy(), equal_nan=True):
        raise ValueError("matched-prior identity check failed")

def main() -> int:
    p = parser("Analyse saved window predictions with matched-prior checks", positional=("predictions",)); args = p.parse_args()
    cfg = compose_config(args.overrides); frame = pd.read_csv(args.predictions); assert_matched_prior(frame)
    with provenance_run(cfg, " ".join(sys.argv)) as out:
        events = aggregate_events(frame); events.to_csv(out / "event_predictions.csv", index=False)
        pd.DataFrame([{"level":"window", "mae": np.mean(np.abs(frame.y_pred-frame.y_true))}, {"level":"event", "mae": np.mean(np.abs(events.y_pred-events.y_true))}]).to_csv(out / "metrics.csv", index=False)
    return 0

if __name__ == "__main__": raise SystemExit(main())

