#!/usr/bin/env bash
#SBATCH --job-name=kol-eval
#SBATCH --time=01:00:00
set -euo pipefail
: "${KOL_REPO_ROOT:?export KOL_REPO_ROOT}"
: "${KOL_CHECKPOINT:?export KOL_CHECKPOINT}"
cd "$KOL_REPO_ROOT"
python -m KOL.cli.evaluate "$KOL_CHECKPOINT" "experiment=${KOL_EXPERIMENT:-L110-2E}" ${KOL_SAVED_SPLITS:+--saved-splits "$KOL_SAVED_SPLITS"}
