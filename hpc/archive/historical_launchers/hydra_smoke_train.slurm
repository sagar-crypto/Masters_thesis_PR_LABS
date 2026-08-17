#!/usr/bin/env bash
#SBATCH --job-name=kol-smoke
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:1
set -euo pipefail
: "${KOL_REPO_ROOT:?export KOL_REPO_ROOT}"
: "${KOL_DATA_ROOT:?export KOL_DATA_ROOT}"
cd "$KOL_REPO_ROOT"
python -m KOL.cli.train --fold 0 "experiment=${KOL_EXPERIMENT:-L110-1E}"
