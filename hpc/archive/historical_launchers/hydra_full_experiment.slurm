#!/usr/bin/env bash
#SBATCH --job-name=kol-full
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
set -euo pipefail
: "${KOL_REPO_ROOT:?export KOL_REPO_ROOT}"
: "${KOL_DATA_ROOT:?export KOL_DATA_ROOT}"
cd "$KOL_REPO_ROOT"
python -m KOL.cli.train "experiment=${KOL_EXPERIMENT:-L110-1E}"

