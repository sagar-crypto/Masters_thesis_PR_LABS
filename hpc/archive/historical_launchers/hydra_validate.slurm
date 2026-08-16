#!/usr/bin/env bash
#SBATCH --job-name=kol-validate
#SBATCH --time=00:10:00
#SBATCH --cpus-per-task=1
set -euo pipefail
: "${KOL_REPO_ROOT:?export KOL_REPO_ROOT}"
cd "$KOL_REPO_ROOT"
# Example only: conda activate Masters_thesis_env_py312
python -m KOL.cli.validate "experiment=${KOL_EXPERIMENT:-C110-1E}"
