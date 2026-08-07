#!/usr/bin/env bash
#SBATCH --job-name=kol-final
#SBATCH --array=0-13
#SBATCH --output=outputs/reproducibility_validation/preflight/slurm/%A_%a.out
#SBATCH --error=outputs/reproducibility_validation/preflight/slurm/%A_%a.err
# Site wrapper must create the log directory before sbatch and may set partition/account.
set -euo pipefail
: "${KOL_REPO_ROOT:?Set KOL_REPO_ROOT}"
: "${KOL_DATA_ROOT:?Set KOL_DATA_ROOT}"
export KOL_OUTPUT_ROOT="${KOL_OUTPUT_ROOT:-$KOL_REPO_ROOT/outputs/reproducibility_validation/full_rerun}"
export PYTHONPATH="$KOL_REPO_ROOT:$KOL_REPO_ROOT/third_party/dl_fault_repo/src${PYTHONPATH:+:$PYTHONPATH}"
cd "$KOL_REPO_ROOT"
experiments=(P90-1E P90-2E P110-1E P110-2E G90 G110 C90-1E L90-1E C90-2E L90-2E C110-1E L110-1E C110-2E L110-2E)
id="${experiments[$SLURM_ARRAY_TASK_ID]}"
if [[ "$id" == P* ]]; then
  python -m KOL.cli.physics "experiment=$id"
else
  python -m KOL.cli.train "experiment=$id"
fi
