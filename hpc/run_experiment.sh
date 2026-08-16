#!/bin/bash -l
#SBATCH --job-name=kol-hydra
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --partition=v100
#SBATCH --gres=gpu:v100:1
#SBATCH --output=./hpc/hpc_logs/%x-%j-on-%N.out
#SBATCH --error=./hpc/hpc_logs/%x-%j-on-%N.err
#SBATCH --export=ALL
set -euo pipefail
usage() { echo "Usage: sbatch hpc/run_experiment.sh EXPERIMENT [validate|smoke|full]" >&2; }
if (( $# < 1 || $# > 2 )); then usage; exit 2; fi
EXPERIMENT="$1"
MODE="${2:-full}"
case "$EXPERIMENT" in
  P90-1E|P90-2E|P110-1E|P110-2E|G90|G110|C90-1E|C90-2E|C110-1E|C110-2E|L90-1E|L90-2E|L110-1E|L110-2E) ;;
  *) echo "ERROR: Unknown experiment: $EXPERIMENT" >&2; usage; exit 2 ;;
esac
case "$MODE" in
  validate|smoke|full) ;;
  *) echo "ERROR: Unknown mode: $MODE" >&2; usage; exit 2 ;;
esac
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=setup_hpc_env.sh
source "$SCRIPT_DIR/setup_hpc_env.sh"
if [[ -n "${SLURM_ARRAY_JOB_ID:-}" ]]; then
  JOB_TAG="${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is missing}"
elif [[ -n "${SLURM_JOB_ID:-}" ]]; then
  JOB_TAG="$SLURM_JOB_ID"
else
  JOB_TAG="local_$$"
fi
export KOL_OUTPUT_ROOT="$THESIS_DIR/outputs/reproducibility_validation/hpc/$MODE/$EXPERIMENT/$JOB_TAG"
mkdir -p "$KOL_OUTPUT_ROOT"
echo "Host: $(hostname)"
echo "Job: ${SLURM_JOB_ID:-local}${SLURM_ARRAY_TASK_ID:+/$SLURM_ARRAY_TASK_ID}"
echo "Experiment: $EXPERIMENT"
echo "Mode: $MODE"
echo "Conda: ${CONDA_DEFAULT_ENV:-unknown}"
echo "Output: $KOL_OUTPUT_ROOT"
echo "Start: $(date --iso-8601=seconds)"
case "$MODE" in
  validate) python -m KOL.cli.validate --deep "experiment=$EXPERIMENT" ;;
  smoke)
    if [[ "$EXPERIMENT" == P* ]]; then
      python -m KOL.cli.physics "experiment=$EXPERIMENT"
    else
      python -m KOL.cli.train --fold 0 --max-epochs 1 \
        --max-train-batches 2 --max-val-batches 2 --max-test-batches 2 \
        --disable-tracking "experiment=$EXPERIMENT"
    fi
    ;;
  full)
    if [[ "$EXPERIMENT" == P* ]]; then
      python -m KOL.cli.physics "experiment=$EXPERIMENT"
    else
      python -m KOL.cli.train --disable-tracking "experiment=$EXPERIMENT"
    fi
    ;;
esac
echo "SUCCESS: $EXPERIMENT ($MODE) completed at $(date --iso-8601=seconds)"
