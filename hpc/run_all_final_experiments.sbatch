#!/bin/bash -l
#SBATCH --job-name=kol-final
#SBATCH --array=0-13
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --partition=v100
#SBATCH --gres=gpu:v100:1
#SBATCH --output=./hpc/hpc_logs/%x-%A_%a-on-%N.out
#SBATCH --error=./hpc/hpc_logs/%x-%A_%a-on-%N.err
#SBATCH --export=ALL
set -euo pipefail
if (( $# > 1 )); then
  echo "Usage: sbatch hpc/run_all_final_experiments.sbatch [validate|smoke|full]" >&2
  exit 2
fi
MODE="${1:-full}"
case "$MODE" in
  validate|smoke|full) ;;
  *) echo "ERROR: Unknown mode: $MODE" >&2; echo "Usage: sbatch hpc/run_all_final_experiments.sbatch [validate|smoke|full]" >&2; exit 2 ;;
esac
: "${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is missing; submit this launcher with sbatch}"
experiments=(
  P90-1E P90-2E P110-1E P110-2E G90 G110
  C90-1E L90-1E C90-2E L90-2E
  C110-1E L110-1E C110-2E L110-2E
)
if (( SLURM_ARRAY_TASK_ID < 0 || SLURM_ARRAY_TASK_ID >= ${#experiments[@]} )); then
  echo "ERROR: SLURM_ARRAY_TASK_ID is out of range: $SLURM_ARRAY_TASK_ID" >&2
  exit 2
fi
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$SCRIPT_DIR/run_experiment.sh" "${experiments[$SLURM_ARRAY_TASK_ID]}" "$MODE"
