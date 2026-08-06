#!/bin/bash -l
#SBATCH --job-name=kol-hydra
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:v100:1
#SBATCH --partition=v100
#SBATCH --output=./hpc/hpc_logs/%x-%j-on-%N.out
#SBATCH --error=./hpc/hpc_logs/%x-%j-on-%N.err
#SBATCH --export=NONE

# Run any canonical experiment through the Hydra configuration in conf/.
#
# Usage:
#   sbatch hpc/run_experiment.sh EXPERIMENT [MODE] [HYDRA_OVERRIDE ...]
#
# Examples:
#   sbatch hpc/run_experiment.sh L110-1E
#   sbatch hpc/run_experiment.sh G90 smoke training.epochs=1
#   sbatch hpc/run_experiment.sh C110-2E validate
#
# EXPERIMENT is one of P90-1E, P90-2E, P110-1E, P110-2E, G90, G110,
# C90-1E, C90-2E, C110-1E, C110-2E, L90-1E, L90-2E, L110-1E, or L110-2E.
# MODE is full (default), smoke, or validate. Arguments after MODE are passed
# unchanged to Hydra, allowing configuration changes without another job file.

set -eo pipefail
unset SLURM_EXPORT_ENV

EXPERIMENT="${1:-L110-1E}"
MODE="${2:-full}"

if [ "$#" -ge 2 ]; then
    shift 2
elif [ "$#" -eq 1 ]; then
    shift 1
fi

case "$EXPERIMENT" in
    P90-1E|P90-2E|P110-1E|P110-2E|G90|G110|C90-1E|C90-2E|C110-1E|C110-2E|L90-1E|L90-2E|L110-1E|L110-2E) ;;
    *)
        echo "Unknown experiment: $EXPERIMENT"
        exit 2
        ;;
esac

case "$MODE" in
    full|smoke|validate) ;;
    *)
        echo "Unknown mode: $MODE (expected full, smoke, or validate)"
        exit 2
        ;;
esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

mkdir -p "$PROJECT_DIR/hpc/hpc_logs"
cd "$PROJECT_DIR"

echo "Host:       $(hostname)"
echo "Job ID:     ${SLURM_JOB_ID:-local}"
echo "Experiment: $EXPERIMENT"
echo "Mode:       $MODE"
echo "Project:    $PROJECT_DIR"
echo "Start:      $(date --iso-8601=seconds)"

# Adjust these two lines if the target cluster exposes Python differently.
module load python/3.12-conda
source /home/hpc/iwi5/iwi5305h/miniconda3/etc/profile.d/conda.sh
conda activate Masters_thesis_env_py312

export KOL_REPO_ROOT="$PROJECT_DIR"
export HYDRA_FULL_ERROR=1
export WANDB_MODE=disabled

case "$MODE" in
    validate)
        python -m KOL.cli.validate "experiment=$EXPERIMENT" "$@"
        ;;
    smoke)
        python -m KOL.cli.train --fold 0 "experiment=$EXPERIMENT" "$@"
        ;;
    full)
        python -m KOL.cli.train "experiment=$EXPERIMENT" "$@"
        ;;
esac

echo "Completed: $(date --iso-8601=seconds)"
