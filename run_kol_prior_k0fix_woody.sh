#!/bin/bash -l
#SBATCH --job-name=kol_prior_k0fix
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=2:00:00
#SBATCH --output=./hpc/hpc_logs/%x/%x-%j-on-%N.out

# If Woody requires explicit GPU request, uncomment this:
# #SBATCH --gres=gpu:1

# -------------------------------
# Setup logging and environment
# -------------------------------
SLURM_JOB_NAME="${SLURM_JOB_NAME:-kol_prior_k0fix}"
SLURM_JOB_ID="${SLURM_JOB_ID:-$$}"
TMPDIR="${TMPDIR:-/tmp}"

mkdir -p "./hpc/hpc_logs/$SLURM_JOB_NAME"

echo "Your job is running on $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "TMPDIR: $TMPDIR"
echo "Start time: $(date)"

export http_proxy=http://proxy.nhr.fau.de:80
export https_proxy=http://proxy.nhr.fau.de:80

source /home/hpc/iwi5/iwi5305h/miniconda3/etc/profile.d/conda.sh
conda activate Masters_thesis_env_py312

# ============================================================
# USER SETTINGS ONLY
# ============================================================

TOPOLOGY="hv_double_line_110kv"
TOPOLOGY_CONFIG="hv_double_line_110kv"

WINDOW_LENGTH="0p060"
W_FLOAT="${WINDOW_LENGTH/p/.}"

DATASET_DIR="/home/vault/iwi5/iwi5305h"
THESIS_DIR="/home/hpc/iwi5/iwi5305h/Masters_thesis_PR_LABS"
DL_FAULT_REPO="$THESIS_DIR/third_party/dl_fault_repo"

WINDOWS_PATH="$DATASET_DIR/windows_tmp"

RAW_OUT="$DATASET_DIR/kol_operator_features_hv_double_line_110kv_both_all_fault_start_i0res_seq_bothfix.csv"
K0FIX_OUT="$DATASET_DIR/kol_operator_features_hv_double_line_110kv_both_all_fault_start_k0fixed.csv"

HYDRA_RUN_DIR="$DATASET_DIR/hydra_prior_runs/k0fix_${SLURM_JOB_ID}"

# ============================================================
# Internal logic below this line
# ============================================================

echo "Topology config: $TOPOLOGY_CONFIG"
echo "Window length: $W_FLOAT"
echo "Dataset dir: $DATASET_DIR"
echo "Thesis dir: $THESIS_DIR"
echo "Windows path: $WINDOWS_PATH"
echo "Hydra run dir: $HYDRA_RUN_DIR"

export PYTHONPATH="${DL_FAULT_REPO}/src:${THESIS_DIR}:${PYTHONPATH:-}"

cd "$THESIS_DIR"

mkdir -p "$HYDRA_RUN_DIR"

echo "Checking required paths..."

if [ ! -d "$WINDOWS_PATH" ]; then
  echo "ERROR: windows path does not exist:"
  echo "$WINDOWS_PATH"
  exit 1
fi

if [ ! -f "KOL/run_kol_physics_baseline.py" ]; then
  echo "ERROR: KOL/run_kol_physics_baseline.py not found in:"
  pwd
  exit 1
fi

echo "Window files found:"
find "$WINDOWS_PATH" -maxdepth 1 -type f -name "*${TOPOLOGY}_W${WINDOW_LENGTH}_*" | head

echo "Starting KOL physics-prior CSV generation..."

python KOL/run_kol_physics_baseline.py \
  dataset="$TOPOLOGY_CONFIG" \
  +training.operator_side_mode=both \
  +training.operator_window_mode=all_fault_start \
  window_extraction.window_length="$W_FLOAT" \
  window_extraction.windows_local_dir="$WINDOWS_PATH" \
  hydra.run.dir="$HYDRA_RUN_DIR"

echo "Physics-prior generation completed."

echo "Checking expected output:"
echo "$RAW_OUT"

if [ -f "$RAW_OUT" ]; then
  cp "$RAW_OUT" "$K0FIX_OUT"
  echo "Copied k0-fixed output to:"
  echo "$K0FIX_OUT"
else
  echo "ERROR: Expected output CSV not found:"
  echo "$RAW_OUT"
  exit 1
fi

echo "Job completed successfully."
echo "End time: $(date)"
echo "-----------------------------------"
