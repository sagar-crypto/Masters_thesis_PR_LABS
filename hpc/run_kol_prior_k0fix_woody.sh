#!/bin/bash -l
#SBATCH --job-name=ch4_fresh_priors
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=./hpc/hpc_logs/%x/%x-%j-on-%N.out

set -euo pipefail

# =============================================================================
# FRESH CHAPTER 4 PHYSICS-OPERATOR EXPORTS
#
# 90 kV:
#   operator_side_mode=both
#   operator_window_mode=single_fault_start
#
# 110 kV:
#   operator_side_mode=both
#   operator_window_mode=all_fault_start
#
# The physics exporter may write into VAULT_DIR. Every file created or modified
# by the two commands is copied into one uniquely named persistent run folder.
# =============================================================================

source \
  /home/hpc/iwi5/iwi5305h/miniconda3/etc/profile.d/conda.sh

conda activate Masters_thesis_env_py312

export THESIS_DIR="/home/hpc/iwi5/iwi5305h/Masters_thesis_PR_LABS"
export DL_FAULT_REPO="$THESIS_DIR/third_party/dl_fault_repo"

export VAULT_DIR="/home/vault/iwi5/iwi5305h"
export WINDOWS_PATH="$VAULT_DIR/windows_tmp"

export WINDOW_LENGTH_TAG="0p060"
export WINDOW_LENGTH_FLOAT="0.060"

export PYTHONPATH="$DL_FAULT_REPO/src:$THESIS_DIR:${PYTHONPATH:-}"
export HYDRA_FULL_ERROR=1
export WANDB_MODE=disabled

SLURM_JOB_ID="${SLURM_JOB_ID:-$$}"
RUN_TIMESTAMP="$(date +%Y%m%d_%H%M%S)"

export RUN_ROOT="$THESIS_DIR/outputs/chapter4/fresh_prior_exports/${SLURM_JOB_ID}_${RUN_TIMESTAMP}"

export RAW_EXPORT_ROOT="$RUN_ROOT/raw_exports"
export HYDRA_ROOT="$RUN_ROOT/hydra"
export COMMAND_LOG_ROOT="$RUN_ROOT/command_logs"
export MANIFEST="$RUN_ROOT/fresh_prior_export_manifest.csv"

mkdir -p \
  "./hpc/hpc_logs/${SLURM_JOB_NAME:-ch4_fresh_priors}" \
  "$RAW_EXPORT_ROOT" \
  "$HYDRA_ROOT" \
  "$COMMAND_LOG_ROOT"

cd "$THESIS_DIR"

if [ ! -d "$WINDOWS_PATH" ]; then
    echo "ERROR: Windows directory does not exist:"
    echo "$WINDOWS_PATH"
    exit 1
fi

if [ ! -f "KOL/run_kol_physics_baseline.py" ]; then
    echo "ERROR: Physics baseline runner is missing:"
    echo "$THESIS_DIR/KOL/run_kol_physics_baseline.py"
    exit 1
fi

python -m py_compile \
  KOL/run_kol_physics_baseline.py \
  KOL/common/operator_features.py \
  KOL/datasets/kol_data_preparation.py

printf \
'topology,window_mode,source_path,copied_path,size_bytes,sha256\n' \
> "$MANIFEST"

cat > "$RUN_ROOT/run_environment.txt" <<EOF
slurm_job_id=$SLURM_JOB_ID
hostname=$(hostname)
start_time=$(date --iso-8601=seconds)
python=$(which python)
conda_environment=${CONDA_DEFAULT_ENV:-unknown}
thesis_dir=$THESIS_DIR
vault_dir=$VAULT_DIR
windows_path=$WINDOWS_PATH
window_length=$WINDOW_LENGTH_FLOAT
EOF

echo "============================================================"
echo "Fresh Chapter 4 physics-prior generation"
echo "============================================================"
echo "Job ID:       $SLURM_JOB_ID"
echo "Host:         $(hostname)"
echo "Run root:     $RUN_ROOT"
echo "Windows:      $WINDOWS_PATH"
echo "Start time:   $(date)"
echo "============================================================"


run_topology() {
    local topology="$1"
    local window_mode="$2"

    local topology_export_dir="$RAW_EXPORT_ROOT/$topology"
    local hydra_dir="$HYDRA_ROOT/$topology"
    local command_log="$COMMAND_LOG_ROOT/${topology}.log"
    local marker="$RUN_ROOT/.marker_${topology}"

    mkdir -p \
      "$topology_export_dir" \
      "$hydra_dir"

    echo
    echo "============================================================"
    echo "RUNNING PHYSICS EXPORT"
    echo "============================================================"
    echo "Topology:     $topology"
    echo "Window mode:  $window_mode"
    echo "Hydra dir:    $hydra_dir"
    echo "Copied files: $topology_export_dir"
    echo "============================================================"

    echo
    echo "Matching waveform files:"

    find "$WINDOWS_PATH" \
      -maxdepth 1 \
      -type f \
      -name "*${topology}_W${WINDOW_LENGTH_TAG}_*" \
      -printf "%f\n" \
      | sort \
      | head -50

    local window_file_count

    window_file_count="$(
      find "$WINDOWS_PATH" \
        -maxdepth 1 \
        -type f \
        -name "*${topology}_W${WINDOW_LENGTH_TAG}_*" \
        | wc -l
    )"

    if [ "$window_file_count" -eq 0 ]; then
        echo "ERROR: No matching waveform files found for $topology"
        exit 1
    fi

    # Files modified after this marker belong to this fresh invocation.
    touch "$marker"

    python -u KOL/run_kol_physics_baseline.py \
      dataset="$topology" \
      +training.operator_side_mode=both \
      +training.operator_window_mode="$window_mode" \
      window_extraction.window_length="$WINDOW_LENGTH_FLOAT" \
      window_extraction.windows_local_dir="$WINDOWS_PATH" \
      hydra.run.dir="$hydra_dir" \
      2>&1 | tee "$command_log"

    local copied_count=0

    while IFS= read -r -d '' source_path
    do
        local filename
        local copied_path
        local size_bytes
        local checksum

        filename="$(basename "$source_path")"
        copied_path="$topology_export_dir/$filename"

        cp -p "$source_path" "$copied_path"

        size_bytes="$(stat -c '%s' "$copied_path")"
        checksum="$(sha256sum "$copied_path" | awk '{print $1}')"

        printf \
          '%s,%s,%s,%s,%s,%s\n' \
          "$topology" \
          "$window_mode" \
          "$source_path" \
          "$copied_path" \
          "$size_bytes" \
          "$checksum" \
          >> "$MANIFEST"

        copied_count=$((copied_count + 1))

    done < <(
        find "$VAULT_DIR" \
          -maxdepth 1 \
          -type f \
          -name "kol_operator_features_${topology}_*.csv" \
          -newer "$marker" \
          -print0
    )

    if [ "$copied_count" -eq 0 ]; then
        echo
        echo "ERROR: The command completed, but no newly written operator CSV"
        echo "was detected for $topology."
        echo
        echo "Current potential files are:"

        find "$VAULT_DIR" \
          -maxdepth 1 \
          -type f \
          -name "kol_operator_features_${topology}_*.csv" \
          -printf "%TY-%Tm-%Td %TH:%TM:%TS  %f\n" \
          | sort

        exit 1
    fi

    echo
    echo "Copied $copied_count fresh CSV file(s):"

    find "$topology_export_dir" \
      -maxdepth 1 \
      -type f \
      -printf "%f\t%s bytes\n" \
      | sort
}


# Frozen Chapter 4 protocol.
run_topology \
  hv_double_line_90kv \
  single_fault_start

run_topology \
  hv_double_line_110kv \
  all_fault_start


# Atomically update the latest-run pointer.
LATEST_POINTER_DIR="$THESIS_DIR/outputs/chapter4/fresh_prior_exports"
LATEST_POINTER="$LATEST_POINTER_DIR/LATEST_FRESH_PRIOR_RUN.txt"
LATEST_POINTER_TMP="$LATEST_POINTER_DIR/.LATEST_FRESH_PRIOR_RUN.txt.tmp"

printf '%s\n' "$RUN_ROOT" \
  > "$LATEST_POINTER_TMP"

mv \
  "$LATEST_POINTER_TMP" \
  "$LATEST_POINTER"

echo
echo "============================================================"
echo "FRESH PRIOR EXPORTS COMPLETED"
echo "============================================================"
echo "Run directory:"
echo "$RUN_ROOT"
echo
echo "Manifest:"
echo "$MANIFEST"
echo
echo "Latest-run pointer:"
echo "$LATEST_POINTER"
echo
echo "Files:"
find "$RAW_EXPORT_ROOT" \
  -type f \
  -printf "%P\t%s bytes\n" \
  | sort
echo
echo "End time: $(date)"
BASH
