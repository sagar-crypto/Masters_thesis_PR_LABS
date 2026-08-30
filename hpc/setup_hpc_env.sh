#!/usr/bin/env bash

# Shared runtime environment for the canonical HPC launchers. Source this file.
set -euo pipefail

export THESIS_DIR="${THESIS_DIR:-/home/hpc/iwi5/iwi5305h/Masters_thesis_PR_LABS}"
export VAULT_ROOT="${VAULT_ROOT:-/home/vault/iwi5/iwi5305h}"
export SOURCE_WINDOWS_DIR="${SOURCE_WINDOWS_DIR:-$VAULT_ROOT/windows_tmp}"
export THIRD_PARTY_DIR="${THIRD_PARTY_DIR:-$THESIS_DIR/third_party/dl_fault_repo}"
CONDA_INIT="${CONDA_INIT:-/home/hpc/iwi5/iwi5305h/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENVIRONMENT="${CONDA_ENVIRONMENT:-Masters_thesis_env_py312}"

fail() { echo "ERROR: $*" >&2; return 1; }
require_dir() { [[ -d "$2" ]] || fail "$1 directory is missing: $2"; }
require_file() { [[ -s "$2" ]] || fail "$1 file is missing or empty: $2"; }

require_dir "Thesis repository" "$THESIS_DIR"
require_file "Conda initialization" "$CONDA_INIT"
# shellcheck disable=SC1090
source "$CONDA_INIT"
conda env list | awk '{print $1}' | grep -Fxq "$CONDA_ENVIRONMENT" \
    || fail "Conda environment is unavailable: $CONDA_ENVIRONMENT"
conda activate "$CONDA_ENVIRONMENT" \
    || fail "Could not activate Conda environment: $CONDA_ENVIRONMENT"

require_dir "Vault root" "$VAULT_ROOT"
require_dir "Waveform source" "$SOURCE_WINDOWS_DIR"
require_dir "Third-party dependency" "$THIRD_PARTY_DIR/src"

CANONICAL_ACTIVE_INPUTS_ENV="$THESIS_DIR/outputs/chapter4/model_inputs/unified_active/LATEST_ACTIVE_INPUTS.env"
ACTIVE_INPUTS_ENV="${KOL_ACTIVE_INPUTS_ENV:-$CANONICAL_ACTIVE_INPUTS_ENV}"

set -a
if [[ -n "${KOL_ACTIVE_INPUTS_ENV:-}" || -s "$CANONICAL_ACTIVE_INPUTS_ENV" ]]; then
    require_file "Unified active-input environment" "$ACTIVE_INPUTS_ENV"
    # shellcheck disable=SC1090
    source "$ACTIVE_INPUTS_ENV"
else
    export SINGLE_ENDED_ENV="$THESIS_DIR/outputs/chapter4/model_inputs/caseaware_bestmae/LATEST_CASEAWARE_SINGLE_ENDED_INPUTS.env"
    export TWO_ENDED_ENV="$THESIS_DIR/outputs/chapter4/model_inputs/two_ended_posseq/LATEST_TWO_ENDED_INPUTS.env"
    export INPUT_90_AFS_ENV="$THESIS_DIR/outputs/chapter4/temp_90kv_afs_check/LATEST_INPUTS.env"
    require_file "Single-ended prepared-input environment" "$SINGLE_ENDED_ENV"
    require_file "Two-ended prepared-input environment" "$TWO_ENDED_ENV"
    require_file "90 kV all-fault-start prepared-input environment" "$INPUT_90_AFS_ENV"
    # shellcheck disable=SC1090
    source "$SINGLE_ENDED_ENV"
    # shellcheck disable=SC1090
    source "$TWO_ENDED_ENV"
    # shellcheck disable=SC1090
    source "$INPUT_90_AFS_ENV"
fi
set +a

: "${P90_1E_AFS_PRIOR:?P90_1E_AFS_PRIOR is missing from prepared-input environment}"
: "${P90_2E_AFS_PRIOR:?P90_2E_AFS_PRIOR is missing from prepared-input environment}"
: "${P110_1E_LINE_CASE_PRIOR:?P110_1E_LINE_CASE_PRIOR is missing from prepared-input environment}"
: "${P110_2E_PRIOR:?P110_2E_PRIOR is missing from prepared-input environment}"
require_file "P90-1E selected CSV" "$P90_1E_AFS_PRIOR"
require_file "P90-2E selected CSV" "$P90_2E_AFS_PRIOR"
require_file "P110-1E selected CSV" "$P110_1E_LINE_CASE_PRIOR"
require_file "P110-2E selected CSV" "$P110_2E_PRIOR"

export KOL_REPO_ROOT="$THESIS_DIR"
export KOL_DATA_ROOT="$THESIS_DIR/hpc/runtime_inputs"
export KOL_WAVEFORM_ROOT="$KOL_DATA_ROOT/waveforms"
export KOL_MODEL_INPUT_ROOT="$KOL_DATA_ROOT/model_inputs"
export KOL_TOPOLOGY_FILE="${KOL_TOPOLOGY_FILE:-$VAULT_ROOT/hv_double_line_110kv/graphs/graph_benchmark.pickle}"
export KOL_90KV_LABELS="${KOL_90KV_LABELS:-$VAULT_ROOT/hv_double_line_90kv/labels.csv}"
export KOL_THIRD_PARTY_ROOT="$THIRD_PARTY_DIR"
export HYDRA_FULL_ERROR=1
export WANDB_MODE=disabled
export WANDB_DISABLED=true
export PYTHONPATH="$THESIS_DIR:$THIRD_PARTY_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
require_file "110 kV topology" "$KOL_TOPOLOGY_FILE"
require_file "90 kV labels" "$KOL_90KV_LABELS"

mkdir -p "$THESIS_DIR/hpc/hpc_logs" "$KOL_WAVEFORM_ROOT" "$KOL_MODEL_INPUT_ROOT"
ln -sfn "$SOURCE_WINDOWS_DIR" "$KOL_WAVEFORM_ROOT/protect90"
ln -sfn "$SOURCE_WINDOWS_DIR" "$KOL_WAVEFORM_ROOT/eventbench110"
ln -sfn "$P90_1E_AFS_PRIOR" "$KOL_MODEL_INPUT_ROOT/C90-1E.csv"
ln -sfn "$P90_1E_AFS_PRIOR" "$KOL_MODEL_INPUT_ROOT/L90-1E.csv"
ln -sfn "$P90_2E_AFS_PRIOR" "$KOL_MODEL_INPUT_ROOT/C90-2E.csv"
ln -sfn "$P90_2E_AFS_PRIOR" "$KOL_MODEL_INPUT_ROOT/L90-2E.csv"
ln -sfn "$P110_1E_LINE_CASE_PRIOR" "$KOL_MODEL_INPUT_ROOT/C110-1E.csv"
ln -sfn "$P110_1E_LINE_CASE_PRIOR" "$KOL_MODEL_INPUT_ROOT/L110-1E.csv"
ln -sfn "$P110_2E_PRIOR" "$KOL_MODEL_INPUT_ROOT/C110-2E.csv"
ln -sfn "$P110_2E_PRIOR" "$KOL_MODEL_INPUT_ROOT/L110-2E.csv"
cd "$THESIS_DIR"
