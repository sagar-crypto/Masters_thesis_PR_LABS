# Physics-informed fault location (master thesis)

This repository is the tracked implementation for a master's thesis on transmission-line fault location. It is intended as a handoff snapshot for research engineers: the public tree contains the experiment compositions, orchestration adapters, scientific processing code, HPC entry points, and analysis notebooks, but not the private implementation or datasets needed for deep validation and execution.

## Canonical experiment matrix

The 14 configurations under `conf/experiment/` combine two voltage cohorts with four method families. `1E` and `2E` denote one-ended and two-ended physics inputs.

| Family | Purpose | Canonical experiment IDs |
|---|---|---|
| P | Physics/operator baseline | `P90-1E`, `P90-2E`, `P110-1E`, `P110-2E` |
| G | Waveform-only GRU | `G90`, `G110` |
| C | Legacy direct correction of a physics prior | `C90-1E`, `C90-2E`, `C110-1E`, `C110-2E` |
| L | Learned bounded adaptation/fusion of a physics prior | `L90-1E`, `L90-2E`, `L110-1E`, `L110-2E` |

The canonical waveform tensor for each retained row is `384 × 48` (timesteps × features) at 90 kV and `576 × 48` at 110 kV. All configurations use seed 42, five outer folds, event grouping by `sample_id`, a 0.15 group-level validation fraction, 60 ms windows stepped by 5 ms, and arithmetic means when window predictions are aggregated to events. Protocol files also fix the expected cohort row/event counts and, for hybrid runs, the accepted fault-start window indices.

P-series results are standard physics baselines. They are not necessarily identical to the exact `operator_prior_col` stored in prepared model inputs and used for paired C/L comparisons.

## Installation and external assets

Python 3.12 is required. Install the tracked package with:

```bash
python -m pip install -e '.[dev]'
```

Scientific execution additionally requires assets that are not included in a clean clone:

- the private `third_party/dl_fault_repo` submodule, providing `dl_psp` and `psp_helper`;
- raw/windowed waveform data for the Protect 90 kV and EventBench 110 kV cohorts;
- prepared operator/model-input CSV or Parquet files for hybrid experiments; and
- topology/label files used by the physics workflows.

Initialize the private dependency only if you have access:

```bash
git submodule update --init --recursive
```

Canonical configuration should be overridden through environment variables, not edited for a machine-specific installation:

| Variable | Purpose |
|---|---|
| `KOL_REPO_ROOT` | Repository root (defaults to `.`) |
| `KOL_DATA_ROOT` | Common data root |
| `KOL_WAVEFORM_ROOT` | Windowed waveform root |
| `KOL_MODEL_INPUT_ROOT` | Prepared operator/model-input root |
| `KOL_TOPOLOGY_FILE` | 110 kV topology file |
| `KOL_90KV_LABELS` | 90 kV labels file |
| `KOL_THIRD_PARTY_ROOT` | Private dependency checkout |
| `KOL_OUTPUT_ROOT` | New-run output root |

Example path compositions are provided in `conf/paths/local.example.yaml` and `conf/paths/hpc.example.yaml`. Missing private modules are a hard error; the project does not substitute a public or approximate implementation.

## Validation and execution workflows

These stages have deliberately different requirements and costs.

### 1. Inspect and validate metadata

Configuration printing and shallow validation need no private data:

```bash
python -m KOL.cli.print_config experiment=C110-1E
python -m KOL.cli.validate experiment=C110-1E
```

Shallow validation checks canonical split and aggregation metadata. It does not open waveform, prior, or topology files.

### 2. Deep data validation

```bash
python -m KOL.cli.validate --deep experiment=C110-1E
```

Deep validation requires the private dependency and configured assets. It loads and filters the cohort, verifies the effective waveform shape and expected row/event/window counts, builds all five grouped folds, checks that events never cross fold boundaries, and verifies finite matched priors where applicable. `--check-files` is a lighter path-existence check; it is not a substitute for `--deep`.

### 3. Smoke training

```bash
python -m KOL.cli.train --fold 0 --max-epochs 1 \
  --max-train-batches 1 --max-val-batches 1 --max-test-batches 1 \
  experiment=L110-1E
```

This exercises one fold with capped work and writes a new provenance run. It still needs all assets for that experiment and is not evidence of a full scientific reproduction.

### 4. Read-only checkpoint evaluation

```bash
python -m KOL.cli.evaluate fold0.pt experiment=L110-2E --saved-splits splits/
```

Evaluation loads an existing checkpoint without retraining and may reuse saved splits. Its data and checkpoint metadata must match the selected configuration.

### 5. Full HPC execution

Portable Hydra/Slurm entry points and historical compatibility scripts live in `hpc/`:

```bash
sbatch hpc/run_experiment.sh L110-1E
sbatch hpc/run_experiment.sh G90 smoke training.epochs=1
sbatch hpc/run_experiment.sh C110-1E validate
```

The first command runs all configured folds; the second selects the smoke mode and adds a Hydra override; the third performs validation. The older named scripts preserve the original data-preparation and Chapter 4 invocation patterns and may contain site-specific assumptions.

Analysis of an existing prediction export is separate from model execution:

```bash
python -m KOL.cli.analyse predictions.csv experiment=L90-1E
```

It checks exact matched-prior identity before producing arithmetic event-level predictions.

## Repository contents and clean-clone boundary

- `KOL/`: public adapters, data preparation, scientific operators, models, and training orchestration.
- `conf/`: Hydra configuration groups and the 14 canonical experiment compositions.
- `hpc/`: portable launchers plus historical/site-specific shell and Slurm workflows.
- `third_party/dl_fault_repo`: a private Git submodule reference; its implementation is not documented or modified here.
- `06_chapter4_ablation_evidence.ipynb`, `07_chapter4_line_location_analysis.ipynb`, and `post_hocanalysis.ipynb`: final analysis notebooks; `EDA.ipynb` is exploratory.
- `dataset_insights_plots/`: the tracked cohort-summary figures used by the documentation/analysis.

Local data, prepared inputs, checkpoints, provenance runs, and reproducibility outputs are ignored or otherwise not part of the tracked handoff. Historical Chapter 4 output paths referenced by notebooks and scripts are traceability pointers to the original working environment, not distributed results: a clean clone does not contain `outputs/chapter4/`.

## Provenance and limitations

Mutating CLI runs create a timestamped directory below `KOL_OUTPUT_ROOT`. The provenance context records the resolved configuration, command/overrides, Git state, Python and package versions, host/Slurm context, feasible input hashes, timestamps, completion status, and failure reason. Fold workflows save their split indices and model artifacts separately; the top-level provenance schema currently reserves a `split_hashes` field but does not populate it.

Known limitations:

- No automated test files are tracked in this repository, even though development dependencies include `pytest`; `pytest` is therefore not a clean-clone acceptance check.
- Deep validation, training, physics evaluation, and checkpoint inference require private code and external assets.
- Historical checkpoints, prediction exports, reproducibility directories, and Chapter 4 results are not distributed in a clean clone.
- The notebooks and older HPC scripts can refer to those local historical paths and require adaptation outside the original environment.

For a public-tree sanity check, compile `KOL`, run shallow validation across all experiment IDs, exercise configuration printing and CLI help, and run `git diff --check`.
