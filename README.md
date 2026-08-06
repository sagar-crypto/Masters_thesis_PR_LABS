# Physics-informed fault location (master thesis)

This repository contains the research for a master's thesis on transmission-line fault location. It compares standard physics estimators, waveform-only GRUs, legacy direct correction, and bounded physics adaptation. The frozen Chapter 4 matrix contains 14 experiments across the 90 kV Protect and 110 kV EventBench cohorts.

## Install and inspect

Python 3.12 is required. Initialize the private dependency and install the package:

```bash
git submodule update --init --recursive
python -m pip install -e '.[dev]'
python -m KOL.cli.print_config experiment=C110-1E
python -m KOL.cli.validate experiment=C110-1E
pytest
```

Set `KOL_DATA_ROOT` (and, when needed, `KOL_WAVEFORM_ROOT`, `KOL_MODEL_INPUT_ROOT`, `KOL_TOPOLOGY_FILE`, `KOL_THIRD_PARTY_ROOT`, or `KOL_OUTPUT_ROOT`) rather than editing canonical configs. New runs are isolated in `outputs/reproducibility_validation/hydra_v1/`.

The waveforms, prepared model inputs, and private `dl_fault_repo` implementation are not distributed here. Missing `dl_psp` or `psp_helper` is a hard error; no replacement implementation is used.

## Workflows

Metadata-only validation needs no data. For an analysis-only workflow, run `python -m KOL.cli.analyse predictions.csv experiment=L90-1E`; it checks exact matched-prior identity and creates event-level arithmetic means. Read-only checkpoint validation uses `python -m KOL.cli.evaluate fold0.pt experiment=L110-2E --saved-splits splits/`. A one-fold smoke invocation is `python -m KOL.cli.train --fold 0 experiment=L110-1E`; full execution omits `--fold` and requires the documented external setup.

All Slurm and shell workflows live in `hpc/`. The canonical Hydra launcher replaces per-experiment job files: `sbatch hpc/run_experiment.sh L110-1E` runs a full experiment, while `sbatch hpc/run_experiment.sh G90 smoke training.epochs=1` runs one fold with an additional Hydra override. Use `validate` as the second argument for metadata-only validation. The older named scripts remain in `hpc/` as historical compatibility workflows for reproducing their original data-preparation and Chapter 4 runs.

Historical results remain under `outputs/chapter4/` and are read-only evidence. Standard P-series physics baselines must not be confused with the exact `prior_pp` supplied as model input for paired C/L comparisons. See [DATA_AVAILABILITY.md](DATA_AVAILABILITY.md) for data-access details.

## Repository structure

- `KOL/`: scientific implementation and stable `KOL.cli` adapters.
- `conf/`: Hydra groups and 14 canonical experiment compositions.
- `tests/`: lightweight metadata and scientific-invariant tests.
- `hpc/`: legacy jobs plus portable `hydra_*.slurm` examples.
- `outputs/`: immutable historical evidence; new runs use the isolated reproducibility subtree.
- `submission_audit/`: preservation and readiness reports.
- Notebooks: Chapter 4, line-location, post-hoc, and recovery notebooks are final analyses; `EDA.ipynb` is exploratory.

## Reproducibility

Use Python 3.12 and `pip install -e '.[dev]'`, initialize the private submodule, and then export the `KOL_*` path variables described above. Start with metadata validation for all configurations. Training preserves seed 42, five event-grouped folds, a 0.15 validation fraction, 60 ms windows, 5 ms steps, and arithmetic event aggregation. Use `--fold 0` only for a smoke run; no training was performed during repository refactoring.

Each new mutating CLI creates a fresh directory below the configured reproducibility root and records the resolved configuration, overrides, command, Git/Python/package/host/Slurm state, feasible input hashes, seed, timestamps, split hashes, terminal status, and failure reason. Checkpoint evaluation loads checkpoints read-only and may reuse saved splits.

## Results traceability

| Experiments | Historical root |
|---|---|
| P90-1E, P90-2E, P110-1E, P110-2E | `outputs/chapter4/physics_baselines/1746087_20260714_193712` |
| G90, G110 | `outputs/chapter4/gru_baselines/1747998` |
| C90-1E, C90-2E, L90-1E, L90-2E | `outputs/chapter4/temp_90kv_afs_check/hybrid_runs/1764759_20260729_150855` |
| C110-1E, L110-1E | `outputs/chapter4/hybrid_single_ended/1751690_20260718_001736` |
| C110-2E, L110-2E | `outputs/chapter4/hybrid_double_ended/1751934_20260718_115538` |

`C90-1E-AFS-TMP`, `L90-1E-AFS-TMP`, `C90-2E-AFS-TMP`, and `L90-2E-AFS-TMP` are source-run names for the four canonical 90 kV hybrids, not additional experiments. P-series outputs are standard physics baselines; C/L comparisons use the exact matched model-input prior recorded in predictions.
