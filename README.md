# Graph Imputation for Outage-Resilient Air-Quality Forecasting

This repository contains the compact research code and prepared public data for
**Graph Imputation for Outage-Resilient Probabilistic Forecasting in Air-Quality
Sensor Networks**.

The implementation evaluates station-local probabilistic forecasters after a target
sensor loses part of its recent history. Fixed or adaptive geographic graph imputation
uses only contemporaneously observed neighboring stations to complete the missing
history before forecasting.

## What is included

- the core model, preprocessing, training, calibration, and evaluation code;
- scripts for downloading and validating the original U.S. EPA AQS observations;
- prepared hourly tensors for the seven-station Salt Lake County and Clark County
  networks;
- protocol files defining the chronological splits, outage interventions, comparators,
  seeds, and statistical endpoints; and
- focused unit tests for the released implementation.

This public package intentionally excludes trained checkpoints, predictions, experiment
results, manuscript sources, publication figures and tables, reviewer-only diagnostics,
and scripts that create paper illustrations. Running the experiment scripts writes new
artifacts under `experiment_protocol/`, which is ignored by Git.

## Repository layout

```text
data_external/
  epa_aqs/                    Salt Lake County prepared data and provenance
  epa_aqs_las_vegas/          Clark County prepared data and provenance
journal_protocol/             Frozen experimental specifications
scripts/                      Data, training, evaluation, and audit entry points
src/aqriskformer/             Core implementation
tests/                        Focused code and protocol tests
```

See [DATA.md](DATA.md) for the data contents, units, provenance, and reuse notes.
The four released tensors can be verified against `DATA_SHA256SUMS`.

## Installation

Python 3.10--3.12 is supported. A CUDA-enabled PyTorch installation is recommended for
the full neural-model protocol, but the tests and smoke run can use CPU.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e ".[dev]"
```

On Linux or macOS, activate the environment with `source .venv/bin/activate`.

## Verify the release

```powershell
python -m pytest -q
python scripts\run_journal_development.py --stage plan
python scripts\run_journal_development.py --stage smoke --device cpu `
  --output experiment_protocol\smoke
```

The plan command performs no training. The smoke command is a small execution check and
does not reproduce the reported multi-seed experiment.

## Prepared data

The repository already contains the deterministic prepared tensors:

```text
data_external/epa_aqs/prepared_development/salt_lake_2021_2024.npz
data_external/epa_aqs/prepared_holdout/salt_lake_2025.npz
data_external/epa_aqs_las_vegas/prepared_development/las_vegas_2021_2024.npz
data_external/epa_aqs_las_vegas/prepared_holdout/las_vegas_2025.npz
```

Each tensor includes the hourly time axis, station and pollutant names, scaled causal
inputs, original observation masks, unscaled pollutant targets, training-only scaling
statistics, risk thresholds, MASE scales, coordinates, and split bounds. Corresponding
availability CSV files and preprocessing reports are stored beside each tensor.

The compressed raw AQS responses are not duplicated in this repository. Their redacted
download manifests are retained in each network's `provenance/` directory. To reconstruct
the raw inputs, set `AQS_API_EMAIL` and `AQS_API_KEY`, then use
`scripts/download_epa_aqs.py`; the exact state, county, pollutant, and year scopes are
documented in `journal_protocol/`.

## Main experiment workflow

The following commands show the Salt Lake development workflow. Outputs are created in
the ignored `experiment_protocol/` directory.

```powershell
# Prespecified comparators and fixed-graph parent models.
python scripts\run_journal_development.py --stage run --device cuda:0 `
  --output experiment_protocol\results_journal\development\full

# Adaptive graph-imputation candidate initialized from matching parent checkpoints.
python scripts\run_journal_development.py --stage run --device cuda:0 `
  --output experiment_protocol\results_journal\development\adaptive_graph_impute_5seed `
  --learned-models adaptive_graph_impute_tcn `
  --seeds 42 123 2026 3407 7777 --without-deterministic `
  --adapter-base-runs experiment_protocol\results_journal\development\full\runs

python scripts\analyze_journal_development.py
```

The final-holdout scripts deliberately require the development outputs and generated
freeze records. This prevents accidental evaluation before the model and statistical
protocol have been fixed. Do not use the 2025 data for model selection, hyperparameter
tuning, threshold selection, or calibration fitting.

The Clark County replication uses the analogous `run_external_*` and
`analyze_external_*` entry points. Use `--device cuda:N` to select a particular GPU.

## Core entry points

| Purpose | Script |
|---|---|
| Download public AQS observations | `scripts/download_epa_aqs.py` |
| Validate downloaded responses | `scripts/validate_epa_aqs_download.py` |
| Audit development data | `scripts/audit_epa_aqs_development.py` |
| Build prepared development tensors | `scripts/prepare_epa_aqs_development.py` |
| Train development models | `scripts/run_journal_development.py` |
| Analyze development runs | `scripts/analyze_journal_development.py` |
| Execute a frozen temporal holdout | `scripts/run_journal_holdout.py` |
| Analyze a frozen temporal holdout | `scripts/analyze_journal_holdout.py` |
| Benchmark model inference | `scripts/benchmark_journal_inference.py` |

Network-specific external wrappers and protocol-freezing utilities are also retained in
`scripts/`.

## Scope and claim boundary

The released data cover two seven-station U.S. EPA AQS networks and five pollutants:
PM2.5, NO2, O3, CO, and SO2. The primary 6-hour and 24-hour outages are controlled
interventions rather than verified communication or hardware failures. The code and data
support reproducibility of the evaluated graph-imputation mechanism; they do not establish
universal superiority, deployed warning-system performance, or resilience to every outage
process.

## Data source

Air-quality observations originate from the U.S. Environmental Protection Agency Air
Quality System (AQS). Users are responsible for following the EPA source terms and citing
the original data source in derivative work.
