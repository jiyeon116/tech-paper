# QECO-ADAPT2

QECO-ADAPT2 is a standalone research implementation of density-aware task
offloading for a multi-edge, multi-access-point, multi-channel mobile edge
computing environment. It preserves QECO's local-or-edge action space while the
environment resolves AP-cluster and channel resources after an edge action is
chosen.

The `qeco-adapt2` adapter observes the active-user count at the beginning of each
episode, classifies users-per-edge into a density regime, and selects a
configurable QECO-family reward/gate policy for that episode. All regimes share
one D3QN/LSTM agent implementation and the same replay history.

The opt-in `dynamic_experiment` pilot separately implements step-level activity
changes and independent specialist learners. It does not change the legacy
adapter above. See the [dynamic specialist pilot protocol](docs/dynamic_specialist_pilot.md)
for training/calibration/frozen-evaluation boundaries, CPU launch and recovery.

## Repository Scope

This repository contains executable experiment code, launchers, focused tests,
and design documentation. It intentionally excludes experiment results, logs,
paper drafts, PDFs, and publication build assets. Generated data belongs under
`results/` and `logs/`, which are ignored by Git.

## Requirements

- Python 3.10
- NumPy 1.23.5
- TensorFlow 2.8.0
- Matplotlib 3.5 or later
- `tmux` for the optional parallel launch scripts

TensorFlow 2.8 constrains the supported Python and dependency versions. Use the
isolated environment below rather than a newer system Python.

```bash
git clone https://github.com/HyeonWooNa0861/qeco-adapt2.git
cd qeco-adapt2
make setup
make test
```

To use an existing Python 3.10 environment instead, set `QECO_PYTHON` when using
the workflow helper:

```bash
QECO_PYTHON=/path/to/python3.10 \
  python3 scripts/qeco_workflow.py --family adapt2 --users 300 commands
```

## Algorithms

The main public experiment set is:

- `qeco`: QECO D3QN/LSTM baseline.
- `qeco-adapt-general`: adaptive energy weight with fixed gate strength.
- `qeco-adaptive-gate`: adaptive energy weight with load-conditioned gate.
- `qeco-adapt2`: per-episode density-regime selector over configurable
  QECO-family policies.

The adapter registry also retains focused ablation variants used to isolate
reward weighting and gate behavior.

## Run Experiments

Inspect the resolved commands without starting an experiment:

```bash
./env_channel/bin/python scripts/qeco_workflow.py \
  --family adapt2 \
  --users 300 \
  commands
```

Run a short fixed-density smoke experiment:

```bash
CUDA_VISIBLE_DEVICES=-1 \
QECO_AGENT_SESSION_MODE=per-agent \
./env_channel/bin/python scripts/qeco_workflow.py \
  --family adapt2 \
  --users 30 \
  run \
  --algorithms qeco-adapt2 \
  --episodes 2
```

Run random density with a pool large enough for every candidate count:

```bash
CUDA_VISIBLE_DEVICES=-1 \
QECO_AGENT_SESSION_MODE=per-agent \
./env_channel/bin/python scripts/qeco_workflow.py \
  --family adapt2 \
  --density-mode random \
  --density-choices 100,300,500,800,1200,1500 \
  --users 1500 \
  --allow-1200 \
  run \
  --episodes 500
```

The random-density mode samples one active-user count per episode. `--users` is
the total agent pool, and inactive agents are masked for that episode. Runs are
stored below:

```text
results/adapt2/fixed_density/<algorithm>/run_*/
results/adapt2/random_density/<algorithm>/run_*/
```

For long CPU runs, the repository provides tmux launchers:

```bash
scripts/run_adapt2_phase_a.sh start
scripts/run_adapt2_phase_a.sh status

scripts/run_adapt2_phase_b.sh start
scripts/run_adapt2_phase_b.sh status
```

Each launcher writes worker logs under `logs/adapt2/` and records the child
process exit status.

## Compare or Regenerate Results

Compare the latest matching runs:

```bash
./env_channel/bin/python scripts/qeco_workflow.py \
  --family adapt2 \
  --users 300 \
  compare
```

Regenerate tabular and PNG outputs from an existing timeseries CSV without
rerunning an experiment:

```bash
./env_channel/bin/python compare_results.py \
  --timeseries-csv results/adapt2/comparisons/fixed_density/<scenario>/comparison_timeseries.csv
```

Random-density comparisons additionally emit active-user and regime summaries.

## Artifact Integrity

Result and log files remain outside Git. A checksum manifest can detect local
artifact drift without copying the artifacts:

```bash
./env_channel/bin/python scripts/qeco_artifact_manifest.py write
./env_channel/bin/python scripts/qeco_artifact_manifest.py verify
```

The manifest scope is limited to `results/**` and `logs/**`.

## Documentation and Attribution

- [QECO-ADAPT2 design](docs/qeco_adapt2_design.md)
- [Provenance](PROVENANCE.md)
- [Third-party notices](THIRD_PARTY_NOTICES.md)
- [MIT license](LICENSE)

This is an independent derivative research project and is not endorsed by the
upstream QECO authors.
