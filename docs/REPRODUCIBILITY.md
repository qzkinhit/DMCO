# Reproducibility Guide

This repository now has two levels of reproducibility.

## 1. Fast Code Verification

Use this before every commit:

```bash
ruff check .
pytest -q
dmco smoke-test
```

This verifies the implementation, command-line interface, scheduler, and sampling modules.

## 2. Paper-Scale Runs

The repository includes `data/raw/`. If your copy does not include it, extract the local archive:

```bash
python scripts/prepare_local_data.py \
  --zip /path/to/dmco_code.zip \
  --output data/raw
```

Run one dataset/error pair:

```bash
PYTHONPATH=src python scripts/run_baselines.py \
  --dirty-csv data/raw/inject_all/cancer_random_missing_30.csv \
  --clean-csv data/raw/cancer/cancer_data_vectorized.csv \
  --task classification \
  --metric f1 \
  --output results/cancer_random_missing_30.json
```

Run the full paper grid:

```bash
PYTHONPATH=src python scripts/run_paper_suite.py \
  --data-root data/raw \
  --task classification \
  --metric f1 \
  --output results/paper_suite.csv \
  --skip-missing
```

For regression datasets, rerun with `--task regression --metric mse`.

Run the Table 2 style budget curves:

```bash
PYTHONPATH=src python scripts/run_table2.py \
  --data-root data/raw \
  --output results/table2_reproduction.csv
```

## Paper-Scale Reproduction

For paper-scale reproduction, keep the following settings aligned with the paper environment:

- The same CSV files used for the paper.
- The same Python package versions.
- The same AutoML backend and search space.
- Fixed random seeds.
- Enough wall-clock budget for AutoML.
- The same hardware or sufficiently close timing behavior.

The default scikit-learn backend is the portable artifact path. For the paper-scale setting, install
and select the `auto-sklearn` backend in `configs/dmco.yaml`, then run with the wall-clock budgets
used in the paper.
