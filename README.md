# DMCO

DMCO is a budget-aware framework for co-optimizing data cleaning and AutoML. It follows the paper logic:

1. Split the total resource budget into time slices.
2. Use multi-model gradient scores to prioritize samples for cleaning.
3. Use loss-driven Gumbel-top-k sampling for AutoML under each slice.
4. Reuse and prune promising model families across slices.
5. Use a cost-aware discounted UCB scheduler to allocate each slice to cleaning or AutoML.

The original exploratory scripts are preserved in `legacy/`. The cleaned implementation lives in `src/dmco/`.
See `docs/PAPER_ALIGNMENT.md` for the mapping between paper algorithms and code modules.

## Install

```bash
git clone https://github.com/qzkinhit/dmco-icml.git
cd dmco-icml
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

`auto-sklearn` is optional and can be difficult to install on macOS. The default backend is a pure scikit-learn fallback so the repository remains runnable:

```bash
python -m pip install -e ".[autosklearn,dev]"
```

## Quick Smoke Test

```bash
dmco smoke-test
```

## Run On CSV Data

The CSV format assumes the last column is the target by default. `dirty-csv` is the corrupted training source, and `clean-csv` is the aligned clean reference used to simulate cleaning.

```bash
dmco run \
  --dirty-csv data/raw/cancer/cancer_random_missing_30.csv \
  --clean-csv data/raw/cancer/cancer_data_vectorized.csv \
  --task classification \
  --config configs/dmco.yaml \
  --output results/cancer_dmco.json
```

## Data

This artifact repository includes the aligned CSV files used by the reproducibility scripts under
`data/raw/`. The largest portion is `data/raw/inject_all/`, which contains the injected corruption
grid used for paper-scale experiments.

If you are working from a copy without `data/raw/`, prepare the same folder from a local archive:

```bash
python scripts/prepare_local_data.py \
  --zip /path/to/dmco_code.zip \
  --output data/raw
```

## Reproducing Paper Values

The default backend is a portable scikit-learn search space so the code can run on macOS and CI.
This verifies the DMCO logic and ranking behavior, but it is not expected to bit-match every paper
number because the paper used a heavier AutoML environment and wall-clock budgets. See
`docs/REPRODUCIBILITY.md` for the exact-value caveats and the Table 2 comparison script.

## Repository Layout

- `src/dmco/`: paper-aligned reusable implementation.
- `configs/`: experiment configuration.
- `scripts/`: local utility scripts.
- `tests/`: unit tests for scheduler and sampling.
- `legacy/`: original scripts kept for reproducibility audit, not used by the package.
- `data/raw/`: tracked paper artifact CSV files.
- `results/`: tracked reference CSV summaries plus local generated outputs.

## Release Hygiene

Before pushing, run:

```bash
ruff check .
pytest -q
dmco smoke-test
git status --short
```

Do not commit `.idea/`, virtual environments, caches, temporary logs, or ad hoc experiment dumps.
The curated CSV artifacts under `data/raw/` and the reference CSV summaries under `results/` are
intentionally tracked.
