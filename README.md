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
cd /Users/qianzekai/PyCharmProjects/dmco-icml
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

The private repository may include `data/raw/` for internal reproducibility. If the data are not present after a fresh clone, prepare them from the original zip:

```bash
python scripts/prepare_local_data.py \
  --zip "/Users/qianzekai/Library/Containers/com.tencent.xinWeChat/Data/Documents/xwechat_files/wxid_nvqr9du1hgd422_c207/msg/file/2026-07/dmco_code.zip" \
  --output data/raw
```

## Repository Layout

- `src/dmco/`: paper-aligned reusable implementation.
- `configs/`: experiment configuration.
- `scripts/`: local utility scripts.
- `tests/`: unit tests for scheduler and sampling.
- `legacy/`: original scripts kept for reproducibility audit, not used by the package.
- `data/`, `results/`: local-only folders ignored by Git.

## Notes For Private GitHub Upload

Before pushing, run:

```bash
pytest
git status --short
```

Do not commit `data/`, `results/`, `.idea/`, or virtual environments. They are ignored by default.
