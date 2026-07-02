# Data

This artifact repository tracks the aligned CSV files needed by the reproducibility scripts, for
example:

```text
data/raw/cancer/cancer_random_missing_30.csv
data/raw/cancer/cancer_data_vectorized.csv
```

If `data/raw/` is missing in a fork or lightweight copy, use `scripts/prepare_local_data.py` to
extract the original archive into `data/raw`.
