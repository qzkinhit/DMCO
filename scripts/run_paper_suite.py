from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from dmco.baselines import run_methods
from dmco.config import load_config
from dmco.data import load_aligned_csv_pair, make_clean_validation_split


DEFAULT_DATASETS = ("adult", "cancer", "nasa", "skin", "smartfactory", "soilmoisture")
DEFAULT_ERRORS = (
    "gauss",
    "random_missing",
    "random_outliers",
    "system_missing",
    "system_outliers",
    "white",
)
DEFAULT_LEVELS = (10, 20, 30, 40, 50)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the DMCO paper experiment grid.")
    parser.add_argument("--data-root", default="data/raw")
    parser.add_argument("--config", default="configs/dmco.yaml")
    parser.add_argument("--datasets", default=",".join(DEFAULT_DATASETS))
    parser.add_argument("--errors", default=",".join(DEFAULT_ERRORS))
    parser.add_argument("--levels", default=",".join(str(x) for x in DEFAULT_LEVELS))
    parser.add_argument("--methods", default="DR,DA,CR,CA,DMCO")
    parser.add_argument("--task", choices=["classification", "regression"], default="classification")
    parser.add_argument("--metric", default="f1")
    parser.add_argument("--output", default="results/paper_suite.csv")
    parser.add_argument("--skip-missing", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config, {"task": args.task, "metric": args.metric})
    data_root = Path(args.data_root)
    datasets = _split_csv_arg(args.datasets)
    errors = _split_csv_arg(args.errors)
    levels = [int(x) for x in _split_csv_arg(args.levels)]
    methods = _split_csv_arg(args.methods)

    rows = []
    for dataset in datasets:
        clean_csv = data_root / dataset / f"{dataset}_data_vectorized.csv"
        for error in errors:
            for level in levels:
                dirty_csv = data_root / "inject_all" / f"{dataset}_{error}_{level}.csv"
                if not dirty_csv.exists() or not clean_csv.exists():
                    if args.skip_missing:
                        continue
                    raise FileNotFoundError(f"Missing {dirty_csv} or {clean_csv}")
                X_dirty, y, X_clean = load_aligned_csv_pair(dirty_csv, clean_csv)
                split = make_clean_validation_split(
                    X_dirty,
                    y,
                    X_clean,
                    random_state=config.random_state,
                )
                for result in run_methods(split, config, methods):
                    row = asdict(result)
                    row.pop("details", None)
                    row.update({"dataset": dataset, "error": error, "level": level})
                    rows.append(row)
                    print(
                        f"{dataset}/{error}/{level}/{result.method}: "
                        f"val={result.validation_metric:.4f}, test={result.test_metric:.4f}"
                    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output, index=False)
    print(f"Wrote {output}")
    return 0


def _split_csv_arg(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


if __name__ == "__main__":
    raise SystemExit(main())
