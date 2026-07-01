from __future__ import annotations

import argparse
import copy
from dataclasses import asdict
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from dmco.baselines import run_methods
from dmco.config import load_config
from dmco.data import load_aligned_csv_pair, make_clean_validation_split


CLASSIFICATION_DATASETS = ("adult", "cancer", "skin", "smartfactory")
REGRESSION_DATASETS = ("nasa", "soilmoisture")


def main() -> int:
    parser = argparse.ArgumentParser(description="Reproduce Table 2 style budget curves.")
    parser.add_argument("--data-root", default="data/raw")
    parser.add_argument("--config", default="configs/dmco.yaml")
    parser.add_argument("--datasets", default="adult,cancer,skin,smartfactory,nasa,soilmoisture")
    parser.add_argument("--methods", default="DR,DA,CR,CA,DMCO")
    parser.add_argument("--budget-unit", type=int, default=30)
    parser.add_argument("--points", type=int, default=10)
    parser.add_argument("--output", default="results/table2_reproduction.csv")
    parser.add_argument("--skip-missing", action="store_true")
    parser.add_argument("--families", default=None, help="Comma-separated AutoML family allowlist.")
    parser.add_argument("--max-rows", type=int, default=None, help="Optional reproducible cap for large CSVs.")
    args = parser.parse_args()

    base_config = load_config(args.config)
    data_root = Path(args.data_root)
    datasets = [x.strip() for x in args.datasets.split(",") if x.strip()]
    methods = [x.strip() for x in args.methods.split(",") if x.strip()]
    rows = []

    for dataset in datasets:
        task = "regression" if dataset in REGRESSION_DATASETS else "classification"
        metric = "mse" if task == "regression" else "f1"
        dirty_csv = data_root / dataset / f"{dataset}_25.csv"
        clean_csv = data_root / dataset / f"{dataset}_data_vectorized.csv"
        if not dirty_csv.exists() or not clean_csv.exists():
            if args.skip_missing:
                continue
            raise FileNotFoundError(f"Missing {dirty_csv} or {clean_csv}")

        X_dirty, y, X_clean = load_aligned_csv_pair(dirty_csv, clean_csv)
        if args.max_rows and len(X_dirty) > args.max_rows:
            X_dirty, _, y, _, X_clean, _ = train_test_split(
                X_dirty,
                y,
                X_clean,
                train_size=args.max_rows,
                random_state=base_config.random_state,
                stratify=y if task == "classification" else None,
            )
        split = make_clean_validation_split(
            X_dirty,
            y,
            X_clean,
            random_state=base_config.random_state,
        )
        for point in range(1, args.points + 1):
            config = copy.deepcopy(base_config)
            config.task = task
            config.metric = metric
            config.budget.total_seconds = args.budget_unit * point
            config.sampling.clean_batch_fraction = point / (args.points * config.budget.n_slices)
            if args.families:
                config.automl.family_allowlist = [x.strip() for x in args.families.split(",") if x.strip()]
            for result in run_methods(split, config, methods):
                raw = asdict(result)
                raw.pop("details", None)
                raw.update(
                    {
                        "dataset": dataset,
                        "task": task,
                        "metric": metric,
                        "budget_point": point,
                        "budget_seconds": config.budget.total_seconds,
                        "paper_test_metric": _paper_metric(raw["test_metric"], metric),
                        "paper_validation_metric": _paper_metric(raw["validation_metric"], metric),
                    }
                )
                rows.append(raw)
                _append_row(raw, Path(args.output))
                print(
                    f"{dataset}/{result.method}/{point}: "
                    f"paper_test={raw['paper_test_metric']:.4f}"
                )

    output = Path(args.output)
    if rows:
        pd.DataFrame(rows).to_csv(output, index=False)
    _write_pivot(rows, output.with_name(output.stem + "_pivot.csv"))
    print(f"Wrote {output}")
    return 0


def _paper_metric(value: float, metric: str) -> float:
    return -value if metric == "mse" else value


def _write_pivot(rows: list[dict], output: Path) -> None:
    if not rows:
        return
    df = pd.DataFrame(rows)
    pivot = df.pivot_table(
        index=["dataset", "method"],
        columns="budget_point",
        values="paper_test_metric",
        aggfunc="first",
    )
    pivot.to_csv(output)


def _append_row(row: dict, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    exists = output.exists()
    pd.DataFrame([row]).to_csv(output, mode="a", index=False, header=not exists)


if __name__ == "__main__":
    raise SystemExit(main())
