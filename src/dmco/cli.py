from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from dmco.config import load_config
from dmco.data import load_aligned_csv_pair, make_clean_validation_split
from dmco.metrics import evaluate_metric
from dmco.pipeline import DMCORunner


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="dmco")
    sub = parser.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser("run", help="Run DMCO on aligned dirty/clean CSV files.")
    run_p.add_argument("--dirty-csv", required=True)
    run_p.add_argument("--clean-csv", required=True)
    run_p.add_argument("--target-column", default="-1")
    run_p.add_argument("--task", choices=["classification", "regression"])
    run_p.add_argument("--metric")
    run_p.add_argument("--config", default="configs/dmco.yaml")
    run_p.add_argument("--output", default="results/dmco_run.json")

    smoke_p = sub.add_parser("smoke-test", help="Run a small synthetic classification test.")
    smoke_p.add_argument("--config", default="configs/dmco.yaml")

    args = parser.parse_args(argv)
    if args.command == "run":
        return _run_csv(args)
    if args.command == "smoke-test":
        return _smoke(args)
    return 1


def _run_csv(args) -> int:
    target_column = int(args.target_column) if str(args.target_column).lstrip("-").isdigit() else args.target_column
    overrides = {}
    if args.task:
        overrides["task"] = args.task
    if args.metric:
        overrides["metric"] = args.metric
    config = load_config(args.config, overrides)
    X_dirty, y, X_clean = load_aligned_csv_pair(args.dirty_csv, args.clean_csv, target_column)
    split = make_clean_validation_split(
        X_dirty, y, X_clean, random_state=config.random_state
    )
    runner = DMCORunner(config)
    state = runner.fit(
        split.X_train_dirty,
        split.y_train,
        split.X_clean_reference,
        split.train_indices,
        split.X_val_clean,
        split.y_val,
    )
    test_metric = evaluate_metric(split.y_test, state.best_model.predict(split.X_test_clean), config.metric)
    payload = {
        "validation_metric": state.metric,
        "test_metric": test_metric,
        "cleaned_count": int(state.cleaned_mask.sum()),
        "logs": [log.__dict__ for log in state.logs],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


def _smoke(args) -> int:
    config = load_config(args.config, {"budget": {"n_slices": 3, "total_seconds": 9}})
    X, y = make_classification(
        n_samples=300,
        n_features=12,
        n_informative=6,
        n_redundant=2,
        random_state=config.random_state,
    )
    X = pd.DataFrame(X)
    y = pd.Series(y)
    X_dirty = X.copy()
    X_dirty.iloc[:40, :3] = 0.0
    indices = list(range(len(X_dirty)))
    X_train, _X_val_dirty, y_train, y_val, idx_train, idx_val = train_test_split(
        X_dirty, y, indices, test_size=0.25, random_state=config.random_state, stratify=y
    )
    runner = DMCORunner(config)
    state = runner.fit(
        X_train.reset_index(drop=True),
        y_train.reset_index(drop=True),
        X,
        pd.Series(idx_train).to_numpy(),
        X.iloc[idx_val].reset_index(drop=True),
        y_val.reset_index(drop=True),
    )
    print(f"metric={state.metric:.4f}, cleaned={int(state.cleaned_mask.sum())}, slices={len(state.logs)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
