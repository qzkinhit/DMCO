from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from dmco.baselines import run_methods
from dmco.config import load_config
from dmco.data import load_aligned_csv_pair, make_clean_validation_split


def main() -> int:
    parser = argparse.ArgumentParser(description="Run paper baselines on one aligned CSV pair.")
    parser.add_argument("--dirty-csv", required=True)
    parser.add_argument("--clean-csv", required=True)
    parser.add_argument("--target-column", default="-1")
    parser.add_argument("--task", choices=["classification", "regression"])
    parser.add_argument("--metric")
    parser.add_argument("--config", default="configs/dmco.yaml")
    parser.add_argument("--methods", default="DR,DA,CR,CA,DMCO")
    parser.add_argument("--output", default="results/baselines.json")
    args = parser.parse_args()

    target_column = int(args.target_column) if str(args.target_column).lstrip("-").isdigit() else args.target_column
    overrides = {}
    if args.task:
        overrides["task"] = args.task
    if args.metric:
        overrides["metric"] = args.metric

    config = load_config(args.config, overrides)
    X_dirty, y, X_clean = load_aligned_csv_pair(args.dirty_csv, args.clean_csv, target_column)
    split = make_clean_validation_split(X_dirty, y, X_clean, random_state=config.random_state)
    methods = [name.strip() for name in args.methods.split(",") if name.strip()]
    results = run_methods(split, config, methods)
    payload = [asdict(result) for result in results]

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.suffix.lower() == ".csv":
        flat = [
            {key: value for key, value in row.items() if key != "details"}
            for row in payload
        ]
        pd.DataFrame(flat).to_csv(output, index=False)
    else:
        output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
