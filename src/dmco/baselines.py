from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from dmco.automl import AutoMLAdapter
from dmco.config import DMCOConfig
from dmco.data import DatasetSplit, apply_reference_cleaning
from dmco.metrics import evaluate_metric
from dmco.pipeline import DMCORunner
from dmco.sampling import random_sample, sample_count


@dataclass
class MethodResult:
    method: str
    validation_metric: float
    test_metric: float
    elapsed_seconds: float
    cleaned_count: int
    details: dict


def run_methods(
    split: DatasetSplit,
    config: DMCOConfig,
    methods: Iterable[str] = ("DR", "DA", "CR", "CA", "DMCO"),
) -> list[MethodResult]:
    results = []
    for method in methods:
        method = method.upper()
        if method == "DR":
            results.append(_run_dirty_raw(split, config))
        elif method == "DA":
            results.append(_run_dirty_automl(split, config))
        elif method == "CR":
            results.append(_run_clean_raw(split, config))
        elif method == "CA":
            results.append(_run_clean_automl(split, config))
        elif method == "DMCO":
            results.append(_run_dmco(split, config))
        else:
            raise ValueError(f"Unsupported method: {method}")
    return results


def _run_dirty_raw(split: DatasetSplit, config: DMCOConfig) -> MethodResult:
    start = time.time()
    model = _raw_model(config.task, config.random_state)
    model.fit(split.X_train_dirty, split.y_train)
    return _result("DR", model, split, config, start, cleaned_count=0, details={})


def _run_dirty_automl(split: DatasetSplit, config: DMCOConfig) -> MethodResult:
    start = time.time()
    automl = _automl_adapter(config)
    result = automl.fit(
        split.X_train_dirty,
        split.y_train,
        families=None,
        budget_seconds=config.budget.total_seconds,
    )
    return _result(
        "DA",
        result.best_model,
        split,
        config,
        start,
        cleaned_count=0,
        details={"families": result.selected_families},
    )


def _run_clean_raw(split: DatasetSplit, config: DMCOConfig) -> MethodResult:
    start = time.time()
    X_cleaned, cleaned_mask = _partial_random_clean(split, config)
    model = _raw_model(config.task, config.random_state)
    model.fit(X_cleaned, split.y_train)
    return _result(
        "CR",
        model,
        split,
        config,
        start,
        cleaned_count=int(cleaned_mask.sum()),
        details={},
    )


def _run_clean_automl(split: DatasetSplit, config: DMCOConfig) -> MethodResult:
    start = time.time()
    X_cleaned, cleaned_mask = _partial_random_clean(split, config)
    automl = _automl_adapter(config)
    result = automl.fit(
        X_cleaned,
        split.y_train,
        families=None,
        budget_seconds=config.budget.total_seconds,
    )
    return _result(
        "CA",
        result.best_model,
        split,
        config,
        start,
        cleaned_count=int(cleaned_mask.sum()),
        details={"families": result.selected_families},
    )


def _run_dmco(split: DatasetSplit, config: DMCOConfig) -> MethodResult:
    start = time.time()
    runner = DMCORunner(config)
    state = runner.fit(
        split.X_train_dirty,
        split.y_train,
        split.X_clean_reference,
        split.train_indices,
        split.X_val_clean,
        split.y_val,
    )
    return _result(
        "DMCO",
        state.best_model,
        split,
        config,
        start,
        cleaned_count=int(state.cleaned_mask.sum()),
        details={"slices": [log.__dict__ for log in state.logs]},
    )


def _partial_random_clean(split: DatasetSplit, config: DMCOConfig) -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(config.random_state)
    fraction = min(
        1.0,
        max(
            config.sampling.clean_batch_fraction,
            config.sampling.clean_batch_fraction * config.budget.n_slices,
        ),
    )
    k = sample_count(len(split.X_train_dirty), fraction, minimum=1)
    selected = random_sample(len(split.X_train_dirty), k, rng)
    return apply_reference_cleaning(
        split.X_train_dirty,
        split.X_clean_reference,
        split.train_indices,
        selected,
    )


def _raw_model(task: str, random_state: int):
    if task == "classification":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1000, random_state=random_state),
        )
    return make_pipeline(StandardScaler(), Ridge())


def _automl_adapter(config: DMCOConfig) -> AutoMLAdapter:
    return AutoMLAdapter(
        task=config.task,
        metric=config.metric,
        backend=config.automl.backend,
        top_k_families=config.automl.top_k_families,
        cv_folds=config.automl.cv_folds,
        random_state=config.random_state,
        family_allowlist=config.automl.family_allowlist,
    )


def _result(
    method: str,
    model,
    split: DatasetSplit,
    config: DMCOConfig,
    start: float,
    cleaned_count: int,
    details: dict,
) -> MethodResult:
    val_metric = evaluate_metric(split.y_val, model.predict(split.X_val_clean), config.metric)
    test_metric = evaluate_metric(split.y_test, model.predict(split.X_test_clean), config.metric)
    return MethodResult(
        method=method,
        validation_metric=val_metric,
        test_metric=test_metric,
        elapsed_seconds=time.time() - start,
        cleaned_count=cleaned_count,
        details=details,
    )
