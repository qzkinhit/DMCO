from __future__ import annotations

from dataclasses import dataclass, field
import time
from typing import List

import numpy as np
import pandas as pd

from dmco.automl import AutoMLAdapter, AutoMLResult
from dmco.config import DMCOConfig
from dmco.data import apply_reference_cleaning
from dmco.metrics import evaluate_metric
from dmco.sampling import gradient_cleaning_sample, loss_driven_gumbel_sample, random_sample, sample_count
from dmco.scheduler import CostAwareUCB


@dataclass
class SliceLog:
    slice_id: int
    action: str
    metric: float
    reward: float
    cost_seconds: float
    cleaned_count: int
    families: List[str]
    scheduler: dict


@dataclass
class DMCOState:
    X_train: pd.DataFrame
    cleaned_mask: np.ndarray
    best_model: object | None = None
    candidate_models: List[object] = field(default_factory=list)
    selected_families: List[str] = field(default_factory=list)
    metric: float | None = None
    logs: List[SliceLog] = field(default_factory=list)


class DMCORunner:
    def __init__(self, config: DMCOConfig):
        self.config = config
        self.rng = np.random.default_rng(config.random_state)
        self.automl = AutoMLAdapter(
            task=config.task,
            metric=config.metric,
            backend=config.automl.backend,
            top_k_families=config.automl.top_k_families,
            cv_folds=config.automl.cv_folds,
            random_state=config.random_state,
            family_allowlist=config.automl.family_allowlist,
        )

    def fit(
        self,
        X_train_dirty: pd.DataFrame,
        y_train: pd.Series,
        X_clean_reference: pd.DataFrame,
        train_indices: np.ndarray,
        X_val_clean: pd.DataFrame,
        y_val: pd.Series,
    ) -> DMCOState:
        state = DMCOState(
            X_train=X_train_dirty.reset_index(drop=True).copy(),
            cleaned_mask=np.zeros(len(X_train_dirty), dtype=bool),
        )

        self._automl_step(state, y_train, initial=True)
        state.metric = self._evaluate(state.best_model, X_val_clean, y_val)
        scheduler = CostAwareUCB(
            gamma=self.config.scheduler.gamma,
            exploration=self.config.scheduler.exploration,
            stall_patience=self.config.scheduler.stall_patience,
            stall_boost=self.config.scheduler.stall_boost,
            last_metric=state.metric,
        )

        no_improve = 0
        for slice_id in range(1, self.config.budget.n_slices + 1):
            available = []
            if not state.cleaned_mask.all():
                available.append("clean")
            available.append("automl")
            action = scheduler.choose(available)
            start = time.time()
            cleaned_count = 0

            if action == "clean":
                cleaned_count = self._clean_step(state, y_train, X_clean_reference, train_indices)
            elif action == "automl":
                self._automl_step(state, y_train)
            else:
                break

            metric = self._evaluate(state.best_model, X_val_clean, y_val)
            elapsed = max(time.time() - start, 1e-9)
            reward = scheduler.observe(action, metric, elapsed)
            state.metric = metric
            state.logs.append(
                SliceLog(
                    slice_id=slice_id,
                    action=action,
                    metric=metric,
                    reward=reward,
                    cost_seconds=elapsed,
                    cleaned_count=int(cleaned_count),
                    families=list(state.selected_families),
                    scheduler=scheduler.snapshot(),
                )
            )

            if scheduler.best_metric is not None and metric < scheduler.best_metric:
                no_improve += 1
            else:
                no_improve = 0
            if no_improve >= self.config.budget.early_stop_patience:
                break

        return state

    def _clean_step(
        self,
        state: DMCOState,
        y_train: pd.Series,
        X_clean_reference: pd.DataFrame,
        train_indices: np.ndarray,
    ) -> int:
        k = sample_count(
            len(state.X_train),
            self.config.sampling.clean_batch_fraction,
            minimum=1,
        )
        selected, _ = gradient_cleaning_sample(
            state.X_train,
            y_train,
            state.candidate_models or [state.best_model],
            k=k,
            alpha=self.config.sampling.gradient_variance_weight,
            rng=self.rng,
            exclude_mask=state.cleaned_mask,
            max_samples=self.config.sampling.max_gradient_samples,
        )
        before = int(state.cleaned_mask.sum())
        state.X_train, state.cleaned_mask = apply_reference_cleaning(
            state.X_train,
            X_clean_reference,
            train_indices,
            selected,
            state.cleaned_mask,
        )
        return int(state.cleaned_mask.sum() - before)

    def _automl_step(self, state: DMCOState, y_train: pd.Series, initial: bool = False) -> AutoMLResult:
        n = len(state.X_train)
        k = sample_count(
            n,
            self.config.sampling.automl_sample_fraction,
            minimum=min(n, self.config.sampling.min_automl_samples),
            maximum=min(n, self.config.sampling.max_automl_samples),
        )
        if initial:
            idx = random_sample(n, k, self.rng)
        else:
            idx = loss_driven_gumbel_sample(
                state.X_train,
                y_train,
                state.best_model,
                k=k,
                beta=self.config.sampling.loss_mixing_weight,
                rng=self.rng,
                task=self.config.task,
            )
        budget = max(self.config.budget.min_automl_seconds, self.config.budget.slice_seconds)
        result = self.automl.fit(
            state.X_train.iloc[idx],
            y_train.iloc[idx],
            families=state.selected_families or None,
            budget_seconds=budget,
        )
        state.best_model = result.best_model
        state.candidate_models = result.candidate_models
        state.selected_families = result.selected_families
        return result

    def _evaluate(self, model, X_val, y_val) -> float:
        if model is None:
            return float("-inf")
        pred = model.predict(X_val)
        return evaluate_metric(y_val, pred, self.config.metric)
