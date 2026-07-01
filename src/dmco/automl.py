from __future__ import annotations

from dataclasses import dataclass
import time
from typing import List, Sequence
import warnings

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.linear_model import LogisticRegression, Ridge, SGDClassifier, SGDRegressor
from sklearn.metrics import get_scorer
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, LinearSVR
from sklearn.exceptions import ConvergenceWarning


@dataclass
class AutoMLResult:
    best_model: object
    candidate_models: List[object]
    selected_families: List[str]
    leaderboard: pd.DataFrame
    elapsed_seconds: float


CLASSIFICATION_FAMILIES = {
    "logistic_regression": make_pipeline(
        StandardScaler(), LogisticRegression(max_iter=1000, solver="lbfgs")
    ),
    "sgd": make_pipeline(StandardScaler(), SGDClassifier(loss="log_loss", max_iter=1000, tol=1e-3)),
    "linear_svc": make_pipeline(StandardScaler(), LinearSVC(dual="auto", max_iter=5000)),
    "gaussian_nb": GaussianNB(),
    "lda": LinearDiscriminantAnalysis(),
    "mlp": make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(64,), max_iter=300)),
}

REGRESSION_FAMILIES = {
    "ridge": make_pipeline(StandardScaler(), Ridge()),
    "sgd": make_pipeline(StandardScaler(), SGDRegressor(max_iter=1000, tol=1e-3)),
    "linear_svr": make_pipeline(StandardScaler(), LinearSVR(dual="auto", max_iter=5000)),
    "mlp": make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(64,), max_iter=300)),
}


class AutoMLAdapter:
    def __init__(
        self,
        task: str,
        metric: str,
        backend: str = "sklearn",
        top_k_families: int = 3,
        cv_folds: int = 3,
        random_state: int = 42,
        family_allowlist: Sequence[str] | None = None,
    ):
        self.task = task
        self.metric = metric
        self.backend = backend
        self.top_k_families = top_k_families
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.family_allowlist = list(family_allowlist) if family_allowlist else None

    def fit(self, X, y, families: Sequence[str] | None = None, budget_seconds: int = 30) -> AutoMLResult:
        if self.backend == "autosklearn":
            try:
                return self._fit_autosklearn(X, y, families, budget_seconds)
            except Exception:
                pass
        return self._fit_sklearn(X, y, families, budget_seconds)

    def _fit_sklearn(self, X, y, families: Sequence[str] | None, budget_seconds: int) -> AutoMLResult:
        start = time.time()
        candidates = CLASSIFICATION_FAMILIES if self.task == "classification" else REGRESSION_FAMILIES
        names = list(families) if families else list(self.family_allowlist or candidates)
        names = [name for name in names if name in candidates] or list(candidates)
        scorer = _scorer_name(self.task, self.metric)
        rows = []
        fitted = []
        best_score = -np.inf
        best_model = None

        for name in names:
            if time.time() - start > max(1, budget_seconds):
                break
            estimator = clone(candidates[name])
            _set_random_state(estimator, self.random_state)
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", ConvergenceWarning)
                    scores = cross_val_score(
                        estimator,
                        X,
                        y,
                        scoring=get_scorer(scorer),
                        cv=min(self.cv_folds, max(2, len(y) // 5)),
                        error_score=np.nan,
                    )
                    score = float(np.nanmean(scores))
                    estimator.fit(X, y)
            except Exception:
                score = -np.inf
                estimator = _fallback_model(self.task)
                estimator.fit(X, y)
            rows.append({"family": name, "score": score})
            fitted.append(estimator)
            if score > best_score:
                best_score = score
                best_model = estimator

        if best_model is None:
            best_model = _fallback_model(self.task)
            best_model.fit(X, y)
            fitted = [best_model]
            rows = [{"family": "dummy", "score": -np.inf}]

        leaderboard = pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)
        selected = leaderboard["family"].head(self.top_k_families).tolist()
        return AutoMLResult(
            best_model=best_model,
            candidate_models=fitted,
            selected_families=selected,
            leaderboard=leaderboard,
            elapsed_seconds=time.time() - start,
        )

    def _fit_autosklearn(self, X, y, families: Sequence[str] | None, budget_seconds: int) -> AutoMLResult:
        start = time.time()
        if self.task == "classification":
            from autosklearn.classification import AutoSklearnClassifier

            automl = AutoSklearnClassifier(
                time_left_for_this_task=max(30, int(budget_seconds)),
                per_run_time_limit=max(5, min(30, int(budget_seconds // 2))),
                include={"classifier": list(families or self.family_allowlist)}
                if families or self.family_allowlist
                else None,
                seed=self.random_state,
                n_jobs=1,
            )
        else:
            from autosklearn.regression import AutoSklearnRegressor

            automl = AutoSklearnRegressor(
                time_left_for_this_task=max(30, int(budget_seconds)),
                per_run_time_limit=max(5, min(30, int(budget_seconds // 2))),
                include={"regressor": list(families or self.family_allowlist)}
                if families or self.family_allowlist
                else None,
                seed=self.random_state,
                n_jobs=1,
            )
        automl.fit(X, y)
        leaderboard = _safe_leaderboard(automl)
        selected = _select_families_from_leaderboard(leaderboard, self.top_k_families)
        return AutoMLResult(
            best_model=automl,
            candidate_models=[automl],
            selected_families=selected,
            leaderboard=leaderboard,
            elapsed_seconds=time.time() - start,
        )


def _scorer_name(task: str, metric: str) -> str:
    if task == "classification":
        return "f1_weighted" if metric == "f1" else "accuracy"
    return "neg_mean_squared_error" if metric in {"mse", "rmse"} else "r2"


def _fallback_model(task: str):
    return DummyClassifier(strategy="most_frequent") if task == "classification" else DummyRegressor()


def _set_random_state(estimator, seed: int):
    if hasattr(estimator, "set_params"):
        params = estimator.get_params()
        updates = {key: seed for key in params if key.endswith("random_state")}
        if updates:
            estimator.set_params(**updates)


def _safe_leaderboard(automl) -> pd.DataFrame:
    try:
        return automl.leaderboard(detailed=True).reset_index(drop=True)
    except Exception:
        return pd.DataFrame([{"family": "autosklearn", "score": np.nan}])


def _select_families_from_leaderboard(leaderboard: pd.DataFrame, top_k: int) -> List[str]:
    vocab = sorted(set(CLASSIFICATION_FAMILIES) | set(REGRESSION_FAMILIES))
    selected: List[str] = []
    for _, row in leaderboard.iterrows():
        text = " ".join(str(v) for v in row.values)
        for family in vocab:
            if family in text and family not in selected:
                selected.append(family)
        if len(selected) >= top_k:
            break
    return selected
