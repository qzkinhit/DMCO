from __future__ import annotations

import numpy as np
from sklearn.pipeline import Pipeline


def aggregate_gradient_scores(X, y, models, alpha: float = 0.5) -> np.ndarray:
    model_list = [m for m in models if m is not None]
    if not model_list:
        return np.ones(len(X), dtype=float)

    norms = []
    for model in model_list:
        norms.append(_gradient_norm(model, X, y))
    matrix = np.vstack(norms)
    return matrix.mean(axis=0) + alpha * matrix.var(axis=0)


def _gradient_norm(model, X, y) -> np.ndarray:
    X_arr = np.asarray(X, dtype=float)
    y_arr = np.asarray(y)
    estimator = _final_estimator(model)

    if hasattr(estimator, "coef_"):
        coef = np.asarray(estimator.coef_, dtype=float)
        if coef.ndim == 1:
            coef = coef.reshape(1, -1)
        if coef.shape[-1] == X_arr.shape[1]:
            return _linear_gradient_norm(model, estimator, X, X_arr, y_arr, coef)

    if hasattr(model, "predict"):
        pred = np.asarray(model.predict(X))
        residual = np.abs(_numeric_y(y_arr) - _numeric_y(pred))
        scale = np.linalg.norm(X_arr, axis=1)
        return residual * np.maximum(scale, 1e-12)

    return np.linalg.norm(X_arr, axis=1)


def _linear_gradient_norm(
    model,
    estimator,
    X,
    X_arr: np.ndarray,
    y_arr: np.ndarray,
    coef: np.ndarray,
) -> np.ndarray:
    if hasattr(model, "predict_proba") and hasattr(estimator, "classes_"):
        proba = np.clip(model.predict_proba(X), 1e-12, 1.0)
        classes = np.asarray(estimator.classes_)
        class_to_pos = {label: i for i, label in enumerate(classes)}
        y_idx = np.array([class_to_pos.get(label, 0) for label in y_arr])
        one_hot = np.zeros_like(proba)
        one_hot[np.arange(len(y_idx)), y_idx] = 1.0
        if coef.shape[0] == 1 and proba.shape[1] == 2:
            coef_full = np.vstack([np.zeros((1, coef.shape[1])), coef])
        else:
            coef_full = coef
        grads = (proba - one_hot) @ coef_full
        return np.linalg.norm(grads, axis=1)

    pred = np.asarray(model.predict(X), dtype=float)
    residual = pred - _numeric_y(y_arr)
    base = np.linalg.norm(coef, axis=0 if coef.shape[0] > 1 else 1)
    if np.ndim(base) > 0:
        base_norm = float(np.linalg.norm(base))
    else:
        base_norm = float(base)
    return np.abs(residual) * max(base_norm, 1e-12)


def _final_estimator(model):
    if isinstance(model, Pipeline):
        return model.steps[-1][1]
    return model


def _numeric_y(values) -> np.ndarray:
    arr = np.asarray(values)
    if np.issubdtype(arr.dtype, np.number):
        return arr.astype(float)
    labels, encoded = np.unique(arr, return_inverse=True)
    _ = labels
    return encoded.astype(float)
