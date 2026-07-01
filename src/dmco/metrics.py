from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score


def evaluate_metric(y_true, y_pred, metric: str) -> float:
    metric = metric.lower()
    if metric == "accuracy":
        return float(accuracy_score(y_true, y_pred))
    if metric == "f1":
        return float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    if metric == "r2":
        return float(r2_score(y_true, y_pred))
    if metric == "mse":
        return -float(mean_squared_error(y_true, y_pred))
    if metric == "rmse":
        return -float(np.sqrt(mean_squared_error(y_true, y_pred)))
    raise ValueError(f"Unsupported metric: {metric}")


def per_sample_loss(model, X, y, task: str) -> np.ndarray:
    y_arr = np.asarray(y)
    if task == "classification":
        if hasattr(model, "predict_proba"):
            proba = np.clip(model.predict_proba(X), 1e-12, 1.0)
            classes = np.asarray(getattr(model, "classes_", np.unique(y_arr)))
            class_to_pos = {label: i for i, label in enumerate(classes)}
            indices = np.array([class_to_pos.get(label, 0) for label in y_arr])
            return -np.log(proba[np.arange(len(y_arr)), indices])
        pred = model.predict(X)
        return (pred != y_arr).astype(float)

    pred = np.asarray(model.predict(X))
    return np.square(y_arr.astype(float) - pred.astype(float))
