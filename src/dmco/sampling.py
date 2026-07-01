from __future__ import annotations

import numpy as np

from dmco.gradients import aggregate_gradient_scores
from dmco.metrics import per_sample_loss


def sample_count(n: int, fraction: float, minimum: int = 1, maximum: int | None = None) -> int:
    k = int(np.ceil(max(0.0, fraction) * n))
    k = max(minimum, k)
    if maximum is not None:
        k = min(maximum, k)
    return min(n, k)


def random_sample(n: int, k: int, rng: np.random.Generator, exclude_mask=None) -> np.ndarray:
    candidates = _candidate_indices(n, exclude_mask)
    if len(candidates) == 0:
        return np.array([], dtype=int)
    k = min(k, len(candidates))
    return rng.choice(candidates, size=k, replace=False)


def gradient_cleaning_sample(
    X,
    y,
    models,
    k: int,
    alpha: float,
    rng: np.random.Generator,
    exclude_mask=None,
    max_samples: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    candidates = _candidate_indices(len(X), exclude_mask)
    if len(candidates) == 0:
        return np.array([], dtype=int), np.zeros(len(X), dtype=float)
    if max_samples is not None and len(candidates) > max_samples:
        candidates = rng.choice(candidates, size=max_samples, replace=False)

    scores_part = aggregate_gradient_scores(X.iloc[candidates], y.iloc[candidates], models, alpha=alpha)
    full_scores = np.full(len(X), -np.inf, dtype=float)
    full_scores[candidates] = scores_part
    selected = np.argsort(-full_scores)[: min(k, np.isfinite(full_scores).sum())]
    return selected.astype(int), full_scores


def loss_driven_gumbel_sample(
    X,
    y,
    model,
    k: int,
    beta: float,
    rng: np.random.Generator,
    task: str,
) -> np.ndarray:
    n = len(X)
    if model is None:
        return random_sample(n, k, rng)
    losses = np.asarray(per_sample_loss(model, X, y, task), dtype=float)
    losses = np.nan_to_num(losses, nan=0.0, posinf=0.0, neginf=0.0)
    losses = np.maximum(losses, 0.0)
    if losses.sum() <= 0:
        weights = np.ones(n) / n
    else:
        weights = losses / losses.sum()
    mixed = (1.0 - beta) * (np.ones(n) / n) + beta * weights
    mixed = np.clip(mixed, 1e-12, None)
    gumbel = -np.log(-np.log(rng.uniform(1e-12, 1.0 - 1e-12, size=n)))
    scores = np.log(mixed) + gumbel
    return np.argsort(-scores)[: min(k, n)].astype(int)


def _candidate_indices(n: int, exclude_mask=None) -> np.ndarray:
    if exclude_mask is None:
        return np.arange(n)
    mask = np.asarray(exclude_mask, dtype=bool)
    return np.flatnonzero(~mask)
