import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from dmco.sampling import gradient_cleaning_sample, loss_driven_gumbel_sample


def test_gradient_cleaning_sample_excludes_cleaned_rows():
    X = pd.DataFrame(np.eye(6))
    y = pd.Series([0, 1, 0, 1, 0, 1])
    model = LogisticRegression().fit(X, y)
    selected, scores = gradient_cleaning_sample(
        X,
        y,
        [model],
        k=2,
        alpha=0.5,
        rng=np.random.default_rng(0),
        exclude_mask=np.array([True, False, False, False, False, False]),
    )
    assert len(selected) == 2
    assert 0 not in selected
    assert np.isfinite(scores[selected]).all()


def test_loss_driven_gumbel_sample_returns_unique_indices():
    X = pd.DataFrame(np.random.default_rng(1).normal(size=(20, 4)))
    y = pd.Series([0, 1] * 10)
    model = LogisticRegression().fit(X, y)
    selected = loss_driven_gumbel_sample(
        X, y, model, k=7, beta=0.7, rng=np.random.default_rng(2), task="classification"
    )
    assert len(selected) == 7
    assert len(set(selected.tolist())) == 7
