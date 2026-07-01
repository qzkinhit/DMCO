from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


@dataclass
class DatasetSplit:
    X_train_dirty: pd.DataFrame
    X_val_clean: pd.DataFrame
    X_test_clean: pd.DataFrame
    y_train: pd.Series
    y_val: pd.Series
    y_test: pd.Series
    train_indices: np.ndarray
    val_indices: np.ndarray
    test_indices: np.ndarray
    X_clean_reference: pd.DataFrame


def load_aligned_csv_pair(
    dirty_csv: str | Path,
    clean_csv: str | Path,
    target_column: int | str = -1,
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    dirty = pd.read_csv(dirty_csv)
    clean = pd.read_csv(clean_csv)
    if len(dirty) != len(clean):
        raise ValueError("dirty_csv and clean_csv must contain aligned rows.")

    target = dirty.columns[target_column] if isinstance(target_column, int) else target_column
    X_dirty = dirty.drop(columns=[target])
    y = dirty[target]
    X_clean = clean.drop(columns=[target]) if target in clean.columns else clean.iloc[:, :-1]
    return X_dirty, y, X_clean


def make_clean_validation_split(
    X_dirty: pd.DataFrame,
    y: pd.Series,
    X_clean_reference: pd.DataFrame,
    test_size: float = 0.2,
    val_size: float = 0.2,
    random_state: int = 42,
) -> DatasetSplit:
    indices = np.arange(len(X_dirty))
    X_train_dirty, X_test_dirty, y_train, y_test, idx_train, idx_test = train_test_split(
        X_dirty, y, indices, test_size=test_size, random_state=random_state, stratify=_safe_stratify(y)
    )
    rel_val_size = val_size / max(1e-12, 1.0 - test_size)
    X_train_dirty, X_val_dirty, y_train, y_val, idx_train, idx_val = train_test_split(
        X_train_dirty,
        y_train,
        idx_train,
        test_size=rel_val_size,
        random_state=random_state,
        stratify=_safe_stratify(y_train),
    )

    return DatasetSplit(
        X_train_dirty=X_train_dirty.reset_index(drop=True),
        X_val_clean=X_clean_reference.iloc[idx_val].reset_index(drop=True),
        X_test_clean=X_clean_reference.iloc[idx_test].reset_index(drop=True),
        y_train=y_train.reset_index(drop=True),
        y_val=y_val.reset_index(drop=True),
        y_test=y_test.reset_index(drop=True),
        train_indices=idx_train,
        val_indices=idx_val,
        test_indices=idx_test,
        X_clean_reference=X_clean_reference.reset_index(drop=True),
    )


def apply_reference_cleaning(
    X_current: pd.DataFrame,
    clean_reference: pd.DataFrame,
    train_indices: np.ndarray,
    selected_positions: np.ndarray,
    cleaned_mask: np.ndarray | None = None,
) -> tuple[pd.DataFrame, np.ndarray]:
    X_next = X_current.copy()
    mask = np.zeros(len(X_current), dtype=bool) if cleaned_mask is None else cleaned_mask.copy()
    selected_positions = np.asarray(selected_positions, dtype=int)
    selected_positions = selected_positions[(selected_positions >= 0) & (selected_positions < len(X_current))]
    selected_positions = selected_positions[~mask[selected_positions]]
    if len(selected_positions) == 0:
        return X_next, mask
    clean_rows = clean_reference.iloc[train_indices[selected_positions]].to_numpy()
    X_next.iloc[selected_positions, :] = clean_rows
    mask[selected_positions] = True
    return X_next, mask


def _safe_stratify(y: pd.Series):
    counts = y.value_counts(dropna=False)
    return y if len(counts) > 1 and counts.min() >= 2 else None
