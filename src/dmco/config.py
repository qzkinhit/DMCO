from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict

import yaml


@dataclass
class BudgetConfig:
    total_seconds: int = 300
    n_slices: int = 10
    min_automl_seconds: int = 15
    early_stop_patience: int = 4

    @property
    def slice_seconds(self) -> int:
        return max(1, int(self.total_seconds / max(1, self.n_slices)))


@dataclass
class SamplingConfig:
    clean_batch_fraction: float = 0.05
    automl_sample_fraction: float = 0.30
    min_automl_samples: int = 200
    max_automl_samples: int = 10000
    max_gradient_samples: int | None = 5000
    gradient_variance_weight: float = 0.5
    loss_mixing_weight: float = 0.7


@dataclass
class SchedulerConfig:
    gamma: float = 0.9
    exploration: float = 1.0
    stall_patience: int = 3
    stall_boost: float = 0.5


@dataclass
class AutoMLConfig:
    backend: str = "sklearn"
    top_k_families: int = 3
    cv_folds: int = 3
    per_run_time_limit: int = 15
    family_allowlist: list[str] | None = None


@dataclass
class DMCOConfig:
    task: str = "classification"
    metric: str = "f1"
    random_state: int = 42
    budget: BudgetConfig = field(default_factory=BudgetConfig)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    automl: AutoMLConfig = field(default_factory=AutoMLConfig)


def _merge_dataclass(cls, raw: Dict[str, Any]):
    defaults = cls()
    values = {}
    for name in defaults.__dataclass_fields__:
        values[name] = raw.get(name, getattr(defaults, name))
    return cls(**values)


def load_config(path: str | Path | None = None, overrides: Dict[str, Any] | None = None) -> DMCOConfig:
    raw: Dict[str, Any] = {}
    if path:
        with Path(path).open("r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}
    if overrides:
        raw = _deep_update(raw, overrides)

    return DMCOConfig(
        task=raw.get("task", "classification"),
        metric=raw.get("metric", "f1"),
        random_state=raw.get("random_state", 42),
        budget=_merge_dataclass(BudgetConfig, raw.get("budget", {})),
        sampling=_merge_dataclass(SamplingConfig, raw.get("sampling", {})),
        scheduler=_merge_dataclass(SchedulerConfig, raw.get("scheduler", {})),
        automl=_merge_dataclass(AutoMLConfig, raw.get("automl", {})),
    )


def _deep_update(base: Dict[str, Any], patch: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_update(out[key], value)
        else:
            out[key] = value
    return out
