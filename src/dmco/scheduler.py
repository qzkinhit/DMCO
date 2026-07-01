from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Dict, Iterable


@dataclass
class CostAwareUCB:
    arms: tuple[str, ...] = ("clean", "automl")
    gamma: float = 0.9
    exploration: float = 1.0
    stall_patience: int = 3
    stall_boost: float = 0.5
    last_metric: float | None = None
    counts: Dict[str, float] = field(init=False)
    rewards: Dict[str, float] = field(init=False)
    t: int = 0
    best_metric: float | None = None
    stall: int = 0

    def __post_init__(self):
        self.counts = {arm: 0.0 for arm in self.arms}
        self.rewards = {arm: 0.0 for arm in self.arms}
        self.best_metric = self.last_metric

    def choose(self, available: Iterable[str] | None = None) -> str:
        available_arms = list(available or self.arms)
        if not available_arms:
            return "terminate"
        self.t += 1
        for arm in available_arms:
            if self.counts.get(arm, 0.0) == 0:
                return arm
        scores = self.scores(available_arms)
        return max(scores, key=scores.get)

    def scores(self, available: Iterable[str] | None = None) -> Dict[str, float]:
        available_arms = list(available or self.arms)
        log_t = math.log(max(2, self.t))
        out = {}
        for arm in available_arms:
            count = self.counts[arm]
            mean = self.rewards[arm] / max(count, 1e-12)
            bonus = self.exploration * math.sqrt(log_t / (count + 1.0))
            out[arm] = mean + bonus
        if self.stall >= self.stall_patience and out:
            best_arm = max(out, key=out.get)
            for arm in out:
                if arm != best_arm:
                    out[arm] += self.stall_boost
        return out

    def observe(self, arm: str, metric: float, cost: float) -> float:
        if arm not in self.counts:
            return 0.0
        reward = 0.0 if self.last_metric is None else (metric - self.last_metric) / max(cost, 1e-12)
        self.last_metric = metric
        self.counts[arm] = self.gamma * self.counts[arm] + 1.0
        self.rewards[arm] = self.gamma * self.rewards[arm] + reward
        if self.best_metric is None or metric > self.best_metric:
            self.best_metric = metric
            self.stall = 0
        else:
            self.stall += 1
        return reward

    def snapshot(self) -> dict:
        return {
            "t": self.t,
            "counts": dict(self.counts),
            "rewards": dict(self.rewards),
            "best_metric": self.best_metric,
            "stall": self.stall,
        }
