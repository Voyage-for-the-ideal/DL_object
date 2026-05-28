"""Training callbacks."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class EarlyStopping:
    patience: int
    mode: str = "min"  # "min" for loss (lower is better), "max" for IC (higher is better)
    best_value: float | None = None
    bad_epochs: int = 0

    def step(self, value: float) -> bool:
        if self.best_value is None:
            self.best_value = value
            self.bad_epochs = 0
            return False
        improved = (
            value < self.best_value if self.mode == "min"
            else value > self.best_value
        )
        if improved:
            self.best_value = value
            self.bad_epochs = 0
            return False
        self.bad_epochs += 1
        return self.bad_epochs >= self.patience
