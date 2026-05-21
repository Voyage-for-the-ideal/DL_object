"""Training callbacks."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class EarlyStopping:
    patience: int
    best_value: float | None = None
    bad_epochs: int = 0

    def step(self, value: float) -> bool:
        if self.best_value is None or value < self.best_value:
            self.best_value = value
            self.bad_epochs = 0
            return False
        self.bad_epochs += 1
        return self.bad_epochs >= self.patience
