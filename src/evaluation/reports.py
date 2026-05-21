"""Evaluation report helpers."""

from __future__ import annotations

import pandas as pd


def summarize_baselines(metrics_by_model: dict[str, dict[str, float]]) -> pd.DataFrame:
    rows = [
        {"model_name": model_name, **metrics} for model_name, metrics in metrics_by_model.items()
    ]
    return pd.DataFrame(rows)
