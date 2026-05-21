from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.training.trainer import train_tabular_model


def test_small_tabular_training_outputs_artifacts(tmp_path: Path) -> None:
    train = pd.DataFrame(
        {
            "trade_date": ["20200101", "20200101", "20200102", "20200102"],
            "ts_code": ["A", "B", "A", "B"],
            "feature": [0.0, 1.0, 2.0, 3.0],
            "label_1d": [0.0, 1.0, 2.0, 3.0],
        }
    )
    valid = pd.DataFrame(
        {
            "trade_date": ["20200103", "20200103", "20200104", "20200104"],
            "ts_code": ["A", "B", "A", "B"],
            "feature": [4.0, 5.0, 6.0, 7.0],
            "label_1d": [4.0, 5.0, 6.0, 7.0],
        }
    )
    config = {"model": {"name": "ridge"}, "training": {"seed": 42, "learning_rate": 0.1}}
    result = train_tabular_model(train, valid, config, tmp_path)
    assert (tmp_path / "train_log.csv").exists()
    assert (tmp_path / "valid_predictions.csv").exists()
    assert result["metrics"]["valid_loss"] >= 0
