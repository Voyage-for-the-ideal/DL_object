from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.models.transformer import torch as torch_transformer
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


@pytest.mark.skipif(torch_transformer is None, reason="PyTorch not installed")
def test_small_transformer_training_uses_window_datasets(tmp_path: Path) -> None:
    train = pd.DataFrame(
        {
            "trade_date": [
                "20200101",
                "20200101",
                "20200102",
                "20200102",
                "20200103",
                "20200103",
            ],
            "ts_code": ["A", "B", "A", "B", "A", "B"],
            "feature": [0.0, 10.0, 1.0, 11.0, 2.0, 12.0],
            "label_1d": [0.0, 1.0, 0.1, 1.1, 0.2, 1.2],
        }
    )
    valid = pd.DataFrame(
        {
            "trade_date": ["20200104", "20200104", "20200105", "20200105"],
            "ts_code": ["A", "B", "A", "B"],
            "feature": [3.0, 13.0, 4.0, 14.0],
            "label_1d": [0.3, 1.3, 0.4, 1.4],
        }
    )
    config = {
        "model": {
            "name": "transformer_encoder",
            "hidden_dim": 8,
            "num_layers": 1,
            "num_heads": 2,
            "dropout": 0.0,
        },
        "dataset": {"lookback": 2},
        "training": {
            "seed": 42,
            "epochs": 1,
            "batch_size": 2,
            "learning_rate": 0.001,
            "early_stopping_patience": 2,
        },
    }
    result = train_tabular_model(train, valid, config, tmp_path)
    predictions = pd.read_csv(tmp_path / "valid_predictions.csv")
    summary = (tmp_path / "model_summary.txt").read_text(encoding="utf-8")
    assert len(predictions) == 4
    assert "lookback=2" in summary
    assert result["metrics"]["valid_loss"] >= 0
