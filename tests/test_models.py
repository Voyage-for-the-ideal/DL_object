from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.models.linear import SklearnRegressorModel
from src.models.mlp import torch as torch_mlp
from src.models.transformer import torch as torch_transformer


def test_sklearn_model_save_load_predictions_match(tmp_path: Path) -> None:
    X = np.array([[0.0], [1.0], [2.0]], dtype=np.float32)
    y = np.array([0.0, 1.0, 2.0], dtype=np.float32)
    model = SklearnRegressorModel.ridge().fit(X, y)
    path = model.save(tmp_path / "ridge.joblib")
    loaded = SklearnRegressorModel.load(path)
    np.testing.assert_allclose(model.predict_array(X), loaded.predict_array(X))
    index = pd.DataFrame({"trade_date": ["1", "2", "3"], "ts_code": ["A", "A", "A"]})
    assert list(model.predict(X, index).columns) == ["trade_date", "ts_code", "score", "model_name"]


@pytest.mark.skipif(torch_mlp is None, reason="PyTorch not installed")
def test_mlp_forward_shape() -> None:
    from src.models.mlp import MLPRegressor

    model = MLPRegressor(input_dim=3, hidden_dims=(4,), dropout=0.0)
    output = model(torch_mlp.randn(2, 3))
    assert tuple(output.shape) == (2,)


@pytest.mark.skipif(torch_transformer is None, reason="PyTorch not installed")
def test_transformer_forward_shape() -> None:
    from src.models.transformer import TransformerRegressor

    model = TransformerRegressor(input_dim=3, hidden_dim=8, num_layers=1, num_heads=2)
    output = model(torch_transformer.randn(2, 5, 3))
    assert tuple(output.shape) == (2,)
