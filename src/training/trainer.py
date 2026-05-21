"""Small-data friendly training orchestration."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.datasets.tabular_dataset import TabularDataset
from src.evaluation.metrics import evaluate_predictions
from src.features.preprocess import TrainOnlyPreprocessor
from src.models.base import BaseAlphaModel
from src.models.factory import create_model
from src.models.linear import create_linear_model
from src.training.callbacks import EarlyStopping
from src.utils.io import ensure_dir, save_yaml
from src.utils.seed import set_global_seed


def _mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean((y_true - y_pred) ** 2))


def train_tabular_model(
    train_frame: pd.DataFrame,
    valid_frame: pd.DataFrame,
    config: dict[str, Any],
    run_dir: str | Path,
    label_column: str = "label_1d",
) -> dict[str, Any]:
    """Fit a configured tabular model and write standard artifacts."""
    target = ensure_dir(run_dir)
    set_global_seed(int(config.get("training", {}).get("seed", 42)))
    train_ds = TabularDataset.from_frame(train_frame, label_column=label_column)
    preprocessor = TrainOnlyPreprocessor().fit(train_frame, train_ds.feature_columns)
    train_transformed = preprocessor.transform(train_frame)
    valid_transformed = preprocessor.transform(valid_frame)
    train_ds = TabularDataset.from_frame(
        train_transformed,
        feature_columns=train_ds.feature_columns,
        label_column=label_column,
    )
    valid_ds = TabularDataset.from_frame(
        valid_transformed,
        feature_columns=train_ds.feature_columns,
        label_column=label_column,
    )

    model_config = config.get("model", {})
    training_config = config.get("training", {})
    model_name = str(model_config.get("name", "ridge"))
    if model_name in {"ridge", "elasticnet"}:
        model: BaseAlphaModel = create_linear_model(model_name)
    else:
        model = create_model(model_name, input_dim=len(train_ds.feature_columns), **model_config)
    if model_name == "mlp" and hasattr(model, "model"):
        log = _fit_torch_tabular_with_log(model, train_ds, valid_ds, training_config)
    else:
        model.fit(train_ds.X, train_ds.y, **training_config)
        log = None

    train_pred = model.predict_array(train_ds.X)
    valid_pred = model.predict(valid_ds.X, valid_ds.index)
    valid_pred[label_column] = valid_ds.y
    metrics, _, _ = evaluate_predictions(valid_pred.rename(columns={label_column: "label"}))
    metrics["train_loss"] = _mse(train_ds.y, train_pred)
    metrics["valid_loss"] = _mse(valid_ds.y, valid_pred["score"].to_numpy())

    if log is None:
        log = pd.DataFrame(
            [
                {
                    "epoch": 1,
                    "train_loss": metrics["train_loss"],
                    "valid_loss": metrics["valid_loss"],
                    "valid_ic": metrics.get("ic_mean"),
                    "valid_rank_ic": metrics.get("rank_ic_mean"),
                    "learning_rate": training_config.get("learning_rate", np.nan),
                }
            ]
        )
    log.to_csv(target / "train_log.csv", index=False)
    valid_pred.to_csv(target / "valid_predictions.csv", index=False)
    with (target / "metrics.json").open("w", encoding="utf-8") as file:
        json.dump(metrics, file, ensure_ascii=False, indent=2)
    save_yaml(config, target / "config.yaml")
    preprocessor.save(target / "preprocessor.joblib")
    model.save(
        target / "model.joblib"
        if model_name in {"ridge", "elasticnet", "gbdt", "lightgbm"}
        else target / "model.pt"
    )
    (target / "model_summary.txt").write_text(
        f"model_name={model.model_name}\nfeatures={len(train_ds.feature_columns)}\n",
        encoding="utf-8",
    )
    plt.figure()
    plt.plot(log["epoch"], log["train_loss"], label="train")
    plt.plot(log["epoch"], log["valid_loss"], label="valid")
    plt.legend()
    plt.tight_layout()
    plt.savefig(target / "loss_curve.png")
    plt.close()
    return {"metrics": metrics, "model": model, "preprocessor": preprocessor, "run_dir": target}


def _fit_torch_tabular_with_log(
    model: BaseAlphaModel,
    train_ds: TabularDataset,
    valid_ds: TabularDataset,
    training_config: dict[str, Any],
) -> pd.DataFrame:
    import torch

    torch_model = getattr(model, "model")
    device = getattr(model, "device", torch.device("cpu"))
    epochs = int(training_config.get("epochs", 50))
    batch_size = int(training_config.get("batch_size", 1024))
    learning_rate = float(training_config.get("learning_rate", 0.001))
    patience = int(training_config.get("early_stopping_patience", 5))
    optimizer = torch.optim.Adam(torch_model.parameters(), lr=learning_rate)
    criterion = torch.nn.MSELoss()
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(
            torch.as_tensor(train_ds.X, dtype=torch.float32),
            torch.as_tensor(train_ds.y, dtype=torch.float32),
        ),
        batch_size=batch_size,
        shuffle=True,
    )
    valid_x = torch.as_tensor(valid_ds.X, dtype=torch.float32, device=device)
    valid_y = torch.as_tensor(valid_ds.y, dtype=torch.float32, device=device)
    stopper = EarlyStopping(patience=patience)
    best_state: dict[str, Any] | None = None
    rows: list[dict[str, float | int]] = []
    for epoch in range(1, epochs + 1):
        torch_model.train()
        train_losses: list[float] = []
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            loss = criterion(torch_model(xb), yb)
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))
        torch_model.eval()
        with torch.no_grad():
            valid_scores = torch_model(valid_x)
            valid_loss = float(criterion(valid_scores, valid_y).detach().cpu())
        pred = model.predict(valid_ds.X, valid_ds.index)
        pred["label"] = valid_ds.y
        metrics, _, _ = evaluate_predictions(pred)
        rows.append(
            {
                "epoch": epoch,
                "train_loss": float(np.mean(train_losses)) if train_losses else float("nan"),
                "valid_loss": valid_loss,
                "valid_ic": metrics.get("ic_mean", float("nan")),
                "valid_rank_ic": metrics.get("rank_ic_mean", float("nan")),
                "learning_rate": learning_rate,
            }
        )
        if stopper.best_value is None or valid_loss < stopper.best_value:
            best_state = deepcopy(torch_model.state_dict())
        if stopper.step(valid_loss):
            break
    if best_state is not None:
        torch_model.load_state_dict(best_state)
    return pd.DataFrame(rows)
