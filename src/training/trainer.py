"""Small-data friendly training orchestration."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data.schema import TRADE_DATE
from src.datasets.tabular_dataset import TabularDataset
from src.datasets.window_dataset import WindowDataset
from src.evaluation.metrics import evaluate_predictions
from src.features.preprocess import TrainOnlyPreprocessor
from src.models.base import BaseAlphaModel, format_predictions
from src.models.factory import create_model
from src.models.linear import create_linear_model
from src.training.callbacks import EarlyStopping
from src.training.losses import get_torch_loss, get_torch_optimizer, get_torch_scheduler
from src.utils.io import ensure_dir, save_yaml
from src.utils.seed import set_global_seed


def _mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean((y_true - y_pred) ** 2))


def _base_model_params(model_config: dict[str, Any]) -> dict[str, Any]:
    ignored = {"name", "input_dim", "task", "ridge", "elasticnet", "gbdt", "lookback"}
    return {key: value for key, value in model_config.items() if key not in ignored}


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
    feature_seed_ds = TabularDataset.from_frame(train_frame, label_column=label_column)
    feature_columns = feature_seed_ds.feature_columns
    preprocessor = TrainOnlyPreprocessor().fit(train_frame, feature_columns)
    train_transformed = preprocessor.transform(train_frame)
    valid_transformed = preprocessor.transform(valid_frame)

    model_config = config.get("model", {})
    training_config = config.get("training", {})
    model_name = str(model_config.get("name", "ridge"))
    lookback = int(config.get("dataset", {}).get("lookback", 20))
    if model_name == "transformer_encoder":
        train_ds = WindowDataset.from_frame(
            train_transformed,
            lookback=lookback,
            feature_columns=feature_columns,
            label_column=label_column,
        )
        valid_history = pd.concat([train_transformed, valid_transformed], ignore_index=True)
        valid_end_dates = valid_transformed[TRADE_DATE].astype(str).unique()
        valid_ds = WindowDataset.from_frame(
            valid_history,
            lookback=lookback,
            feature_columns=feature_columns,
            label_column=label_column,
            end_dates=valid_end_dates,
        )
        if len(train_ds) == 0:
            raise ValueError(
                f"No training windows were created with lookback={lookback}; "
                "extend the training date range or reduce dataset.lookback."
            )
        if len(valid_ds) == 0:
            raise ValueError(
                f"No validation windows were created with lookback={lookback}; "
                "provide enough train/validation history or reduce dataset.lookback."
            )
    else:
        train_ds = TabularDataset.from_frame(
            train_transformed,
            feature_columns=feature_columns,
            label_column=label_column,
        )
        valid_ds = TabularDataset.from_frame(
            valid_transformed,
            feature_columns=feature_columns,
            label_column=label_column,
        )
    if model_name in {"ridge", "elasticnet"}:
        params = dict(model_config.get(model_name, {}))
        model: BaseAlphaModel = create_linear_model(model_name, **params)
    elif model_name in {"gbdt", "lightgbm"}:
        params = dict(model_config.get("gbdt", {}))
        model = create_model(model_name, input_dim=len(feature_columns), **params)
    else:
        params = _base_model_params(model_config)
        if model_name == "transformer_encoder":
            params["lookback"] = lookback
        model = create_model(model_name, input_dim=len(feature_columns), **params)
    if model_name in {"mlp", "transformer_encoder", "ft_transformer"} and hasattr(model, "model"):
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
    summary = f"model_name={model.model_name}\nfeatures={len(feature_columns)}\n"
    if model_name == "transformer_encoder":
        summary += (
            f"lookback={lookback}\n"
            f"train_windows={len(train_ds)}\n"
            f"valid_windows={len(valid_ds)}\n"
        )
    (target / "model_summary.txt").write_text(summary, encoding="utf-8")
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
    train_ds: TabularDataset | WindowDataset,
    valid_ds: TabularDataset | WindowDataset,
    training_config: dict[str, Any],
) -> pd.DataFrame:
    import torch

    torch_model = getattr(model, "model")
    device = getattr(model, "device", torch.device("cpu"))
    epochs = int(training_config.get("epochs", 50))
    batch_size = int(training_config.get("batch_size", 1024))
    patience = int(training_config.get("early_stopping_patience", 10))
    base_lr = float(training_config.get("learning_rate", 0.0003))

    optimizer = get_torch_optimizer(
        training_config.get("optimizer", "adamw"),
        torch_model.parameters(),
        **training_config,
    )
    criterion = get_torch_loss(
        training_config.get("loss", "huber"),
        **training_config,
    )
    scheduler_kwargs = {k: v for k, v in training_config.items() if k != "optimizer"}
    scheduler = get_torch_scheduler(
        training_config.get("scheduler"),
        optimizer,
        **scheduler_kwargs,
    )

    warmup_epochs = int(training_config.get("warmup_epochs", 5))
    max_grad_norm = float(training_config.get("max_grad_norm", 1.0))
    early_stop_metric = training_config.get("early_stop_metric", "valid_loss")
    stopper_mode = "max" if early_stop_metric in ("valid_ic", "valid_rank_ic") else "min"
    best_metric = float("-inf") if early_stop_metric in ("valid_ic", "valid_rank_ic") else float("inf")
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(
            torch.as_tensor(train_ds.X, dtype=torch.float32),
            torch.as_tensor(train_ds.y, dtype=torch.float32),
        ),
        batch_size=batch_size,
        shuffle=True,
    )
    valid_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(
            torch.as_tensor(valid_ds.X, dtype=torch.float32),
            torch.as_tensor(valid_ds.y, dtype=torch.float32),
        ),
        batch_size=batch_size,
        shuffle=False,
    )
    stopper = EarlyStopping(patience=patience, mode=stopper_mode)
    best_state: dict[str, Any] | None = None
    rows: list[dict[str, float | int]] = []
    for epoch in range(1, epochs + 1):
        torch_model.train()
        # Warmup
        if warmup_epochs > 0 and epoch <= warmup_epochs:
            warmup_lr = base_lr * (epoch / warmup_epochs)
            for param_group in optimizer.param_groups:
                param_group["lr"] = warmup_lr
        train_losses: list[float] = []
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            loss = criterion(torch_model(xb), yb)
            loss.backward()
            if max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(torch_model.parameters(), max_grad_norm)
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))
        torch_model.eval()
        valid_scores_list: list[torch.Tensor] = []
        with torch.no_grad():
            for xb, _yb in valid_loader:
                xb = xb.to(device)
                _yb = _yb.to(device)
                valid_scores_list.append(torch_model(xb))
        valid_scores_gpu = torch.cat(valid_scores_list, dim=0)
        valid_y = torch.as_tensor(valid_ds.y, dtype=torch.float32, device=device)
        valid_loss = float(criterion(valid_scores_gpu, valid_y).detach().cpu())
        pred = format_predictions(valid_ds.index, valid_scores_gpu.detach().cpu().numpy(), model.model_name)
        pred["label"] = valid_ds.y
        metrics, _, _ = evaluate_predictions(pred)
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(valid_loss)
            else:
                scheduler.step()
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(train_losses)) if train_losses else float("nan"),
            "valid_loss": valid_loss,
            "valid_ic": metrics.get("ic_mean", float("nan")),
            "valid_rank_ic": metrics.get("rank_ic_mean", float("nan")),
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        rows.append(row)
        epoch_width = max(3, len(str(epochs)))
        print(
            f"epoch {epoch:0{epoch_width}d}/{epochs:0{epoch_width}d} | "
            f"train_loss={row['train_loss']:.6f} | "
            f"valid_loss={row['valid_loss']:.6f} | "
            f"valid_ic={row['valid_ic']:.6f} | "
            f"valid_rank_ic={row['valid_rank_ic']:.6f} | "
            f"lr={row['learning_rate']:.6g}",
            flush=True,
        )
        if early_stop_metric == "valid_rank_ic":
            monitor_value = row.get("valid_rank_ic", row["valid_ic"])
        elif early_stop_metric == "valid_ic":
            monitor_value = row["valid_ic"]
        else:
            monitor_value = row["valid_loss"]

        is_best = False
        if early_stop_metric == "valid_rank_ic":
            is_best = row.get("valid_rank_ic", -999) > best_metric
            best_metric = max(best_metric, row.get("valid_rank_ic", -999))
        elif early_stop_metric == "valid_ic":
            is_best = row["valid_ic"] > best_metric
            best_metric = max(best_metric, row["valid_ic"])
        else:
            is_best = valid_loss < best_metric
            best_metric = min(best_metric, valid_loss)

        if is_best:
            best_state = {k: v.cpu().clone() for k, v in torch_model.state_dict().items()}

        if stopper.step(monitor_value):
            print(
                f"early stopping at epoch {epoch} | "
                f"best_{early_stop_metric}={stopper.best_value:.6f} | patience={patience}",
                flush=True,
            )
            break
    if best_state is not None:
        torch_model.load_state_dict(best_state)
    return pd.DataFrame(rows)
