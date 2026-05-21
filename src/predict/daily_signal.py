"""Generate latest daily signal files."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.config.loader import load_config
from src.data.calendar import TradingCalendar
from src.data.loader import CsvDataLoader
from src.data.schema import NEXT_TRADE_DATE, SIGNAL_DATE, TS_CODE
from src.datasets.tabular_dataset import TabularDataset
from src.features.pipeline import build_latest_feature_frame
from src.features.preprocess import TrainOnlyPreprocessor
from src.models.base import BaseAlphaModel
from src.models.gbdt import GbdtRegressorModel
from src.models.linear import SklearnRegressorModel
from src.models.mlp import TorchMLPAlphaModel
from src.models.transformer import TorchTransformerAlphaModel
from src.utils.io import ensure_dir


def latest_available_signal_date(loader: CsvDataLoader, max_date: str | None = None) -> str:
    latest = loader.latest_partition_date("daily", max_date=max_date)
    if latest is None:
        raise FileNotFoundError("No daily CSV files found")
    return latest


def load_model_artifact(path: str | Path, model_name: str) -> BaseAlphaModel:
    if model_name in {"ridge", "elasticnet"}:
        return SklearnRegressorModel.load(path)
    if model_name in {"gbdt", "lightgbm", "hist_gradient_boosting"}:
        return GbdtRegressorModel.load(path)
    if model_name == "mlp":
        return TorchMLPAlphaModel.load(path)
    if model_name == "transformer_encoder":
        return TorchTransformerAlphaModel.load(path)
    raise ValueError(f"Unsupported model artifact type: {model_name}")


def generate_daily_signal(
    feature_frame: pd.DataFrame,
    model: BaseAlphaModel,
    preprocessor: TrainOnlyPreprocessor,
    signal_date: str,
    next_trade_date: str,
    label_column: str = "label_1d",
) -> pd.DataFrame:
    transformed = preprocessor.transform(feature_frame)
    dataset = TabularDataset.from_frame(
        transformed,
        feature_columns=preprocessor.feature_columns,
        label_column=label_column,
        dropna_label=False,
    )
    predictions = model.predict(dataset.X, dataset.index)
    predictions = predictions.rename(columns={"trade_date": SIGNAL_DATE})
    predictions[SIGNAL_DATE] = signal_date
    predictions[NEXT_TRADE_DATE] = next_trade_date
    predictions["rank"] = predictions["score"].rank(ascending=False, method="first").astype(int)
    return predictions[[SIGNAL_DATE, NEXT_TRADE_DATE, TS_CODE, "score", "rank", "model_name"]]


def save_signal(signal: pd.DataFrame, output_root: str | Path, signal_date: str) -> Path:
    target = ensure_dir(Path(output_root) / "signals")
    path = target / f"{signal_date}_signal.csv"
    signal.to_csv(path, index=False)
    return path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--signal-date", default=None)
    parser.add_argument("--feature-file", default=None)
    parser.add_argument("--preprocessor", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-name", default="ridge")
    args = parser.parse_args(argv)
    config = load_config(args.config)
    loader = CsvDataLoader(config["data"]["root"])
    calendar = TradingCalendar.from_frame(loader.load_trade_calendar())
    signal_date = args.signal_date or latest_available_signal_date(loader)
    next_trade_date = calendar.next_trade_date(signal_date)
    if args.feature_file:
        features = pd.read_csv(args.feature_file)
    else:
        features = build_latest_feature_frame(
            loader,
            calendar,
            signal_date,
            lookback=int(config.get("dataset", {}).get("lookback", 20)),
            universe_mode=str(config.get("data", {}).get("universe_mode", "official")),
        )
    preprocessor = TrainOnlyPreprocessor.load(args.preprocessor)
    model = load_model_artifact(args.model, args.model_name)
    signal = generate_daily_signal(features, model, preprocessor, signal_date, next_trade_date)
    save_signal(signal, config["outputs"]["root"], signal_date)


if __name__ == "__main__":
    main()
