"""Generate latest daily signal files."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.config.loader import load_config
from src.data.calendar import TradingCalendar
from src.data.loader import CsvDataLoader
from src.data.schema import NEXT_TRADE_DATE, SIGNAL_DATE, TRADE_DATE, TS_CODE, normalize_trade_date
from src.datasets.tabular_dataset import TabularDataset
from src.datasets.window_dataset import WindowDataset
from src.features.pipeline import build_latest_feature_frame, build_recent_feature_frame
from src.features.preprocess import TrainOnlyPreprocessor
from src.features.news_features import NewsFinbertFeatureGenerator
from src.models.base import BaseAlphaModel
from src.models.factory import load_model_artifact
from src.utils.io import ensure_dir


def latest_available_signal_date(loader: CsvDataLoader, max_date: str | None = None) -> str:
    latest = loader.latest_partition_date("daily", max_date=max_date)
    if latest is None:
        raise FileNotFoundError("No daily CSV files found")
    return latest


def generate_daily_signal(
    feature_frame: pd.DataFrame,
    model: BaseAlphaModel,
    preprocessor: TrainOnlyPreprocessor,
    signal_date: str,
    next_trade_date: str,
    label_column: str = "label_1d",
) -> pd.DataFrame:
    transformed = preprocessor.transform(feature_frame)
    if label_column not in transformed.columns:
        transformed[label_column] = 0.0
    if model.model_name == "transformer_encoder":
        signal_date = normalize_trade_date(signal_date)
        lookback = int(getattr(model, "lookback", 20))
        unique_dates = sorted({normalize_trade_date(date) for date in transformed[TRADE_DATE]})
        if len(unique_dates) < lookback:
            raise ValueError(
                "Transformer prediction requires a feature panel with at least "
                f"{lookback} dates; received {len(unique_dates)} unique trade dates. "
                "Pass a feature file containing recent history or omit --feature-file."
            )
        dataset = WindowDataset.from_frame(
            transformed,
            lookback=lookback,
            feature_columns=preprocessor.feature_columns,
            label_column=label_column,
            drop_missing_history=True,
            end_dates=[signal_date],
        )
        if len(dataset) == 0:
            raise ValueError(
                f"No transformer prediction windows ended at signal_date={signal_date}; "
                "ensure the feature panel contains that date and enough per-stock history."
            )
    else:
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
    parser.add_argument("--news-generator", default=None)
    parser.add_argument("--stock-news-generator", default=None)
    parser.add_argument("--ensemble-mode", action="store_true",
                        help="Fuse multiple model signals using ICIR-weighted rank fusion")
    parser.add_argument("--ensemble-weights", default=None,
                        help="Path to pre-computed ensemble weights JSON")
    args = parser.parse_args(argv)
    config = load_config(args.config)
    loader = CsvDataLoader(config["data"]["root"])
    calendar = TradingCalendar.from_frame(loader.load_trade_calendar())
    signal_date = args.signal_date or latest_available_signal_date(loader)
    next_trade_date = calendar.next_trade_date(signal_date)
    preprocessor = TrainOnlyPreprocessor.load(args.preprocessor)
    model = load_model_artifact(args.model, args.model_name)

    news_gen = None
    gen_path = args.news_generator
    if gen_path is None:
        default_gen = Path(config["outputs"]["root"]) / "cache" / "panels" / "news_finbert_generator.joblib"
        if default_gen.exists():
            gen_path = str(default_gen)
    if gen_path:
        news_gen = NewsFinbertFeatureGenerator.load(gen_path)

    stock_news_gen = None
    stock_gen_path = args.stock_news_generator
    if stock_gen_path is None:
        default_stock_gen = (
            Path(config["outputs"]["root"]) / "cache" / "panels" / "stock_news_generator.joblib"
        )
        if default_stock_gen.exists():
            stock_gen_path = str(default_stock_gen)
    if stock_gen_path:
        from src.features.stock_news_features import StockNewsFeatureGenerator

        stock_news_gen = StockNewsFeatureGenerator.load(stock_gen_path)

    strategy_config = config.get("strategy", {})
    data_config = config.get("data", {})
    if args.feature_file:
        features = pd.read_csv(args.feature_file)
    elif args.model_name == "transformer_encoder":
        features = build_recent_feature_frame(
            loader,
            calendar,
            signal_date,
            lookback=int(getattr(model, "lookback", config.get("dataset", {}).get("lookback", 20))),
            universe_mode=str(data_config.get("universe_mode", "official")),
            feature_windows=tuple(config.get("features", {}).get("lookback_windows", [5, 10, 20])),
            min_amount=float(strategy_config.get("min_amount", 0)),
            min_amount_pct=float(strategy_config.get("min_amount_pct", 0)),
            news_generator=news_gen,
            stock_news_generator=stock_news_gen,
            exclude_st=bool(data_config.get("official_universe_exclude_st", True)),
            exclude_bse=bool(data_config.get("official_universe_exclude_bse", True)),
        )
    else:
        features = build_latest_feature_frame(
            loader,
            calendar,
            signal_date,
            lookback=int(config.get("dataset", {}).get("lookback", 20)),
            universe_mode=str(data_config.get("universe_mode", "official")),
            min_amount=float(strategy_config.get("min_amount", 0)),
            min_amount_pct=float(strategy_config.get("min_amount_pct", 0)),
            news_generator=news_gen,
            stock_news_generator=stock_news_gen,
            exclude_st=bool(data_config.get("official_universe_exclude_st", True)),
            exclude_bse=bool(data_config.get("official_universe_exclude_bse", True)),
        )
    signal = generate_daily_signal(features, model, preprocessor, signal_date, next_trade_date)
    save_signal(signal, config["outputs"]["root"], signal_date)


if __name__ == "__main__":
    main()
