"""CLI entry point for building train/validation panel CSV files."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.config.loader import load_config, parse_cli_overrides
from src.data.calendar import TradingCalendar
from src.data.loader import CsvDataLoader
from src.data.news_features import aggregate_news
from src.datasets.labels import add_forward_return_labels, merge_labels
from src.datasets.split import split_from_config
from src.features.pipeline import build_feature_panel


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--output-dir", default="outputs/cache")
    parser.add_argument("--override", action="append", default=[])
    args = parser.parse_args()

    config = load_config(args.config, overrides=parse_cli_overrides(args.override))
    loader = CsvDataLoader(config["data"]["root"])
    calendar = TradingCalendar.from_frame(loader.load_trade_calendar())
    trade_dates = calendar.trade_dates_between(
        config["data"]["start_date"],
        config["data"]["valid_end_date"],
    )

    features_config = config.get("features", {})
    news_gen = None
    stock_news_gen = None

    # Load all raw news for fitting (fit-on-train-only invariant)
    raw_news_frames = []
    for date in trade_dates:
        raw = loader.load_news(date)
        if not raw.empty:
            raw_news_frames.append(raw)
    all_raw = pd.concat(raw_news_frames) if raw_news_frames else pd.DataFrame()

    # Build stock code lookup from basic.csv
    basic = loader.load_basic()
    from src.data.stock_news_extractor import (
        build_stock_lookup,
        extract_stock_codes,
        split_market_stock_news,
    )
    lookup = build_stock_lookup(basic)

    # Extract stock codes from all news and split
    train_end = str(config["data"]["train_end_date"])
    if not all_raw.empty:
        news_coded = extract_stock_codes(all_raw, lookup)
        market_news_all, stock_news_all = split_market_stock_news(news_coded)
        # Add trade_date column (from datetime) for train/valid filtering
        market_news_all["trade_date"] = pd.to_datetime(
            market_news_all["datetime"]
        ).dt.strftime("%Y%m%d")
        stock_news_all["trade_date"] = pd.to_datetime(
            stock_news_all["datetime"]
        ).dt.strftime("%Y%m%d")
    else:
        market_news_all = pd.DataFrame(columns=["datetime", "content", "title", "ts_code", "trade_date"])
        stock_news_all = pd.DataFrame(columns=["datetime", "content", "title", "ts_code", "trade_date"])

    # --- Market-level FinBERT (titles only, unmatched news, fit on train) ---
    if features_config.get("use_news_finbert"):
        from src.data.news_features import aggregate_news_titles
        from src.features.news_features import NewsFinbertFeatureGenerator

        train_market = market_news_all[
            market_news_all["trade_date"] <= train_end
        ] if not market_news_all.empty else market_news_all

        agg_train_market = aggregate_news_titles(train_market)

        news_gen = NewsFinbertFeatureGenerator(
            model_name=str(features_config["news_finbert_model"]),
            svd_dim=int(features_config["news_finbert_svd_dim"]),
        )
        print(
            f"fitting FinBERT on {len(agg_train_market)} market-level news days "
            f"(titles only) ..."
        )
        news_gen.fit(agg_train_market)
        print("FinBERT fit done.")

    # --- Stock-level TF-IDF (matched news, fit on train) ---
    if features_config.get("use_stock_news"):
        from src.data.news_features import aggregate_stock_news
        from src.features.stock_news_features import StockNewsFeatureGenerator

        train_stock = stock_news_all[
            stock_news_all["trade_date"] <= train_end
        ] if not stock_news_all.empty else stock_news_all

        agg_train_stock = aggregate_stock_news(train_stock)

        stock_news_gen = StockNewsFeatureGenerator(
            tfidf_max_features=int(
                features_config.get("stock_news_tfidf_max_features", 256)
            ),
            svd_dim=int(features_config.get("stock_news_svd_dim", 16)),
        )
        n_pairs = len(agg_train_stock)
        n_stocks = agg_train_stock["ts_code"].nunique() if n_pairs > 0 else 0
        print(
            f"fitting StockNews TF-IDF on {n_pairs} stock-news pairs "
            f"({n_stocks} unique stocks) ..."
        )
        stock_news_gen.fit(agg_train_stock)
        print("StockNews TF-IDF fit done.")

    features = build_feature_panel(
        loader,
        trade_dates,
        universe_mode=str(config["data"].get("universe_mode", "official")),
        windows=tuple(features_config.get("lookback_windows", [5, 10, 20])),
        min_amount=float(config.get("strategy", {}).get("min_amount", 0)),
        show_progress=True,
        news_generator=news_gen,
        stock_news_generator=stock_news_gen,
        exclude_st=bool(config["data"].get("official_universe_exclude_st", True)),
        exclude_bse=bool(config["data"].get("official_universe_exclude_bse", True)),
    )
    labels = add_forward_return_labels(
        loader.load_many_daily(
            trade_dates,
            show_progress=True,
            desc="load labels daily",
        ),
        horizons=tuple(config.get("label", {}).get("horizons", [1, 3, 5])),
    )
    panel = merge_labels(features, labels)
    train, valid = split_from_config(panel, config)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if news_gen is not None:
        news_gen.save(output_dir / "news_finbert_generator.joblib")
        print(f"news_generator saved to {output_dir / 'news_finbert_generator.joblib'}")
    if stock_news_gen is not None:
        stock_news_gen.save(output_dir / "stock_news_generator.joblib")
        print(f"stock_news_generator saved to {output_dir / 'stock_news_generator.joblib'}")
    train_path = output_dir / "train_panel.csv"
    valid_path = output_dir / "valid_panel.csv"
    train.to_csv(train_path, index=False)
    valid.to_csv(valid_path, index=False)

    print(f"train_panel: {train_path} {train.shape}")
    print(f"valid_panel: {valid_path} {valid.shape}")


if __name__ == "__main__":
    main()
