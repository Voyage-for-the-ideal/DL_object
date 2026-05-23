"""CLI entry point for building train/validation panel CSV files."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.config.loader import load_config, parse_cli_overrides
from src.data.calendar import TradingCalendar
from src.data.loader import CsvDataLoader
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

    features = build_feature_panel(
        loader,
        trade_dates,
        universe_mode=str(config["data"].get("universe_mode", "official")),
        windows=tuple(config.get("features", {}).get("lookback_windows", [5, 10, 20])),
        min_amount=float(config.get("strategy", {}).get("min_amount", 0)),
        show_progress=True,
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
    train_path = output_dir / "train_panel.csv"
    valid_path = output_dir / "valid_panel.csv"
    train.to_csv(train_path, index=False)
    valid.to_csv(valid_path, index=False)

    print(f"train_panel: {train_path} {train.shape}")
    print(f"valid_panel: {valid_path} {valid.shape}")


if __name__ == "__main__":
    main()
