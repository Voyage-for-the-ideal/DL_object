"""Run a full historical backtest over a date range.

Signal generation respects the project time-split convention: features for each
signal_date are built from data on or before that date only, and the backtest
engine consumes signals via ``calendar.previous_trade_date(trade_date)`` so
that no future information leaks into a trade decision.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from src.backtest.engine import BacktestEngine
from src.backtest.execution import ExecutionConfig, ExecutionEngine
from src.backtest.metrics import benchmark_nav, summarize_nav
from src.backtest.strategy import ScoreWeightedRiskControlStrategy, TopKEqualWeightStrategy
from src.config.loader import load_config
from src.data.calendar import TradingCalendar
from src.data.loader import CsvDataLoader
from src.data.schema import NEXT_TRADE_DATE, SIGNAL_DATE, TRADE_DATE, TS_CODE
from src.features.pipeline import build_feature_panel
from src.features.preprocess import TrainOnlyPreprocessor
from src.models.factory import load_model_artifact
from src.predict.daily_signal import generate_daily_signal
from src.utils.io import ensure_dir

BENCHMARK_LABEL_MAP = {
    "000300.SH": "沪深300",
    "000001.SH": "上证指数",
    "399006.SZ": "创业板指",
}


def run_backtest(
    config: dict,
    model,
    preprocessor: TrainOnlyPreprocessor,
    calendar: TradingCalendar,
    loader: CsvDataLoader,
    start_date: str,
    end_date: str,
    output_dir: str | Path,
    *,
    news_generator=None,
    stock_news_generator=None,
    show_progress: bool = False,
) -> Path:
    output_root = ensure_dir(output_dir)

    # ── 1. Trading date range ──────────────────────────────────────────
    trade_dates = calendar.trade_dates_between(start_date, end_date)
    if not trade_dates:
        raise ValueError(f"No trading days between {start_date} and {end_date}")

    # ── 2. Unique signal dates needed ──────────────────────────────────
    signal_dates_needed = sorted({
        calendar.previous_trade_date(td) for td in trade_dates
    })

    # ── 3. Pre-build feature panel (warmup + all signal dates) ─────────
    model_name = str(config["model"].get("name", "mlp"))
    features_config = config.get("features", {})
    feature_windows = tuple(features_config.get("lookback_windows", [5, 10, 20, 60]))
    lookback = int(config.get("dataset", {}).get("lookback", 20))
    max_warmup = max(max(feature_windows) - 1, lookback - 1)
    all_calendar_dates = calendar.trade_dates
    min_signal_idx = all_calendar_dates.index(min(signal_dates_needed))
    warmup_start_idx = max(0, min_signal_idx - max_warmup)
    warmup_start_date = all_calendar_dates[warmup_start_idx]
    panel_dates = calendar.trade_dates_between(warmup_start_date, max(signal_dates_needed))

    data_config = config.get("data", {})

    print(f"Building feature panel: {len(panel_dates)} dates "
          f"({warmup_start_date} – {max(signal_dates_needed)})")
    feature_panel = build_feature_panel(
        loader,
        panel_dates,
        universe_mode=str(data_config.get("universe_mode", "official")),
        windows=feature_windows,
        min_amount=float(config.get("strategy", {}).get("min_amount", 0)),
        min_amount_pct=float(config.get("strategy", {}).get("min_amount_pct", 0)),
        show_progress=show_progress,
        news_generator=news_generator,
        stock_news_generator=stock_news_generator,
        exclude_st=bool(data_config.get("official_universe_exclude_st", True)),
        exclude_bse=bool(data_config.get("official_universe_exclude_bse", True)),
    )

    # ── 4. Generate signals per signal_date ────────────────────────────
    label_column = str(config.get("label", {}).get("main", "label_5d"))
    all_signals: list[pd.DataFrame] = []
    signal_iter = tqdm(signal_dates_needed, desc="signals", disable=not show_progress)

    for signal_date in signal_iter:
        try:
            next_trade_date = calendar.next_trade_date(signal_date)
        except ValueError:
            # last trading day — no next date to trade into
            continue

        if model_name == "transformer_encoder":
            signal_idx = all_calendar_dates.index(signal_date)
            history_start_idx = max(0, signal_idx - lookback + 1)
            history_dates = {
                normalize_str(d)
                for d in all_calendar_dates[history_start_idx:signal_idx + 1]
            }
            features = feature_panel[
                feature_panel[TRADE_DATE].astype(str).isin(history_dates)
            ]
        else:
            features = feature_panel[
                feature_panel[TRADE_DATE].astype(str) == signal_date
            ]

        if features.empty:
            continue

        try:
            signal = generate_daily_signal(
                features, model, preprocessor, signal_date, next_trade_date,
                label_column=label_column,
            )
            all_signals.append(signal)
        except Exception:
            # some stocks may lack sufficient history for a given date
            continue

    if not all_signals:
        raise RuntimeError("No signals generated — check model compatibility with feature data")

    signals = pd.concat(all_signals, ignore_index=True)
    print(f"Generated {len(signals)} signal rows across {len(all_signals)} dates")

    # ── 5. Load daily market data for backtest execution ───────────────
    daily_panel = loader.load_many_daily(
        trade_dates, show_progress=show_progress, desc="load daily"
    )

    # ── 6. Execution config ────────────────────────────────────────────
    bt_config = config.get("backtest", {})
    exec_config = ExecutionConfig(
        commission_rate=float(bt_config.get("commission_rate", 0.0003)),
        stamp_tax_rate=float(bt_config.get("stamp_tax_rate", 0.001)),
        slippage_rate=float(bt_config.get("slippage_rate", 0.0005)),
        min_lot_size=int(bt_config.get("min_lot_size", 100)),
        execution_price=str(bt_config.get("execution_price", "open")),
        block_limit_up_buy=bool(bt_config.get("block_limit_up_buy", True)),
        block_limit_down_sell=bool(bt_config.get("block_limit_down_sell", True)),
        enforce_t_plus_one=bool(bt_config.get("enforce_t_plus_one", True)),
    )
    execution = ExecutionEngine(exec_config)

    strategy_config = config.get("strategy", {})
    top_k = int(strategy_config.get("top_k", 20))
    max_single = float(strategy_config.get("max_single_weight", 0.10))
    initial_cash = float(bt_config.get("initial_cash", 1_000_000))

    strategy_name = strategy_config.get("name", "score_weighted_risk_control")
    if strategy_name == "score_weighted_risk_control":
        strategy = ScoreWeightedRiskControlStrategy(top_k=top_k, max_single_weight=max_single)
    else:
        strategy = TopKEqualWeightStrategy(top_k=top_k)

    # ── 7. Strategy backtest ───────────────────────────────────────────
    print(f"Running strategy backtest ({strategy_name}) ...")
    engine = BacktestEngine(
        calendar=calendar,
        strategy=strategy,
        execution=execution,
        initial_cash=initial_cash,
    )
    strategy_outputs = engine.run(signals, daily_panel, start_date, end_date)

    # ── 8. Benchmark strategy backtest ─────────────────────────────────
    benchmark_strat = TopKEqualWeightStrategy(top_k=top_k)
    print("Running benchmark strategy backtest (topk_equal_weight) ...")
    bench_engine = BacktestEngine(
        calendar=calendar,
        strategy=benchmark_strat,
        execution=execution,
        initial_cash=initial_cash,
    )
    benchmark_outputs = bench_engine.run(signals, daily_panel, start_date, end_date)

    # ── 9. Market index benchmark ──────────────────────────────────────
    benchmark_index = "000300.SH"
    benchmark_label = "沪深300"
    benchmark_nav_frame: pd.DataFrame | None = None
    try:
        market_data = loader.load_market(benchmark_index)
        benchmark_nav_frame = benchmark_nav(market_data, start_date, end_date)
        if benchmark_nav_frame.empty:
            print(f"Warning: no {benchmark_label} data for backtest range")
            benchmark_nav_frame = None
    except Exception:
        print(f"Warning: could not load {benchmark_label} index data")

    # ── 10. Save outputs ───────────────────────────────────────────────
    strategy_dir = ensure_dir(output_root / "strategy")
    engine.save_outputs(
        strategy_outputs, strategy_dir,
        benchmark_nav_frame=benchmark_nav_frame,
        benchmark_label=benchmark_label,
        strategy_label=strategy_name,
    )

    bench_dir = ensure_dir(output_root / "benchmark_strategy")
    bench_engine.save_outputs(benchmark_outputs, bench_dir)

    signals.to_csv(output_root / "signals.csv", index=False)

    # ── 11. Summary ────────────────────────────────────────────────────
    summary: dict = {
        "run_id": output_root.name,
        "start_date": start_date,
        "end_date": end_date,
        "model": model_name,
        "strategy": strategy_name,
        "benchmark_strategy": "topk_equal_weight",
        "benchmark_index": benchmark_index,
        "strategy_metrics": summarize_nav(strategy_outputs["nav"]),
        "benchmark_strategy_metrics": summarize_nav(benchmark_outputs["nav"]),
    }
    if benchmark_nav_frame is not None:
        bench_col = (
            "benchmark_nav" if "benchmark_nav" in benchmark_nav_frame.columns
            else benchmark_nav_frame.columns[-1]
        )
        summary["market_benchmark_metrics"] = summarize_nav(
            benchmark_nav_frame.rename(columns={bench_col: "nav"})
        )

    with (output_root / "backtest_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\nBacktest complete: {output_root}")
    print(f"  Strategy ({strategy_name}): "
          f"cumulative={summary['strategy_metrics']['cumulative_return']:.2%}, "
          f"annualized={summary['strategy_metrics']['annualized_return']:.2%}, "
          f"sharpe={summary['strategy_metrics']['sharpe']:.2f}, "
          f"max_dd={summary['strategy_metrics']['max_drawdown']:.2%}")
    print(f"  Benchmark (topk_equal_weight): "
          f"cumulative={summary['benchmark_strategy_metrics']['cumulative_return']:.2%}, "
          f"annualized={summary['benchmark_strategy_metrics']['annualized_return']:.2%}, "
          f"sharpe={summary['benchmark_strategy_metrics']['sharpe']:.2f}")
    if "market_benchmark_metrics" in summary:
        print(f"  {benchmark_label}: "
              f"cumulative={summary['market_benchmark_metrics']['cumulative_return']:.2%}")

    return output_root


def normalize_str(value: object) -> str:
    return str(value).strip()


def _load_optional_news_generator(
    gen_path: str | None, config: dict, label: str
):
    """Try to load a news generator from explicit path or default cache location."""
    from src.features.news_features import NewsFinbertFeatureGenerator

    if gen_path is None:
        default = (
            Path(config["outputs"]["root"]) / "cache" / "panels" / f"{label}_generator.joblib"
        )
        if default.exists():
            gen_path = str(default)
    if gen_path:
        return NewsFinbertFeatureGenerator.load(gen_path)
    return None


def _load_optional_stock_news_generator(
    gen_path: str | None, config: dict
):
    """Try to load a stock-news generator from explicit path or default cache location."""
    from src.features.stock_news_features import StockNewsFeatureGenerator

    if gen_path is None:
        default = (
            Path(config["outputs"]["root"]) / "cache" / "panels" / "stock_news_generator.joblib"
        )
        if default.exists():
            gen_path = str(default)
    if gen_path:
        return StockNewsFeatureGenerator.load(gen_path)
    return None


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run full historical backtest")
    parser.add_argument("--config", default=None, help="path to config YAML")
    parser.add_argument("--model", required=True, help="path to model artifact")
    parser.add_argument("--preprocessor", required=True, help="path to preprocessor artifact")
    parser.add_argument("--model-name", default=None, help="model type override")
    parser.add_argument("--run-id", default=None, help="run identifier (default: timestamp)")
    parser.add_argument("--start-date", default=None, help="backtest start (default: config)")
    parser.add_argument("--end-date", default=None, help="backtest end (default: config)")
    parser.add_argument("--benchmark-index", default="000300.SH",
                        help="market index code for benchmark comparison")
    parser.add_argument("--override", action="append", default=[],
                        help="config overrides (key=value)")
    parser.add_argument("--news-generator", default=None)
    parser.add_argument("--stock-news-generator", default=None)
    parser.add_argument("--show-progress", action="store_true", default=False)
    args = parser.parse_args(argv)

    overrides = _parse_overrides(args.override)
    if args.model_name:
        overrides["model.name"] = args.model_name
    config = load_config(args.config, overrides=overrides)

    loader = CsvDataLoader(config["data"]["root"])
    calendar = TradingCalendar.from_frame(loader.load_trade_calendar())
    preprocessor = TrainOnlyPreprocessor.load(args.preprocessor)
    model_name = str(config["model"].get("name", "mlp"))
    model = load_model_artifact(args.model, model_name)

    data_config = config.get("data", {})
    start_date = args.start_date or str(data_config.get("valid_start_date", "2026-01-01"))
    end_date = args.end_date or str(data_config.get("valid_end_date", "2026-05-20"))

    output_root = Path(config["outputs"]["root"])
    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    backtest_dir = output_root / "backtest" / run_id

    news_gen = _load_optional_news_generator(args.news_generator, config, "news_finbert")
    stock_news_gen = _load_optional_stock_news_generator(args.stock_news_generator, config)

    run_backtest(
        config,
        model,
        preprocessor,
        calendar,
        loader,
        start_date,
        end_date,
        backtest_dir,
        news_generator=news_gen,
        stock_news_generator=stock_news_gen,
        show_progress=args.show_progress,
    )


def _parse_overrides(overrides: list[str]) -> dict[str, object]:
    result: dict[str, object] = {}
    for item in overrides:
        if "=" not in item:
            continue
        key, value = item.split("=", 1)
        result[key.strip()] = value.strip()
    return result


if __name__ == "__main__":
    main()
