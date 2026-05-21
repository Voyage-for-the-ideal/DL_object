# Agent Notes

## Project

This workspace is for a deep-learning course assignment: predict short-term A-share stock trends, evaluate the model, run historical backtests, and support a simulated trading competition.

Primary brief: `大作业.md`

## Data Layout

All provided data lives under `A股数据/`.

- `basic.csv`: stock metadata, including market and listing date.
- `trade_cal.csv`: trading calendar.
- `daily/`: date-partitioned daily OHLCV data, one CSV per trading day.
- `metric/`: date-partitioned daily fundamental/valuation metrics.
- `moneyflow/`: date-partitioned daily money-flow features.
- `market/`: benchmark index daily data.
- `index_weight/`: index constituents and weights.
- `news/`: daily news text, available from 2019 onward.
- `stock_st/`: daily ST stock lists, available from 2016-08 onward.

## Core Requirements

- Use at least one neural-network/deep-learning model.
- Split train/validation by time. Do not randomly shuffle time-series samples across dates.
- Avoid future information leakage in every preprocessing step.
- For a prediction made for trading on date `T`, use only data available after market close on `T-1` or earlier.
- Do not fit scalers, imputers, rankings, or feature statistics on the full dataset when evaluating historical dates.
- Filter the official trading universe for competition work: exclude ST stocks and Beijing Stock Exchange stocks.
- Include reproducibility files when code is added: `README.md` and `requirements.txt`.

## Suggested Implementation Shape

- Keep source code in `src/` or a similarly clear package directory.
- Keep experiments/configs separate from large generated outputs.
- Do not commit or submit raw data, trained weights, caches, or bulky derived artifacts.
- Prefer date-indexed pipelines that can be replayed for any historical date.
- Validate that the latest available date can produce predictions without requiring future files.

## Metrics And Backtest

Expected outputs include:

- Training and validation loss curves.
- IC and ICIR against the target future return.
- Optional direction accuracy.
- Backtest metrics such as annualized return, Sharpe ratio, and maximum drawdown.
- Baseline comparisons, for example index benchmarks or simple MLP/rule baselines.

## Environment Notes

- This directory is currently not a Git repository.
- `rg` may be unavailable due to Windows app permission issues; use PowerShell native commands as a fallback.
