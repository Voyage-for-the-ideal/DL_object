# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

A-share (中国A股) short-term trend prediction using deep learning — a university DL course assignment. Supports feature engineering, model training, evaluation, backtesting, and daily simulated trading signal/order generation.

## Commands

```bash
# Build feature panels (prerequisite for training)
python prepare_panels.py --output-dir outputs/cache/panels

# Train a single model
python train.py --train-panel outputs/cache/panels/train_panel.csv \
                --valid-panel outputs/cache/panels/valid_panel.csv \
                --model-name mlp

# Train with config overrides
python train.py ... --override training.epochs=50 --override model.dropout=0.2

# Batch train all models
bash scripts/train_all_models.sh
REBUILD_PANELS=1 bash scripts/train_all_models.sh  # with panel rebuild

# Evaluate predictions
python evaluate.py --predictions outputs/runs/<run_id>/valid_predictions.csv \
                   --output-dir outputs/evaluation/<run_id>

# Generate daily trading signals
python -m src.predict.daily_signal --config src/config/default.yaml --model-name mlp

# Generate daily orders from signals
python -m src.predict.daily_order --signal-file outputs/signals/<date>_signal.csv

# Run all tests
python -m pytest

# Run a single test file
python -m pytest tests/test_models.py

# Lint and type check
python -m ruff check .
python -m mypy src tests
```

## Architecture

The pipeline flows in strict order: **raw CSV → features → labels → dataset → model → predictions → evaluation/backtest**.

### Data flow

1. `CsvDataLoader` reads date-partitioned CSVs from `A股数据/` (daily OHLCV, metrics, moneyflow, news, index weights)
2. `build_feature_panel()` orchestrates feature construction per trading day: price features, metric features, moneyflow features, TF-IDF news features — all using rolling windows with no future leakage
3. `add_forward_return_labels()` computes `label_hd(T) = close(T+h+2)/close(T+1) − 1` (respects A-share T+1 settlement)
4. `TrainOnlyPreprocessor` fits winsorize/impute/scale parameters **only on training data**, then transforms validation/test
5. For tabular models: `TabularDataset` produces 2D `[samples, features]`. For transformers: `WindowDataset` produces 3D `[samples, lookback, features]` via per-stock sliding windows with left-zero-padding

### Critical invariants

- **Time-based split only.** `train_end_date < valid_start_date` is enforced. Never shuffle time-series samples across dates.
- **Fit-on-train-only.** Scalers, imputers, quantile thresholds, TF-IDF vocabulary — all fitted exclusively on training data. Use `.transform()` only for validation/backtest/prediction.
- **No future leakage.** For a prediction on date T, only data available after market close on T−1 or earlier is used. Rolling windows are strictly backward-looking.
- **Date normalization.** All dates are normalized to `YYYYMMDD` strings via `normalize_trade_date()`.
- **Official universe.** Excludes ST stocks and Beijing Stock Exchange (北交所) stocks.

### Model dispatch

`create_model(name, input_dim, **kwargs)` in `src/models/factory.py` dispatches by name:
- `ridge` / `elasticnet` → scikit-learn Ridge/ElasticNet
- `gbdt` → LightGBM (with `HistGradientBoostingRegressor` fallback)
- `mlp` → PyTorch MLP with BatchNorm+ReLU+Dropout blocks
- `transformer_encoder` → TransformerEncoder + PositionalEncoding, takes 3D windowed input
- `gru` / `lstm` / `tcn` → reserved names, not yet implemented

### Training orchestration

`train_tabular_model()` handles the full flow: seed → preprocessor fit → dataset construction (choosing TabularDataset vs WindowDataset based on model type) → model creation → training loop with early stopping → artifact saving (metrics.json, model weights, preprocessor, config, loss curve).

### Configuration

Single source of truth: `src/config/default.yaml`. Loaded via `load_config()` which resolves paths relative to project root and applies CLI `--override key=value` pairs (e.g., `--override training.lr=0.001`).

### Backtest

`BacktestEngine` runs a daily loop: reads signals from previous trade date → executes strategy orders via `ExecutionEngine` (enforces T+1, lot rounding, limit-up/down blocking) → tracks NAV. Two strategies: `TopKEqualWeightStrategy` and `ScoreWeightedRiskControlStrategy` (with single-stock and industry caps).

### Output structure

```
outputs/
  cache/         # Prepared panels
  runs/<id>/     # Training artifacts per run
  evaluation/    # IC curves, metrics
  signals/       # Daily signal CSVs
  orders/        # Daily order CSVs
  logs/          # Training logs
```

### Code quality

- Python 3.10 target, all public functions have type hints
- Ruff (line-length 100, rules E/F/I/UP/B), mypy (ignore_missing_imports), isort (black profile)
- Tests use `tmp_path` fixtures with synthetic data — no real data required
