# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

每次做出代码改动后，在最后说明有哪些文件被改动了

## Project

A-share (中国A股) short-term cross-sectional ranking prediction using deep learning — a university DL course assignment. Supports feature engineering, model training, label comparison experiments, evaluation, backtesting, multi-signal ensemble fusion, and daily simulated trading signal/order generation.

## Commands

```bash
# Smoke test — verify all modules work (no panel data needed)
python scripts/test_pipeline.py --quick

# Build feature panels with alpha factors (prerequisite for training)
python prepare_panels.py --output-dir outputs/cache/panels

# Train a single model
python train.py --train-panel outputs/cache/panels/train_panel.csv \
                --valid-panel outputs/cache/panels/valid_panel.csv \
                --model-name ft_transformer

# Train with config overrides
python train.py ... --override training.epochs=50 --override model.ft_transformer.d_token=64

# Label comparison experiment (grid search over labels × models)
python train.py --train-panel outputs/cache/panels/train_panel.csv \
                --valid-panel outputs/cache/panels/valid_panel.csv \
                --label-experiment \
                --label-types label_5d label_5d_cs_rank \
                --label-models gbdt ft_transformer

# Batch train all models (legacy)
bash scripts/train_all_models.sh

# Full pipeline SLURM submission (30h, A100)
sbatch scripts/train_full_pipeline.sbatch

# Evaluate predictions
python evaluate.py --predictions outputs/runs/<run_id>/valid_predictions.csv \
                   --output-dir outputs/evaluation/<run_id>

# Generate daily trading signals
python -m src.predict.daily_signal --config src/config/default.yaml --model-name ft_transformer

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

The pipeline flows in strict order: **raw CSV → features → labels → dataset → model → predictions → evaluation/backtest → ensemble**.

### Data flow

1. `CsvDataLoader` reads date-partitioned CSVs from `A股数据/` (daily OHLCV, metrics, moneyflow, news, index weights)
2. `build_feature_panel()` orchestrates feature construction per trading day: price features, metric features, moneyflow features, **alpha factors** (~144 factors: price-derived, metric-derived, moneyflow-derived, cross-sectional rank, momentum/reversal hybrids), and optionally TF-IDF/news-sentiment features — all using rolling windows with no future leakage
3. `add_forward_return_labels()` computes `label_hd(T) = close(T+h+2)/close(T+1) − 1` (respects A-share T+1 settlement). Supports 4 label types: `label_5d`, `label_5d_vol_norm`, `label_5d_cs_rank`, `label_5d_excess` (excess vs market, requires `market_df`)
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
- `gbdt` → LightGBM (with `HistGradientBoostingRegressor` fallback). Supports `feature_importance()` for feature selection.
- `mlp` → PyTorch MLP with BatchNorm+ReLU+Dropout blocks
- `ft_transformer` → FT-Transformer (Gorishniy et al. 2021): FeatureTokenizer maps each feature to a d_token-dim embedding → [CLS] token prepended → learnable feature positional encoding → Pre-LN Transformer blocks with DropPath (stochastic depth, linearly increasing) → CLS head regression. Takes 2D tabular input `[B, F]`.
- `transformer_encoder` → Legacy temporal Transformer with PositionalEncoding, takes 3D windowed input `[B, lookback, F]`

### Training orchestration

`train_tabular_model()` handles the full flow: seed → preprocessor fit → dataset construction (choosing TabularDataset vs WindowDataset based on model type) → model creation → training loop → artifact saving (metrics.json, model weights, preprocessor, config, loss curve).

The training loop (`_fit_torch_tabular_with_log()`) supports:
- **Configurable loss**: `mse` or `huber` (with `huber_delta`)
- **Configurable optimizer**: `adam` or `adamw` (with `weight_decay`)
- **Configurable scheduler**: `cosine_restart` (CosineAnnealingWarmRestarts), `cosine`, `reduce_on_plateau`, or `none`
- **Linear warmup**: `warmup_epochs` controls learning rate ramp-up
- **Gradient clipping**: `max_grad_norm` clip norm before optimizer step
- **Metric-aware early stopping**: monitors `valid_loss`, `valid_ic`, or `valid_rank_ic` — mode auto-switches between min/max

### Alpha factors

`build_alpha_factors()` in `src/features/alpha_factors.py` generates ~144 factors from daily/ metric/ moneyflow data:
1. **Price-derived**: ret_1d, ret_overnight, ret_intraday, hl_range, gap, vol/amount changes, multi-horizon returns (1/5/10/20d), rolling mean/std/sharpe/skew of ret_1d, downside volatility, max drawdown, volume-price correlation
2. **Metric-derived**: pe, pb, ps, total_mv, circ_mv, turnover_rate, volume_ratio with log transforms, multi-horizon changes, rolling means
3. **Moneyflow-derived**: net flows per order-size level (sm/md/lg/elg), imbalance ratios, large-order dominance, consecutive net inflow days, rolling stats with acceleration
4. **Cross-sectional**: per-date rank percentiles of top-50 factors
5. **Momentum/reversal hybrids**: short-term × long-term return interactions

### Label comparison

`run_label_comparison()` in `src/training/label_experiment.py` runs a systematic grid search over (label_type × model_name) pairs, training each combination and ranking by `valid_rank_ic`. Results saved to `outputs/label_experiment/label_comparison.csv`.

### GBDT feature selection

`gbdt_select_and_train_ftt()` in `src/models/gbdt_selector.py`: (1) train GBDT on all features → (2) extract top-k by importance → (3) train FT-Transformer on reduced feature set. Reduces noise for FT-Transformer input.

### News sentiment

`FinBERTSentimentModel` in `src/models/news_sentiment.py`: dual-task model based on `yiyanghkust/finbert-tone-chinese` with LoRA (r=16, target=query/key/value). Sentiment classification (3-class) + intensity regression (-1 to 1). `NewsSentimentFeatureGenerator` aggregates per-stock per-day.

### Ensemble fusion

`SignalEnsemble` in `src/models/ensemble.py`: computes ICIR-based weights from validation predictions → fuses signals via cross-sectional rank percentile weighted sum. Handles scale differences between GBDT (return space), FT-Transformer (learned space), and News ([-1, 1]).

### Configuration

Single source of truth: `src/config/default.yaml`. Loaded via `load_config()` with CLI `--override key=value` pairs.

Key defaults:
- **dataset.type**: `tabular`
- **model.name**: `ft_transformer`
- **label.main**: `label_5d_cs_rank`
- **training**: AdamW, Huber loss, CosineWarmRestarts, warmup=5, grad_clip=1.0, early_stop_metric=valid_rank_ic
- **ensemble**: gbdt=0.50, ft_transformer=0.30, news=0.20

### Backtest

`BacktestEngine` runs a daily loop: reads signals from previous trade date → executes strategy orders via `ExecutionEngine` (enforces T+1, lot rounding, limit-up/down blocking) → tracks NAV. Two strategies: `TopKEqualWeightStrategy` and `ScoreWeightedRiskControlStrategy` (with single-stock and industry caps).

### Output structure

```
outputs/
  cache/panels/        # Prepared feature panels
  runs/<id>/           # Training artifacts per run
  evaluation/          # IC curves, metrics
  signals/             # Daily signal CSVs
  orders/              # Daily order CSVs
  logs/                # Training logs
  label_experiment/    # Label comparison results
  feature_selection/   # GBDT importance + selected features
  ensemble/            # ICIR ensemble weights
  backtest/            # Backtest NAV and reports
```

### Code quality

- Python 3.10 target, all public functions have type hints
- Ruff (line-length 100, rules E/F/I/UP/B), mypy (ignore_missing_imports), isort (black profile)
- Tests use `tmp_path` fixtures with synthetic data — no real data required
