#!/bin/bash
set -euo pipefail
PROJECT_ROOT="D:/code/DL_foundation/DL_object"
cd "$PROJECT_ROOT"
export CUDA_VISIBLE_DEVICES=0

echo "=== [Hour 0-2] Building feature panels ==="
python prepare_panels.py --output-dir outputs/cache/panels_v2

echo "=== [Hour 2-12] Label experiment ==="
python train.py \
  --train-panel outputs/cache/panels_v2/train_panel.csv \
  --valid-panel outputs/cache/panels_v2/valid_panel.csv \
  --label-experiment \
  --label-types label_5d label_5d_vol_norm label_5d_cs_rank label_5d_excess \
  --label-models gbdt ft_transformer

# Read best label from outputs/label_experiment/label_comparison.csv
BEST_LABEL=$(python -c "
import pandas as pd
df = pd.read_csv('outputs/label_experiment/label_comparison.csv')
best = df.sort_values('Rank_IC', ascending=False).iloc[0]
print(best['label'])
")
echo "Best label: $BEST_LABEL"

echo "=== [Hour 12-14] GBDT full training + feature selection ==="
python -c "
from src.config.loader import load_config
from src.models.gbdt_selector import gbdt_select_and_train_ftt
import pandas as pd
config = load_config()
config['label']['main'] = '$BEST_LABEL'
train = pd.read_csv('outputs/cache/panels_v2/train_panel.csv')
valid = pd.read_csv('outputs/cache/panels_v2/valid_panel.csv')
top_features, metrics = gbdt_select_and_train_ftt(
    config, train, valid, top_k=128,
    output_dir='outputs/feature_selection'
)
print(f'GBDT Rank IC: {metrics[\"rank_ic_mean\"]}')
"

echo "=== [Hour 14-24] FT-Transformer multi-seed training ==="
for SEED in 42 123 456; do
  echo "Training FT-Transformer with seed=$SEED"
  python train.py \
    --train-panel outputs/cache/panels_v2/train_panel.csv \
    --valid-panel outputs/cache/panels_v2/valid_panel.csv \
    --output-dir outputs/runs/ftt_seed_${SEED} \
    --model-name ft_transformer \
    --override label.main=$BEST_LABEL \
    --override training.seed=$SEED \
    --override training.epochs=100 \
    --override training.early_stopping_patience=10
done

echo "=== [Hour 24-27] News sentiment (if news data available) ==="
# Optional: Fine-tune FinBERT on A-share news
# python scripts/train_news_sentiment.py

echo "=== [Hour 27-28] Ensemble weights ==="
python -c "
from src.models.ensemble import compute_ensemble_weights_from_dir
ensemble = compute_ensemble_weights_from_dir(
    predictions_dir='outputs/runs',
    model_names=['gbdt', 'ftt_seed_42', 'ftt_seed_123', 'ftt_seed_456'],
)
ensemble.save('outputs/ensemble/ensemble_weights.json')
print(f'Weights: {ensemble.weights}')
print(f'ICIR: {ensemble._icir}')
"

echo "=== [Hour 28-30] Full backtest ==="
python backtest.py \
  --config src/config/default.yaml \
  --model-name ft_transformer \
  --preprocessor outputs/runs/ftt_seed_42/preprocessor.joblib \
  --ensemble-weights outputs/ensemble/ensemble_weights.json \
  --start-date 2025-01-02 \
  --end-date 2026-05-20 \
  --output-dir outputs/backtest/final

echo "=== Done ==="
