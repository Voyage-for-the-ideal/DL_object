#!/bin/bash
#SBATCH --job-name=dl-full-pipeline
#SBATCH --partition=Students
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=30:00:00
#SBATCH --output=outputs/logs/full_pipeline_%j.log

set -euo pipefail

PROJECT_DIR=~/Desktop/DL_object
cd "$PROJECT_DIR"

# 激活 conda 环境
export PATH="/home/scc/pb23151808/software/miniconda3/bin:$PATH"
source /home/scc/pb23151808/software/miniconda3/etc/profile.d/conda.sh
conda activate DL
set -u

# 检测 GPU
DEVICE="cpu"
if python -c "import torch; print(torch.cuda.is_available())" 2>/dev/null | grep -q True; then
    DEVICE="cuda"
fi

PANEL_DIR="$PROJECT_DIR/outputs/cache/panels_v2"
LOG_DIR="$PROJECT_DIR/outputs/logs"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

mkdir -p "$PANEL_DIR" "$LOG_DIR"

echo "=========================================="
echo "project:   $PROJECT_DIR"
echo "device:    $DEVICE"
echo "start:     $(date)"
echo "job_id:    ${SLURM_JOB_ID:-local}"
echo "=========================================="

# ═══════════════════════════════════════════════════════════════
# Step 0: Smoke test (5 min)
# ═══════════════════════════════════════════════════════════════

echo ""
echo "[Step 0/7] Running smoke tests..."
echo ""

python scripts/test_pipeline.py --quick 2>&1 | tee "$LOG_DIR/${TIMESTAMP}_smoke_test.log"

echo ""
echo "Smoke test done: $(date)"

# ═══════════════════════════════════════════════════════════════
# Step 1: Build feature panels (1-2 hours)
# ═══════════════════════════════════════════════════════════════

echo ""
echo "[Step 1/7] Building feature panels..."
echo ""

python prepare_panels.py \
    --config src/config/default.yaml \
    --output-dir "$PANEL_DIR" \
    2>&1 | tee "$LOG_DIR/${TIMESTAMP}_prepare_panels.log"

# 快速列数检查
python -c "
import pandas as pd
df = pd.read_csv('$PANEL_DIR/train_panel.csv', nrows=5)
print(f'Panel columns: {len(df.columns)}')
assert len(df.columns) >= 200, f'Expected >=200 cols, got {len(df.columns)}'
assert df.isna().all().sum() == 0, 'Found all-NaN columns'
print('Panel check OK')
"

echo ""
echo "Panels built: $(date)"

# ═══════════════════════════════════════════════════════════════
# Step 2: Label experiment (8-10 hours)
# ═══════════════════════════════════════════════════════════════

echo ""
echo "[Step 2/7] Running label comparison experiment..."
echo ""

python train.py \
    --config src/config/default.yaml \
    --train-panel "$PANEL_DIR/train_panel.csv" \
    --valid-panel "$PANEL_DIR/valid_panel.csv" \
    --label-experiment \
    --label-types label_5d label_5d_vol_norm label_5d_cs_rank \
    --label-models gbdt ft_transformer \
    2>&1 | tee "$LOG_DIR/${TIMESTAMP}_label_experiment.log"

BEST_LABEL=$(python -c "
import pandas as pd
df = pd.read_csv('outputs/label_experiment/label_comparison.csv')
best = df.sort_values('Rank_IC', ascending=False).iloc[0]
print(best['label'])
")
echo "Best label: $BEST_LABEL"

echo ""
echo "Label experiment done: $(date)"

# ═══════════════════════════════════════════════════════════════
# Step 3: GBDT full training + feature selection (2 hours)
# ═══════════════════════════════════════════════════════════════

echo ""
echo "[Step 3/7] GBDT training + feature selection..."
echo ""

python -c "
from src.config.loader import load_config
from src.models.gbdt_selector import gbdt_select_and_train_ftt
import pandas as pd

config = load_config()
config['label']['main'] = '$BEST_LABEL'

train = pd.read_csv('$PANEL_DIR/train_panel.csv')
valid = pd.read_csv('$PANEL_DIR/valid_panel.csv')

top_features, metrics = gbdt_select_and_train_ftt(
    config, train, valid, top_k=128,
    output_dir='outputs/feature_selection'
)
print(f'GBDT Rank IC: {metrics.get(\"rank_ic_mean\", float(\"nan\"))}')
print(f'FTT  Rank IC: {metrics.get(\"rank_ic_mean\", float(\"nan\"))}')
" 2>&1 | tee "$LOG_DIR/${TIMESTAMP}_feature_selection.log"

echo ""
echo "Feature selection done: $(date)"

# ═══════════════════════════════════════════════════════════════
# Step 4: FT-Transformer multi-seed training (10 hours)
# ═══════════════════════════════════════════════════════════════

echo ""
echo "[Step 4/7] FT-Transformer multi-seed training..."
echo ""

for SEED in 42 123 456; do
    echo "--- Training FT-Transformer with seed=$SEED ---"
    python train.py \
        --config src/config/default.yaml \
        --train-panel "$PANEL_DIR/train_panel.csv" \
        --valid-panel "$PANEL_DIR/valid_panel.csv" \
        --run-id "ftt_seed_${SEED}" \
        --model-name ft_transformer \
        --override "label.main=$BEST_LABEL" \
        --override "training.seed=$SEED" \
        --override "training.epochs=100" \
        --override "training.early_stopping_patience=10" \
        --override "model.device=$DEVICE" \
        --override "training.device=$DEVICE" \
        2>&1 | tee "$LOG_DIR/${TIMESTAMP}_ftt_seed_${SEED}.log"
    echo "FT-Transformer seed=$SEED done: $(date)"
done

echo ""
echo "FT-Transformer training done: $(date)"

# ═══════════════════════════════════════════════════════════════
# Step 5: Train single best GBDT (for ensemble) (1 hour)
# ═══════════════════════════════════════════════════════════════

echo ""
echo "[Step 5/7] Training final GBDT for ensemble..."
echo ""

python train.py \
    --config src/config/default.yaml \
    --train-panel "$PANEL_DIR/train_panel.csv" \
    --valid-panel "$PANEL_DIR/valid_panel.csv" \
    --run-id "gbdt_full" \
    --model-name gbdt \
    --override "label.main=$BEST_LABEL" \
    --override "model.gbdt.n_estimators=1000" \
    --override "model.device=$DEVICE" \
    2>&1 | tee "$LOG_DIR/${TIMESTAMP}_gbdt_full.log"

# ═══════════════════════════════════════════════════════════════
# Step 6: Compute ensemble weights (5 min)
# ═══════════════════════════════════════════════════════════════

echo ""
echo "[Step 6/7] Computing ensemble weights..."
echo ""

python -c "
from src.models.ensemble import compute_ensemble_weights_from_dir
from pathlib import Path

ensemble = compute_ensemble_weights_from_dir(
    predictions_dir='outputs/runs',
    model_names=['gbdt_full', 'ftt_seed_42', 'ftt_seed_123', 'ftt_seed_456'],
)
Path('outputs/ensemble').mkdir(parents=True, exist_ok=True)
ensemble.save('outputs/ensemble/ensemble_weights.json')
print(f'Weights: {ensemble.weights}')
print(f'ICIR: {ensemble._icir}')
"

# ═══════════════════════════════════════════════════════════════
# Step 7: Full backtest (2 hours)
# ═══════════════════════════════════════════════════════════════

echo ""
echo "[Step 7/7] Running full backtest..."
echo ""

# 用 Rank IC 最高的 seed 做 backtest
BEST_SEED=$(python -c "
import json
from pathlib import Path
best = {'seed': '42', 'ic': -999}
for s in ['42', '123', '456']:
    p = Path('outputs/runs/ftt_seed_' + s + '/metrics.json')
    if p.exists():
        m = json.load(open(p))
        ic = m.get('rank_ic_mean', -999)
        if ic > best['ic']:
            best = {'seed': s, 'ic': ic}
print(best['seed'])
")

echo "Best FT-Transformer seed for backtest: $BEST_SEED"

python -m src.backtest.engine \
    --config src/config/default.yaml \
    --model-name ft_transformer \
    --preprocessor outputs/runs/ftt_seed_${BEST_SEED}/preprocessor.joblib \
    --start-date 2025-01-02 \
    --end-date 2026-05-20 \
    --output-dir outputs/backtest/final \
    2>&1 | tee "$LOG_DIR/${TIMESTAMP}_backtest.log"

# ═══════════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════════

echo ""
echo "=========================================="
echo "PIPELINE COMPLETE"
echo "=========================================="
echo "start:    $TIMESTAMP"
echo "end:      $(date)"
echo "best_label: $BEST_LABEL"
echo "best_fit_seed: $BEST_SEED"

# Quick metrics summary
python -c "
import json
from pathlib import Path

runs_dir = Path('outputs/runs')
for run_dir in sorted(runs_dir.glob('ftt_seed_*')):
    mf = run_dir / 'metrics.json'
    if mf.exists():
        m = json.load(open(mf))
        print(f'{run_dir.name}: rank_ic={m.get(\"rank_ic_mean\", float(\"nan\")):.6f}, '
              f'ic={m.get(\"ic_mean\", float(\"nan\")):.6f}, '
              f'icir={m.get(\"icir\", float(\"nan\")):.6f}')

# GBDT summary
gbdt_mf = runs_dir / 'gbdt_full' / 'metrics.json'
if gbdt_mf.exists():
    m = json.load(open(gbdt_mf))
    print(f'gbdt_full:  rank_ic={m.get(\"rank_ic_mean\", float(\"nan\")):.6f}, '
          f'ic={m.get(\"ic_mean\", float(\"nan\")):.6f}, '
          f'icir={m.get(\"icir\", float(\"nan\")):.6f}')

print()
print('Ensemble weights:', end=' ')
ew = Path('outputs/ensemble/ensemble_weights.json')
if ew.exists():
    w = json.load(open(ew))
    print(w.get('weights', {}))
"

echo ""
echo "=========================================="
