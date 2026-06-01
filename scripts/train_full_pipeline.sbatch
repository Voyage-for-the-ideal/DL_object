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

# ── 预先读取 BEST_LABEL（如果 label experiment 已跑过） ──
LABEL_RESULT="outputs/label_experiment/label_comparison.csv"
if [ -f "$LABEL_RESULT" ]; then
    BEST_LABEL=$(python -c "
import pandas as pd
df = pd.read_csv('$LABEL_RESULT')
best = df.sort_values('Rank_IC', ascending=False).iloc[0]
print(best['label'])
")
    echo "Using existing BEST_LABEL: $BEST_LABEL"
else
    BEST_LABEL=""
fi

# ═══════════════════════════════════════════════════════════════
# Step 1: Build feature panels (1-2 hours)
# ═══════════════════════════════════════════════════════════════

echo ""
echo "[Step 1/7] Building feature panels..."
echo ""

TRAIN_PANEL="$PANEL_DIR/train_panel.csv"
VALID_PANEL="$PANEL_DIR/valid_panel.csv"

if [ -f "$TRAIN_PANEL" ] && [ -f "$VALID_PANEL" ]; then
    echo "Panels already exist, skipping generation."
else
    python prepare_panels.py \
        --config src/config/default.yaml \
        --output-dir "$PANEL_DIR" \
        2>&1 | tee "$LOG_DIR/${TIMESTAMP}_prepare_panels.log"

    echo ""
    echo "Panels built: $(date)"
fi

# 快速列数检查
python -c "
import pandas as pd
df = pd.read_csv('$TRAIN_PANEL', nrows=5)
print(f'Panel columns: {len(df.columns)}')
assert len(df.columns) >= 200, f'Expected >=200 cols, got {len(df.columns)}'
print('Panel check OK')
"

# ═══════════════════════════════════════════════════════════════
# Step 2: Label experiment (8-10 hours)
# ═══════════════════════════════════════════════════════════════

echo ""
echo "[Step 2/7] Running label comparison experiment..."

if [ -f "$LABEL_RESULT" ]; then
    echo "Label experiment already done, skipping."
else
    echo ""
    python train.py \
        --config src/config/default.yaml \
        --train-panel "$TRAIN_PANEL" \
        --valid-panel "$VALID_PANEL" \
        --label-experiment \
        --label-types label_5d label_5d_vol_norm label_5d_cs_rank \
        --label-models gbdt ft_transformer \
        2>&1 | tee "$LOG_DIR/${TIMESTAMP}_label_experiment_log"

    echo ""
    echo "Label experiment done: $(date)"
fi

# 确保 BEST_LABEL 已设定
BEST_LABEL=$(python -c "
import pandas as pd
df = pd.read_csv('$LABEL_RESULT')
best = df.sort_values('Rank_IC', ascending=False).iloc[0]
print(best['label'])
")
echo "Best label: $BEST_LABEL"

# ═══════════════════════════════════════════════════════════════
# Step 3: GBDT full training + feature selection (2 hours)
# ═══════════════════════════════════════════════════════════════

echo ""
echo "[Step 3/7] GBDT training + feature selection..."

FS_SUMMARY="outputs/feature_selection/selection_summary.json"

if [ -f "$FS_SUMMARY" ]; then
    echo "Feature selection already done, skipping."
else
    echo ""
    python -c "
from src.config.loader import load_config
from src.models.gbdt_selector import gbdt_select_and_train_ftt
import pandas as pd

config = load_config()
config['label']['main'] = '$BEST_LABEL'

train = pd.read_csv('$TRAIN_PANEL')
valid = pd.read_csv('$VALID_PANEL')

top_features, metrics = gbdt_select_and_train_ftt(
    config, train, valid, top_k=128,
    output_dir='outputs/feature_selection'
)
print(f'GBDT Rank IC: {metrics.get(\"rank_ic_mean\", float(\"nan\"))}')
print(f'FTT  Rank IC: {metrics.get(\"rank_ic_mean\", float(\"nan\"))}')
" 2>&1 | tee "$LOG_DIR/${TIMESTAMP}_feature_selection.log"

    echo ""
    echo "Feature selection done: $(date)"
fi

# ═══════════════════════════════════════════════════════════════
# Step 4: FT-Transformer single-seed training (3-4 hours)
# ═══════════════════════════════════════════════════════════════

echo ""
echo "[Step 4/7] FT-Transformer training..."
echo ""

FTT_SEED=42
FTT_METRICS="outputs/runs/ftt_seed_${FTT_SEED}/metrics.json"

if [ -f "$FTT_METRICS" ]; then
    echo "FT-Transformer already done, skipping."
else
    echo "--- Training FT-Transformer with seed=$FTT_SEED ---"
    python train.py \
        --config src/config/default.yaml \
        --train-panel "$TRAIN_PANEL" \
        --valid-panel "$VALID_PANEL" \
        --run-id "ftt_seed_${FTT_SEED}" \
        --model-name ft_transformer \
        --override "label.main=$BEST_LABEL" \
        --override "training.seed=$FTT_SEED" \
        --override "training.epochs=100" \
        --override "training.early_stopping_patience=10" \
        --override "model.device=$DEVICE" \
        --override "training.device=$DEVICE" \
        2>&1 | tee "$LOG_DIR/${TIMESTAMP}_ftt_seed_${FTT_SEED}.log"
    echo "FT-Transformer done: $(date)"
fi

# ═══════════════════════════════════════════════════════════════
# Step 5: Train single best GBDT (for ensemble) (1 hour)
# ═══════════════════════════════════════════════════════════════

echo ""
echo "[Step 5/7] Training final GBDT for ensemble..."

GBDT_METRICS="outputs/runs/gbdt_full/metrics.json"

if [ -f "$GBDT_METRICS" ]; then
    echo "GBDT ensemble training already done, skipping."
else
    echo ""
    python train.py \
        --config src/config/default.yaml \
        --train-panel "$TRAIN_PANEL" \
        --valid-panel "$VALID_PANEL" \
        --run-id "gbdt_full" \
        --model-name gbdt \
        --override "label.main=$BEST_LABEL" \
        --override "model.gbdt.n_estimators=1000" \
        --override "model.device=$DEVICE" \
        2>&1 | tee "$LOG_DIR/${TIMESTAMP}_gbdt_full.log"
fi

# ═══════════════════════════════════════════════════════════════
# Step 6: Compute ensemble weights (5 min)
# ═══════════════════════════════════════════════════════════════

echo ""
echo "[Step 6/7] Computing ensemble weights..."

ENSEMBLE_WEIGHTS="outputs/ensemble/ensemble_weights.json"

if [ -f "$ENSEMBLE_WEIGHTS" ]; then
    echo "Ensemble weights already computed, skipping."
else
    echo ""
    python -c "
from src.models.ensemble import compute_ensemble_weights_from_dir
from pathlib import Path

ensemble = compute_ensemble_weights_from_dir(
    predictions_dir='outputs/runs',
    model_names=['gbdt_full', 'ftt_seed_42'],
    label_column='$BEST_LABEL',
)
Path('outputs/ensemble').mkdir(parents=True, exist_ok=True)
ensemble.save('outputs/ensemble/ensemble_weights.json')
print(f'Weights: {ensemble.weights}')
print(f'ICIR: {ensemble._icir}')
"
fi

# ═══════════════════════════════════════════════════════════════
# Step 7: Full backtest (2 hours)
# ═══════════════════════════════════════════════════════════════

echo ""
echo "[Step 7/7] Running full backtest..."

BACKTEST_SUMMARY="outputs/backtest/final/backtest_summary.json"

if [ -f "$BACKTEST_SUMMARY" ]; then
    echo "Backtest already done, skipping."
else
    echo ""
    python backtest.py \
        --config src/config/default.yaml \
        --model-name ft_transformer \
        --model outputs/runs/ftt_seed_${FTT_SEED}/model.pt \
        --preprocessor outputs/runs/ftt_seed_${FTT_SEED}/preprocessor.joblib \
        --start-date 2025-01-02 \
        --end-date 2026-05-20 \
        --run-id final \
        --benchmark-index 000300.SH \
        2>&1 | tee "$LOG_DIR/${TIMESTAMP}_backtest.log"
fi

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

# Quick metrics summary
python -c "
import json
from pathlib import Path

runs_dir = Path('outputs/runs')
ftt_mf = runs_dir / 'ftt_seed_42' / 'metrics.json'
if ftt_mf.exists():
    m = json.load(open(ftt_mf))
    print(f'ftt_seed_42: rank_ic={m.get(\"rank_ic_mean\", float(\"nan\")):.6f}, '
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
