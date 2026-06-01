#!/bin/bash
#SBATCH --job-name=dl-hp-tuning
#SBATCH --partition=Students
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=30:00:00
#SBATCH --output=outputs/logs/hp_tuning_%j.log

set -euo pipefail

PROJECT_DIR=~/Desktop/DL_object
cd "$PROJECT_DIR"

export PATH="/home/scc/pb23151808/software/miniconda3/bin:$PATH"
source /home/scc/pb23151808/software/miniconda3/etc/profile.d/conda.sh
conda activate DL
set -u

DEVICE="cpu"
if python -c "import torch; print(torch.cuda.is_available())" 2>/dev/null | grep -q True; then
    DEVICE="cuda"
fi

PANEL_DIR="$PROJECT_DIR/outputs/cache/panels_v2"
LOG_DIR="$PROJECT_DIR/outputs/logs"
TRAIN_PANEL="$PANEL_DIR/train_panel.csv"
VALID_PANEL="$PANEL_DIR/valid_panel.csv"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "=========================================="
echo "HP Tuning — 4 separate hyperparameter tests"
echo "project:   $PROJECT_DIR"
echo "device:    $DEVICE"
echo "start:     $(date)"
echo "=========================================="

# 读取 BEST_LABEL（沿用 label experiment 结果）
LABEL_RESULT="outputs/label_experiment/label_comparison.csv"
if [ -f "$LABEL_RESULT" ]; then
    BEST_LABEL=$(python -c "
import pandas as pd
df = pd.read_csv('$LABEL_RESULT')
best = df.sort_values('Rank_IC', ascending=False).iloc[0]
print(best['label'])
")
    echo "Using BEST_LABEL: $BEST_LABEL"
else
    echo "ERROR: label experiment result not found, run full pipeline first"
    exit 1
fi

SEEDS=(42 123 456)
NUM_SEEDS=${#SEEDS[@]}

# ═══════════════════════════════════════════════════════════════
# 定义 4 组超参数测试
# 每组只改动一个参数，其余保持 default.yaml 的默认值
# ═══════════════════════════════════════════════════════════════

declare -A HP_NAME
declare -A HP_OVERRIDE

# Test 1: n_blocks 2 → 4
HP_NAME[1]="nblocks4"
HP_OVERRIDE[1]="model.ft_transformer.n_blocks=4"

# Test 2: dropout 0.2 → 0.35
HP_NAME[2]="dropout35"
HP_OVERRIDE[2]="model.ft_transformer.dropout=0.35"

# Test 3: drop_path 0.1 → 0.2
HP_NAME[3]="droppath2"
HP_OVERRIDE[3]="model.ft_transformer.drop_path=0.2"

# Test 4: weight_decay 0.001 → 0.005
HP_NAME[4]="wd005"
HP_OVERRIDE[4]="training.weight_decay=0.005"

NUM_TESTS=4

# ═══════════════════════════════════════════════════════════════
# 执行每个超参数测试
# ═══════════════════════════════════════════════════════════════

for ((i=1; i<=NUM_TESTS; i++)); do
    HP_LABEL="${HP_NAME[$i]}"
    HP_EXTRA="${HP_OVERRIDE[$i]}"

    echo ""
    echo "=========================================="
    echo "[HP Test $i/$NUM_TESTS] $HP_LABEL"
    echo "  override: $HP_EXTRA"
    echo "=========================================="

    # ── 训练 3 个 seed ────────────────────────────────────────
    for SEED in "${SEEDS[@]}"; do
        RUN_ID="ftt_hp_${HP_LABEL}_seed${SEED}"
        METRICS_FILE="outputs/runs/${RUN_ID}/metrics.json"

        if [ -f "$METRICS_FILE" ]; then
            echo "  [$HP_LABEL] seed=$SEED already done, skipping."
            continue
        fi

        echo "  [$HP_LABEL] Training FT-Transformer with seed=$SEED ..."
        python train.py \
            --config src/config/default.yaml \
            --train-panel "$TRAIN_PANEL" \
            --valid-panel "$VALID_PANEL" \
            --run-id "$RUN_ID" \
            --model-name ft_transformer \
            --override "label.main=$BEST_LABEL" \
            --override "training.seed=$SEED" \
            --override "training.epochs=100" \
            --override "training.early_stopping_patience=10" \
            --override "model.device=$DEVICE" \
            --override "training.device=$DEVICE" \
            --override "$HP_EXTRA" \
            2>&1 | tee "$LOG_DIR/${TIMESTAMP}_hp_${HP_LABEL}_seed${SEED}.log"

        echo "  [$HP_LABEL] seed=$SEED done: $(date)"
    done

    # ── 计算 ensemble 权重 ────────────────────────────────────
    echo ""
    echo "  [$HP_LABEL] Computing ensemble weights..."

    RUN_IDS=()
    for SEED in "${SEEDS[@]}"; do
        RUN_IDS+=("ftt_hp_${HP_LABEL}_seed${SEED}")
    done

    python -c "
from src.models.ensemble import compute_ensemble_weights_from_dir
from pathlib import Path, os

ensemble = compute_ensemble_weights_from_dir(
    predictions_dir='outputs/runs',
    model_names=['ftt_hp_${HP_LABEL}_seed42', 'ftt_hp_${HP_LABEL}_seed123', 'ftt_hp_${HP_LABEL}_seed456'],
    label_column='$BEST_LABEL',
)
Path('outputs/hp_tuning/${HP_LABEL}').mkdir(parents=True, exist_ok=True)
ensemble.save('outputs/hp_tuning/${HP_LABEL}/ensemble_weights.json')
print(f'Weights: {ensemble.weights}')
print(f'ICIR: {ensemble._icir}')
"

    # ── 选最佳 seed 跑回测 ────────────────────────────────────
    BEST_SEED=$(python -c "
import json
from pathlib import Path
best = {'seed': '42', 'ic': -999}
for s in ['42', '123', '456']:
    p = Path('outputs/runs/ftt_hp_${HP_LABEL}_seed' + s + '/metrics.json')
    if p.exists():
        m = json.load(open(p))
        ic = m.get('rank_ic_mean', -999)
        if ic > best['ic']:
            best = {'seed': s, 'ic': ic}
print(best['seed'])
")

    echo "  [$HP_LABEL] Best seed for backtest: $BEST_SEED"

    BACKTEST_DIR="outputs/backtest/hp_${HP_LABEL}"
    if [ -f "$BACKTEST_DIR/backtest_summary.json" ]; then
        echo "  [$HP_LABEL] Backtest already complete, skipping."
    else
        python backtest.py \
            --config src/config/default.yaml \
            --model-name ft_transformer \
            --model "outputs/runs/ftt_hp_${HP_LABEL}_seed${BEST_SEED}/model.pt" \
            --preprocessor "outputs/runs/ftt_hp_${HP_LABEL}_seed${BEST_SEED}/preprocessor.joblib" \
            --start-date 2025-01-02 \
            --end-date 2026-05-20 \
            --run-id "hp_${HP_LABEL}" \
            --benchmark-index 000300.SH \
            2>&1 | tee "$LOG_DIR/${TIMESTAMP}_hp_${HP_LABEL}_backtest.log"

        echo "  [$HP_LABEL] Backtest done: $(date)"
    fi

done

# ═══════════════════════════════════════════════════════════════
# 汇总所有测试结果
# ═══════════════════════════════════════════════════════════════

echo ""
echo "=========================================="
echo "HP TUNING SUMMARY"
echo "=========================================="
echo ""

python -c "
import json
from pathlib import Path

tests = ['nblocks4', 'dropout35', 'droppath2', 'wd005']
baseline_runs = ['ftt_seed_42', 'ftt_seed_123', 'ftt_seed_456']

print(f'{"Test":<16} {"Seed":<8} {"Rank_IC":>10} {"IC":>10} {"ICIR":>10} {"Loss":>10}')
print('-' * 70)

# Baseline
for run_id in baseline_runs:
    mf = Path('outputs/runs') / run_id / 'metrics.json'
    if mf.exists():
        m = json.load(open(mf))
        print(f'{\"BASELINE\":<16} {run_id[-3:]:<8} {m.get(\"rank_ic_mean\", 0):>10.6f} {m.get(\"ic_mean\", 0):>10.6f} {m.get(\"icir\", 0):>10.6f} {m.get(\"valid_loss\", 0):>10.6f}')

for hp in tests:
    for s in ['42', '123', '456']:
        run_id = f'ftt_hp_{hp}_seed{s}'
        mf = Path('outputs/runs') / run_id / 'metrics.json'
        if mf.exists():
            m = json.load(open(mf))
            print(f'{hp:<16} {s:<8} {m.get(\"rank_ic_mean\", 0):>10.6f} {m.get(\"ic_mean\", 0):>10.6f} {m.get(\"icir\", 0):>10.6f} {m.get(\"valid_loss\", 0):>10.6f}')

print()
print('--- Backtest Comparison ---')
print(f'{\"Test\":<16} {\"Cum.Ret\":>10} {\"Ann.Ret\":>10} {\"Sharpe\":>10} {\"MaxDD\":>10}')
print('-' * 62)

for hp in ['baseline'] + tests:
    if hp == 'baseline':
        bt_dir = Path('outputs/backtest/final')
        label = 'BASELINE'
    else:
        bt_dir = Path('outputs/backtest/hp_' + hp)
        label = hp
    sf = bt_dir / 'strategy' / 'backtest_metrics.json'
    if sf.exists():
        m = json.load(open(sf))
        if 'strategy' in str(type(m)):
            # nested dict with strategy label
            keys = [k for k in m.keys() if 'score_weighted' in k or 'topk' not in k.lower()]
            strat_key = keys[0] if keys else list(m.keys())[0]
            sm = m[strat_key]
        else:
            sm = m
        print(f'{label:<16} {sm.get(\"cumulative_return\", 0):>10.4f} {sm.get(\"annualized_return\", 0):>10.4f} {sm.get(\"sharpe\", 0):>10.4f} {sm.get(\"max_drawdown\", 0):>10.4f}')

print()
print('All results saved under outputs/hp_tuning/ and outputs/backtest/hp_*/')
"

echo ""
echo "HP tuning complete: $(date)"
