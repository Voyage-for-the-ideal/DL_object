#!/bin/bash
#SBATCH --job-name=dl-daily-predict
#SBATCH --partition=Students
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=2:00:00
#SBATCH --output=outputs/logs/daily_predict_%j.log

set -euo pipefail

# ═══════════════════════════════════════════════════════════════
# 每日选股预测脚本
# 生成 ensemble 信号 + 交易订单，输出应该买哪几只股票
# ═══════════════════════════════════════════════════════════════

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

# ── 可配置参数 ──
SIGNAL_DATE="${1:-}"                          # 信号日期，不指定则自动使用最新数据日期
TOP_K="${2:-20}"                              # 最终输出 top K 只股票
OUTPUT_ROOT="$PROJECT_DIR/outputs"

# ── 已训练好的模型 ──
# 来自 20260531 pipeline，label_5d_cs_rank，ensemble 权重已计算好
PREPROCESSOR="outputs/runs/ftt_seed_42/preprocessor.joblib"

declare -A MODELS=(
    ["ftt_seed_42"]="outputs/runs/ftt_seed_42/model.pt:ft_transformer"
    ["ftt_seed_123"]="outputs/runs/ftt_seed_123/model.pt:ft_transformer"
    ["ftt_seed_456"]="outputs/runs/ftt_seed_456/model.pt:ft_transformer"
    ["gbdt_full"]="outputs/runs/gbdt_full/model.joblib:gbdt"
)

declare -A ENSEMBLE_WEIGHTS=(
    ["gbdt_full"]="0.1673"
    ["ftt_seed_42"]="0.2451"
    ["ftt_seed_123"]="0.2849"
    ["ftt_seed_456"]="0.3027"
)

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="$OUTPUT_ROOT/logs"
SIGNAL_DIR="$OUTPUT_ROOT/signals"
ORDER_DIR="$OUTPUT_ROOT/orders"
mkdir -p "$LOG_DIR" "$SIGNAL_DIR" "$ORDER_DIR"

echo "=========================================="
echo "DL Daily Stock Prediction"
echo "=========================================="
echo "project:     $PROJECT_DIR"
echo "device:      $DEVICE"
echo "signal_date: ${SIGNAL_DATE:-auto (latest)}"
echo "top_k:       $TOP_K"
echo "start:       $(date)"
echo "job_id:      ${SLURM_JOB_ID:-local}"
echo "=========================================="

# ═══════════════════════════════════════════════════════════════
# Step 1: 确定信号日期
# ═══════════════════════════════════════════════════════════════

echo ""
echo "[Step 1/4] Determining signal date..."

if [ -z "$SIGNAL_DATE" ]; then
    SIGNAL_DATE=$(python -c "
from src.data.loader import CsvDataLoader
loader = CsvDataLoader('A股数据')
latest = loader.latest_partition_date('daily')
print(latest)
")
fi

echo "Signal date: $SIGNAL_DATE"

# ═══════════════════════════════════════════════════════════════
# Step 2: 对每个模型生成 signal
# ═══════════════════════════════════════════════════════════════

echo ""
echo "[Step 2/4] Generating signals for each model..."

SIGNAL_FILES=""

for MODEL_NAME in "${!MODELS[@]}"; do
    IFS=':' read -r MODEL_PATH MODEL_TYPE <<< "${MODELS[$MODEL_NAME]}"
    SIGNAL_FILE="$SIGNAL_DIR/${SIGNAL_DATE}_signal_${MODEL_NAME}.csv"

    if [ -f "$SIGNAL_FILE" ]; then
        echo "  [$MODEL_NAME] signal exists, skipping."
    else
        echo "  [$MODEL_NAME] generating signal..."
        python -m src.predict.daily_signal \
            --config src/config/default.yaml \
            --preprocessor "$PREPROCESSOR" \
            --model "$MODEL_PATH" \
            --model-name "$MODEL_TYPE" \
            --signal-date "$SIGNAL_DATE"

        # daily_signal 输出固定命名为 {date}_signal.csv，需要重命名
        DEFAULT_SIGNAL="$SIGNAL_DIR/${SIGNAL_DATE}_signal.csv"
        if [ -f "$DEFAULT_SIGNAL" ]; then
            mv "$DEFAULT_SIGNAL" "$SIGNAL_FILE"
            echo "  [$MODEL_NAME] done → $SIGNAL_FILE"
        else
            echo "  [$MODEL_NAME] ERROR: signal file not generated"
            exit 1
        fi
    fi

    SIGNAL_FILES="$SIGNAL_FILES $SIGNAL_FILE"
done

echo ""
echo "All signals generated: $(date)"

# ═══════════════════════════════════════════════════════════════
# Step 3: Ensemble 融合
# ═══════════════════════════════════════════════════════════════

echo ""
echo "[Step 3/4] Fusing ensemble signals..."

ENSEMBLE_SIGNAL="$SIGNAL_DIR/${SIGNAL_DATE}_signal_ensemble.csv"

python -c "
import pandas as pd
import numpy as np
import json

weights = {
    'gbdt_full': ${ENSEMBLE_WEIGHTS[gbdt_full]},
    'ftt_seed_42': ${ENSEMBLE_WEIGHTS[ftt_seed_42]},
    'ftt_seed_123': ${ENSEMBLE_WEIGHTS[ftt_seed_123]},
    'ftt_seed_456': ${ENSEMBLE_WEIGHTS[ftt_seed_456]},
}

print(f'Ensemble weights: {json.dumps(weights, indent=2)}')

signals = {}
for name in weights:
    path = f'${SIGNAL_DIR}/${SIGNAL_DATE}_signal_{name}.csv'
    df = pd.read_csv(path)
    signals[name] = df
    print(f'  {name}: {len(df)} stocks, score range [{df[\"score\"].min():.4f}, {df[\"score\"].max():.4f}]')

# 取第一个 signal 作为基准 (用 gbdt_full 的列结构)
base = signals['gbdt_full'][['signal_date', 'next_trade_date', 'ts_code']].copy()

# 对每个模型做横截面 rank percentile 后加权求和
fused_score = pd.Series(0.0, index=base.index)
total_w = sum(weights.values())

for name, w in weights.items():
    rank_pct = signals[name]['score'].rank(pct=True)
    fused_score += rank_pct * w / total_w

base['score'] = fused_score.values
base['rank'] = base['score'].rank(ascending=False, method='first').astype(int)
base['model_name'] = 'ensemble'

# 按 rank 排序输出
base = base.sort_values('rank')

base.to_csv('${ENSEMBLE_SIGNAL}', index=False)
print(f'\nEnsemble signal saved: ${ENSEMBLE_SIGNAL}')
print(f'Top 10 stocks:')
print(base.head(10)[['ts_code', 'score', 'rank']].to_string(index=False))
"

echo ""
echo "Ensemble fusion done: $(date)"

# ═══════════════════════════════════════════════════════════════
# Step 4: 生成订单 (买什么股票)
# ═══════════════════════════════════════════════════════════════

echo ""
echo "[Step 4/4] Generating orders..."

# 从 daily 数据中提取 quotes（需要用信号日期对应的行情数据）
QUOTES_FILE="$SIGNAL_DIR/${SIGNAL_DATE}_quotes.csv"

python -c "
import pandas as pd
from pathlib import Path
from src.data.loader import CsvDataLoader
from src.data.schema import normalize_trade_date

loader = CsvDataLoader('A股数据')
df = loader.load_daily(date='${SIGNAL_DATE}')
if df is None or len(df) == 0:
    # fallback: 尝试找最近的一个交易日数据
    dates = sorted(Path('A股数据/daily').glob('*.csv'))
    print(f'No daily data for ${SIGNAL_DATE}, using latest available')
    latest = dates[-1]
    df = pd.read_csv(latest)
else:
    print(f'Loaded daily data for ${SIGNAL_DATE}: {len(df)} stocks')

# 确保有 ts_code 列
if 'ts_code' not in df.columns:
    raise KeyError(f'ts_code not found in daily data. Columns: {list(df.columns)[:10]}')

# 提取 quotes 需要的列
quote_cols = ['ts_code']
for col in ['open', 'close', 'high', 'low', 'amount', 'vol']:
    if col in df.columns:
        quote_cols.append(col)

quotes = df[quote_cols].copy()
quotes.to_csv('${QUOTES_FILE}', index=False)
print(f'Quotes saved: ${QUOTES_FILE} ({len(quotes)} stocks)')
"

python -m src.predict.daily_order \
    --signal-file "$ENSEMBLE_SIGNAL" \
    --quotes-file "$QUOTES_FILE" \
    --output-root "$OUTPUT_ROOT"

ORDER_FILE="$ORDER_DIR/${SIGNAL_DATE}_orders.csv"

echo ""
echo "Order generation done: $(date)"

# ═══════════════════════════════════════════════════════════════
# 最终输出：推荐买入的股票列表
# ═══════════════════════════════════════════════════════════════

echo ""
echo "=========================================="
echo "RECOMMENDED STOCKS TO BUY (TOP $TOP_K)"
echo "=========================================="

python -c "
import pandas as pd

orders = pd.read_csv('${ORDER_FILE}')
# 筛选买入操作
buy_orders = orders[orders['action'] == 'BUY'].head(${TOP_K})

print(f'Total buy orders: {len(orders[orders[\"action\"] == \"BUY\"])}')
print(f'Showing top ${TOP_K}:')
print()

for i, (_, row) in enumerate(buy_orders.iterrows(), 1):
    print(f'{i:3d}. {row[\"ts_code\"]:10s}  '
          f'target_weight={row[\"target_weight\"]:6.2%}  '
          f'shares={int(row[\"target_shares\"]):6d}  '
          f'est_price={row[\"estimated_price\"]:8.2f}  '
          f'est_amount={row[\"estimated_amount\"]:12.2f}')

print()
print(f'Order file: ${ORDER_FILE}')
print(f'Signal file: ${ENSEMBLE_SIGNAL}')
"

echo ""
echo "=========================================="
echo "PREDICTION COMPLETE"
echo "=========================================="
echo "start:       $TIMESTAMP"
echo "end:         $(date)"
echo "signal_date: $SIGNAL_DATE"
echo "order_file:  $ORDER_FILE"
echo "signal_file: $ENSEMBLE_SIGNAL"
echo "=========================================="
