#!/bin/bash
#SBATCH --job-name=dl-train-all
#SBATCH --partition=Students
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G

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

PANEL_DIR="$PROJECT_DIR/outputs/cache/panels"
LOG_DIR="$PROJECT_DIR/outputs/logs"
RUN_PREFIX="slurm_$(date +%Y%m%d_%H%M%S)"
MODELS=("ridge" "elasticnet" "gbdt" "mlp" "transformer_encoder")
REBUILD_PANELS="${REBUILD_PANELS:-0}"

mkdir -p "$PANEL_DIR" "$LOG_DIR"

echo "=========================================="
echo "project:   $PROJECT_DIR"
echo "device:    $DEVICE"
echo "models:    ${MODELS[*]}"
echo "prefix:    $RUN_PREFIX"
echo "start:     $(date)"
echo "=========================================="

# Step 1: 构建特征面板（如果不存在或强制重建）
echo ""
echo "[1/2] preparing feature panels..."
echo ""
if [[ "$REBUILD_PANELS" == "1" || ! -f "$PANEL_DIR/train_panel.csv" || ! -f "$PANEL_DIR/valid_panel.csv" ]]; then
    python prepare_panels.py \
        --config src/config/default.yaml \
        --output-dir "$PANEL_DIR" \
        2>&1 | tee "$LOG_DIR/${RUN_PREFIX}_prepare_panels.log"
    echo "panels built: $PANEL_DIR"
else
    echo "reusing existing panels in $PANEL_DIR"
fi

# Step 2: 依次训练所有模型
echo ""
echo "[2/2] training models..."
echo ""

TOTAL=${#MODELS[@]}
for i in "${!MODELS[@]}"; do
    model="${MODELS[$i]}"
    run_id="${RUN_PREFIX}_${model}"
    log_file="$LOG_DIR/${run_id}.log"

    echo "----------------------------------------"
    echo "[$((i+1))/$TOTAL] training $model -> outputs/runs/$run_id"
    echo "start: $(date)"
    echo "----------------------------------------"

    python train.py \
        --config src/config/default.yaml \
        --train-panel "$PANEL_DIR/train_panel.csv" \
        --valid-panel "$PANEL_DIR/valid_panel.csv" \
        --model-name "$model" \
        --run-id "$run_id" \
        --override "model.device=$DEVICE" \
        --override "training.device=$DEVICE" \
        2>&1 | tee "$log_file"

    echo ""
    echo "[$((i+1))/$TOTAL] $model done: $(date)"
    echo ""
done

# 汇总 metrics
echo ""
echo "=========================================="
echo "metrics summary"
echo "=========================================="
python - "$RUN_PREFIX" <<'PY'
import json, sys
from pathlib import Path

prefix = sys.argv[1]
runs = sorted(Path("outputs/runs").glob(f"{prefix}_*/metrics.json"))
for metrics_path in runs:
    run_name = metrics_path.parent.name
    with metrics_path.open("r", encoding="utf-8") as f:
        m = json.load(f)
    print(
        f"  {run_name}: "
        f"mse={m.get('mse', float('nan')):.9f}, "
        f"mae={m.get('mae', float('nan')):.9f}, "
        f"ic={m.get('ic_mean', float('nan')):.6f}, "
        f"icir={m.get('icir', float('nan')):.6f}, "
        f"dir_acc={m.get('direction_accuracy', float('nan')):.4f}"
    )
PY

echo ""
echo "=========================================="
echo "all done: $(date)"
echo "=========================================="
