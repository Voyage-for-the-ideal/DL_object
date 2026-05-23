#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON="${PYTHON:-python}"
CONFIG="${CONFIG:-src/config/default.yaml}"
PANEL_DIR="${PANEL_DIR:-outputs/cache/full_20160104_20260520}"
LOG_DIR="${LOG_DIR:-outputs/logs}"
RUN_PREFIX="${RUN_PREFIX:-full_$(date +%Y%m%d_%H%M%S)}"
REBUILD_PANELS="${REBUILD_PANELS:-0}"
MODELS="${MODELS:-ridge elasticnet gbdt mlp transformer_encoder}"

detect_device() {
  "$PYTHON" - <<'PY'
try:
    import torch
    print("cuda" if torch.cuda.is_available() else "cpu")
except Exception:
    print("cpu")
PY
}

DEVICE="${DEVICE:-$(detect_device)}"

mkdir -p "$PANEL_DIR" "$LOG_DIR"

echo "root: $ROOT_DIR"
echo "config: $CONFIG"
echo "panel_dir: $PANEL_DIR"
echo "run_prefix: $RUN_PREFIX"
echo "device: $DEVICE"
echo "models: $MODELS"

if [[ "$REBUILD_PANELS" == "1" || ! -f "$PANEL_DIR/train_panel.csv" || ! -f "$PANEL_DIR/valid_panel.csv" ]]; then
  echo "building full train/valid panels..."
  "$PYTHON" prepare_panels.py \
    --config "$CONFIG" \
    --output-dir "$PANEL_DIR" \
    2>&1 | tee "$LOG_DIR/${RUN_PREFIX}_prepare_panels.log"
else
  echo "reusing existing panels in $PANEL_DIR"
fi

read -r -a MODEL_ARRAY <<< "$MODELS"
for model in "${MODEL_ARRAY[@]}"; do
  run_id="${RUN_PREFIX}_${model}"
  log_file="$LOG_DIR/${run_id}.log"
  echo "training $model -> outputs/runs/$run_id"
  "$PYTHON" train.py \
    --config "$CONFIG" \
    --train-panel "$PANEL_DIR/train_panel.csv" \
    --valid-panel "$PANEL_DIR/valid_panel.csv" \
    --model-name "$model" \
    --run-id "$run_id" \
    --override "model.device=${DEVICE}" \
    --override "training.device=${DEVICE}" \
    2>&1 | tee "$log_file"
done

echo "metrics summary:"
"$PYTHON" - "$RUN_PREFIX" <<'PY'
import json
import sys
from pathlib import Path

prefix = sys.argv[1]
runs = sorted(Path("outputs/runs").glob(f"{prefix}_*/metrics.json"))
for metrics_path in runs:
    run_name = metrics_path.parent.name
    with metrics_path.open("r", encoding="utf-8") as file:
        metrics = json.load(file)
    print(
        f"{run_name}: "
        f"mse={metrics.get('mse', float('nan')):.9f}, "
        f"mae={metrics.get('mae', float('nan')):.9f}, "
        f"ic={metrics.get('ic_mean', float('nan')):.6f}, "
        f"icir={metrics.get('icir', float('nan')):.6f}, "
        f"rank_ic={metrics.get('rank_ic_mean', float('nan')):.6f}, "
        f"rank_icir={metrics.get('rank_icir', float('nan')):.6f}, "
        f"dir_acc={metrics.get('direction_accuracy', float('nan')):.4f}"
    )
PY
