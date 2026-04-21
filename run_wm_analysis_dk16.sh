#!/usr/bin/env bash
# Tier 1.1 — WM semantic-accuracy analysis on DoorKey-16x16 seed sweep.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="$PROJECT_ROOT/../miniforge3/envs/stvqvae/bin/python"
MODELS_DIR="$PROJECT_ROOT/discrete_mbrl/model_free/models/MiniGrid-DoorKey-16x16-v0"
OUT_DIR="$PROJECT_ROOT/logs/wm_analysis_dk16"
DEVICE="${DEVICE:-cuda}"

mkdir -p "$OUT_DIR"
cd "$PROJECT_ROOT/discrete_mbrl"
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
export TORCHDYNAMO_DISABLE=1

for SEED in 1 2 3; do
    LABEL="s${SEED}"
    MODEL="$MODELS_DIR/sweep_doorkey16_v6_s${SEED}_best_model.pt"
    OUT_JSON="$OUT_DIR/dk16_${LABEL}.json"
    LOG="$OUT_DIR/dk16_${LABEL}.log"
    echo ""
    echo "=== DK16 WM analysis seed=${SEED} ==="
    if [[ ! -f "$MODEL" ]]; then
        echo "  ⚠ Missing checkpoint: $MODEL"
        continue
    fi
    "$PYTHON_BIN" -u analyze_wm_semantic_accuracy.py \
        --model_path "$MODEL" \
        --n_frames 5000 \
        --device "$DEVICE" \
        --output_json "$OUT_JSON" \
        > "$LOG" 2>&1 || { echo "  ✗ failed; tail:"; tail -20 "$LOG"; continue; }
    echo "  → $OUT_JSON"
done

echo ""
echo "DK16 WM analysis done. Results in $OUT_DIR"
