#!/usr/bin/env bash
# Post-analysis for full-rigor sweep:
#  1. Continuous probe on VAE seeds 1+2 (discrete probe doesn't support AEModel)
#  2. WM semantic accuracy on DK16 rigor runs (A1, A2)
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="$PROJECT_ROOT/../miniforge3/envs/stvqvae/bin/python"
RIGOR_DIR="$PROJECT_ROOT/logs/full_rigor"
DEVICE="${DEVICE:-cuda}"

cd "$PROJECT_ROOT/discrete_mbrl"
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
export TORCHDYNAMO_DISABLE=1

# ── VAE continuous probes ────────────────────────────────────────────────────
DK8_DIR="$PROJECT_ROOT/discrete_mbrl/model_free/models/MiniGrid-DoorKey-8x8-v0"
for SEED in 1 2; do
    MODEL="$DK8_DIR/rigor_dk8_vae_s${SEED}_best_model.pt"
    OUT="$RIGOR_DIR/rigor_dk8_vae_s${SEED}_contprobe.json"
    LOG="$RIGOR_DIR/rigor_dk8_vae_s${SEED}_contprobe.log"
    echo "=== VAE continuous probe seed=$SEED ==="
    "$PYTHON_BIN" -u probe_continuous.py \
        --model_path "$MODEL" --n_frames 30000 --device "$DEVICE" \
        --output_json "$OUT" > "$LOG" 2>&1 || { echo "  ✗"; tail -10 "$LOG"; continue; }
    MACRO=$(grep -aoP 'macro-mean\):\s+\K[0-9.]+' "$LOG" | tail -1 || echo NA)
    echo "  → macro=${MACRO}%"
done

# ── DK16 WM analysis ────────────────────────────────────────────────────────
DK16_DIR="$PROJECT_ROOT/discrete_mbrl/model_free/models/MiniGrid-DoorKey-16x16-v0"
for RUN in "rigor_dk16_cb512_thr2:A1_cb512_thr2" "rigor_dk16_cb256_thr1:A2_cb256_thr1"; do
    RUN_NAME="${RUN%%:*}"; LABEL="${RUN##*:}"
    MODEL="$DK16_DIR/${RUN_NAME}_best_model.pt"
    OUT="$RIGOR_DIR/${RUN_NAME}_wm.json"
    LOG="$RIGOR_DIR/${RUN_NAME}_wm.log"
    echo "=== DK16 WM analysis: $LABEL ==="
    "$PYTHON_BIN" -u analyze_wm_semantic_accuracy.py \
        --model_path "$MODEL" --n_frames 5000 --device "$DEVICE" \
        --output_json "$OUT" > "$LOG" 2>&1 || { echo "  ✗"; tail -10 "$LOG"; continue; }
    echo "  → $OUT"
done

echo ""
echo "Post-rigor analysis done."
