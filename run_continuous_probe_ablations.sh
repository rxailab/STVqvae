#!/usr/bin/env bash
# Tier 1.3 — Continuous probes on v6a/v6b/v6c ablation checkpoints.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="$PROJECT_ROOT/../miniforge3/envs/stvqvae/bin/python"
MODELS_DIR="$PROJECT_ROOT/discrete_mbrl/model_free/models/MiniGrid-DoorKey-8x8-v0"
OUT_DIR="$PROJECT_ROOT/logs/continuous_probe"
DEVICE="${DEVICE:-cuda}"

mkdir -p "$OUT_DIR"
cd "$PROJECT_ROOT/discrete_mbrl"
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
export TORCHDYNAMO_DISABLE=1

for VAR in v6a v6b v6c; do
    LABEL="${VAR}_ablation"
    MODEL="$MODELS_DIR/doorkey_${VAR}enc_ablation_best_model.pt"
    OUT_JSON="$OUT_DIR/${LABEL}.json"
    LOG="$OUT_DIR/${LABEL}.log"
    echo ""
    echo "=== Continuous probe ${LABEL} ==="
    if [[ ! -f "$MODEL" ]]; then
        echo "  ⚠ Missing: $MODEL"
        continue
    fi
    "$PYTHON_BIN" -u probe_continuous.py \
        --model_path "$MODEL" \
        --n_frames 30000 \
        --device "$DEVICE" \
        --output_json "$OUT_JSON" \
        > "$LOG" 2>&1 || { echo "  ✗ failed; tail:"; tail -20 "$LOG"; continue; }
    MACRO=$(grep -aoP 'macro-mean\):\s+\K[0-9.]+' "$LOG" | tail -1 || echo "NA")
    echo "  → macro=${MACRO}%   ($OUT_JSON)"
done

echo ""
echo "Continuous probe ablations done. Results in $OUT_DIR"
