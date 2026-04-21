#!/usr/bin/env bash
# Phase 8: Codebook analysis on all Crafter checkpoints
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="$PROJECT_ROOT/../miniforge3/envs/stvqvae/bin/python"
MODELS_DIR="$PROJECT_ROOT/discrete_mbrl/model_free/models/crafter"
OUT_DIR="$PROJECT_ROOT/logs/codebook_analysis"
DEVICE="${DEVICE:-cuda}"

mkdir -p "$OUT_DIR"
cd "$PROJECT_ROOT/discrete_mbrl"
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
export TORCHDYNAMO_DISABLE=1

run_cb() {
    local label="$1"
    local model="$2"
    local out_json="$OUT_DIR/crafter_${label}.json"
    echo ""
    echo "=== Crafter codebook: ${label} ==="
    "$PYTHON_BIN" -u analyze_codebook_usage.py \
        --model_path "$model" \
        --n_frames 10000 \
        --device "$DEVICE" \
        --output_json "$out_json" \
        2>&1 | grep -v "FutureWarn\|pynvml\|Gym has\|replaced\|upgrade\|migration\|Users" \
             | tee "$OUT_DIR/crafter_${label}.log"
    echo "  → Saved to $out_json"
}

# Run all 4 Crafter models
run_cb "v6enc_original"  "$MODELS_DIR/mf_e2e_semantic_crafter_v6enc_best_model.pt"
run_cb "v6enc_fix"       "$MODELS_DIR/mf_e2e_semantic_crafter_v6enc_fix_best_model.pt"
run_cb "v6enc_cal"       "$MODELS_DIR/mf_e2e_semantic_crafter_v6enc_cal_best_model.pt"
run_cb "v6enc_cal2"      "$MODELS_DIR/mf_e2e_semantic_crafter_v6enc_cal2_best_model.pt"

echo ""
echo "════════════════════════════════════════════════════════"
echo "  CRAFTER CODEBOOK SUMMARY"
echo "════════════════════════════════════════════════════════"
"$PYTHON_BIN" -c "
import json, os, glob
out_dir = '$OUT_DIR'
for jf in sorted(glob.glob(os.path.join(out_dir, 'crafter_*.json'))):
    label = os.path.basename(jf).replace('.json','')
    with open(jf) as f:
        d = json.load(f)
    n_dead = d.get('n_dead_codes', 'N/A')
    cb = d.get('codebook_size', '?')
    util = d.get('n_active_codes', '?')
    top_cls = [(k,v) for k,v in sorted(d.get('codes_per_class',{}).items(), key=lambda x:-x[1]) if v>0][:5]
    print(f'  {label:<25}  dead={n_dead}/{cb}  active={util}')
    for cls,n in top_cls:
        print(f'    {cls:<15} {n} codes')
"
echo ""
echo "Results saved to $OUT_DIR"
