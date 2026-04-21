#!/usr/bin/env bash
# Phase 5: WM semantic accuracy analysis on all encoder variants
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="$PROJECT_ROOT/../miniforge3/envs/stvqvae/bin/python"
MODELS_DIR="$PROJECT_ROOT/discrete_mbrl/model_free/models/MiniGrid-DoorKey-8x8-v0"
OUT_DIR="$PROJECT_ROOT/logs/wm_analysis"
DEVICE="${DEVICE:-cuda}"

mkdir -p "$OUT_DIR"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
export TORCHDYNAMO_DISABLE=1

run_analysis() {
    local label="$1"
    local model="$2"
    local out_json="$OUT_DIR/${label}.json"
    echo ""
    echo "=== ${label} ==="
    TORCHDYNAMO_DISABLE=1 "$PYTHON_BIN" -u discrete_mbrl/analyze_wm_semantic_accuracy.py \
        --model_path "$model" \
        --n_frames 8000 \
        --probe_frames 30000 \
        --device "$DEVICE" \
        --output_json "$out_json" 2>&1 \
      | grep -v "FutureWarn\|pynvml\|Gym has\|replaced\|upgrade\|migration\|Users\|ConvergenceWarn\|Increase\|scale\|documentation\|sklearn\|n_iter_i" \
      | tee "$OUT_DIR/${label}.log"
}

# ── Core encoder comparison ──
run_analysis "v2_baseline"    "$MODELS_DIR/mf_e2e_semantic_doorkey_best_model.pt"
run_analysis "v5_spatial"     "$MODELS_DIR/mf_e2e_semantic_doorkey_v5enc_best_model.pt"
run_analysis "v5_deadcode"    "$MODELS_DIR/mf_e2e_semantic_doorkey_v5enc_deadcode_best_model.pt"
run_analysis "v6_full"        "$MODELS_DIR/mf_e2e_semantic_doorkey_v6enc_goal_best_model.pt"
run_analysis "v9_patch"       "$MODELS_DIR/mf_e2e_semantic_doorkey_v9enc_patch_best_model.pt"

# ── v6 ablations ──
run_analysis "v6a_rgb_only"   "$MODELS_DIR/doorkey_v6aenc_ablation_best_model.pt"
run_analysis "v6b_coord_only" "$MODELS_DIR/doorkey_v6benc_ablation_best_model.pt"
run_analysis "v6c_wide_trunk" "$MODELS_DIR/doorkey_v6cenc_ablation_best_model.pt"

# ── Aggregate summary ──
echo ""
echo "════════════════════════════════════════════════════════"
echo "  SUMMARY: Pearson r (probe vs WM accuracy per class)"
echo "════════════════════════════════════════════════════════"
"$PYTHON_BIN" -c "
import json, os, glob
out_dir = '$OUT_DIR'
rows = []
for jf in sorted(glob.glob(os.path.join(out_dir, '*.json'))):
    label = os.path.basename(jf).replace('.json','')
    with open(jf) as f:
        d = json.load(f)
    r = d.get('pearson_r')
    scatter = d.get('scatter_data', [])
    r_str = f'{r:.3f}' if r is not None else '  N/A'
    classes_str = ', '.join(f\"{s['name']}(p={s['probe_acc']*100:.0f}%,w={s['wm_acc']*100:.0f}%)\"
                            for s in scatter)
    print(f'  {label:<20}  r={r_str}  |  {classes_str}')
"
echo ""
echo "Results saved to $OUT_DIR"
