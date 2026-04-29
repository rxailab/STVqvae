#!/usr/bin/env bash
# Re-run multi-step WM analysis on all 12 existing DK-8 sweep checkpoints.
# Writes per-checkpoint JSON + summary CSV.
set -euo pipefail

MAIN_TREE="/home/xiar3/experiments/STVqvae"
CKPT_DIR="$MAIN_TREE/discrete_mbrl/model_free/models/MiniGrid-DoorKey-8x8-v0"
OUT_DIR="$MAIN_TREE/logs/wm_multistep"
SUMMARY_CSV="$OUT_DIR/reprobe_summary.csv"
PYTHON_BIN="/home/xiar3/experiments/miniforge3/envs/stvqvae/bin/python"

mkdir -p "$OUT_DIR"

# CSV header
if [[ ! -f "$SUMMARY_CSV" ]]; then
  echo "encoder,seed,probe_wall,probe_door,probe_key,probe_goal,probe_agent,wm_1_wall,wm_1_door,wm_1_key,wm_1_goal,wm_1_agent,wm_5_wall,wm_5_door,wm_5_key,wm_5_goal,wm_5_agent,r_1,r_3,r_5,r_10" > "$SUMMARY_CSV"
fi

export PYTHONPATH="$MAIN_TREE:${PYTHONPATH:-}"
export TORCHDYNAMO_DISABLE=1
export MPLCONFIGDIR="/tmp"

cd "$MAIN_TREE"

TOTAL=0; COUNT=0
for ENC in v2 v5 v6 v9; do
  for SEED in 1 2 3; do
    TOTAL=$((TOTAL + 1))
  done
done

for ENC in v2 v5 v6 v9; do
  for SEED in 1 2 3; do
    COUNT=$((COUNT + 1))
    MODEL="$CKPT_DIR/sweep_doorkey8_${ENC}_s${SEED}_best_model.pt"
    OUT_JSON="$OUT_DIR/sweep_doorkey8_${ENC}_s${SEED}.json"

    if [[ ! -f "$MODEL" ]]; then
      echo "[$COUNT/$TOTAL] MISSING: $MODEL"
      continue
    fi

    if [[ -f "$OUT_JSON" ]]; then
      echo "[$COUNT/$TOTAL] $ENC s$SEED — already exists, skipping"
      continue
    fi

    echo ""
    echo "════════════════════════════════════════════════════════════════"
    echo "  [$COUNT/$TOTAL] Re-probe: encoder=$ENC seed=$SEED"
    echo "════════════════════════════════════════════════════════════════"

    "$PYTHON_BIN" -u discrete_mbrl/analyze_wm_multistep.py \
        --model_path "$MODEL" \
        --n_rollouts 500 \
        --probe_frames 20000 \
        --horizons 1 3 5 10 \
        --device cuda \
        --output_json "$OUT_JSON" 2>&1 | tee "$OUT_DIR/sweep_doorkey8_${ENC}_s${SEED}.log" | \
        grep -E "Pearson|wall|door|key|goal|agent|rollouts: 500" || true

    if [[ ! -f "$OUT_JSON" ]]; then
      echo "  ⚠ Analysis failed for $ENC s$SEED"
      continue
    fi

    # Extract summary fields from JSON via python one-liner
    "$PYTHON_BIN" -c "
import json
d = json.load(open('$OUT_JSON'))
def g(per_class_dict, cls):
    # cls IDs: wall=2, door=4, key=5, goal=8, agent=10
    return per_class_dict.get(str(cls), {}).get('acc', None)
pa = d.get('probe_acc_per_class', {})
wm = d.get('wm_acc_per_class_per_horizon', {})
row = ['$ENC', '$SEED']
for cls in [2, 4, 5, 8, 10]:
    v = g(pa, cls); row.append(f'{v:.4f}' if v is not None else 'NA')
for k in [1, 5]:
    for cls in [2, 4, 5, 8, 10]:
        v = g(wm.get(str(k), {}), cls); row.append(f'{v:.4f}' if v is not None else 'NA')
for k in [1, 3, 5, 10]:
    r = d.get('pearson_r_per_horizon', {}).get(str(k))
    row.append(f'{r:+.3f}' if r is not None else 'NA')
print(','.join(row))
" >> "$SUMMARY_CSV"

  done
done

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "  RE-PROBE COMPLETE — summary in $SUMMARY_CSV"
echo "════════════════════════════════════════════════════════════════"
column -t -s, "$SUMMARY_CSV"
