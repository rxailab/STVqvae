#!/usr/bin/env bash
# DoorKey-16x16 multi-seed sweep with encoder arg — supports v6 (default) and
# v5dc. Calls analyze_wm_multistep.py after training.
#
# Usage:
#   SEEDS="1 2 3" bash sweep_doorkey16_v2.sh v6
#   SEEDS="1 2 3" bash sweep_doorkey16_v2.sh v5dc
set -euo pipefail

# Use main tree paths (dashboard, checkpoints live there)
PROJECT_ROOT="/home/xiar3/experiments/STVqvae"
LOG_DIR="$PROJECT_ROOT/logs/sweep_dk16"
RESULTS_CSV="$LOG_DIR/sweep_dk16_results_v2.csv"
WM_OUT_DIR="$PROJECT_ROOT/logs/wm_multistep"
ENV_NAME="MiniGrid-DoorKey-16x16-v0"
DEVICE="${DEVICE:-cuda}"
SEEDS="${SEEDS:-1 2 3}"
ENC="${1:-v6}"

DEFAULT_PYTHON="/home/xiar3/experiments/miniforge3/envs/stvqvae/bin/python"
PYTHON_BIN="$DEFAULT_PYTHON"
[[ -x "$PYTHON_BIN" ]] || { echo "No Python at $PYTHON_BIN" >&2; exit 1; }

mkdir -p "$LOG_DIR" "$WM_OUT_DIR" "$PROJECT_ROOT/.mplconfig"

if [[ ! -f "$RESULTS_CSV" ]]; then
  echo "encoder,seed,best_reward,final_reward,overall_avg,probe_wall,probe_door,probe_key,probe_goal,probe_agent,wm_1_wall,wm_1_door,wm_1_key,wm_1_goal,wm_1_agent,wm_10_wall,wm_10_door,wm_10_key,wm_10_goal,wm_10_agent,r_1,r_5,r_10" > "$RESULTS_CSV"
fi

# Encoder-specific args for DK-16 (larger filter, bigger codebook for v6)
case "$ENC" in
    v6)
        ENC_ARGS="--ae_model_version 6 --codebook_size 512 --filter_size 16 --dead_code_threshold 1.0"
        ;;
    v5dc)
        ENC_ARGS="--ae_model_version 5 --codebook_size 512 --filter_size 16 --dead_code_threshold 2.0"
        ;;
    *)
        echo "Unknown encoder: $ENC (supported: v6, v5dc)" >&2; exit 1
        ;;
esac

TOTAL=$(echo $SEEDS | wc -w)
COUNT=0

for SEED in $SEEDS; do
  COUNT=$((COUNT + 1))
  RUN_NAME="sweep_doorkey16_${ENC}_s${SEED}"
  LOG_FILE="$LOG_DIR/${RUN_NAME}.log"
  MODEL_PATH="$PROJECT_ROOT/discrete_mbrl/model_free/models/${ENV_NAME}/${RUN_NAME}_best_model.pt"

  echo ""
  echo "════════════════════════════════════════════════════════════════"
  echo "  [$COUNT/$TOTAL] DK-16  Enc=${ENC}  Seed=${SEED}  $(date)"
  echo "════════════════════════════════════════════════════════════════"

  if grep -q "^${ENC},${SEED}," "$RESULTS_CSV" 2>/dev/null; then
    echo "  → already in CSV, skipping"
    continue
  fi

  cd "$PROJECT_ROOT/discrete_mbrl/model_free"
  export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
  export TORCHDYNAMO_DISABLE=1
  export MPLCONFIGDIR="$PROJECT_ROOT/.mplconfig"

  echo "  Training 8M steps..."
  "$PYTHON_BIN" -u train.py \
    --env_name "$ENV_NAME" \
    --ae_model_type vqvae \
    $ENC_ARGS \
    --embedding_dim 64 \
    --seed "$SEED" \
    --mf_steps 8000000 \
    --batch_size 4096 \
    --num_envs 16 \
    --ppo_iters 10 \
    --ppo_batch_size 64 \
    --ppo_entropy_coef 0.01 \
    --ppo_gae_lambda 0.95 \
    --ppo_norm_advantages \
    --ppo_max_grad_norm 0.5 \
    --learning_rate 1e-4 \
    --e2e_loss \
    --encoder_lr 1e-5 \
    --encoder_lr_cosine \
    --encoder_snapback \
    --snapback_threshold 0.5 \
    --snapback_patience 100 \
    --snapback_min_reward 0.05 \
    --ortho_init \
    --model_dir .. \
    --run_name "$RUN_NAME" \
    --device "$DEVICE" \
    --save \
    --use_world_model \
    --wm_aux_coef 0.1 \
    --wm_train_freq 1 \
    --trans_model_type discrete \
    --trans_model_version 1 \
    --trans_hidden 256 \
    --trans_depth 3 \
    --use_semantic_aux \
    --sem_aux_coef 0.08 \
    --sem_head_hidden 128 \
    --sem_head_version 2 \
    --sem_n_classes 11 \
    --sem_aux_start_reward 0.0 \
    --sem_class_weights \
    --sem_focal_gamma 0.0 \
    --sem_pre_vq \
    > "$LOG_FILE" 2>&1

  if [[ $? -ne 0 ]]; then
    echo "  ✗ Training failed. See $LOG_FILE"
    continue
  fi

  BEST=$(grep -aoP 'New best average reward:\s+\K[0-9.]+' "$LOG_FILE" | tail -1 || echo "NA")
  FINAL=$(grep -aoP 'Final \d+-episode average:\s+\K[0-9.]+' "$LOG_FILE" | tail -1 || echo "NA")
  AVG=$(grep -aoP 'Overall average reward:\s+\K[0-9.]+' "$LOG_FILE" | tail -1 || echo "NA")
  echo "  ✓ best=$BEST final=$FINAL avg=$AVG"

  # ── Multi-step WM analysis ──
  OUT_JSON="$WM_OUT_DIR/${RUN_NAME}.json"
  if [[ -f "$MODEL_PATH" ]]; then
    echo "  Running multi-step WM analysis..."
    cd "$PROJECT_ROOT/discrete_mbrl"
    "$PYTHON_BIN" -u analyze_wm_multistep.py \
      --model_path "$MODEL_PATH" \
      --n_rollouts 500 \
      --probe_frames 20000 \
      --horizons 1 3 5 10 \
      --device "$DEVICE" \
      --output_json "$OUT_JSON" \
      > "$LOG_DIR/${RUN_NAME}_analysis.log" 2>&1 || true

    if [[ -f "$OUT_JSON" ]]; then
      "$PYTHON_BIN" -c "
import json
d = json.load(open('$OUT_JSON'))
def g(pc, cls): return pc.get(str(cls), {}).get('acc', None)
pa = d.get('probe_acc_per_class', {})
wm = d.get('wm_acc_per_class_per_horizon', {})
row = ['$ENC','$SEED','$BEST','$FINAL','$AVG']
for cls in [2, 4, 5, 8, 10]:
    v = g(pa, cls); row.append(f'{v:.4f}' if v is not None else 'NA')
for k in [1, 10]:
    for cls in [2, 4, 5, 8, 10]:
        v = g(wm.get(str(k), {}), cls); row.append(f'{v:.4f}' if v is not None else 'NA')
for k in [1, 5, 10]:
    r = d.get('pearson_r_per_horizon', {}).get(str(k))
    row.append(f'{r:+.3f}' if r is not None else 'NA')
print(','.join(row))
" >> "$RESULTS_CSV"
      echo "  → CSV updated"
    fi
  fi
done

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "  DK-16 sweep ${ENC} complete"
echo "════════════════════════════════════════════════════════════════"
cat "$RESULTS_CSV"
