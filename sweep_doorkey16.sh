#!/usr/bin/env bash
# Phase 6: DoorKey-16x16 multi-seed sweep (seeds 1-3, v6 encoder)
#
# Configuration mirrors exp42 (run_mf_e2e_semantic_doorkey16_v6enc_restart.sh)
# with the addition of --seed and run-name parameterisation.
#
# Usage:
#   bash sweep_doorkey16.sh          # seeds 1 2 3
#   SEEDS="1 2" bash sweep_doorkey16.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"
LOG_DIR="$PROJECT_ROOT/logs/sweep_dk16"
RESULTS_CSV="$LOG_DIR/sweep_dk16_results.csv"
ENV_NAME="MiniGrid-DoorKey-16x16-v0"
DEVICE="${DEVICE:-cuda}"
SEEDS="${SEEDS:-1 2 3}"

DEFAULT_PYTHON="$PROJECT_ROOT/../miniforge3/envs/stvqvae/bin/python"

pick_python() {
  for c in "$DEFAULT_PYTHON" "python3"; do
    if [[ -x "$c" ]] && "$c" -c "import torch" >/dev/null 2>&1; then
      printf '%s\n' "$c"; return 0
    fi
  done
  return 1
}

PYTHON_BIN="$(pick_python)" || { echo "No Python with torch found." >&2; exit 1; }

mkdir -p "$LOG_DIR"
mkdir -p "$PROJECT_ROOT/.mplconfig"

if [[ ! -f "$RESULTS_CSV" ]]; then
  echo "encoder,seed,best_reward,final_reward,overall_avg,probe_overall,probe_wall,probe_floor,probe_door,probe_key,probe_goal,probe_agent" \
    > "$RESULTS_CSV"
fi

ENC="v6"
TOTAL=$(echo $SEEDS | wc -w)
COUNT=0

for SEED in $SEEDS; do
  COUNT=$((COUNT + 1))
  RUN_NAME="sweep_doorkey16_${ENC}_s${SEED}"
  LOG_FILE="$LOG_DIR/${RUN_NAME}.log"
  MODEL_PATH="$PROJECT_ROOT/discrete_mbrl/model_free/models/${ENV_NAME}/${RUN_NAME}_best_model.pt"

  echo ""
  echo "════════════════════════════════════════════════════════════════"
  echo "  [$COUNT/$TOTAL] DoorKey-16x16  Encoder=${ENC}  Seed=${SEED}"
  echo "  Run: $RUN_NAME"
  echo "════════════════════════════════════════════════════════════════"

  # Skip if already in CSV
  if grep -q "^${ENC},${SEED}," "$RESULTS_CSV" 2>/dev/null; then
    echo "  → Already in results CSV, skipping."
    continue
  fi

  # ── Train ──
  cd "$PROJECT_ROOT/discrete_mbrl/model_free"
  export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
  export TORCHDYNAMO_DISABLE=1
  export MPLCONFIGDIR="$PROJECT_ROOT/.mplconfig"

  echo "  Training 8M steps... (log: $LOG_FILE)"
  "$PYTHON_BIN" -u train.py \
    --env_name "$ENV_NAME" \
    --ae_model_type vqvae \
    --ae_model_version 6 \
    --codebook_size 512 \
    --embedding_dim 64 \
    --filter_size 16 \
    --dead_code_threshold 1.0 \
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

  TRAIN_EXIT=$?
  if [[ $TRAIN_EXIT -ne 0 ]]; then
    echo "  ✗ Training failed (exit $TRAIN_EXIT). See $LOG_FILE"
    continue
  fi

  # Extract RL metrics
  BEST_REWARD=$(grep -aoP 'New best average reward:\s+\K[0-9.]+' "$LOG_FILE" | tail -1 || echo "NA")
  FINAL_REWARD=$(grep -aoP 'Final \d+-episode average:\s+\K[0-9.]+' "$LOG_FILE" | tail -1 || echo "NA")
  OVERALL_AVG=$(grep -aoP 'Overall average reward:\s+\K[0-9.]+' "$LOG_FILE" | tail -1 || echo "NA")
  echo "  ✓ Training done: best=$BEST_REWARD  final=$FINAL_REWARD  avg=$OVERALL_AVG"

  # ── Semantic probe ──
  if [[ -f "$MODEL_PATH" ]]; then
    PROBE_LOG="$LOG_DIR/${RUN_NAME}_probe.log"
    echo "  Running semantic probe..."
    cd "$PROJECT_ROOT/discrete_mbrl"
    TORCHDYNAMO_DISABLE=1 "$PYTHON_BIN" -u probe_semantics.py \
      --model_path "$MODEL_PATH" \
      --n_frames 50000 \
      --probe_epochs 20 \
      --device "$DEVICE" \
      > "$PROBE_LOG" 2>&1 || true

    PROBE_OVERALL=$(grep -a "Overall per-position accuracy" "$PROBE_LOG" | grep -oP '[\d.]+(?=%)' | head -1 || echo "NA")
    PROBE_WALL=$(grep  -a "wall"  "$PROBE_LOG" | grep "id=" | grep -oP '[\d.]+(?=%)' | head -1 || echo "NA")
    PROBE_FLOOR=$(grep -a "floor\|empty" "$PROBE_LOG" | grep "id=" | grep -oP '[\d.]+(?=%)' | head -1 || echo "NA")
    PROBE_DOOR=$(grep  -a "door"  "$PROBE_LOG" | grep "id=" | grep -oP '[\d.]+(?=%)' | head -1 || echo "NA")
    PROBE_KEY=$(grep   -a " key " "$PROBE_LOG" | grep "id=" | grep -oP '[\d.]+(?=%)' | head -1 || echo "NA")
    PROBE_GOAL=$(grep  -a "goal"  "$PROBE_LOG" | grep "id=" | grep -oP '[\d.]+(?=%)' | head -1 || echo "NA")
    PROBE_AGENT=$(grep -a "agent" "$PROBE_LOG" | grep "id=" | grep -oP '[\d.]+(?=%)' | head -1 || echo "NA")
    echo "  ✓ Probe: overall=${PROBE_OVERALL}%  goal=${PROBE_GOAL}%  door=${PROBE_DOOR}%"
  else
    echo "  ⚠ No model found at $MODEL_PATH, skipping probe"
    PROBE_OVERALL="NA"; PROBE_WALL="NA"; PROBE_FLOOR="NA"
    PROBE_DOOR="NA"; PROBE_KEY="NA"; PROBE_GOAL="NA"; PROBE_AGENT="NA"
  fi

  # ── Codebook analysis ──
  if [[ -f "$MODEL_PATH" ]]; then
    CB_LOG="$LOG_DIR/${RUN_NAME}_codebook.log"
    echo "  Running codebook analysis..."
    cd "$PROJECT_ROOT/discrete_mbrl"
    TORCHDYNAMO_DISABLE=1 "$PYTHON_BIN" -u analyze_codebook_usage.py \
      --model_path "$MODEL_PATH" \
      --n_frames 10000 \
      --device "$DEVICE" \
      --output_json "$LOG_DIR/${RUN_NAME}_codebook.json" \
      > "$CB_LOG" 2>&1 || true
    echo "  ✓ Codebook analysis done"
  fi

  # ── Append to CSV ──
  echo "${ENC},${SEED},${BEST_REWARD},${FINAL_REWARD},${OVERALL_AVG},${PROBE_OVERALL},${PROBE_WALL},${PROBE_FLOOR},${PROBE_DOOR},${PROBE_KEY},${PROBE_GOAL},${PROBE_AGENT}" \
    >> "$RESULTS_CSV"
  echo "  → Saved to $RESULTS_CSV"
done

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "  DoorKey-16x16 sweep complete!"
echo "════════════════════════════════════════════════════════════════"
cat "$RESULTS_CSV"
