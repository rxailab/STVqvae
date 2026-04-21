#!/usr/bin/env bash
# Phase 3: v6 ablation sweep — runs v6a, v6b, v6c sequentially then probes each
# v6a = RGB shortcut only  (no coords)
# v6b = coord grid only    (no RGB)
# v6c = both + full trunk  (no capacity split)
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="$PROJECT_ROOT/../miniforge3/envs/stvqvae/bin/python"
LOG_DIR="$PROJECT_ROOT/logs/ablation_v6"
RESULTS_CSV="$LOG_DIR/ablation_results.csv"
ENV_NAME="MiniGrid-DoorKey-8x8-v0"
DEVICE="${DEVICE:-cuda}"
TOTAL_BATCHES=1221

mkdir -p "$LOG_DIR"

# Reference: v6 (full) and v5+deadcode results for comparison column
echo "encoder,seed,best_reward,final_reward,overall_avg,probe_overall,probe_wall,probe_empty,probe_door,probe_key,probe_goal,probe_agent" \
  > "$RESULTS_CSV"

VARIANTS=(v6a v6b v6c)
DESCS=(
  "RGB shortcut only (no coord grid)"
  "Coord grid only (no RGB shortcut)"
  "RGB + coords + full-width trunk"
)

for i in "${!VARIANTS[@]}"; do
  ENC="${VARIANTS[$i]}"
  DESC="${DESCS[$i]}"
  RUN_NAME="doorkey_${ENC}enc_ablation"
  LOG="$LOG_DIR/${RUN_NAME}.log"
  MODEL="$PROJECT_ROOT/discrete_mbrl/model_free/models/${ENV_NAME}/${RUN_NAME}_best_model.pt"

  echo ""
  echo "════════════════════════════════════════════════════════════════"
  echo "  [$(($i+1))/3] Encoder=${ENC}  —  ${DESC}"
  echo "════════════════════════════════════════════════════════════════"
  echo "  Training... (log: $LOG)"

  cd "$PROJECT_ROOT/discrete_mbrl/model_free"
  export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
  export TORCHDYNAMO_DISABLE=1

  "$PYTHON_BIN" -u train.py \
    --env_name "$ENV_NAME" \
    --ae_model_type vqvae \
    --ae_model_version "${ENC#v}" \
    --codebook_size 64 \
    --embedding_dim 64 \
    --filter_size 8 \
    --dead_code_threshold 2.0 \
    --seed 42 \
    --mf_steps 5000000 \
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
    --snapback_min_reward 0.0 \
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
    --sem_aux_coef 0.05 \
    --sem_head_hidden 128 \
    --sem_head_version 2 \
    --sem_n_classes 11 \
    --sem_aux_start_reward 0.0 \
    --sem_class_weights \
    --sem_focal_gamma 2.0 \
    --sem_pre_vq \
    > "$LOG" 2>&1

  # ── Extract training metrics ──
  BEST_REWARD=$(grep -aoP 'New best average reward:\s+\K[0-9.]+' "$LOG" | tail -1 || echo "NA")
  FINAL_REWARD=$(grep -aoP 'Final \d+-episode average:\s+\K[0-9.]+' "$LOG" | tail -1 || echo "NA")
  OVERALL_AVG=$(grep -aoP 'Overall average reward:\s+\K[0-9.]+' "$LOG" | tail -1 || echo "NA")
  echo "  ✓ Training done: best=${BEST_REWARD}  final=${FINAL_REWARD}"

  # ── Semantic probe ──
  PROBE_LOG="$LOG_DIR/${RUN_NAME}_probe.log"
  if [[ -f "$MODEL" ]]; then
    echo "  Running semantic probe..."
    cd "$PROJECT_ROOT/discrete_mbrl"
    "$PYTHON_BIN" -u probe_semantics.py \
      --model_path "$MODEL" \
      --n_frames 50000 \
      --probe_epochs 20 \
      --device "$DEVICE" \
      > "$PROBE_LOG" 2>&1

    PROBE_OVERALL=$(grep -a "Overall per-position accuracy" "$PROBE_LOG" | grep -oP '[\d.]+(?=%)' | head -1 || echo "NA")
    PROBE_WALL=$(grep  -a "wall"  "$PROBE_LOG" | grep "id=" | grep -oP '[\d.]+(?=%)' | head -1 || echo "NA")
    PROBE_EMPTY=$(grep -a "empty" "$PROBE_LOG" | grep "id=" | grep -oP '[\d.]+(?=%)' | head -1 || echo "NA")
    PROBE_DOOR=$(grep  -a "door"  "$PROBE_LOG" | grep "id=" | grep -oP '[\d.]+(?=%)' | head -1 || echo "NA")
    PROBE_KEY=$(grep   -a " key " "$PROBE_LOG" | grep "id=" | grep -oP '[\d.]+(?=%)' | head -1 || echo "NA")
    PROBE_GOAL=$(grep  -a "goal"  "$PROBE_LOG" | grep "id=" | grep -oP '[\d.]+(?=%)' | head -1 || echo "NA")
    PROBE_AGENT=$(grep -a "agent" "$PROBE_LOG" | grep "id=" | grep -oP '[\d.]+(?=%)' | head -1 || echo "NA")
    echo "  ✓ Probe: overall=${PROBE_OVERALL}%  goal=${PROBE_GOAL}%  key=${PROBE_KEY}%"
  else
    echo "  ⚠ No model found, skipping probe"
    PROBE_OVERALL="NA"; PROBE_WALL="NA"; PROBE_EMPTY="NA"
    PROBE_DOOR="NA"; PROBE_KEY="NA"; PROBE_GOAL="NA"; PROBE_AGENT="NA"
  fi

  # ── Codebook analysis ──
  CB_LOG="$LOG_DIR/${RUN_NAME}_codebook.log"
  if [[ -f "$MODEL" ]]; then
    echo "  Running codebook analysis..."
    cd "$PROJECT_ROOT/discrete_mbrl"
    "$PYTHON_BIN" -u analyze_codebook_usage.py \
      --model_path "$MODEL" \
      --n_frames 10000 --device "$DEVICE" \
      --output_json "$LOG_DIR/${RUN_NAME}_codebook.json" \
      > "$CB_LOG" 2>&1 || true
    echo "  ✓ Codebook analysis done"
  fi

  echo "${ENC},42,${BEST_REWARD},${FINAL_REWARD},${OVERALL_AVG},${PROBE_OVERALL},${PROBE_WALL},${PROBE_EMPTY},${PROBE_DOOR},${PROBE_KEY},${PROBE_GOAL},${PROBE_AGENT}" \
    >> "$RESULTS_CSV"
  echo "  → Results saved to $RESULTS_CSV"
done

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "  v6 Ablation sweep complete!"
echo "════════════════════════════════════════════════════════════════"
cat "$RESULTS_CSV"
