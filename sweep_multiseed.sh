#!/usr/bin/env bash
set -euo pipefail

# Multi-seed sweep runner for DoorKey-8x8 encoder comparison.
# Runs each (encoder, seed) combination sequentially on a single GPU.
# After each training run, automatically runs the semantic probe.
#
# Usage:
#   bash sweep_multiseed.sh                    # all encoders, seeds 1-3
#   bash sweep_multiseed.sh v6 v9              # specific encoders only
#   SEEDS="1 2 3 4 5" bash sweep_multiseed.sh  # 5 seeds

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"
LOG_DIR="$PROJECT_ROOT/logs/sweep"
RESULTS_CSV="$LOG_DIR/sweep_results.csv"
DEFAULT_LOCAL_PYTHON="$PROJECT_ROOT/../miniforge3/envs/stvqvae/bin/python"
DEVICE="${DEVICE:-cuda}"

# ── Configuration ──
SEEDS="${SEEDS:-1 2 3}"
ENV_NAME="MiniGrid-DoorKey-8x8-v0"
MF_STEPS=5000000
BATCH_SIZE=4096
NUM_ENVS=16

# Which encoders to sweep (override via CLI args)
if [[ $# -gt 0 ]]; then
    ENCODERS=("$@")
else
    ENCODERS=(v2 v5 v6 v9)
fi

pick_python() {
  local candidates=("$DEFAULT_LOCAL_PYTHON" "python3")
  for c in "${candidates[@]}"; do
    if [[ -x "$c" ]] && "$c" -c "import torch" >/dev/null 2>&1; then
      printf '%s\n' "$c"; return 0
    fi
  done
  return 1
}

PYTHON_BIN="$(pick_python)" || { echo "No Python with torch found." >&2; exit 1; }
mkdir -p "$LOG_DIR"

# ── CSV header ──
if [[ ! -f "$RESULTS_CSV" ]]; then
    echo "encoder,seed,best_reward,final_reward,overall_avg,probe_overall,probe_wall,probe_floor,probe_door,probe_key,probe_goal,probe_agent,codebook_util" > "$RESULTS_CSV"
fi

# ── Encoder-specific args ──
get_encoder_args() {
    local enc="$1"
    case "$enc" in
        v2)
            echo "--ae_model_version 2 --codebook_size 64 --embedding_dim 64 --filter_size 9"
            ;;
        v5)
            echo "--ae_model_version 5 --codebook_size 64 --embedding_dim 64 --filter_size 8"
            ;;
        v6)
            echo "--ae_model_version 6 --codebook_size 64 --embedding_dim 64 --filter_size 8"
            ;;
        v9)
            echo "--ae_model_version 9 --codebook_size 64 --embedding_dim 64 --filter_size 8"
            ;;
        v6a)
            echo "--ae_model_version 6a --codebook_size 64 --embedding_dim 64 --filter_size 8"
            ;;
        v6b)
            echo "--ae_model_version 6b --codebook_size 64 --embedding_dim 64 --filter_size 8"
            ;;
        v6c)
            echo "--ae_model_version 6c --codebook_size 64 --embedding_dim 64 --filter_size 8"
            ;;
        *)
            echo "Unknown encoder: $enc" >&2; exit 1
            ;;
    esac
}

# ── Main sweep loop ──
TOTAL=$(( ${#ENCODERS[@]} * $(echo $SEEDS | wc -w) ))
COUNT=0

for ENC in "${ENCODERS[@]}"; do
    ENC_ARGS=$(get_encoder_args "$ENC")
    for SEED in $SEEDS; do
        COUNT=$((COUNT + 1))
        RUN_NAME="sweep_doorkey8_${ENC}_s${SEED}"
        LOG_FILE="$LOG_DIR/${RUN_NAME}.log"

        echo ""
        echo "════════════════════════════════════════════════════════════════"
        echo "  [$COUNT/$TOTAL] Encoder=$ENC  Seed=$SEED  Run=$RUN_NAME"
        echo "════════════════════════════════════════════════════════════════"

        # Skip if already completed
        if grep -q "^${ENC},${SEED}," "$RESULTS_CSV" 2>/dev/null; then
            echo "  → Already in results CSV, skipping."
            continue
        fi

        # ── Train ──
        cd "$PROJECT_ROOT/discrete_mbrl/model_free"
        export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
        export TORCHDYNAMO_DISABLE=1
        export MPLCONFIGDIR="$PROJECT_ROOT/.mplconfig"

        echo "  Training... (log: $LOG_FILE)"
        "$PYTHON_BIN" -u train.py \
            --env_name "$ENV_NAME" \
            --ae_model_type vqvae \
            $ENC_ARGS \
            --seed "$SEED" \
            --mf_steps $MF_STEPS \
            --batch_size $BATCH_SIZE \
            --num_envs $NUM_ENVS \
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
            --sem_class_weight_power 1.0 \
            --sem_focal_gamma 2.0 \
            --sem_pre_vq \
            > "$LOG_FILE" 2>&1

        TRAIN_EXIT=$?
        if [[ $TRAIN_EXIT -ne 0 ]]; then
            echo "  ✗ Training failed (exit $TRAIN_EXIT). See $LOG_FILE"
            continue
        fi

        # Extract RL metrics from log
        BEST_REWARD=$(grep -a "Best rolling average" "$LOG_FILE" | tail -1 | grep -oP '[\d.]+$' || echo "NA")
        FINAL_REWARD=$(grep -a "Final 10-episode average" "$LOG_FILE" | tail -1 | grep -oP '[\d.]+$' || echo "NA")
        OVERALL_AVG=$(grep -a "Overall average reward" "$LOG_FILE" | tail -1 | grep -oP '[\d.]+$' || echo "NA")

        echo "  ✓ Training done: best=$BEST_REWARD final=$FINAL_REWARD avg=$OVERALL_AVG"

        # ── Semantic probe ──
        MODEL_DIR="models/${ENV_NAME}"
        BEST_MODEL="${MODEL_DIR}/${RUN_NAME}_best_model.pt"

        if [[ -f "$BEST_MODEL" ]]; then
            echo "  Running semantic probe..."
            cd "$PROJECT_ROOT/discrete_mbrl"
            PROBE_LOG="$LOG_DIR/${RUN_NAME}_probe.log"

            # Use --model_path for direct checkpoint loading (avoids hash-lookup deps)
            BEST_MODEL_ABS="$PROJECT_ROOT/discrete_mbrl/${BEST_MODEL}"
            "$PYTHON_BIN" -u probe_semantics.py \
                --model_path "$BEST_MODEL_ABS" \
                --n_frames 50000 \
                --probe_epochs 20 \
                --device "$DEVICE" \
                > "$PROBE_LOG" 2>&1 || true

            # Extract probe metrics from "  classname  (id=N):  XX.X%  (N samples)" lines
            PROBE_OVERALL=$(grep -a "Overall per-position accuracy" "$PROBE_LOG" | grep -oP '[\d.]+(?=%)' | head -1 || echo "NA")
            PROBE_WALL=$(grep -a "wall" "$PROBE_LOG" | grep -oP '[\d.]+(?=%)' | head -1 || echo "NA")
            PROBE_FLOOR=$(grep -a "floor\|empty" "$PROBE_LOG" | grep "id=" | grep -oP '[\d.]+(?=%)' | head -1 || echo "NA")
            PROBE_DOOR=$(grep -a "door" "$PROBE_LOG" | grep "id=" | grep -oP '[\d.]+(?=%)' | head -1 || echo "NA")
            PROBE_KEY=$(grep -a "key" "$PROBE_LOG" | grep "id=" | grep -oP '[\d.]+(?=%)' | head -1 || echo "NA")
            PROBE_GOAL=$(grep -a "goal" "$PROBE_LOG" | grep "id=" | grep -oP '[\d.]+(?=%)' | head -1 || echo "NA")
            PROBE_AGENT=$(grep -a "agent" "$PROBE_LOG" | grep "id=" | grep -oP '[\d.]+(?=%)' | head -1 || echo "NA")
            CODEBOOK_UTIL="NA"  # TODO: run analyze_codebook_usage.py

            echo "  ✓ Probe: overall=${PROBE_OVERALL}%"
        else
            echo "  ⚠ No best model found at $BEST_MODEL, skipping probe"
            PROBE_OVERALL="NA"; PROBE_WALL="NA"; PROBE_FLOOR="NA"
            PROBE_DOOR="NA"; PROBE_KEY="NA"; PROBE_GOAL="NA"; PROBE_AGENT="NA"
            CODEBOOK_UTIL="NA"
        fi

        # ── Append to CSV ──
        echo "${ENC},${SEED},${BEST_REWARD},${FINAL_REWARD},${OVERALL_AVG},${PROBE_OVERALL},${PROBE_WALL},${PROBE_FLOOR},${PROBE_DOOR},${PROBE_KEY},${PROBE_GOAL},${PROBE_AGENT},${CODEBOOK_UTIL}" >> "$RESULTS_CSV"

        echo "  → Saved to $RESULTS_CSV"
    done
done

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "  SWEEP COMPLETE — Results in $RESULTS_CSV"
echo "════════════════════════════════════════════════════════════════"
cat "$RESULTS_CSV"
