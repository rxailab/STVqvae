#!/usr/bin/env bash
# Tier 1.2 — DoorKey-16x16 with codebook_size=1024 (single seed).
# Tests whether doubling codebook capacity fixes dead-code collapse on DK16.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="$PROJECT_ROOT/../miniforge3/envs/stvqvae/bin/python"
ENV_NAME="MiniGrid-DoorKey-16x16-v0"
RUN_NAME="dk16_v6_cb1024_s1"
LOG_DIR="$PROJECT_ROOT/logs/dk16_cb1024"
LOG_FILE="$LOG_DIR/${RUN_NAME}.log"
DEVICE="${DEVICE:-cuda}"

mkdir -p "$LOG_DIR" "$PROJECT_ROOT/.mplconfig"
cd "$PROJECT_ROOT/discrete_mbrl/model_free"
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
export TORCHDYNAMO_DISABLE=1
export MPLCONFIGDIR="$PROJECT_ROOT/.mplconfig"

echo "Training DK16 cb=1024 seed=1 (8M steps). Log: $LOG_FILE"

"$PYTHON_BIN" -u train.py \
    --env_name "$ENV_NAME" \
    --ae_model_type vqvae \
    --ae_model_version 6 \
    --codebook_size 1024 \
    --embedding_dim 64 \
    --filter_size 16 \
    --dead_code_threshold 1.0 \
    --seed 1 \
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

BEST=$(grep -aoP 'New best average reward:\s+\K[0-9.]+' "$LOG_FILE" | tail -1 || echo "NA")
FINAL=$(grep -aoP 'Final \d+-episode average:\s+\K[0-9.]+' "$LOG_FILE" | tail -1 || echo "NA")
echo "Done: best=$BEST final=$FINAL"

# Follow-up probe + codebook analysis
MODEL="$PROJECT_ROOT/discrete_mbrl/model_free/models/${ENV_NAME}/${RUN_NAME}_best_model.pt"
if [[ -f "$MODEL" ]]; then
    cd "$PROJECT_ROOT/discrete_mbrl"
    echo "Probing..."
    "$PYTHON_BIN" -u probe_semantics.py \
        --model_path "$MODEL" --n_frames 50000 --probe_epochs 20 \
        --device "$DEVICE" > "$LOG_DIR/${RUN_NAME}_probe.log" 2>&1 || true
    echo "Codebook analysis..."
    "$PYTHON_BIN" -u analyze_codebook_usage.py \
        --model_path "$MODEL" --n_frames 10000 --device "$DEVICE" \
        --output_json "$LOG_DIR/${RUN_NAME}_codebook.json" \
        > "$LOG_DIR/${RUN_NAME}_codebook.log" 2>&1 || true
    echo "All done. Outputs in $LOG_DIR"
fi
