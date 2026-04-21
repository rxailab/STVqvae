#!/usr/bin/env bash
# Phase 7: Continuous VAE baseline on DoorKey-8x8
#
# Trains a continuous VAE encoder (no VQ bottleneck) end-to-end with PPO +
# semantic aux loss. Used to show that discrete VQ codes are necessary for
# hard semantic specialization — a continuous encoder can achieve high probe
# accuracy but doesn't develop dedicated codes per class.
#
# Key differences vs v6 VQVAE:
#   --ae_model_type vae    (continuous encoder, no quantization)
#   no --use_world_model   (WM requires discrete codes)
#   --sem_pre_vq           (semantic loss on encoder output directly)
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="$PROJECT_ROOT/../miniforge3/envs/stvqvae/bin/python"
ENV_NAME="MiniGrid-DoorKey-8x8-v0"
RUN_NAME="vae_baseline_doorkey_v6enc"
LOG_DIR="$PROJECT_ROOT/logs"
LOG_FILE="$LOG_DIR/${RUN_NAME}.log"
DEVICE="${DEVICE:-cuda}"

mkdir -p "$LOG_DIR" "$PROJECT_ROOT/.mplconfig"

cd "$PROJECT_ROOT/discrete_mbrl/model_free"
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
export TORCHDYNAMO_DISABLE=1
export MPLCONFIGDIR="$PROJECT_ROOT/.mplconfig"

echo "Phase 7: VAE baseline DoorKey-8x8"
echo "Log: $LOG_FILE"

"$PYTHON_BIN" -u train.py \
    --env_name "$ENV_NAME" \
    --ae_model_type vae \
    --ae_model_version 6 \
    --embedding_dim 64 \
    --filter_size 8 \
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
    --use_semantic_aux \
    --sem_aux_coef 0.05 \
    --sem_head_hidden 128 \
    --sem_head_version 2 \
    --sem_n_classes 11 \
    --sem_aux_start_reward 0.0 \
    --sem_class_weights \
    --sem_focal_gamma 2.0 \
    --sem_pre_vq \
    > "$LOG_FILE" 2>&1

BEST=$(grep -aoP 'New best average reward:\s+\K[0-9.]+' "$LOG_FILE" | tail -1 || echo "NA")
FINAL=$(grep -aoP 'Final \d+-episode average:\s+\K[0-9.]+' "$LOG_FILE" | tail -1 || echo "NA")
echo "Done: best=$BEST  final=$FINAL"
echo "Model: discrete_mbrl/model_free/models/${ENV_NAME}/${RUN_NAME}_best_model.pt"
