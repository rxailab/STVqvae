#!/usr/bin/env bash
set -euo pipefail

# Experiment 45: Crafter — fix class-weight calibration
#
# PROBLEM (Exp 44):
#   Dead-code restart worked: 1024/1024 codebook entries active (100%).
#   Encoder DOES learn semantic features — trained sem_head achieved:
#     fence 33.5%, player 24.3%, lava 14%, path 13.8%, sand 12%, cow 11.7%
#   BUT overall accuracy = 6.4% because the class weights are misbalanced:
#     - grass (42% of data) gets weight ≈ 0.125 → barely penalised for missing grass
#     - fence (0.6% of data) gets weight ≈ 8.8 → heavily penalised → head over-predicts fence (21%!)
#   New linear probe (no weighting) collapsed to always-grass → 44% "accuracy"
#   This is a CALIBRATION failure, not a representation failure.
#
# FIX (this experiment):
#   1. sem_class_weight_power=0.5: sqrt of inverse-frequency weighting.
#      Before: grass=0.125, fence=8.8 (70× ratio).
#      After:  grass≈0.35, fence≈2.97 (normalized → ~8× ratio). Much gentler.
#      Weights still normalized to mean=1 and capped at 10×.
#   2. No focal loss (sem_focal_gamma=0.0): focal loss was adding on top of already
#      aggressive class weights, compounding over-confidence on rare classes.
#   3. Keep all Exp 44 fixes: dead_code_threshold=2.0, sem_aux_coef=0.5,
#      codebook_size=1024, v6 encoder, 8M steps.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"
RUN_NAME="mf_e2e_semantic_crafter_v6enc_cal"
DEFAULT_LOCAL_PYTHON="$PROJECT_ROOT/../miniforge3/envs/stvqvae/bin/python"
DEVICE="${DEVICE:-cuda}"

pick_python() {
  local candidates=()
  if [[ -n "${PYTHON_BIN:-}" ]]; then
    candidates+=("$PYTHON_BIN")
  fi
  candidates+=(
    "$DEFAULT_LOCAL_PYTHON"
    "/storage/hpc/11/xiar3/vit5/bin/python"
    "python3"
  )
  local candidate
  for candidate in "${candidates[@]}"; do
    if [[ "$candidate" == "python3" ]]; then
      if command -v python3 >/dev/null 2>&1 && python3 -c "import torch" >/dev/null 2>&1; then
        printf '%s\n' "python3"; return 0
      fi
      continue
    fi
    if [[ -x "$candidate" ]] && "$candidate" -c "import torch" >/dev/null 2>&1; then
      printf '%s\n' "$candidate"; return 0
    fi
  done
  return 1
}

PYTHON_BIN="$(pick_python || true)"
if [[ -z "$PYTHON_BIN" ]]; then
  echo "No Python with torch found. Set PYTHON_BIN." >&2; exit 1
fi

mkdir -p "$PROJECT_ROOT/.mplconfig"

cd "$PROJECT_ROOT/discrete_mbrl/model_free"
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
export TORCHDYNAMO_DISABLE=1
export MPLCONFIGDIR="$PROJECT_ROOT/.mplconfig"

exec "$PYTHON_BIN" -u train.py \
  --env_name crafter \
  --ae_model_type vqvae \
  --ae_model_version 6 \
  --codebook_size 1024 \
  --embedding_dim 64 \
  --filter_size 8 \
  --dead_code_threshold 2.0 \
  --mf_steps 8000000 \
  --batch_size 4096 \
  --num_envs 16 \
  --ppo_iters 10 \
  --ppo_batch_size 64 \
  --ppo_entropy_coef 0.02 \
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
  --sem_aux_coef 0.5 \
  --sem_head_hidden 128 \
  --sem_head_version 2 \
  --sem_n_classes 19 \
  --sem_aux_start_reward 0.0 \
  --sem_class_weights \
  --sem_class_weight_power 0.5 \
  --sem_focal_gamma 0.0 \
  --sem_pre_vq
