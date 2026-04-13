#!/usr/bin/env bash
set -euo pipefail

# Experiment 37: DoorKey-8x8 with strided encoder v5 + pre-VQ semantic loss + larger codebook
#
# BASELINE: Exp 36 (v5 encoder, post-VQ sem) — Probe 82.9%, goal 0%
#
# DIAGNOSIS:
#   Goal maps to codebook entry #4 (99.8%), which is also wall's dominant code (81%).
#   100% of goal samples share codes with empty/wall. Even PRE-VQ continuous features
#   show goal=0%, meaning the encoder itself never learned to separate goal from empty.
#   Root cause: STE gradients from post-VQ semantic loss are too noisy for the 1.6%
#   prevalence of goal to override the majority class signal.
#
# KEY CHANGES FROM EXP 36:
#   1. --sem_pre_vq: semantic loss on pre-VQ encoder output (direct gradient, no STE)
#   2. --codebook_size 256 (was 64): more codes so goal can get its own entry
#   3. --sem_focal_gamma 0 (was 2.0): drop focal loss — it may over-focus on
#      already-hard cases; standard CE with class weights is more stable
#   All other settings identical to exp 36.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"
RUN_NAME="mf_e2e_semantic_doorkey_v5enc_prevq"
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
  --env_name MiniGrid-DoorKey-8x8-v0 \
  --ae_model_type vqvae \
  --ae_model_version 5 \
  --codebook_size 256 \
  --embedding_dim 64 \
  --filter_size 8 \
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
  --snapback_min_reward 0.1 \
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
  --sem_aux_start_reward 0.1 \
  --sem_class_weights \
  --sem_focal_gamma 0.0 \
  --sem_pre_vq
