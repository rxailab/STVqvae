#!/usr/bin/env bash
set -euo pipefail

# Experiment 36: DoorKey-8x8 with strided encoder v5 (no AdaptiveAvgPool2d)
#
# BASELINE: Exp 35 (DoorKey + SemanticHeadV2 + v2 encoder) — Probe 58.4%
# CONTROL:  Exp 32 (DoorKey wm-only + v2 encoder) — Probe 62.4%
#
# HYPOTHESIS:
#   The v2 encoder's AdaptiveAvgPool2d upsamples 5×5 → 9×9, creating blurry
#   interpolated features where spatial boundaries between grid cells are lost.
#   The v5 strided encoder produces 8×8 output from 64×64 input using clean
#   stride-2 convolutions (no pooling). Each of the 64 latent tokens maps to
#   exactly one 8×8-pixel tile = one DoorKey grid cell.
#
#   This 1:1 alignment should dramatically improve semantic probe accuracy,
#   especially for small/rare objects (key, door, goal, agent) that get averaged
#   away by adaptive pooling.
#
# KEY CHANGES FROM EXP 35:
#   1. ae_model_version=5 (strided conv encoder, no AdaptiveAvgPool2d)
#   2. filter_size=8 (natural 8×8 output for DoorKey-8x8, NOT 9)
#   3. Semantic labels no longer need resize (grid 8×8 == latent 8×8)
#   All other hyperparameters kept identical for fair comparison.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"
RUN_NAME="mf_e2e_semantic_doorkey_v5enc"
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
  --codebook_size 64 \
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
  --sem_focal_gamma 2.0
