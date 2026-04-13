#!/usr/bin/env bash
set -euo pipefail

# Experiment 42: DoorKey-16x16 with v6 encoder + dead-code restart
#
# SCALING TEST: Move from DoorKey-8x8 to DoorKey-16x16 with the best encoder.
#
# WHY THIS MATTERS:
#   DoorKey-16x16 is significantly harder than 8x8:
#     - 256 tokens vs 64 (4× larger state space)
#     - Rare classes are 0.4% of tokens (vs 1.6% in 8x8)
#     - Longer exploration horizon (16x16 rooms, more steps to solve)
#     - VQ codebook pressure: 256 tokens/step dominate EMA statistics
#
# KEY CHANGES:
#   1. Environment: DoorKey-16x16 (128x128 images, 16x16 grid = 256 tokens)
#   2. Encoder: v6 (best proven: 88.9% overall on 8x8)
#   3. Dead-code restart in VQ: codes with usage < 1.0 are reinitialized
#      from random encoder outputs — prevents codebook collapse on rare classes
#   4. Codebook size 512 (up from 256) — more codes for more diverse 16x16 grids
#   5. filter_size=16 for 1:1 token-to-cell alignment on 16x16 grid
#   6. 8M steps (up from 5M) — harder task needs more training

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"
RUN_NAME="mf_e2e_semantic_doorkey16_v6enc_restart"
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
  --env_name MiniGrid-DoorKey-16x16-v0 \
  --ae_model_type vqvae \
  --ae_model_version 6 \
  --codebook_size 512 \
  --embedding_dim 64 \
  --filter_size 16 \
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
  --sem_pre_vq
