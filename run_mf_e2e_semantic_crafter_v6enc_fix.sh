#!/usr/bin/env bash
set -euo pipefail

# Experiment 44: Crafter with v6 encoder — FIX codebook collapse
#
# PROBLEM (Exp 43):
#   Semantic probe showed 42.1% overall accuracy — everything classified as
#   Grass. 14 out of 15 classes at 0% probe accuracy. Root cause:
#     1. VectorQuantizerEMA had NO dead-code restart, so dominant classes
#        (grass=42% of pixels) absorbed all EMA mass, killing rare codes.
#     2. sem_aux_coef=0.08 was far too weak for 15 diverse terrain classes.
#     3. codebook_size=512 may be insufficient for Crafter's visual variety.
#
# FIX (this experiment):
#   1. dead_code_threshold=2.0: Codes with EMA usage < 2.0 get replaced
#      with random encoder outputs. This prevents codebook collapse.
#   2. sem_aux_coef=0.5: 6x stronger semantic grounding signal — needed
#      because Crafter has 15 visually distinct classes vs MiniGrid's 5-6.
#   3. codebook_size=1024: Double the codes for richer visual vocabulary.
#   4. sem_focal_gamma=1.0: Mild focal loss to down-weight easy (grass)
#      examples and up-weight rare classes (diamond, lava, arrow).
#
# UNCHANGED: v6 encoder, filter_size=8 (8x8 grid), 8M steps, PPO settings.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"
RUN_NAME="mf_e2e_semantic_crafter_v6enc_fix"
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
  --sem_focal_gamma 1.0 \
  --sem_pre_vq
