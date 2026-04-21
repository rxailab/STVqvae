#!/usr/bin/env bash
set -euo pipefail

# Experiment 46: Crafter — fix mid-frequency class collapse (stone/path/sand/tree)
#
# PROBLEM (Exp 45):
#   Best RL reward = 8.2 (new record). Trained sem_head: 35.7% overall.
#   grass (72.4%), water (24.3%), plant (33.1%) now properly detected.
#   BUT stone (0.3%), path (0.5%), sand (0%), tree (0%) all collapsed.
#
#   Root cause: absent classes (table, furnace, zombie, skeleton) inflated the
#   normalization mean, crushing ALL common-class weights to 0.1. With equal
#   weight, the loss is dominated by volume — grass (42%) + water (22%) = 64%
#   of gradient, leaving stone/path/sand/tree with too weak a signal.
#
# FIX (this experiment):
#   1. Normalize class weights over PRESENT classes only (count > 50).
#      Absent classes get weight=1.0 (neutral). This prevents absent classes
#      from crushing all common-class weights through the mean.
#   2. sem_class_weight_power=0.75: Stronger than 0.5 (Exp 45), gentler than
#      1.0 (Exp 44). With present-only normalization:
#        stone/grass ratio ≈ 2.8× (was 1× in Exp 45, 70× in Exp 44)
#        fence/grass ratio ≈ 31× (was 4× in Exp 45, 70× in Exp 44)
#   3. sem_focal_gamma=1.0: Mild focal loss to focus on hard examples
#      (stone/grass confusion, path/sand distinction).
#   4. No --sem_pre_vq: Apply semantic loss to QUANTIZED features (post-VQ)
#      so the codebook structure is directly influenced by semantic gradients.
#      In Exp 44-45, pre-VQ features had semantic info but the codebook didn't.
#   5. Keep all Exp 44-45 fixes: dead_code_threshold=2.0, codebook_size=1024.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"
RUN_NAME="mf_e2e_semantic_crafter_v6enc_cal2"
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
  --sem_class_weight_power 0.75 \
  --sem_focal_gamma 1.0
