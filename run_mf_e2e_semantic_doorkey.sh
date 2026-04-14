#!/usr/bin/env bash
set -euo pipefail

# Experiment 35: DoorKey-8x8 with improved semantic architecture (SemanticHeadV2)
#
# BASELINE: Exp 32 (DoorKey wm-only) — Peak 0.9988, Final 0.9979, Avg 0.9979
#
# HYPOTHESIS:
#   DoorKey-8x8 requires the agent to find a key, pick it up, open a door, and
#   reach the goal — semantically rich task where distinguishing door/key/goal/wall
#   from visual features alone is harder than LavaCrossing. An improved SemanticHeadV2
#   with positional encoding, local 3x3 context conv, deeper MLP, and focal loss
#   should produce strongly semantically-grounded representations that aid policy
#   learning on this task.
#
# KEY CHANGES FROM EXP 32 (DoorKey wm-only):
#   1. SemanticHeadV2: pos encoding + 3x3 depth-wise conv + 3-layer MLP + LayerNorm
#   2. sem_aux_coef = 0.05 (aggressive — proven safe on LavaCrossing)
#   3. Focal loss (gamma=2) + class weights: focuses on hard/rare objects
#   4. Gate at 0.1 (early activation)
#   5. DoorKey has 7+ active classes: empty, wall, floor, door, key, goal, agent
#
# KEY CHANGES FROM EXP 34 (LavaCrossing sem v2):
#   1. Environment: DoorKey-8x8 (harder, more semantic diversity)
#   2. No pretrained VQVAE (random init, like exp 32)
#   3. SemanticHeadV2 (v1 in exp 34)
#   4. Higher coef (0.05 vs 0.01) + focal loss (gamma=2)
#   5. snapback_min_reward=0.1 (matches exp 32 for DoorKey difficulty)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"
RUN_NAME="mf_e2e_semantic_doorkey"
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
  --ae_model_version 2 \
  --codebook_size 64 \
  --embedding_dim 64 \
  --filter_size 9 \
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
