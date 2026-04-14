#!/usr/bin/env bash
set -euo pipefail

# Experiment 34: mf_e2e_semantic_aux_v2 (improved semantic grounding)
#
# BASELINE: Experiment 33 run 3 — Peak 0.9988, Final 0.9985, Avg 0.9592
#           Probe: 85.2% overall (wall 92.7%, lava 46.1%, goal 0%, agent 1.1%)
#
# PROBLEM: Class imbalance. Unweighted CE is dominated by empty (51%) and wall (40%).
#   Rare but critical classes (lava 7%, goal 1%, agent 1%) get negligible gradient.
#
# CHANGES FROM EXP 33:
#   1. Inverse-frequency class weights (capped at 20x) in semantic CE loss
#      - lava ~7x upweight, goal/agent ~20x (capped)
#   2. sem_aux_coef 0.005 -> 0.01 (2x increase; stronger semantic signal)
#   3. sem_aux_start_reward 0.3 -> 0.1 (earlier activation; more training budget)
#   4. Checkpoint now saves sem_head + trans_model weights
#
# EXPECTED: lava probe acc 46% -> 80%+, goal/agent > 0%, overall > 90%
#           RL performance should stay at ~0.999 peak (coef still conservative)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"
RUN_NAME="mf_e2e_semantic_aux_v2"
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
  --env_name MiniGrid-LavaCrossingS9N1-v0 \
  --ae_model_type vqvae \
  --ae_model_version 2 \
  --ae_model_hash ea136dc75d389f7b850959cd1f78eb6a \
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
  --snapback_min_reward 0.6 \
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
  --sem_aux_coef 0.01 \
  --sem_head_hidden 64 \
  --sem_n_classes 11 \
  --sem_aux_start_reward 0.1 \
  --sem_class_weights
