#!/usr/bin/env bash
set -euo pipefail

# Experiment 30: mf_e2e_wm_aux_v1
#
# BASELINE: Experiment 29 (mf_e2e_snapback_v1)
#   - E2E PPO + pretrained VQVAE + snapback
#   - Result: 0.9988 peak, 0.9984 final
#
# WHAT'S NEW:
#   World model auxiliary predictive loss. A discrete transition model is
#   trained online (on real PPO rollout data) to predict next-state tokens
#   from current-state + action. The prediction loss (cross-entropy on
#   discrete latent indices) is added to the PPO objective with coefficient
#   --wm_aux_coef, and gradients flow back through the VQVAE encoder via
#   straight-through estimator.
#
#   This shapes the encoder to produce representations that are:
#     1. Useful for reward prediction (from PPO policy/value loss)
#     2. Temporally predictable (from world model auxiliary loss)
#
#   The transition model is also trained independently via its own optimizer
#   to keep its predictions on-distribution as the encoder evolves.
#
# HYPOTHESIS:
#   The auxiliary predictive loss provides a complementary learning signal
#   that may improve sample efficiency and/or final performance by
#   encouraging the encoder to learn dynamics-aware features.
#
# All other hyperparameters identical to exp 29.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"
RUN_NAME="mf_e2e_wm_aux_v1"
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
  --trans_learning_rate 1e-3
