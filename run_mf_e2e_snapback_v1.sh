#!/usr/bin/env bash
set -euo pipefail

# Experiment 29: mf_e2e_snapback_v1
#
# ROOT CAUSE of exp 28 (mf_frozen_vqvae_v2) FAILURE:
#   Experiment 28 replicated the WRONG approach. It mirrored exp 11
#   (frozen VQVAE + PPO), which also failed (0.1927 best, 0.0000 final).
#
#   Frozen VQVAE fails because:
#     1. Random-policy data barely reaches the goal (~1-2% success rate).
#        VQVAE codebook encodes non-goal states. "One step from goal" and
#        "far from goal" may quantize to identical tokens.
#     2. Reconstruction loss != task-relevant features. Navigation needs
#        relative agent-to-goal position; reconstruction needs all pixel detail.
#     3. No RL signal shapes the representation. Frozen = useless for control.
#
# WHAT ACTUALLY WORKED (Experiment 12: vqvae_preinit_snapback_ppo):
#   - Pretrained VQVAE loaded as starting point (hash ea136dc7)
#   - e2e gradients: PPO gradients flow through VQVAE encoder
#   - Very low encoder LR (1e-5 vs 1e-4 for policy): slow representation drift
#   - Cosine schedule: encoder LR decays to 0 → eventually quasi-frozen
#   - Snapback: if reward drops > 50% from peak, restore encoder to best
#     state and freeze permanently
#   - Result: 0.9988 peak, 0.8988 final (best result in all 28 experiments)
#
# This experiment replicates exp 12 EXACTLY using the same:
#   - Pretrained VQVAE: model_ea136dc75d389f7b850959cd1f78eb6a.pt
#   - Architecture: filter_size=9, codebook_size=64, embedding_dim=64
#   - Script: discrete_mbrl/model_free/train.py
#   - Hyperparameters: identical to exp 12
#
# Goal: confirm replicability on current hardware before exploring variants.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"
RUN_NAME="mf_e2e_snapback_v1"
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

# model_free/train.py saves models relative to its CWD:
#   ./models/{env_name}/{run_name}_{suffix}_model.pt
# --model_dir .. tells construct_ae_model to look for pretrained encoder in:
#   ../models/{env_name}/model_{hash}.pt
# = discrete_mbrl/models/MiniGrid-LavaCrossingS9N1-v0/model_ea136dc7...pt (EXISTS)

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
  --save
