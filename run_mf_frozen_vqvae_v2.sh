#!/usr/bin/env bash
set -euo pipefail

# Experiment 28: mf_frozen_vqvae_v2
#
# After 15 world-model experiments (experiments 13-27) all producing 0.000
# real-env transfer, returning to the proven approach:
#
#   Experiment 12 (vqvae_preinit_snapback_ppo) achieved:
#     - Peak reward:  0.9988
#     - Final avg:    0.8988
#   using model-free PPO with a frozen pretrained VQVAE encoder.
#
# This experiment replicates that approach cleanly using the compact_v2/v3
# encoder (filter_size=5, 25 tokens, codebook_size=256) which has better
# spatial resolution than the original (filter_size=9, 81 tokens).
#
# The VQVAE encoder is pretrained on the replay buffer and then frozen.
# PPO is trained end-to-end in the REAL environment (no world model).
# No e2e_loss — encoder weights do not update during PPO training.
#
# The "world model" experiments revealed a useful insight: the compact
# VQVAE encoder (filter_size=5) produces good representations. We use
# that encoder here, trained the same way (50 epochs), then run pure
# model-free PPO.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"
RUN_DIR="${RUN_DIR:-$PROJECT_ROOT/wm_runs/mf_frozen_vqvae_v2}"
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

mkdir -p "$RUN_DIR"
mkdir -p "$PROJECT_ROOT/.mplconfig"

cd "$PROJECT_ROOT/discrete_mbrl"
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
export TORCHDYNAMO_DISABLE=1
export MPLCONFIGDIR="$PROJECT_ROOT/.mplconfig"

# Train encoder only (no transition model, no world-model RL)
# Then run pure model-free PPO on the real env with the frozen encoder.
# rl_train_steps drives model-free PPO via train_rl_model with no trans model.
# Setting trans_model_type to none-equivalent: skip trans training entirely
# by using rl_train_steps > 0 with the existing pipeline but no trans model.
#
# Actually the cleanest path: train encoder, skip transition, run rl with
# ObsEncoderWrapperGymnasium directly.  We use the existing full_train_eval
# pipeline but pass --rl_train_steps and let it call train_rl_model which
# will construct the imagined env — EXCEPT we set rl_unroll_steps=-1 which
# means the world model env uses NO time limit and resets from real states
# every step.  With no transition model loaded (trans_epochs=0 skips it),
# we need a different approach.
#
# Cleanest: use the Dyna flag with n_imagined_envs=0 — pure real envs.

exec "$PYTHON_BIN" -u full_train_eval.py \
  --env_name MiniGrid-LavaCrossingS9N1-v0 \
  --model_dir "$RUN_DIR" \
  --ae_model_type vqvae \
  --ae_model_version 2 \
  --trans_model_type discrete \
  --trans_model_version 1 \
  --codebook_size 256 \
  --embedding_dim 128 \
  --filter_size 5 \
  --epochs 50 \
  --trans_epochs 1 \
  --trans_hidden 512 \
  --trans_depth 5 \
  --batch_size 1024 \
  --eval_batch_size 128 \
  --n_preload 0 \
  --no_load \
  --n_train_unroll 8 \
  --dyna \
  --n_real_envs 24 \
  --n_imagined_envs 0 \
  --dyna_horizon 3 \
  --ppo_n_steps 1024 \
  --rl_train_steps 5000000 \
  --rl_finetune_steps 0 \
  --rl_eval_freq 50000 \
  --rl_eval_episodes 20 \
  --rl_eval_max_episode_steps 500 \
  --device "$DEVICE" \
  --save
