#!/usr/bin/env bash
set -euo pipefail

# Experiment 27: vqvae_wm_dyna_v1
#
# ROOT CAUSE OF ALL PRIOR WORLD-MODEL FAILURES (experiments 13-26):
#   PPO trained in pure imagination NEVER sees real rewards. The transition
#   model's reward predictions hallucinate, and compounding state errors
#   make imagined trajectories diverge from reality. The policy learns to
#   navigate a dream that doesn't exist. All 13 experiments: 0.000 real transfer.
#
# FIX: Dyna-style training (Sutton 1991, MBPO 2019)
#   Train PPO on a MIX of real and imagined environments simultaneously:
#   - 4 real envs: provide grounded reward signal from the actual MDP
#   - 8 imagined envs: provide data augmentation via short-horizon (3-step)
#     rollouts from fresh real states, within the reliable OL zone (97.9% acc)
#
#   The PPO collects experience from ALL 12 envs each rollout. Real envs
#   provide the sparse-but-real reward signal that the policy needs to learn
#   actual navigation. Imagined envs provide 2x more training data per update.
#
#   After Dyna training, an optional Phase 2 finetunes on pure real envs
#   to squeeze out any remaining transfer gap.
#
# WHY THIS SHOULD WORK:
#   Model-free PPO with frozen VQVAE (experiment 12) achieved 0.9988 in 5M steps.
#   Dyna keeps the same real-reward signal but augments it with imagined data,
#   which should improve sample efficiency. Even if the imagined data adds zero
#   value, the 4 real envs alone replicate the proven model-free approach.
#
# KEY SETTINGS (encoder + transition unchanged from compact_v2/v3):
#   --dyna                  : enable mixed real+imagined training
#   --n_real_envs 4         : 4 real environments
#   --n_imagined_envs 8     : 8 imagined environments
#   --dyna_horizon 3        : max 3 imagined steps (within 97.9% OL accuracy zone)
#   --rl_train_steps 2000000: 2M total steps (across 12 envs = ~167k per env)
#   --rl_finetune_steps 500000: 500k additional pure real-env finetuning

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"
RUN_DIR="${RUN_DIR:-$PROJECT_ROOT/wm_runs/vqvae_wm_dyna_v1}"
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
    "/opt/miniconda3/envs/vit5/bin/python"
    "/home/xiar3/miniconda3/envs/vit5/bin/python"
    "/home/xiar3/.conda/envs/vit5/bin/python"
    "python3"
  )

  local candidate
  for candidate in "${candidates[@]}"; do
    if [[ "$candidate" == "python3" ]]; then
      if command -v python3 >/dev/null 2>&1 && python3 -c "import torch" >/dev/null 2>&1; then
        printf '%s\n' "python3"
        return 0
      fi
      continue
    fi

    if [[ -x "$candidate" ]] && "$candidate" -c "import torch" >/dev/null 2>&1; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done

  return 1
}

PYTHON_BIN="$(pick_python || true)"

if [[ -z "$PYTHON_BIN" ]]; then
  cat >&2 <<'EOF'
Could not find a Python interpreter with torch.
Set PYTHON_BIN to the correct interpreter and rerun.
EOF
  exit 1
fi

mkdir -p "$RUN_DIR"
mkdir -p "$PROJECT_ROOT/.mplconfig"

cd "$PROJECT_ROOT/discrete_mbrl"
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
export TORCHDYNAMO_DISABLE=1
export MPLCONFIGDIR="$PROJECT_ROOT/.mplconfig"

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
  --trans_epochs 80 \
  --trans_hidden 512 \
  --trans_depth 5 \
  --batch_size 1024 \
  --eval_batch_size 128 \
  --n_preload 0 \
  --no_load \
  --n_train_unroll 8 \
  --dyna \
  --n_real_envs 8 \
  --n_imagined_envs 16 \
  --dyna_horizon 3 \
  --ppo_n_steps 1024 \
  --rl_train_steps 3000000 \
  --rl_finetune_steps 3000000 \
  --rl_eval_freq 50000 \
  --rl_eval_episodes 20 \
  --rl_eval_max_episode_steps 500 \
  --device "$DEVICE" \
  --save
