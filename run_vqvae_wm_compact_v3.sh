#!/usr/bin/env bash
set -euo pipefail

# Experiment 26: vqvae_wm_compact_v3
#
# Diagnostic findings from compact_v2 (filter_size=5, 25 tokens,
# rl_unroll_steps=20):
#
#   Problem: rl_unroll_steps=20 is far outside the model's reliable
#   open-loop prediction horizon.
#
#   compact_v2 diagnostic (diagnose_compact_v2.py on policy_20k buffer):
#     Horizon  TF loss  TF acc   OL loss  OL acc
#          1   0.3792   0.9998    0.3792  0.9998   ← reliable
#          3   0.3791   0.9998    2.3242  0.9792   ← marginal
#          5   0.3792   0.9998    7.9409  0.9410   ← degraded
#         10   0.3792   0.9998   23.5861  0.8597   ← corrupted
#
#   At horizon 20+ (which rl_unroll_steps=20 uses), states are heavily
#   corrupted (~4-5 tokens wrong per state). PPO learned to navigate a
#   hallucinated dreamscape that doesn't match real-env dynamics.
#
#   The imagined ep_rew_mean=0.5-0.6 was an artifact: PPO exploited
#   hallucinated rewards in corrupted imagined states far from real dynamics.
#   All 97 real-env evals returned 0.000.
#
# Root cause diagnosis:
#   compact_v2 made the WRONG fix by increasing rl_unroll_steps from 5 to 20.
#   The reliable open-loop zone is ≤3 steps. Even at horizon 5, the OL loss
#   is already 21× higher than teacher-forced.
#
# Fix (targeted, minimal change from compact_v2):
#   - rl_unroll_steps: 20 → 5  (back within the reliable prediction zone,
#     OL acc=94.1% vs 86% at h=10 and much worse at h=20)
#
# Why rl_unroll_steps=5 should work for rewards:
#   The training buffer includes BFS data with goal-reaching transitions.
#   Imagined rollouts starting from states 1-5 steps from the goal will
#   encounter the goal in the reliable prediction horizon. Expected episode
#   reward ≈ 5 × P(in reliable zone with goal) > 0 when initialised from
#   BFS trajectories near the goal.
#
# All other settings identical to compact_v2 (filter_size=5 confirmed good).

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"
RUN_DIR="${RUN_DIR:-$PROJECT_ROOT/wm_runs/vqvae_wm_compact_v3}"
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
  --rl_unroll_steps 5 \
  --rl_train_steps 600000 \
  --rl_eval_freq 5000 \
  --rl_eval_episodes 25 \
  --device "$DEVICE" \
  --save
