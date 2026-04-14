#!/usr/bin/env bash
set -euo pipefail

# Experiment 25: vqvae_wm_compact_v2
#
# Diagnoses from compact_v1 (filter_size=3, 9 tokens):
#
#   Problem 1 — representation too coarse:
#     Each token covers a 3×3 region of the 9×9 MiniGrid grid. A single-step
#     agent move often doesn't cross a token boundary, so the state looks static.
#     The transition model trivially learned "predict no change" → flat 0.21 loss
#     at all horizons, but PPO gets no meaningful state signal from it.
#     ep_rew_mean inside the world model stayed at 0.07–0.14 the entire run.
#
#   Problem 2 — imagined rollouts too short for sparse rewards:
#     With rl_unroll_steps=5 and only 1.8% positive reward rate in replay data,
#     expected reward per imagined episode = 5 × 0.018 = 0.09. PPO starves.
#
# Fixes:
#   - filter_size: 3 → 5  (9 → 25 tokens, ~1.8 grid cells/token)
#     Most single-step moves now cross a token boundary → visible state changes.
#     MLP output grows from 2304 to 6400 logits — still well within learnable range.
#
#   - rl_unroll_steps: 5 → 20
#     Longer imagined episodes give PPO more steps to encounter the goal.
#     Safe because compact_v1 proved flat transition loss across all 8 steps;
#     filter_size=5 should be at least as stable.
#
# Everything else identical to compact_v1.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"
RUN_DIR="${RUN_DIR:-$PROJECT_ROOT/wm_runs/vqvae_wm_compact_v2}"
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
  --rl_unroll_steps 20 \
  --rl_train_steps 600000 \
  --rl_eval_freq 5000 \
  --rl_eval_episodes 25 \
  --device "$DEVICE" \
  --save
