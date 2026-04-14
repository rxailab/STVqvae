#!/usr/bin/env bash
set -euo pipefail

# Experiment 24: vqvae_wm_compact_v1
#
# Core hypothesis: The 81-token VQVAE spatial grid (filter_size=9) is too complex
# for the transition model. Even 1-step teacher-forced loss on policy trajectories
# was ~475 (experiment 16 diagnostic). Reducing to 9 tokens (filter_size=3) with a
# richer codebook (256 entries, 128-dim embeddings) makes the prediction target 9x
# simpler while preserving representational capacity.
#
# Changes vs vqvae_wm_rl_v2_bfs (experiment 15) and v3_shortroll_inpolicy (experiment 16):
#   - filter_size: 9 -> 3 (81 -> 9 spatial tokens)
#   - codebook_size: 64 -> 256 (richer vocabulary to compensate)
#   - embedding_dim: 64 -> 128 (richer per-token features)
#   - trans_hidden: 256 -> 512, trans_depth: 3 -> 5 (match v5 capacity)
#   - trans_epochs: 40 -> 80 (more training for the new encoder)
#   - n_train_unroll: 8 -> 8 (keep same, report multi-step losses)
#   - rl_unroll_steps: 5 (short, within model's reliable range)
#   - rl_eval_freq: 5000 (frequent real-env eval to catch transfer)
#   - rl_train_steps: 600000

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"
RUN_DIR="${RUN_DIR:-$PROJECT_ROOT/wm_runs/vqvae_wm_compact_v1}"
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
  --filter_size 3 \
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
