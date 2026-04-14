#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"
RUN_DIR="${RUN_DIR:-$PROJECT_ROOT/wm_runs/vae_wm_rl_v6_diag}"
DEFAULT_LOCAL_PYTHON="$PROJECT_ROOT/../miniforge3/envs/stvqvae/bin/python"
DEVICE="${DEVICE:-cuda}"
N_EPISODES="${N_EPISODES:-25}"
MAX_STEPS="${MAX_STEPS:-500}"
HORIZONS="${HORIZONS:-1 3 5 8 12}"

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
      if command -v python3 >/dev/null 2>&1 && python3 -c "import torch, stable_baselines3" >/dev/null 2>&1; then
        printf '%s\n' "python3"
        return 0
      fi
      continue
    fi

    if [[ -x "$candidate" ]] && "$candidate" -c "import torch, stable_baselines3" >/dev/null 2>&1; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done

  return 1
}

PYTHON_BIN="$(pick_python || true)"

if [[ -z "$PYTHON_BIN" ]]; then
  cat >&2 <<'EOF'
Could not find a Python interpreter with torch and stable_baselines3.
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

exec "$PYTHON_BIN" -u diagnose_transition_consistency_continuous.py \
  --env_name MiniGrid-LavaCrossingS9N1-v0 \
  --policy_path "$PROJECT_ROOT/discrete_mbrl/models/MiniGrid-LavaCrossingS9N1-v0/ppo_world_model.zip" \
  --model_dir "$PROJECT_ROOT/wm_runs/vae_wm_rl_v5_curriculum" \
  --ae_model_type vae \
  --ae_model_version 2 \
  --ae_model_hash 393f184899f1c7bd6740a092b342902c \
  --latent_dim 128 \
  --embedding_dim 64 \
  --filter_size 9 \
  --trans_model_type continuous \
  --trans_model_version 1 \
  --trans_model_hash a4934ab750b13c3dc13e17f928e5cfe4 \
  --trans_hidden 512 \
  --trans_depth 5 \
  --device "$DEVICE" \
  --n_episodes "$N_EPISODES" \
  --max_steps "$MAX_STEPS" \
  --horizons $HORIZONS \
  --output_json "$RUN_DIR/results.json"
