#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"
RUN_DIR="${RUN_DIR:-$PROJECT_ROOT/wm_runs/vae_wm_rl_v3_actionfix_seq}"
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
Tried:
  - $PYTHON_BIN (if set)
  - /storage/hpc/11/xiar3/vit5/bin/python
  - /opt/miniconda3/envs/vit5/bin/python
  - /home/xiar3/miniconda3/envs/vit5/bin/python
  - /home/xiar3/.conda/envs/vit5/bin/python
  - python3

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
  --ae_model_type vae \
  --ae_model_version 2 \
  --trans_model_type continuous \
  --trans_model_version 1 \
  --latent_dim 128 \
  --filter_size 9 \
  --epochs 20 \
  --trans_epochs 40 \
  --batch_size 1024 \
  --eval_batch_size 128 \
  --n_preload 0 \
  --no_load \
  --n_train_unroll 8 \
  --rl_unroll_steps 40 \
  --rl_train_steps 600000 \
  --device "$DEVICE" \
  --save
