#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"
DEFAULT_LOCAL_PYTHON="$PROJECT_ROOT/../miniforge3/envs/stvqvae/bin/python"
DEVICE="${DEVICE:-cuda}"
RUN_NAME="${RUN_NAME:-vae_encoder_frozen_ppo_v1}"
MODEL_DIR="${MODEL_DIR:-$PROJECT_ROOT/wm_runs/vae_wm_rl_v11_conservative_reward_longshort}"
MF_STEPS="${MF_STEPS:-5000000}"
NUM_ENVS="${NUM_ENVS:-16}"
BATCH_SIZE="${BATCH_SIZE:-4096}"
PPO_BATCH_SIZE="${PPO_BATCH_SIZE:-256}"

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

mkdir -p "$PROJECT_ROOT/.mplconfig"

echo "[$(date -u '+%Y-%m-%dT%H:%M:%SZ')] Starting vae_encoder_frozen_ppo_v1"
echo "PYTHON_BIN=$PYTHON_BIN"
echo "DEVICE=$DEVICE"
echo "RUN_NAME=$RUN_NAME"
echo "MODEL_DIR=$MODEL_DIR"
echo "MF_STEPS=$MF_STEPS NUM_ENVS=$NUM_ENVS BATCH_SIZE=$BATCH_SIZE PPO_BATCH_SIZE=$PPO_BATCH_SIZE"

cd "$PROJECT_ROOT/discrete_mbrl/model_free"
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
export TORCHDYNAMO_DISABLE=1
export MPLCONFIGDIR="$PROJECT_ROOT/.mplconfig"

exec "$PYTHON_BIN" -u train.py \
  --env_name MiniGrid-LavaCrossingS9N1-v0 \
  --model_dir "$MODEL_DIR" \
  --ae_model_type vae \
  --ae_model_version 2 \
  --ae_model_hash 393f184899f1c7bd6740a092b342902c \
  --latent_dim 128 \
  --embedding_dim 64 \
  --filter_size 9 \
  --batch_size "$BATCH_SIZE" \
  --learning_rate 3e-4 \
  --mf_steps "$MF_STEPS" \
  --num_envs "$NUM_ENVS" \
  --ppo_batch_size "$PPO_BATCH_SIZE" \
  --ppo_iters 20 \
  --ppo_clip 0.2 \
  --ppo_value_coef 0.5 \
  --ppo_entropy_coef 0.003 \
  --ppo_gae_lambda 0.95 \
  --ppo_norm_advantages \
  --ppo_max_grad_norm 0.5 \
  --ortho_init \
  --run_name "$RUN_NAME" \
  --device "$DEVICE" \
  --save
