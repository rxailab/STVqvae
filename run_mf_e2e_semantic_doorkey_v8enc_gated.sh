#!/usr/bin/env bash
set -euo pipefail

# Experiment 40: DoorKey-8x8 with v8 gated input-skip encoder
#
# BASELINE: Exp 38 (v6) — 91.1% overall, goal 98.8%, but door 59.4%, key 81.8%
#           Exp 39 (v7) — 78.3% overall, goal 94%, key 99.1%, but wall 78.5%
#
# DIAGNOSIS:
#   v5's full-width trunk (64ch) is best for wall/door/key/agent.
#   v6's input pooling cracked goal but its narrow trunk (40ch) hurt door/key.
#   v7's SE + positions + multi-scale skips hurt wall badly (96% -> 78%).
#
# V8 APPROACH — minimal intervention on v5:
#   1. Keep v5 trunk EXACTLY (64->128->64, stride-2 x3) — proven best for
#      structural classes.
#   2. Add ONE parallel path: pooled input -> 2-layer 1x1 projection (16ch).
#      This is the domain-agnostic version of v6's RGB shortcut.
#   3. Gated fusion: a learned sigmoid gate controls per-token blending.
#      gate=0 → pure trunk (v5 behaviour); gate>0 → skip is mixed in.
#      The network learns to open the gate for tokens needing raw input
#      identity (goal) and close it for tokens where the trunk suffices.
#   4. No SE, no coordinates, no multi-scale — those hurt in v7.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"
RUN_NAME="mf_e2e_semantic_doorkey_v8enc_gated"
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
  --env_name MiniGrid-DoorKey-8x8-v0 \
  --ae_model_type vqvae \
  --ae_model_version 8 \
  --codebook_size 256 \
  --embedding_dim 64 \
  --filter_size 8 \
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
  --snapback_min_reward 0.1 \
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
  --use_semantic_aux \
  --sem_aux_coef 0.08 \
  --sem_head_hidden 128 \
  --sem_head_version 2 \
  --sem_n_classes 11 \
  --sem_aux_start_reward 0.0 \
  --sem_class_weights \
  --sem_focal_gamma 0.0 \
  --sem_pre_vq
