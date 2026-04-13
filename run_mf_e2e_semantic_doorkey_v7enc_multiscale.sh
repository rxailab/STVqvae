#!/usr/bin/env bash
set -euo pipefail

# Experiment 39: DoorKey-8x8 with v7 multi-scale encoder (generalised)
#
# BASELINE: Exp 38 (v6: RGB shortcut + coords) — Probe 91.1%, goal 98.8%
#           but door dropped to 59.4% and key to 81.8%
#
# PROBLEM WITH V6:
#   The v6 encoder was MiniGrid-specific: hand-crafted AvgPool2d RGB shortcut
#   with exact tile-size alignment, hard-coded x/y coordinate channels, and a
#   narrowed conv trunk (40 channels). This cracked goal but hurt door/key
#   because (a) averaging over 8x8 tiles washes out object texture, and
#   (b) the trunk lost capacity.
#
# V7 APPROACH (domain-agnostic):
#   1. Multi-scale skip connections from every intermediate backbone layer,
#      pooled and projected — a general mechanism that preserves both fine
#      texture (door, key) and coarse colour (goal) without assuming tile sizes.
#   2. Squeeze-and-Excitation (SE) channel attention — lets the network learn
#      per-sample which feature channels matter, replacing the hard-coded
#      colour/coord split.
#   3. Learnable positional embeddings (like ViT) instead of coordinate grids.
#   4. Full-width backbone (64 channels in last conv) restores capacity.
#
# All other settings match exp 38 to isolate the encoder change.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"
RUN_NAME="mf_e2e_semantic_doorkey_v7enc_multiscale"
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
  --ae_model_version 7 \
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
