#!/usr/bin/env bash
set -euo pipefail

# Experiment 33: mf_e2e_semantic_probe (Phase 2 — e2e training with semantic aux loss)
#
# BASELINE: Experiment 31 (mf_e2e_wm_aux_v2) — Peak 0.9988, Final 0.9986, Avg 0.9681
#
# HYPOTHESIS:
#   A small SemanticHead (64→64→ReLU→11) added as an auxiliary loss forces each
#   of the 81 spatial VQVAE tokens to predict the MiniGrid object type (wall/lava/
#   goal/empty/agent) at the corresponding grid cell. This grounds each spatial
#   code semantically, preventing the codebook from conflating visually similar
#   objects. The world model aux loss from exp #31 is retained for additive comparison.
#
# KEY CHANGES FROM EXP 31:
#   1. --use_semantic_aux enables SemanticLabelWrapper + SemanticHead
#   2. --sem_aux_coef 0.05 (half the wm_aux coefficient — conservative start)
#   3. --sem_head_hidden 64 (small MLP, 64→64→ReLU→11 per position)
#   World model aux kept: --use_world_model --wm_aux_coef 0.1
#
# NOTE: Run probe_semantics.py first (Phase 1) to diagnose existing codebook
# semantics without training, then run this script for Phase 2.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"
RUN_NAME="mf_e2e_semantic_aux"
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
  --env_name MiniGrid-LavaCrossingS9N1-v0 \
  --ae_model_type vqvae \
  --ae_model_version 2 \
  --ae_model_hash ea136dc75d389f7b850959cd1f78eb6a \
  --codebook_size 64 \
  --embedding_dim 64 \
  --filter_size 9 \
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
  --snapback_min_reward 0.6 \
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
  --sem_aux_coef 0.005 \
  --sem_head_hidden 64 \
  --sem_n_classes 11 \
  --sem_aux_start_reward 0.3
  # Note: --wm_standalone_train omitted (default=False)
  # Run 3 changes: sem_aux_coef 0.05->0.005, added reward gate at 0.3
