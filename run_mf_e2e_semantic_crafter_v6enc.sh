#!/usr/bin/env bash
set -euo pipefail

# Experiment 43: Crafter with v6 encoder (color-coord shortcut)
#
# FIRST NON-MINIGRID EXPERIMENT: Test whether the v6 encoder + semantic
# auxiliary approach transfers to a fundamentally different environment.
#
# WHY CRAFTER:
#   - 2D survival game with rich semantic diversity (19 classes)
#   - Materials (water, grass, stone, tree, lava, coal, iron, diamond, ...)
#   - Objects (player, cow, zombie, skeleton, arrow, plant)
#   - 64×64 RGB observations (same input pipeline as MiniGrid)
#   - Built-in info['semantic'] provides ground-truth 64×64 semantic maps
#   - 17 discrete actions (movement, crafting, combat)
#   - Much harder exploration than DoorKey: open-ended survival
#
# KEY SETTINGS:
#   1. v6 encoder: RGB shortcut works on any image — pools average color per
#      tile, which distinguishes water (blue) from grass (green) from stone
#      (grey) just as it distinguishes goal (green) from wall (grey) in MiniGrid.
#   2. filter_size=8: 64/8 = 8×8 latent grid (64 tokens). Each token maps to
#      an 8×8 pixel patch. Crafter's procedural terrain is spatially varied
#      enough that 64 tokens should capture the local scene.
#   3. sem_n_classes=19: Crafter has 19 semantic classes (0=None through 18=Plant)
#   4. codebook_size=512: Crafter has more visual variety than DoorKey — more
#      codes needed to represent water/grass/stone/tree/lava + all objects.
#   5. 8M steps: Open-ended exploration needs more training than key-door task.
#   6. ppo_entropy_coef=0.02: Higher entropy for exploration-heavy environment.
#   7. snapback_min_reward=0.0: Crafter reward is sparse, don't gate on it.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"
RUN_NAME="mf_e2e_semantic_crafter_v6enc"
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
  --env_name crafter \
  --ae_model_type vqvae \
  --ae_model_version 6 \
  --codebook_size 512 \
  --embedding_dim 64 \
  --filter_size 8 \
  --mf_steps 8000000 \
  --batch_size 4096 \
  --num_envs 16 \
  --ppo_iters 10 \
  --ppo_batch_size 64 \
  --ppo_entropy_coef 0.02 \
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
  --snapback_min_reward 0.0 \
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
  --sem_n_classes 19 \
  --sem_aux_start_reward 0.0 \
  --sem_class_weights \
  --sem_focal_gamma 0.0 \
  --sem_pre_vq
