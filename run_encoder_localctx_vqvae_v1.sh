#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"
RUN_DIR="${RUN_DIR:-$PROJECT_ROOT/wm_runs/encoder_localctx_vqvae_v1}"
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

echo "[$(date -u '+%Y-%m-%dT%H:%M:%SZ')] Starting encoder_localctx_vqvae_v1"
echo "RUN_DIR=$RUN_DIR"
echo "PYTHON_BIN=$PYTHON_BIN"
echo "DEVICE=$DEVICE"

cd "$PROJECT_ROOT/discrete_mbrl"
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
export TORCHDYNAMO_DISABLE=1
export MPLCONFIGDIR="$PROJECT_ROOT/.mplconfig"

"$PYTHON_BIN" -u train_encoder.py \
  --env_name MiniGrid-LavaCrossingS9N1-v0 \
  --model_dir "$RUN_DIR" \
  --ae_model_type local_ctx_vqvae \
  --ae_model_version 2 \
  --embedding_dim 128 \
  --codebook_size 256 \
  --filter_size 9 \
  --ctx_channels 128 \
  --ctx_cond_type concat \
  --ctx_aux_coef 2.0 \
  --entropy_penalty_coef 0.05 \
  --code_dropout_rate 0.15 \
  --mae_mask_ratio 0.5 \
  --mae_patch_size 4 \
  --mae_loss_coef 1.0 \
  --epochs 40 \
  --batch_size 1024 \
  --eval_batch_size 128 \
  --n_preload 0 \
  --no_load \
  --device "$DEVICE" \
  --save

AE_HASH="$("$PYTHON_BIN" - <<'PY'
import os, torch
run_dir = "/home/xiar3/experiments/STVqvae/wm_runs/encoder_localctx_vqvae_v1/models/MiniGrid-LavaCrossingS9N1-v0"
files = sorted([f for f in os.listdir(run_dir) if f.startswith("model_") and f.endswith(".pt")], key=lambda f: os.path.getmtime(os.path.join(run_dir, f)))
print(files[-1][6:-3])
PY
)"

echo "VALIDATING_AE_HASH=$AE_HASH"

exec "$PYTHON_BIN" -u validate_encoder_checkpoint.py \
  --env_name MiniGrid-LavaCrossingS9N1-v0 \
  --model_dir "$RUN_DIR" \
  --ae_model_type local_ctx_vqvae \
  --ae_model_version 2 \
  --ae_model_hash "$AE_HASH" \
  --embedding_dim 128 \
  --codebook_size 256 \
  --filter_size 9 \
  --ctx_channels 128 \
  --ctx_cond_type concat \
  --ctx_aux_coef 2.0 \
  --entropy_penalty_coef 0.05 \
  --code_dropout_rate 0.15 \
  --mae_mask_ratio 0.5 \
  --mae_patch_size 4 \
  --mae_loss_coef 1.0 \
  --batch_size 256 \
  --device "$DEVICE" \
  --output_json "$RUN_DIR/encoder_validation.json"
