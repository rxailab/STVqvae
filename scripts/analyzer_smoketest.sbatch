#!/bin/bash
#SBATCH -J analyzer_smoke
#SBATCH -p gpu-short
#SBATCH --gres=gpu:nvidia_h200_nvl:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 0:30:00
#SBATCH -o logs/phaseA/analyzer_smoke.%j.out
#SBATCH -e logs/phaseA/analyzer_smoke.%j.err

set -euo pipefail

REPO="${SLURM_SUBMIT_DIR:-/mmfs1/storage/users/xiar3/exp/STVqvae}"
cd "${REPO}"
mkdir -p logs/phaseA

source /usr/shared_apps/packages/anaconda3-2023.09/etc/profile.d/conda.sh
conda activate vit5

# Tiny pilot: 50 rollouts, 2k probe frames -- ~3 min on H200.
# Pick a v6 DK-8 best model that we know exists (override via $1 if you want).
DEFAULT_CKPT="${REPO}/discrete_mbrl/model_free/models/MiniGrid-DoorKey-8x8-v0/sweep_dk8_v6_s4_best_model.pt"
CKPT="${1:-${DEFAULT_CKPT}}"

if [[ ! -s "${CKPT}" ]]; then
  echo "checkpoint not found: ${CKPT}" >&2
  exit 2
fi

echo "smoke checkpoint: ${CKPT}"
cd "${REPO}/discrete_mbrl"
python analyze_wm_multistep.py \
  --model_path   "${CKPT}" \
  --n_rollouts   50 \
  --probe_frames 2000 \
  --horizons     1 3 5 10 \
  --device       cuda \
  --output_json  "${REPO}/logs/phaseA/_smoke.json"

echo
echo "=== JSON head ==="
python - <<'PY'
import json, sys
with open('logs/phaseA/_smoke.json') as f:
    d = json.load(f)
print('encoder_version :', d.get('encoder_version'))
print('ae_model_type   :', d.get('ae_model_type'))
print('trans_model_type:', d.get('trans_model_type'))
print('horizons        :', d.get('horizons'))
print('probe classes   :', sorted(d.get('probe_acc_per_class', {}).keys()))
print('extended block? :', 'extended_metrics_per_horizon' in d)
ext = d.get('extended_metrics_per_horizon', {})
for k in sorted(ext.keys(), key=int):
    pr = ext[k].get('pearson_r', {})
    print(f'  k={k}  pearson_r={pr}  variants={[v for v in ext[k] if v != "confusion" and v != "pearson_r"]}')
PY
echo
echo "DONE"
