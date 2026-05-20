#!/bin/bash
#SBATCH -J phaseD_smoke
#SBATCH -p gpu-short
#SBATCH --gres=gpu:nvidia_h200_nvl:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 0:30:00
#SBATCH -o logs/phaseD/phaseD_smoke.%j.out
#SBATCH -e logs/phaseD/phaseD_smoke.%j.err

set -euo pipefail

REPO="${SLURM_SUBMIT_DIR:-/mmfs1/storage/users/xiar3/exp/STVqvae}"
cd "${REPO}"
mkdir -p logs/phaseD

source /usr/shared_apps/packages/anaconda3-2023.09/etc/profile.d/conda.sh
conda activate vit5
export LANG=en_US.UTF-8 LC_ALL=en_US.UTF-8 PYTHONIOENCODING=utf-8

cd "${REPO}/discrete_mbrl"

echo "=== Phase D smoke: VQ v6 DK-8 s4 ==="
python analyze_wm_planning.py \
  --model_path "${REPO}/discrete_mbrl/model_free/models/MiniGrid-DoorKey-8x8-v0/sweep_dk8_v6_s4_best_model.pt" \
  --n_episodes 30 --max_steps 100 \
  --horizons 5 10 20 \
  --device cuda \
  --output_json "${REPO}/logs/phaseD/_smoke_v6.json"

echo
echo "=== Phase D smoke: VAE DK-8 s1 ==="
python analyze_wm_planning.py \
  --model_path "${REPO}/discrete_mbrl/model_free/models/MiniGrid-DoorKey-8x8-v0/sweep_dk8_vae_s1_best_model.pt" \
  --n_episodes 30 --max_steps 100 \
  --horizons 5 10 20 \
  --device cuda \
  --output_json "${REPO}/logs/phaseD/_smoke_vae.json"

echo
echo "DONE"
