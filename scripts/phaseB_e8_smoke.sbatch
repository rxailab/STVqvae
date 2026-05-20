#!/bin/bash
#SBATCH -J phaseB_e8_smoke
#SBATCH -p gpu-short
#SBATCH --gres=gpu:nvidia_h200_nvl:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 0:30:00
#SBATCH -o logs/phaseB/phaseB_e8_smoke.%j.out
#SBATCH -e logs/phaseB/phaseB_e8_smoke.%j.err

set -euo pipefail

REPO="${SLURM_SUBMIT_DIR:-/mmfs1/storage/users/xiar3/exp/STVqvae}"
cd "${REPO}"
mkdir -p logs/phaseB

source /usr/shared_apps/packages/anaconda3-2023.09/etc/profile.d/conda.sh
conda activate vit5

export LANG=en_US.UTF-8
export LC_ALL=en_US.UTF-8
export PYTHONIOENCODING=utf-8

cd "${REPO}/discrete_mbrl"

echo "=== E8 smoke 1/2: VQ v6 DK-8 ==="
python analyze_wm_phaseB_e8.py \
  --model_path "${REPO}/discrete_mbrl/model_free/models/MiniGrid-DoorKey-8x8-v0/sweep_dk8_v6_s4_best_model.pt" \
  --n_episodes 20 --max_steps 100 \
  --horizons 1 3 5 10 \
  --device cuda \
  --output_json "${REPO}/logs/phaseB/_smoke_e8_v6.json"

echo
echo "=== E8 smoke 2/2: VAE DK-8 ==="
python analyze_wm_phaseB_e8.py \
  --model_path "${REPO}/discrete_mbrl/model_free/models/MiniGrid-DoorKey-8x8-v0/sweep_dk8_vae_s1_best_model.pt" \
  --n_episodes 20 --max_steps 100 \
  --horizons 1 3 5 10 \
  --device cuda \
  --output_json "${REPO}/logs/phaseB/_smoke_e8_vae.json"

echo
echo "DONE"
