#!/bin/bash
#SBATCH -J phaseE_smoke
#SBATCH -p gpu-short
#SBATCH --gres=gpu:nvidia_h200_nvl:1
#SBATCH -c 16
#SBATCH --mem=128G
#SBATCH -t 1:00:00
#SBATCH -o logs/phaseE/phaseE_smoke.%j.out
#SBATCH -e logs/phaseE/phaseE_smoke.%j.err

set -euo pipefail

REPO="${SLURM_SUBMIT_DIR:-/mmfs1/storage/users/xiar3/exp/STVqvae}"
cd "${REPO}"
mkdir -p logs/phaseE

source /usr/shared_apps/packages/anaconda3-2023.09/etc/profile.d/conda.sh
conda activate vit5
export LANG=en_US.UTF-8 LC_ALL=en_US.UTF-8 PYTHONIOENCODING=utf-8

CKPT="${REPO}/discrete_mbrl/model_free/models/MiniGrid-DoorKey-8x8-v0/sweep_dk8_v6_s4_best_model.pt"

cd "${REPO}/discrete_mbrl"

echo "=== E1a smoke: teacher-forced on v6 DK-8 s4 (50 rollouts) ==="
python analyze_wm_teacher_forced.py \
    --model_path "${CKPT}" \
    --n_rollouts 100 \
    --horizons 1 3 5 10 \
    --device cuda \
    --output_json "${REPO}/logs/phaseE/_smoke_tf_v6_s4.json"

echo
echo "=== E2 smoke: train oracle WM offline on v6 DK-8 s4 (small-scale) ==="
ORACLE_CKPT="${REPO}/discrete_mbrl/model_free/models/MiniGrid-DoorKey-8x8-v0/oracle_v6_dk8_s4_best_model.pt"
python train_oracle_wm.py \
    --model_path  "${CKPT}" \
    --output_path "${ORACLE_CKPT}" \
    --n_random_eps 100 --n_policy_eps 100 \
    --max_steps 100 \
    --epochs 30 --batch_size 512 --lr 1e-4 \
    --device cuda

echo
echo "=== E2 smoke part 2: analyzer on the oracle ckpt ==="
python analyze_wm_multistep.py \
    --model_path "${ORACLE_CKPT}" \
    --n_rollouts 100 --probe_frames 5000 \
    --horizons 1 3 5 10 \
    --device cuda \
    --output_json "${REPO}/logs/phaseE/_smoke_oracle_analyzer_v6_s4.json"

echo
echo "DONE"
