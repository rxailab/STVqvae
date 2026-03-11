#!/bin/bash
#SBATCH --mem=50G
#SBATCH -J e2ebaseline
#SBATCH -p gpu-short
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
source /etc/profile

echo Job running on compute node `uname -n`

module add anaconda3/2023.09
module add cuda
source activate vit5

PYTHONPATH=/scratch/hpc/11/xiar3/vqvaeEntPan:$PYTHONPATH /storage/hpc/11/xiar3/vit5/bin/python /storage/users/xiar3/exp/Vqvae2Path/discrete_mbrl/model_free/train.py --env_name MiniGrid-LavaCrossingS9N1-v0 --ae_model_type vqvae --ae_model_version 2 --codebook_size 64 --embedding_dim 64 --filter_size 9 --mf_steps 5000000 --batch_size 256 --e2e_loss --ppo_iters 20 --ppo_batch_size 32 --ppo_entropy_coef 0.01 --device cuda --save          