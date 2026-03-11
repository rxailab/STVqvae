#!/bin/bash
#SBATCH --mem=50G
#SBATCH -J e2erecon
#SBATCH -p gpu-short
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
source /etc/profile

echo Job running on compute node `uname -n`

module add anaconda3/2023.09
module add cuda
source activate vit5

# Direction: joint reconstruction loss alongside e2e PPO
#   The key idea: anchor the VQVAE with a reconstruction objective so the
#   encoder cannot drift purely toward RL-useful features and lose structure.
#
# Key differences vs e2ebaseline:
#   --ae_recon_loss          : add reconstruction loss to keep VQVAE well-structured
#   --ae_er_train            : train AE from replay buffer (more stable, diverse samples)
#   --encoder_lr 3e-5        : slower encoder updates to balance two loss signals
#   --ppo_gae_lambda 0.95    : GAE for better advantage estimates
#   --ppo_norm_advantages    : normalize advantages per mini-batch
#   --ppo_max_grad_norm 0.5  : gradient clipping
PYTHONPATH=/scratch/hpc/11/xiar3/vqvaeEntPan:$PYTHONPATH /storage/hpc/11/xiar3/vit5/bin/python /storage/users/xiar3/exp/Vqvae2Path/discrete_mbrl/model_free/train.py \
  --env_name MiniGrid-LavaCrossingS9N1-v0 \
  --ae_model_type vqvae \
  --ae_model_version 2 \
  --codebook_size 64 \
  --embedding_dim 64 \
  --filter_size 9 \
  --mf_steps 5000000 \
  --batch_size 256 \
  --e2e_loss \
  --ae_recon_loss \
  --ae_er_train \
  --encoder_lr 3e-5 \
  --ppo_iters 20 \
  --ppo_batch_size 32 \
  --ppo_entropy_coef 0.01 \
  --ppo_gae_lambda 0.95 \
  --ppo_norm_advantages \
  --ppo_max_grad_norm 0.5 \
  --device cuda \
  --save
