#!/bin/bash
#SBATCH --mem=24G
#SBATCH -J ppocnn
#SBATCH -p gpu-short
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
source /etc/profile

echo Job running on compute node `uname -n`

module add anaconda3/2023.09
module add cuda
source activate vit5

# PPO CnnPolicy baseline (SB3)
# Policy    : CnnPolicy (NatureCNN, 72x72x3 RGB obs)
# Envs      : 8 x DummyVecEnv
# Steps     : 300,000
# Eval freq : 30,000 steps, 20 episodes
# LR        : 3e-4, n_steps=2048, batch=64, epochs=10
# gamma=0.99, gae_lambda=0.95, clip=0.2, ent_coef=0.0, vf_coef=0.5
PYTHONPATH=/scratch/hpc/11/xiar3/vqvaeEntPan:$PYTHONPATH /storage/hpc/11/xiar3/vit5/bin/python /storage/users/xiar3/exp/Vqvae2Path/discrete_mbrl/model_free/ppo_cnn_baseline.py \
  --run_name ppo_cnn_baseline \
  --total_steps 300000 \
  --seed 42
