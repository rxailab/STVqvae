#!/bin/bash
#SBATCH --mem=24G
#SBATCH -J ppocnn3
#SBATCH -p gpu-short
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
source /etc/profile

echo Job running on compute node `uname -n`

module add anaconda3/2023.09
module add cuda
source activate vit5

cd /mmfs1/storage/users/xiar3/exp/Vqvae2Path/discrete_mbrl/eval_policies

PYTHONPATH=/scratch/hpc/11/xiar3/vqvaeEntPan:/mmfs1/storage/users/xiar3/exp/Vqvae2Path/discrete_mbrl:$PYTHONPATH \
  /storage/hpc/11/xiar3/vit5/bin/python train_policy.py \
  --env_name MiniGrid-LavaCrossingS9N1-v0 \
  --goal_type goal \
  --n_envs 8 \
  --train_steps 300000 \
  --run_name ppo_cnn_baseline3
