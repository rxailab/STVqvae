#!/bin/bash
#SBATCH --mem=24G
#SBATCH -J ppocnn2
#SBATCH -p gpu-short
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
source /etc/profile

echo Job running on compute node `uname -n`

module add anaconda3/2023.09
module add cuda
source activate vit5

# PPO CnnPolicy baseline v2 — uses eval_policies/train_policy.py (correct obs pipeline)
# Fix vs v1: n_steps=128 -> ~292 PPO updates (was 18 with n_steps=2048)
# Obs: (3,72,72) uint8 via make_env + SB3ObsWrapper (already channel-first, no VecTransposeImage needed)
cd /mmfs1/storage/users/xiar3/exp/Vqvae2Path/discrete_mbrl/eval_policies

PYTHONPATH=/scratch/hpc/11/xiar3/vqvaeEntPan:/mmfs1/storage/users/xiar3/exp/Vqvae2Path/discrete_mbrl:$PYTHONPATH \
  /storage/hpc/11/xiar3/vit5/bin/python train_policy.py \
  --env_name MiniGrid-LavaCrossingS9N1-v0 \
  --goal_type goal \
  --n_envs 8 \
  --train_steps 300000 \
  --n_steps 128 \
  --batch_size 64 \
  --n_epochs 10 \
  --learning_rate 3e-4 \
  --gamma 0.99 \
  --gae_lambda 0.95 \
  --clip_range 0.2 \
  --ent_coef 0.0 \
  --vf_coef 0.5 \
  --eval_freq 10000 \
  --eval_episodes 20 \
  --run_name ppo_cnn_baseline2
