#!/bin/bash
#SBATCH --mem=50G
#SBATCH -J e2ephased
#SBATCH -p gpu-short
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
source /etc/profile

echo Job running on compute node `uname -n`

module add anaconda3/2023.09
module add cuda
source activate vit5

# Changes vs e2estable:
#   --encoder_lr_cosine          : cosine anneal encoder LR 3e-5 → 0 over first 2.5M steps
#   --freeze_encoder_after 2500000 : hard-freeze encoder at halfway point (step 2.5M)
#                                    policy trains alone in second half, no encoder drift
#   --run_name e2ephased         : saves e2ephased_best_model.pt / e2ephased_final_model.pt
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
  --encoder_lr 3e-5 \
  --encoder_lr_cosine \
  --freeze_encoder_after 2500000 \
  --ppo_iters 10 \
  --ppo_batch_size 64 \
  --ppo_entropy_coef 0.01 \
  --ppo_gae_lambda 0.95 \
  --ppo_norm_advantages \
  --ppo_max_grad_norm 0.5 \
  --entropy_penalty_coef 0.05 \
  --ortho_init \
  --run_name e2ephased \
  --device cuda \
  --save
