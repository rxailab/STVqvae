#!/bin/bash
#SBATCH --mem=50G
#SBATCH -J e2ecosine2
#SBATCH -p gpu-short
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
source /etc/profile

echo Job running on compute node `uname -n`

module add anaconda3/2023.09
module add cuda
source activate vit5

# Changes vs e2ecosine:
#   BUG FIX: cosine scheduler now only anneals the encoder param group (not policy/critic).
#            Previously CosineAnnealingLR was applied to the whole optimizer, decaying
#            both encoder AND policy LR to 0 — this suppressed policy learning and
#            explains why e2ecosine peaked lower (0.698) than e2estable (0.897).
#   All other settings identical to e2ecosine.
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
  --ppo_iters 10 \
  --ppo_batch_size 64 \
  --ppo_entropy_coef 0.01 \
  --ppo_gae_lambda 0.95 \
  --ppo_norm_advantages \
  --ppo_max_grad_norm 0.5 \
  --entropy_penalty_coef 0.05 \
  --ortho_init \
  --run_name e2ecosine2 \
  --device cuda \
  --save
