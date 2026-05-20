#!/usr/bin/env bash
# Phase C1: train v6 DK-8 with wm_aux_coef=0 (the §4.7 "VQ-no-aux confound" run).
# Single-seed launcher. Args: $1 = seed integer.
#
# Mirrors sweep_multiseed_v2.sh's v6 DK-8 invocation except:
#   --wm_aux_coef 0.0   (instead of 0.1)
#   --run_name phaseC1_v6_noaux_dk8_s<seed>

set -euo pipefail

SEED="${1:?usage: $0 <seed>}"
REPO="${SLURM_SUBMIT_DIR:-/mmfs1/storage/users/xiar3/exp/STVqvae}"
cd "${REPO}"

source /usr/shared_apps/packages/anaconda3-2023.09/etc/profile.d/conda.sh
conda activate vit5
export LANG=en_US.UTF-8 LC_ALL=en_US.UTF-8 PYTHONIOENCODING=utf-8
export PYTHONPATH="${REPO}:${PYTHONPATH:-}"
export TORCHDYNAMO_DISABLE=1
mkdir -p "${REPO}/.mplconfig" "${REPO}/logs/phaseC1"
export MPLCONFIGDIR="${REPO}/.mplconfig"

ENV_NAME="MiniGrid-DoorKey-8x8-v0"
RUN_NAME="phaseC1_v6_noaux_dk8_s${SEED}"
LOG_FILE="${REPO}/logs/phaseC1/${RUN_NAME}.log"
MF_STEPS=5000000

cd "${REPO}/discrete_mbrl/model_free"

echo "=== Phase C1 train  v6 DK-8  wm_aux_coef=0.0  seed=${SEED} ==="
echo "log: ${LOG_FILE}"
date

python -u train.py \
    --env_name "${ENV_NAME}" \
    --ae_model_type vqvae --ae_model_version 6 \
    --codebook_size 64 --embedding_dim 64 --filter_size 8 \
    --trans_model_type discrete \
    --seed "${SEED}" \
    --mf_steps ${MF_STEPS} \
    --batch_size 4096 \
    --num_envs 16 \
    --ppo_iters 10 \
    --ppo_batch_size 64 \
    --ppo_entropy_coef 0.01 \
    --ppo_gae_lambda 0.95 \
    --ppo_norm_advantages \
    --ppo_max_grad_norm 0.5 \
    --learning_rate 1e-4 \
    --e2e_loss \
    --encoder_lr 1e-5 \
    --encoder_lr_cosine \
    --encoder_snapback \
    --snapback_threshold 0.5 \
    --snapback_patience 100 \
    --snapback_min_reward 0.0 \
    --ortho_init \
    --model_dir .. \
    --run_name "${RUN_NAME}" \
    --device cuda \
    --save \
    --use_world_model \
    --wm_aux_coef 0.0 \
    --wm_train_freq 1 \
    --trans_model_version 1 \
    --trans_hidden 256 \
    --trans_depth 3 \
    --use_semantic_aux \
    --sem_aux_coef 0.05 \
    --sem_head_hidden 128 \
    --sem_head_version 2 \
    --sem_n_classes 11 \
    --sem_aux_start_reward 0.0 \
    --sem_class_weights \
    --sem_class_weight_power 1.0 \
    --sem_focal_gamma 2.0 \
    --sem_pre_vq \
    > "${LOG_FILE}" 2>&1
TRAIN_EXIT=$?
echo "train exit ${TRAIN_EXIT}"
date
if [[ ${TRAIN_EXIT} -ne 0 ]]; then
  echo "TRAIN FAILED -- see ${LOG_FILE}"
  exit ${TRAIN_EXIT}
fi

# Auto-analyze: drop a Phase A JSON for this fresh checkpoint so the §4.7
# claim ("VQ-no-aux matches v6 to within seed-to-seed noise") can be checked.
BEST_MODEL="${REPO}/discrete_mbrl/model_free/models/${ENV_NAME}/${RUN_NAME}_best_model.pt"
if [[ -f "${BEST_MODEL}" ]]; then
  cd "${REPO}/discrete_mbrl"
  python -u analyze_wm_multistep.py \
      --model_path  "${BEST_MODEL}" \
      --n_rollouts  500 \
      --probe_frames 20000 \
      --horizons    1 3 5 10 \
      --device      cuda \
      --output_json "${REPO}/logs/phaseA/${RUN_NAME}_best_model.json" \
      >> "${REPO}/logs/phaseC1/${RUN_NAME}_analysis.log" 2>&1 || \
      echo "WARN: analyzer failed for ${BEST_MODEL}"
else
  echo "WARN: best_model not found at ${BEST_MODEL}"
fi

echo "DONE  seed=${SEED}"
