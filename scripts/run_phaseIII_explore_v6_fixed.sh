#!/usr/bin/env bash
# Phase III RE-RUN: the original ε=0.15 step-level run was effectively §4 baseline
# because the injection patch only fired in the non-vectorized rollout path
# (train.py:611). The vec-env path (used at num_envs=16) was unaffected.
#
# This runner exercises the corrected patch (vec-env path now overrides actions
# when explore_random_prob > 0 or random_episode_prob > 0).
#
# Single-run validation: v6 DK-8 encoder, seed=1, ε_step=0.15.
set -euo pipefail
REPO="/mmfs1/storage/users/xiar3/exp/STVqvae"
LOG_DIR="${REPO}/logs/phaseIII"
mkdir -p "${LOG_DIR}"

cd "${REPO}/discrete_mbrl/model_free"
export PYTHONPATH="${REPO}:${PYTHONPATH:-}"
export TORCHDYNAMO_DISABLE=1
export MPLCONFIGDIR="${REPO}/.mplconfig"

RUN_NAME="v6_eps015_fixed_s1"
LOG="${LOG_DIR}/${RUN_NAME}.log"
MARKER="${REPO}/discrete_mbrl/model_free/models/MiniGrid-DoorKey-8x8-v0/${RUN_NAME}_best_model.pt"

if [[ -s "${MARKER}" ]]; then
  echo "[skip] ${RUN_NAME}"
else
  echo "[phaseIII-fixed] ${RUN_NAME}  (eps_step=0.15, seed=1)"
  set +e
  python -u train.py \
    --env_name MiniGrid-DoorKey-8x8-v0 \
    --ae_model_type vqvae --ae_model_version 6 \
    --codebook_size 64 --embedding_dim 64 --filter_size 8 \
    --dead_code_threshold 2.0 \
    --seed 1 \
    --mf_steps 5000000 \
    --batch_size 4096 --num_envs 16 \
    --ppo_iters 10 --ppo_batch_size 64 \
    --ppo_entropy_coef 0.01 --ppo_gae_lambda 0.95 \
    --ppo_norm_advantages --ppo_max_grad_norm 0.5 \
    --learning_rate 1e-4 \
    --e2e_loss --encoder_lr 1e-5 --encoder_lr_cosine \
    --encoder_snapback --snapback_threshold 0.5 --snapback_patience 100 \
    --snapback_min_reward 0.0 \
    --ortho_init \
    --model_dir .. \
    --run_name "${RUN_NAME}" \
    --device cuda --save \
    --use_world_model --wm_aux_coef 0.1 --wm_train_freq 1 \
    --trans_model_type discrete --trans_model_version 1 \
    --trans_hidden 256 --trans_depth 3 \
    --use_semantic_aux --sem_aux_coef 0.05 \
    --sem_head_hidden 128 --sem_head_version 2 --sem_n_classes 11 \
    --sem_aux_start_reward 0.0 --sem_class_weights --sem_focal_gamma 2.0 \
    --sem_pre_vq \
    --explore_random_prob 0.15 > "${LOG}" 2>&1
  rc=$?; set -e
  if [[ $rc -ne 0 ]]; then echo "[fail] ${RUN_NAME} (rc=$rc)"; exit 1; fi
fi

# Analyse
CKPT="${REPO}/discrete_mbrl/model_free/models/MiniGrid-DoorKey-8x8-v0/${RUN_NAME}_best_model.pt"
cd "${REPO}/discrete_mbrl"
for analyzer in analyze_wm_semantic_accuracy analyze_wm_action_cond_v2; do
  out="${LOG_DIR}/${analyzer}_${RUN_NAME}.json"
  log="${LOG_DIR}/${analyzer}_${RUN_NAME}.log"
  if [[ -s "${out}" ]]; then continue; fi
  echo "[analyze] ${RUN_NAME} ${analyzer}"
  set +e
  case "${analyzer}" in
    analyze_wm_semantic_accuracy)
      python ${analyzer}.py --model_path "${CKPT}" --n_frames 5000 --device cuda --output_json "${out}" > "${log}" 2>&1
      ;;
    analyze_wm_action_cond_v2)
      python ${analyzer}.py --model_path "${CKPT}" --n_states 2000 --device cuda --output_json "${out}" > "${log}" 2>&1
      ;;
  esac
done
echo Done.
