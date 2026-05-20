#!/usr/bin/env bash
# Phase VIII-C: two-buffer architecture. Pre-collect a static random-action
# transition buffer (Cell A-style, 600 episodes); do K WM-only gradient
# steps on it per PPO update. PPO rollout stays clean (no random injection);
# WM gets dedicated broad-coverage data; encoder gets gradient from BOTH
# (PPO policy/value during PPO update + WM aux during both PPO and WM-only
# updates).
#
# Args: run_name, seed
set -euo pipefail
REPO="/mmfs1/storage/users/xiar3/exp/STVqvae"
LOG_DIR="${REPO}/logs/phaseVIII_C"
mkdir -p "${LOG_DIR}"

cd "${REPO}/discrete_mbrl/model_free"
export PYTHONPATH="${REPO}:${PYTHONPATH:-}"
export TORCHDYNAMO_DISABLE=1
export MPLCONFIGDIR="${REPO}/.mplconfig"

RUN_NAME="$1"
SEED="$2"
LOG="${LOG_DIR}/${RUN_NAME}.log"
SENTINEL="${REPO}/discrete_mbrl/model_free/models/MiniGrid-DoorKey-8x8-v0/${RUN_NAME}_final_model.pt"

if [[ -s "${SENTINEL}" ]]; then
  echo "[skip] ${RUN_NAME}"
else
  echo "[phaseVIII-C] ${RUN_NAME}  seed=${SEED}  two-buffer (random_wm_buffer_eps=600, per_update=1)"
  set +e
  python -u train.py \
    --env_name MiniGrid-DoorKey-8x8-v0 \
    --ae_model_type vqvae --ae_model_version 6 \
    --codebook_size 64 --embedding_dim 64 --filter_size 8 \
    --dead_code_threshold 2.0 \
    --seed ${SEED} \
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
    --random_wm_buffer_eps 600 \
    --random_wm_per_update 1 \
    --random_wm_batch_size 256 > "${LOG}" 2>&1
  rc=$?; set -e
  if [[ $rc -ne 0 ]]; then echo "[fail] ${RUN_NAME} (rc=$rc)"; exit 1; fi
fi

# Analyse both
cd "${REPO}/discrete_mbrl"
for suffix in best final; do
  CKPT="${REPO}/discrete_mbrl/model_free/models/MiniGrid-DoorKey-8x8-v0/${RUN_NAME}_${suffix}_model.pt"
  if [[ ! -s "${CKPT}" ]]; then echo "[no-${suffix}] ${RUN_NAME}"; continue; fi
  for analyzer in analyze_wm_semantic_accuracy analyze_wm_action_cond_v2; do
    out="${LOG_DIR}/${analyzer}_${RUN_NAME}_${suffix}.json"
    log="${LOG_DIR}/${analyzer}_${RUN_NAME}_${suffix}.log"
    if [[ -s "${out}" ]]; then continue; fi
    echo "[analyze-${suffix}] ${RUN_NAME} ${analyzer}"
    set +e
    case "${analyzer}" in
      analyze_wm_semantic_accuracy) python ${analyzer}.py --model_path "${CKPT}" --n_frames 5000 --device cuda --output_json "${out}" > "${log}" 2>&1 ;;
      analyze_wm_action_cond_v2)    python ${analyzer}.py --model_path "${CKPT}" --n_states 2000 --device cuda --output_json "${out}" > "${log}" 2>&1 ;;
    esac
  done
done
echo "Done ${RUN_NAME}."
