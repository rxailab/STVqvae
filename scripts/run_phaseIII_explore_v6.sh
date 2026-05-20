#!/usr/bin/env bash
# Phase III: online recipe — ε-greedy random-action injection during PPO+WM
# training. Tests whether action-diversity restoration during *online* training
# (not just frozen-policy oracle WM, as in Cell D) closes the §4 per-class WM
# gap on a fresh PPO run.
#
# Single-run validation first: v6 DK-8 encoder, seed=1, ε=0.15.
# If this trains to >0.9 reward AND has WM_1 door >0.5, scale up to full sweep.
set -euo pipefail
REPO="/mmfs1/storage/users/xiar3/exp/STVqvae"
LOG_DIR="${REPO}/logs/phaseIII"
mkdir -p "${LOG_DIR}"

cd "${REPO}/discrete_mbrl/model_free"
export PYTHONPATH="${REPO}:${PYTHONPATH:-}"
export TORCHDYNAMO_DISABLE=1
export MPLCONFIGDIR="${REPO}/.mplconfig"

# Grid (validation = first row only)
declare -a GRID=(
  "v6_eps015_s1   5   0.15   1"
  # "v6_eps000_s1   5   0.00   1"  # control = §4 baseline; uncomment if needed
  # "v6_eps020_s1   5   0.20   1"
  # "v6_eps015_s2   5   0.15   2"
)

for row in "${GRID[@]}"; do
  read -r run_name aeversion eps seed <<< "${row}"
  log="${LOG_DIR}/${run_name}.log"
  marker="${REPO}/discrete_mbrl/model_free/models/MiniGrid-DoorKey-8x8-v0/${run_name}_best_model.pt"
  if [[ -s "${marker}" ]]; then echo "[skip] ${run_name}"; continue; fi
  echo "[phaseIII] ${run_name}  (eps=${eps}, seed=${seed})"

  set +e
  python -u train.py \
    --env_name MiniGrid-DoorKey-8x8-v0 \
    --ae_model_type vqvae --ae_model_version ${aeversion} \
    --codebook_size 64 --embedding_dim 64 --filter_size 8 \
    --dead_code_threshold 2.0 \
    --seed ${seed} \
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
    --run_name "${run_name}" \
    --device cuda --save \
    --use_world_model --wm_aux_coef 0.1 --wm_train_freq 1 \
    --trans_model_type discrete --trans_model_version 1 \
    --trans_hidden 256 --trans_depth 3 \
    --use_semantic_aux --sem_aux_coef 0.05 \
    --sem_head_hidden 128 --sem_head_version 2 --sem_n_classes 11 \
    --sem_aux_start_reward 0.0 --sem_class_weights --sem_focal_gamma 2.0 \
    --sem_pre_vq \
    --explore_random_prob ${eps} > "${log}" 2>&1
  rc=$?; set -e
  if [[ $rc -ne 0 ]]; then echo "[fail] ${run_name} (rc=$rc)"; fi
done

# Analyze the resulting ckpts.
echo
echo "=== Analysing Phase III ckpts ==="
for row in "${GRID[@]}"; do
  read -r run_name _ _ _ <<< "${row}"
  ckpt="${REPO}/discrete_mbrl/model_free/models/MiniGrid-DoorKey-8x8-v0/${run_name}_best_model.pt"
  if [[ ! -s "${ckpt}" ]]; then continue; fi
  for analyzer in analyze_wm_semantic_accuracy analyze_wm_action_cond_v2; do
    out="${LOG_DIR}/${analyzer}_${run_name}.json"
    log="${LOG_DIR}/${analyzer}_${run_name}.log"
    if [[ -s "${out}" ]]; then continue; fi
    echo "[analyze] ${run_name} ${analyzer}"
    cd "${REPO}/discrete_mbrl"
    set +e
    case "${analyzer}" in
      analyze_wm_semantic_accuracy)
        python ${analyzer}.py --model_path "${ckpt}" --n_frames 5000 --device cuda --output_json "${out}" > "${log}" 2>&1
        ;;
      analyze_wm_action_cond_v2)
        python ${analyzer}.py --model_path "${ckpt}" --n_states 2000 --device cuda --output_json "${out}" > "${log}" 2>&1
        ;;
    esac
    rc=$?; set -e
    if [[ $rc -ne 0 ]]; then echo "[fail-ana] ${run_name} ${analyzer} (rc=$rc)"; fi
    cd "${REPO}/discrete_mbrl/model_free"
  done
done
echo Done.
