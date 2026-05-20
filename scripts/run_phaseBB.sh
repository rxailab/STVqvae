#!/usr/bin/env bash
# Phase BB: train fresh encoders on additional MiniGrid environments
# (MultiRoom-N2-S4 and KeyCorridor-S3R1), then run the causal λ-sweep
# on each. Adds 2 new MiniGrid environments to the existing 3
# (DK-8, DK-16, LavaCrossing-S9N1) for cross-task generalization.
#
# Per env: ~2h training (5M steps) + ~2h sweep (4 cells × ~30min) = ~4h
# Two envs sequentially: ~8h total compute.
#
# Idempotent: skip-guards on encoder ckpts and per-cell logs allow resume.

set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="${REPO}/logs/phaseBB"
mkdir -p "${LOG_DIR}"

LAMBDAS=(0.0 0.5 2.0 10.0)

# ── Step 1: Train fresh encoders ────────────────────────────────────────

train_encoder() {
  local env_name="$1" seed="$2" run_name="$3"
  local model_dir="${REPO}/discrete_mbrl/model_free/models/${env_name}"
  local target_ckpt="${model_dir}/${run_name}_best_model.pt"
  if [[ -s "${target_ckpt}" ]]; then echo "[skip-train] ${run_name}"; return; fi

  echo "[train] ${env_name} seed=${seed} run=${run_name}"
  cd "${REPO}/discrete_mbrl/model_free"
  set +e
  # Use continuous spatial VAE (vae_spatial) — more flexible to observation
  # sizes than the v6 VQ encoder which has hardcoded shape constraints.
  # Mirrors phaseC1_train_one.sh with --save and full PPO+e2e flags.
  python -u train.py \
    --env_name "${env_name}" \
    --ae_model_type vae_spatial --embedding_dim 64 \
    --trans_model_type continuous \
    --seed "${seed}" \
    --mf_steps 2000000 \
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
    --run_name "${run_name}" \
    --device cuda \
    --save \
    --use_world_model \
    --wm_aux_coef 0.0 \
    --wm_train_freq 1 \
    --trans_model_version 1 \
    --trans_hidden 256 \
    --trans_depth 3 \
    > "${LOG_DIR}/train_${run_name}.log" 2>&1
  rc=$?; set -e
  cd "${REPO}"
  if [[ $rc -ne 0 ]]; then echo "[fail-train] ${run_name}"; return 1; fi
  echo "[ok-train] ${run_name}"
}

# ── Step 2: Run λ-sweep pipeline per (env, encoder, λ) ─────────────────

run_pipeline() {
  local env_name="$1" label="$2" src_ckpt="$3" lam="$4"
  local model_dir="${REPO}/discrete_mbrl/model_free/models/${env_name}"
  local pr_dir="${model_dir}/phaseBB"
  mkdir -p "${pr_dir}"
  local enc_ckpt="${pr_dir}/finetuned_${label}_lam${lam}.pt"
  local oracle_ckpt="${pr_dir}/oracle_finetuned_${label}_lam${lam}.pt"
  local phaseA_out="${LOG_DIR}/phaseA_${label}_lam${lam}.json"
  local phaseN_out="${LOG_DIR}/phaseN_${label}_lam${lam}.json"
  local ft_log="${LOG_DIR}/finetune_${label}_lam${lam}.log"
  local oracle_log="${LOG_DIR}/oracle_${label}_lam${lam}.log"
  local A_log="${LOG_DIR}/phaseA_${label}_lam${lam}.log"
  local N_log="${LOG_DIR}/phaseN_${label}_lam${lam}.log"

  cd "${REPO}/discrete_mbrl"

  if [[ -s "${enc_ckpt}" ]]; then echo "[skip] finetune ${label} lam=${lam}"; else
    echo "[finetune] ${label} lam=${lam}"
    set +e
    python finetune_encoder_action_cond.py \
      --model_path "${src_ckpt}" --output_path "${enc_ckpt}" \
      --action_aux_weight "${lam}" --epochs 10 \
      --n_random_eps 200 --n_policy_eps 200 \
      --batch_size 128 --lr 1e-4 \
      --device cuda > "${ft_log}" 2>&1
    rc=$?; set -e
    if [[ $rc -ne 0 ]]; then echo "[fail-ft] ${label} lam=${lam}"; rm -f "${enc_ckpt}"; return; fi
  fi

  if [[ -s "${oracle_ckpt}" ]]; then echo "[skip] oracle ${label} lam=${lam}"; else
    echo "[oracle] ${label} lam=${lam}"
    set +e
    python train_oracle_wm.py \
      --model_path "${enc_ckpt}" --output_path "${oracle_ckpt}" \
      --n_random_eps 250 --n_policy_eps 250 \
      --epochs 50 --batch_size 256 --lr 1e-4 \
      --device cuda > "${oracle_log}" 2>&1
    rc=$?; set -e
    if [[ $rc -ne 0 ]]; then echo "[fail-oracle] ${label} lam=${lam}"; rm -f "${oracle_ckpt}"; return; fi
  fi

  if [[ ! -s "${phaseA_out}" ]]; then
    echo "[phaseA] ${label} lam=${lam}"
    set +e
    python analyze_wm_multistep.py --model_path "${oracle_ckpt}" \
      --n_rollouts 300 --probe_frames 10000 --horizons 1 3 5 10 \
      --device cuda --output_json "${phaseA_out}" > "${A_log}" 2>&1
    rc=$?; set -e
    if [[ $rc -ne 0 ]]; then echo "[fail-A] ${label} lam=${lam}"; rm -f "${phaseA_out}"; fi
  fi

  if [[ ! -s "${phaseN_out}" ]]; then
    echo "[phaseN] ${label} lam=${lam}"
    set +e
    python analyze_wm_action_cond.py --model_path "${oracle_ckpt}" \
      --n_states 2000 --device cuda \
      --output_json "${phaseN_out}" > "${N_log}" 2>&1
    rc=$?; set -e
    if [[ $rc -ne 0 ]]; then echo "[fail-N] ${label} lam=${lam}"; rm -f "${phaseN_out}"; fi
  fi
  echo "[ok] ${label} lam=${lam}"
}

# ── Configuration: 2 new MiniGrid environments ─────────────────────────

declare -a CONFIGS=(
  "MiniGrid-Empty-8x8-v0|empty8_vae_s1|phaseBB_empty8_vae_s1|1"
  "MiniGrid-FourRooms-v0|fourrooms_vae_s1|phaseBB_fourrooms_vae_s1|1"
  "MiniGrid-LavaCrossingS9N2-v0|lavaS9N2_vae_s1|phaseBB_lavaS9N2_vae_s1|1"
)

for entry in "${CONFIGS[@]}"; do
  IFS='|' read -r env_name label run_name seed <<< "${entry}"
  echo "================================================"
  echo "Environment: ${env_name}  label=${label}  seed=${seed}"
  echo "================================================"

  # Train fresh encoder
  if ! train_encoder "${env_name}" "${seed}" "${run_name}"; then
    echo "[skip-env] training failed for ${env_name}"
    continue
  fi

  src_ckpt="${REPO}/discrete_mbrl/model_free/models/${env_name}/${run_name}_best_model.pt"
  if [[ ! -f "${src_ckpt}" ]]; then echo "[skip-miss] ${src_ckpt}"; continue; fi

  # Run λ-sweep pipeline
  for lam in "${LAMBDAS[@]}"; do
    run_pipeline "${env_name}" "${label}" "${src_ckpt}" "${lam}"
  done
done

echo "Phase BB complete."
