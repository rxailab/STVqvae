#!/usr/bin/env bash
# Phase Z: extend the causal λ-sweep to 9 additional source encoders for
# larger N. Targets:
#   - DK-8 VAE s4, s5 (continuous VAE)
#   - DK-16 v6_s2, v6_s3 (cross-task within MiniGrid)
#   - DK-8 VQ v6_s5, v5dc_s5 (cross-architecture)
#   - Crafter v6enc_cal_best, v6enc_cal2_best, v6enc_fix_best (cross-environment)
# 9 source encoders × 4 λ values = 36 new (encoder, λ) cells.
# Combined with existing N=28 (Phase R/R-multi/W/X/Y), final N=64.
#
# Idempotent: each step checks for existing artifacts and skips. Safe to rerun.

set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DK8="${REPO}/discrete_mbrl/model_free/models/MiniGrid-DoorKey-8x8-v0"
DK16="${REPO}/discrete_mbrl/model_free/models/MiniGrid-DoorKey-16x16-v0"
CRAFTER="${REPO}/discrete_mbrl/model_free/models/crafter"

OUT="${REPO}/logs/phaseZ"
PR_DK8="${DK8}/phaseZ"
PR_DK16="${DK16}/phaseZ"
PR_CRAFTER="${CRAFTER}/phaseZ"
mkdir -p "${OUT}" "${PR_DK8}" "${PR_DK16}" "${PR_CRAFTER}"

LAMBDAS=(0.0 0.5 2.0 10.0)

cd "${REPO}/discrete_mbrl"

run_pipeline() {
  local label="$1" src_ckpt="$2" enc_dir="$3" lam="$4" env_kind="$5"
  local enc_ckpt="${enc_dir}/finetuned_${label}_lam${lam}.pt"
  local oracle_ckpt="${enc_dir}/oracle_finetuned_${label}_lam${lam}.pt"
  local phaseA_out="${OUT}/phaseA_${label}_lam${lam}.json"
  local phaseN_out="${OUT}/phaseN_${label}_lam${lam}.json"
  local ft_log="${OUT}/finetune_${label}_lam${lam}.log"
  local oracle_log="${OUT}/oracle_${label}_lam${lam}.log"
  local A_log="${OUT}/phaseA_${label}_lam${lam}.log"
  local N_log="${OUT}/phaseN_${label}_lam${lam}.log"

  if [[ -s "${enc_ckpt}" ]]; then
    echo "[skip] finetune ${label} lam=${lam}"
  else
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

  if [[ -s "${oracle_ckpt}" ]]; then
    echo "[skip] oracle ${label} lam=${lam}"
  else
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
    if [[ "${env_kind}" == "crafter" ]]; then
      python analyze_wm_crafter.py \
        --model_path "${oracle_ckpt}" \
        --probe_frames 8000 --n_rollouts 200 --horizons 1 3 5 10 \
        --device cuda --output_json "${phaseA_out}" > "${A_log}" 2>&1
    else
      python analyze_wm_multistep.py \
        --model_path "${oracle_ckpt}" \
        --n_rollouts 300 --probe_frames 10000 --horizons 1 3 5 10 \
        --device cuda --output_json "${phaseA_out}" > "${A_log}" 2>&1
    fi
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

# Source list: (label, ckpt_path, target_dir, env_kind)
declare -a SOURCES=(
  "vae_s4|${DK8}/sweep_dk8_vae_s4_best_model.pt|${PR_DK8}|minigrid"
  "vae_s5|${DK8}/sweep_dk8_vae_s5_best_model.pt|${PR_DK8}|minigrid"
  "dk16_v6_s2|${DK16}/sweep_doorkey16_v6_s2_best_model.pt|${PR_DK16}|minigrid"
  "dk16_v6_s3|${DK16}/sweep_doorkey16_v6_s3_best_model.pt|${PR_DK16}|minigrid"
  "v6_s5|${DK8}/sweep_dk8_v6_s5_best_model.pt|${PR_DK8}|minigrid"
  "v5dc_s5|${DK8}/sweep_dk8_v5dc_s5_best_model.pt|${PR_DK8}|minigrid"
  "crafter_cal_best|${CRAFTER}/mf_e2e_semantic_crafter_v6enc_cal_best_model.pt|${PR_CRAFTER}|crafter"
  "crafter_cal2_best|${CRAFTER}/mf_e2e_semantic_crafter_v6enc_cal2_best_model.pt|${PR_CRAFTER}|crafter"
  "crafter_fix_best|${CRAFTER}/mf_e2e_semantic_crafter_v6enc_fix_best_model.pt|${PR_CRAFTER}|crafter"
)

for entry in "${SOURCES[@]}"; do
  IFS='|' read -r label ckpt enc_dir env_kind <<< "${entry}"
  if [[ ! -f "${ckpt}" ]]; then echo "[skip-miss] ${label}"; continue; fi
  for lam in "${LAMBDAS[@]}"; do
    run_pipeline "${label}" "${ckpt}" "${enc_dir}" "${lam}" "${env_kind}"
  done
done

echo "Phase Z complete."
