#!/usr/bin/env bash
# Phase X (DK-16) + Phase Y (Crafter): cross-task and cross-environment
# causal λ-sweep. Together with R-multi (DK-8 VAE) and W (DK-8 VQ), this
# extends the causal Pareto to:
#   - 2 architecture classes (continuous VAE + VQ-VAE)
#   - 3 environments (DK-8, DK-16, Crafter)
#   - 6 source encoders × 4 λ = 24 data points (combined with previous 20)

set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DK8="${REPO}/discrete_mbrl/model_free/models/MiniGrid-DoorKey-8x8-v0"
DK16="${REPO}/discrete_mbrl/model_free/models/MiniGrid-DoorKey-16x16-v0"
CRAFTER="${REPO}/discrete_mbrl/model_free/models/crafter"

OUT="${REPO}/logs/phaseXY"
PR_CKPTS_X="${DK16}/phaseX"
PR_CKPTS_Y="${CRAFTER}/phaseY"
mkdir -p "${OUT}" "${PR_CKPTS_X}" "${PR_CKPTS_Y}"

LAMBDAS=(0.0 0.5 2.0 10.0)

cd "${REPO}/discrete_mbrl"

run_pipeline() {
  local label="$1" src_ckpt="$2" enc_dir="$3" lam="$4"
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
    # Choose analyzer based on environment
    if [[ "${label}" == crafter* ]]; then
      python analyze_wm_crafter.py \
        --model_path "${oracle_ckpt}" \
        --probe_frames 8000 --n_rollouts 200 \
        --horizons 1 3 5 10 \
        --device cuda --output_json "${phaseA_out}" > "${A_log}" 2>&1
    else
      python analyze_wm_multistep.py \
        --model_path "${oracle_ckpt}" \
        --n_rollouts 300 --probe_frames 10000 \
        --horizons 1 3 5 10 \
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

# --- Phase X: DK-16 v6_s1 (cross-task replication) ---
SRC_X="${DK16}/sweep_doorkey16_v6_s1_best_model.pt"
if [[ -f "${SRC_X}" ]]; then
  for lam in "${LAMBDAS[@]}"; do
    run_pipeline "dk16_v6_s1" "${SRC_X}" "${PR_CKPTS_X}" "${lam}"
  done
else
  echo "[skip-miss] DK-16 source"
fi

# --- Phase Y: Crafter v6enc (cross-environment replication) ---
SRC_Y="${CRAFTER}/mf_e2e_semantic_crafter_v6enc_best_model.pt"
if [[ -f "${SRC_Y}" ]]; then
  for lam in "${LAMBDAS[@]}"; do
    run_pipeline "crafter_v6enc" "${SRC_Y}" "${PR_CKPTS_Y}" "${lam}"
  done
else
  echo "[skip-miss] Crafter source"
fi

echo "Phase XY complete."
