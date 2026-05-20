#!/usr/bin/env bash
# Phase AA: causal λ-sweep on MiniGrid LavaCrossing-S9N1 (cross-env beyond
# DoorKey within MiniGrid). Uses pre-trained LavaCrossing checkpoints from
# the original codebase to avoid the ~2h-per-encoder training cost.
#
# 2 source encoders (different training procedures) × 4 λ values = 8 cells.
# Combined with R/R-multi/W/X/Y/Z, brings the causal claim to >=4 environments.

set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LAVA="${REPO}/discrete_mbrl/model_free/models/MiniGrid-LavaCrossingS9N1-v0"
OUT="${REPO}/logs/phaseAA"
PR_LAVA="${LAVA}/phaseAA"
mkdir -p "${OUT}" "${PR_LAVA}"

LAMBDAS=(0.0 0.5 2.0 10.0)
cd "${REPO}/discrete_mbrl"

run_pipeline() {
  local label="$1" src_ckpt="$2" lam="$3"
  local enc_ckpt="${PR_LAVA}/finetuned_${label}_lam${lam}.pt"
  local oracle_ckpt="${PR_LAVA}/oracle_finetuned_${label}_lam${lam}.pt"
  local phaseA_out="${OUT}/phaseA_${label}_lam${lam}.json"
  local phaseN_out="${OUT}/phaseN_${label}_lam${lam}.json"
  local ft_log="${OUT}/finetune_${label}_lam${lam}.log"
  local oracle_log="${OUT}/oracle_${label}_lam${lam}.log"
  local A_log="${OUT}/phaseA_${label}_lam${lam}.log"
  local N_log="${OUT}/phaseN_${label}_lam${lam}.log"

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

# Three LavaCrossing encoders covering distinct training procedures.
# (lava_snapback had a dtype bug on one variant, swapped for the v2 + ema_tau)
SOURCES=(
  "lava_semaux|${LAVA}/mf_e2e_semantic_aux_best_model.pt"
  "lava_semaux_v2|${LAVA}/mf_e2e_semantic_aux_v2_best_model.pt"
  "lava_ema_tau|${LAVA}/e2e_ema_tau_best_model.pt"
)

for entry in "${SOURCES[@]}"; do
  IFS='|' read -r label ckpt <<< "${entry}"
  if [[ ! -f "${ckpt}" ]]; then echo "[skip-miss] ${label}"; continue; fi
  for lam in "${LAMBDAS[@]}"; do
    run_pipeline "${label}" "${ckpt}" "${lam}"
  done
done

echo "Phase AA complete."
