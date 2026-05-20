#!/usr/bin/env bash
# Phase R-multi: causal λ-sweep on additional source encoders to test
# whether the (probe accuracy) ↔ (action_dep_ratio) trade-off generalizes
# beyond vae_s1.
#
# Source encoders: vae_s2 (action_dep_ratio_orig ≈ 0.68), vae_s3 (≈ 0.34)
# These have different starting points than vae_s1's ≈ 0.0003. If the
# Pareto trade-off still holds (probe drops as λ rises), the causal claim
# is robust across encoder seeds.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CKPT_DIR="${REPO}/discrete_mbrl/model_free/models/MiniGrid-DoorKey-8x8-v0"
OUT="${REPO}/logs/phaseR_multi"
PR_CKPTS="${CKPT_DIR}/phaseR_multi"
mkdir -p "${OUT}" "${PR_CKPTS}"

LAMBDAS=(0.0 0.5 2.0 10.0)
SOURCES=("vae_s2" "vae_s3")

cd "${REPO}/discrete_mbrl"

for src in "${SOURCES[@]}"; do
  SRC_CKPT="${CKPT_DIR}/sweep_dk8_${src}_best_model.pt"
  if [[ ! -f "${SRC_CKPT}" ]]; then echo "[skip-miss] ${src}"; continue; fi

  for lam in "${LAMBDAS[@]}"; do
    ENC_CKPT="${PR_CKPTS}/finetuned_${src}_lam${lam}.pt"
    ORACLE_CKPT="${PR_CKPTS}/oracle_finetuned_${src}_lam${lam}.pt"
    PHASEA_OUT="${OUT}/phaseA_${src}_lam${lam}.json"
    PHASEN_OUT="${OUT}/phaseN_${src}_lam${lam}.json"
    FT_LOG="${OUT}/finetune_${src}_lam${lam}.log"
    ORACLE_LOG="${OUT}/oracle_${src}_lam${lam}.log"
    A_LOG="${OUT}/phaseA_${src}_lam${lam}.log"
    N_LOG="${OUT}/phaseN_${src}_lam${lam}.log"

    if [[ -s "${ENC_CKPT}" ]]; then
      echo "[skip] finetune ${src} lam=${lam}"
    else
      echo "[finetune] ${src} lam=${lam}"
      set +e
      python finetune_encoder_action_cond.py \
        --model_path "${SRC_CKPT}" --output_path "${ENC_CKPT}" \
        --action_aux_weight "${lam}" --epochs 10 \
        --n_random_eps 200 --n_policy_eps 200 \
        --batch_size 128 --lr 1e-4 \
        --device cuda > "${FT_LOG}" 2>&1
      rc=$?; set -e
      if [[ $rc -ne 0 ]]; then echo "[fail-ft] ${src} lam=${lam}"; rm -f "${ENC_CKPT}"; continue; fi
    fi

    if [[ -s "${ORACLE_CKPT}" ]]; then
      echo "[skip] oracle ${src} lam=${lam}"
    else
      echo "[oracle] ${src} lam=${lam}"
      set +e
      python train_oracle_wm.py \
        --model_path "${ENC_CKPT}" --output_path "${ORACLE_CKPT}" \
        --n_random_eps 250 --n_policy_eps 250 \
        --epochs 50 --batch_size 256 --lr 1e-4 \
        --device cuda > "${ORACLE_LOG}" 2>&1
      rc=$?; set -e
      if [[ $rc -ne 0 ]]; then echo "[fail-oracle] ${src} lam=${lam}"; rm -f "${ORACLE_CKPT}"; continue; fi
    fi

    if [[ ! -s "${PHASEA_OUT}" ]]; then
      echo "[phaseA] ${src} lam=${lam}"
      set +e
      python analyze_wm_multistep.py \
        --model_path "${ORACLE_CKPT}" \
        --n_rollouts 300 --probe_frames 10000 \
        --horizons 1 3 5 10 \
        --device cuda --output_json "${PHASEA_OUT}" > "${A_LOG}" 2>&1
      rc=$?; set -e
      if [[ $rc -ne 0 ]]; then echo "[fail-A] ${src} lam=${lam}"; rm -f "${PHASEA_OUT}"; fi
    fi

    if [[ ! -s "${PHASEN_OUT}" ]]; then
      echo "[phaseN] ${src} lam=${lam}"
      set +e
      python analyze_wm_action_cond.py \
        --model_path "${ORACLE_CKPT}" \
        --n_states 2000 --device cuda \
        --output_json "${PHASEN_OUT}" > "${N_LOG}" 2>&1
      rc=$?; set -e
      if [[ $rc -ne 0 ]]; then echo "[fail-N] ${src} lam=${lam}"; rm -f "${PHASEN_OUT}"; fi
    fi
    echo "[ok] ${src} lam=${lam}"
  done
done
echo "Phase R-multi complete."
