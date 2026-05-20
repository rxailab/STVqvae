#!/usr/bin/env bash
# Phase R: causal test of probe-vs-action-conditioning trade-off.
#
# 1. Fine-tune vae_s1 encoder with auxiliary action-prediction loss at
#    λ ∈ {0, 0.5, 2.0, 10.0} (4 fine-tuned encoders).
# 2. Train an oracle WM on each fine-tuned encoder.
# 3. Run analyze_wm_multistep (Phase A: probe + WM accuracy) and
#    analyze_wm_action_cond (Phase N: action_dep_ratio) on each.
# 4. Aggregate into a (λ, probe_acc, action_dep_ratio) table.
#
# Hypothesis: as λ ↑, action_dep_ratio ↑ AND probe_accuracy ↓ (causal trade-off).
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CKPT_DIR="${REPO}/discrete_mbrl/model_free/models/MiniGrid-DoorKey-8x8-v0"
OUT="${REPO}/logs/phaseR"
PR_CKPTS="${CKPT_DIR}/phaseR"
mkdir -p "${OUT}" "${PR_CKPTS}"

SRC="${CKPT_DIR}/sweep_dk8_vae_s1_best_model.pt"
LAMBDAS=(0.0 0.5 2.0 10.0)

cd "${REPO}/discrete_mbrl"

for lam in "${LAMBDAS[@]}"; do
  ENC_CKPT="${PR_CKPTS}/finetuned_lam${lam}.pt"
  ORACLE_CKPT="${PR_CKPTS}/oracle_finetuned_lam${lam}.pt"
  PHASEA_OUT="${OUT}/phaseA_lam${lam}.json"
  PHASEN_OUT="${OUT}/phaseN_lam${lam}.json"
  FT_LOG="${OUT}/finetune_lam${lam}.log"
  ORACLE_LOG="${OUT}/oracle_lam${lam}.log"
  A_LOG="${OUT}/phaseA_lam${lam}.log"
  N_LOG="${OUT}/phaseN_lam${lam}.log"

  # --- Step 1: fine-tune encoder ---
  if [[ -s "${ENC_CKPT}" ]]; then
    echo "[skip] finetune lam=${lam}"
  else
    echo "[finetune] lam=${lam}"
    set +e
    python finetune_encoder_action_cond.py \
      --model_path "${SRC}" --output_path "${ENC_CKPT}" \
      --action_aux_weight "${lam}" --epochs 10 \
      --n_random_eps 200 --n_policy_eps 200 \
      --batch_size 128 --lr 1e-4 \
      --device cuda > "${FT_LOG}" 2>&1
    rc=$?; set -e
    if [[ $rc -ne 0 ]]; then echo "[fail-ft] lam=${lam}"; rm -f "${ENC_CKPT}"; continue; fi
  fi

  # --- Step 2: train oracle WM on fine-tuned encoder ---
  if [[ -s "${ORACLE_CKPT}" ]]; then
    echo "[skip] oracle WM lam=${lam}"
  else
    echo "[oracle] lam=${lam}"
    set +e
    python train_oracle_wm.py \
      --model_path "${ENC_CKPT}" --output_path "${ORACLE_CKPT}" \
      --n_random_eps 250 --n_policy_eps 250 \
      --epochs 50 --batch_size 256 --lr 1e-4 \
      --device cuda > "${ORACLE_LOG}" 2>&1
    rc=$?; set -e
    if [[ $rc -ne 0 ]]; then echo "[fail-oracle] lam=${lam}"; rm -f "${ORACLE_CKPT}"; continue; fi
  fi

  # --- Step 3a: Phase A (probe + WM accuracy) ---
  if [[ -s "${PHASEA_OUT}" ]]; then
    echo "[skip] Phase A lam=${lam}"
  else
    echo "[phaseA] lam=${lam}"
    set +e
    python analyze_wm_multistep.py \
      --model_path "${ORACLE_CKPT}" \
      --n_rollouts 300 --probe_frames 10000 \
      --horizons 1 3 5 10 \
      --device cuda --output_json "${PHASEA_OUT}" > "${A_LOG}" 2>&1
    rc=$?; set -e
    if [[ $rc -ne 0 ]]; then echo "[fail-A] lam=${lam}"; rm -f "${PHASEA_OUT}"; fi
  fi

  # --- Step 3b: Phase N (action-conditioning probe) ---
  if [[ -s "${PHASEN_OUT}" ]]; then
    echo "[skip] Phase N lam=${lam}"
  else
    echo "[phaseN] lam=${lam}"
    set +e
    python analyze_wm_action_cond.py \
      --model_path "${ORACLE_CKPT}" \
      --n_states 2000 --device cuda \
      --output_json "${PHASEN_OUT}" > "${N_LOG}" 2>&1
    rc=$?; set -e
    if [[ $rc -ne 0 ]]; then echo "[fail-N] lam=${lam}"; rm -f "${PHASEN_OUT}"; fi
  fi

  echo "[ok] lam=${lam}"
done

echo "Phase R complete."
