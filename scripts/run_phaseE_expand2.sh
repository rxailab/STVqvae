#!/usr/bin/env bash
# Action 2: extend Phase E2 from N=10 to N=15. Pick 5 more source ckpts
# beyond the existing 10. Mixed buffer (300 random + 300 policy), same recipe.

set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CKPT_DIR="${REPO}/discrete_mbrl/model_free/models/MiniGrid-DoorKey-8x8-v0"
OUT="${REPO}/logs/phaseE/oracle_analyzer"
TRAIN_LOG="${REPO}/logs/phaseE/oracle_train"
mkdir -p "${OUT}" "${TRAIN_LOG}"

# 5 fresh source ckpts not yet in the oracle pool
SOURCES=(
  "sweep_dk8_vae_s4_best_model"
  "sweep_dk8_vae_s5_best_model"
  "rigor_dk8_v6_s1_best_model"
  "rigor_dk8_v6_s2_best_model"
  "mf_e2e_semantic_doorkey_v6enc_goal_best_model"
)

cd "${REPO}/discrete_mbrl"

for src in "${SOURCES[@]}"; do
  src_path="${CKPT_DIR}/${src}.pt"
  oracle="${CKPT_DIR}/oracle_${src}.pt"
  out="${OUT}/oracle_${src}.json"
  train_log="${TRAIN_LOG}/oracle_${src}_train.log"
  ana_log="${OUT}/oracle_${src}.log"

  if [[ ! -f "${src_path}" ]]; then echo "[skip-miss] ${src}"; continue; fi
  if [[ -s "${out}" ]]; then echo "[skip] ${src}"; continue; fi

  echo "[train] ${src}"
  set +e
  python train_oracle_wm.py \
    --model_path "${src_path}" --output_path "${oracle}" \
    --n_random_eps 300 --n_policy_eps 300 \
    --epochs 100 --batch_size 512 --lr 1e-4 \
    --device cuda > "${train_log}" 2>&1
  rc=$?; set -e
  if [[ $rc -ne 0 ]]; then echo "[fail-train] ${src}"; continue; fi

  echo "[analyze] ${src}"
  set +e
  python analyze_wm_multistep.py \
    --model_path "${oracle}" \
    --n_rollouts 300 --probe_frames 10000 \
    --horizons 1 3 5 10 \
    --device cuda --output_json "${out}" > "${ana_log}" 2>&1
  rc=$?; set -e
  if [[ $rc -ne 0 ]]; then echo "[fail-analyze] ${src}"; rm -f "${out}"; else echo "[ok] ${src}"; fi
done
