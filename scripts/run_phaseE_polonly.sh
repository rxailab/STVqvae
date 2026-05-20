#!/usr/bin/env bash
# Action 1: policy-only-buffer control. Train oracle WMs with n_random_eps=0,
# n_policy_eps=600 on the same 5 source ckpts, then evaluate under
# random-action rollouts. If gap closure holds, distribution-overlap is
# definitively ruled out.

set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CKPT_DIR="${REPO}/discrete_mbrl/model_free/models/MiniGrid-DoorKey-8x8-v0"
OUT="${REPO}/logs/phaseE/oracle_analyzer"
TRAIN_LOG="${REPO}/logs/phaseE/oracle_train"
mkdir -p "${OUT}" "${TRAIN_LOG}"

SOURCES=(
  "sweep_dk8_v6_s4_best_model"
  "sweep_dk8_v6_s5_best_model"
  "sweep_dk8_v5dc_s4_best_model"
  "sweep_dk8_vae_s1_best_model"
  "sweep_dk8_vae_s2_best_model"
)

cd "${REPO}/discrete_mbrl"

for src in "${SOURCES[@]}"; do
  src_path="${CKPT_DIR}/${src}.pt"
  oracle="${CKPT_DIR}/oracle_${src}_polonly.pt"
  out="${OUT}/oracle_${src}_polonly.json"
  train_log="${TRAIN_LOG}/oracle_${src}_polonly_train.log"
  ana_log="${OUT}/oracle_${src}_polonly.log"

  if [[ ! -f "${src_path}" ]]; then echo "[skip-miss] ${src}"; continue; fi
  if [[ -s "${out}" ]]; then echo "[skip] ${src}_polonly"; continue; fi

  echo "[train] ${src}_polonly  (n_random=0, n_policy=600)"
  set +e
  python train_oracle_wm.py \
    --model_path "${src_path}" --output_path "${oracle}" \
    --n_random_eps 0 --n_policy_eps 600 \
    --epochs 100 --batch_size 512 --lr 1e-4 \
    --device cuda > "${train_log}" 2>&1
  rc=$?; set -e
  if [[ $rc -ne 0 ]]; then echo "[fail-train] ${src}_polonly"; continue; fi

  echo "[analyze] ${src}_polonly"
  set +e
  python analyze_wm_multistep.py \
    --model_path "${oracle}" \
    --n_rollouts 300 --probe_frames 10000 \
    --horizons 1 3 5 10 \
    --device cuda --output_json "${out}" > "${ana_log}" 2>&1
  rc=$?; set -e
  if [[ $rc -ne 0 ]]; then echo "[fail-analyze] ${src}_polonly"; rm -f "${out}"; else echo "[ok] ${src}_polonly"; fi
done
