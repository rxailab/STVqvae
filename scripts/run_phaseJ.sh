#!/usr/bin/env bash
# Phase J: V(ẑ_k) vs V(z*_k) trajectory diagnostic.
# Extends E8 from {k=1,5,10} to {k=1..10} on online + oracle ckpts to locate
# where in the rollout value coherence breaks. Same analyzer, different horizons.

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CKPT_DIR="${REPO}/discrete_mbrl/model_free/models/MiniGrid-DoorKey-8x8-v0"
OUT_DIR="${REPO}/logs/phaseJ"
N_EPISODES="${N_EPISODES:-50}"
MAX_STEPS="${MAX_STEPS:-150}"
HORIZONS="${HORIZONS:-1 2 3 4 5 6 7 8 9 10}"
DEVICE="${DEVICE:-cuda}"

mkdir -p "${OUT_DIR}"

# 5 online + 5 oracle = 10 ckpts to map v_corr(k) curve
CKPTS=(
  # Online
  "${CKPT_DIR}/sweep_dk8_v6_s4_best_model.pt"
  "${CKPT_DIR}/sweep_dk8_v6_s5_best_model.pt"
  "${CKPT_DIR}/sweep_dk8_v2_s4_best_model.pt"
  "${CKPT_DIR}/sweep_dk8_v5dc_s4_best_model.pt"
  "${CKPT_DIR}/sweep_dk8_vae_s1_best_model.pt"
  # Oracle
  "${CKPT_DIR}/oracle_sweep_dk8_v6_s4_best_model.pt"
  "${CKPT_DIR}/oracle_sweep_dk8_v6_s5_best_model.pt"
  "${CKPT_DIR}/oracle_sweep_dk8_v5dc_s4_best_model.pt"
  "${CKPT_DIR}/oracle_sweep_dk8_vae_s1_best_model.pt"
  "${CKPT_DIR}/oracle_sweep_dk8_vae_s2_best_model.pt"
)

cd "${REPO}/discrete_mbrl"

for src in "${CKPTS[@]}"; do
  if [[ ! -f "${src}" ]]; then echo "[skip-miss] $(basename ${src})"; continue; fi
  name="$(basename "${src}" .pt)"
  out="${OUT_DIR}/${name}.json"
  log="${OUT_DIR}/${name}.log"
  if [[ -s "${out}" ]]; then echo "[skip] ${name}"; continue; fi

  echo "[run] ${name}"
  set +e
  python analyze_wm_phaseB_e8.py \
    --model_path "${src}" \
    --n_episodes "${N_EPISODES}" --max_steps "${MAX_STEPS}" \
    --horizons ${HORIZONS} \
    --device "${DEVICE}" \
    --output_json "${out}" > "${log}" 2>&1
  rc=$?; set -e
  if [[ $rc -ne 0 ]]; then echo "[fail] ${name}"; rm -f "${out}"; else echo "[ok] ${name}"; fi
done
