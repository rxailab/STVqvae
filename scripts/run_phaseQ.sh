#!/usr/bin/env bash
# Phase Q: action-conditioning probe on Crafter (cross-environment replication
# of Phase N). Tests whether the encoder-determined action-blindness pattern
# generalizes beyond DK-8x8.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CKPT_DIR="${REPO}/discrete_mbrl/model_free/models/crafter"
OUT="${REPO}/logs/phaseQ"
mkdir -p "${OUT}"
cd "${REPO}/discrete_mbrl"
for ckpt in "${CKPT_DIR}"/mf_e2e_semantic_crafter_*.pt; do
  base=$(basename "${ckpt}" .pt)
  out="${OUT}/phaseQ_${base}.json"
  log="${OUT}/phaseQ_${base}.log"
  if [[ -s "${out}" ]]; then echo "[skip] ${base}"; continue; fi
  echo "[run] ${base}"
  set +e
  python analyze_wm_action_cond.py --model_path "${ckpt}" \
    --n_states 2000 --device cuda --output_json "${out}" > "${log}" 2>&1
  rc=$?; set -e
  if [[ $rc -ne 0 ]]; then echo "[fail] ${base}"; rm -f "${out}"; else echo "[ok] ${base}"; fi
done
