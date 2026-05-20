#!/usr/bin/env bash
# Crafter sweep v2: fixed get_semantic() wrapper unwrapping bug.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CKPT_DIR="${REPO}/discrete_mbrl/model_free/models/crafter"
OUT="${REPO}/logs/crafter"
mkdir -p "${OUT}"
cd "${REPO}/discrete_mbrl"
for ckpt in "${CKPT_DIR}"/mf_e2e_semantic_crafter_*.pt; do
  base=$(basename "${ckpt}" .pt)
  out="${OUT}/${base}.json"
  log="${OUT}/${base}.log"
  if [[ -s "${out}" ]]; then echo "[skip] ${base}"; continue; fi
  echo "[run] ${base}"
  set +e
  python analyze_wm_crafter.py \
    --model_path "${ckpt}" \
    --probe_frames 8000 --n_rollouts 200 \
    --horizons 1 3 5 10 \
    --device cuda --output_json "${out}" > "${log}" 2>&1
  rc=$?; set -e
  if [[ $rc -ne 0 ]]; then echo "[fail] ${base}"; rm -f "${out}"; else echo "[ok] ${base}"; fi
done
