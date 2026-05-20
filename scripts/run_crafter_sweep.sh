#!/usr/bin/env bash
# Action 4: Crafter cross-environment sweep — replicate Phase A on the 8
# Crafter ckpts already in the repo. Tests whether the probe-WM dissociation
# transfers beyond MiniGrid DoorKey.

set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CKPT_DIR="${REPO}/discrete_mbrl/model_free/models/crafter"
OUT="${REPO}/logs/crafter"
mkdir -p "${OUT}"

cd "${REPO}/discrete_mbrl"
mapfile -t CKPTS < <(find "${CKPT_DIR}" -maxdepth 1 -name '*.pt' | sort)

for ckpt in "${CKPTS[@]}"; do
  name="$(basename "${ckpt}" .pt)"
  out="${OUT}/${name}.json"
  log="${OUT}/${name}.log"
  if [[ -s "${out}" ]]; then echo "[skip] ${name}"; continue; fi

  echo "[run] ${name}"
  set +e
  python analyze_wm_crafter.py \
    --model_path "${ckpt}" \
    --probe_frames 8000 --n_rollouts 200 \
    --horizons 1 3 5 10 \
    --device cuda --output_json "${out}" > "${log}" 2>&1
  rc=$?; set -e
  if [[ $rc -ne 0 ]]; then echo "[fail] ${name}"; rm -f "${out}"; else echo "[ok] ${name}"; fi
done
