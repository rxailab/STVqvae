#!/usr/bin/env bash
# Phase T-more: extend Dyna seeds from {1..5} to {1..15} so each (lam, mode)
# cell has N=15 total. Reuses Phase R fine-tuned encoders + oracle WMs.
#
# 10 new seeds x 4 lambda x 2 modes = 80 runs.
# ~5.5 min/run x 80 = ~7.3h wall.
#
# Idempotent: skip-guard on per-(lam, mode, seed) ckpt path; existing seeds
# 1-5 (from run_phaseT.sh + run_phaseT_extra.sh) are untouched.

set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CKPT_DIR="${REPO}/discrete_mbrl/model_free/models/MiniGrid-DoorKey-8x8-v0"
PR_CKPTS="${CKPT_DIR}/phaseR"
OUT="${REPO}/logs/phaseT"
DYNA_CKPTS="${CKPT_DIR}/phaseT"
mkdir -p "${OUT}" "${DYNA_CKPTS}"

LAMBDAS=(0.0 0.5 2.0 10.0)
MORE_SEEDS=(6 7 8 9 10 11 12 13 14 15)

cd "${REPO}/discrete_mbrl"

run_dyna() {
  local lam="$1" mode="$2" wm_path="$3" seed="$4"
  local out_ckpt="${DYNA_CKPTS}/dyna_lam${lam}_${mode}_s${seed}.pt"
  local log="${OUT}/dyna_lam${lam}_${mode}_s${seed}.log"
  local enc_ckpt="${PR_CKPTS}/finetuned_lam${lam}.pt"
  if [[ -s "${out_ckpt}" ]]; then echo "[skip] dyna lam=${lam} mode=${mode} s=${seed}"; return; fi
  if [[ ! -f "${enc_ckpt}" ]]; then echo "[skip-miss] enc lam=${lam}"; return; fi
  echo "[dyna-T-more] lam=${lam} mode=${mode} seed=${seed}"
  set +e
  PYTHONHASHSEED=${seed} python train_dyna.py \
    --src_ckpt "${enc_ckpt}" \
    --wm_ckpt  "${wm_path}" \
    --wm_mode  "${mode}" \
    --output_path "${out_ckpt}" \
    --total_steps 300000 \
    --num_envs 16 --n_steps_per_update 128 \
    --imagine_K 5 --imagine_ratio 0.5 \
    --ppo_entropy_coef 0.02 \
    --lr 3e-4 --device cuda > "${log}" 2>&1
  rc=$?; set -e
  if [[ $rc -ne 0 ]]; then echo "[fail] dyna lam=${lam} mode=${mode} s=${seed}"; rm -f "${out_ckpt}"; fi
}

for lam in "${LAMBDAS[@]}"; do
  ORACLE="${PR_CKPTS}/oracle_finetuned_lam${lam}.pt"
  for seed in "${MORE_SEEDS[@]}"; do
    run_dyna "${lam}" none   ""          "${seed}"
    run_dyna "${lam}" oracle "${ORACLE}" "${seed}"
  done
done
echo "Phase T-more complete."
