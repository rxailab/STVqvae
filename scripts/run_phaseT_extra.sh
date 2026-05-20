#!/usr/bin/env bash
# Phase T-extra: extend Dyna seeds from {1,2} to {3,4,5} so each (λ, mode)
# cell has N=5 seeds. Reuses the existing Phase R fine-tuned encoders and
# oracle WMs in models/MiniGrid-DoorKey-8x8-v0/phaseR/.
#
# 3 new seeds × 4 λ values × 2 modes (none, oracle) = 24 runs
# ~5.5 min/run × 24 = ~2.2h wall.
#
# Idempotent: existing dyna_lam*_*_s{1,2}.pt are untouched; new seeds 3-5
# skip-guard on dyna_lam*_*_s{3,4,5}.pt.

set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CKPT_DIR="${REPO}/discrete_mbrl/model_free/models/MiniGrid-DoorKey-8x8-v0"
PR_CKPTS="${CKPT_DIR}/phaseR"
OUT="${REPO}/logs/phaseT"
DYNA_CKPTS="${CKPT_DIR}/phaseT"
mkdir -p "${OUT}" "${DYNA_CKPTS}"

LAMBDAS=(0.0 0.5 2.0 10.0)
EXTRA_SEEDS=(3 4 5)

cd "${REPO}/discrete_mbrl"

run_dyna() {
  local lam="$1" mode="$2" wm_path="$3" seed="$4"
  local out_ckpt="${DYNA_CKPTS}/dyna_lam${lam}_${mode}_s${seed}.pt"
  local log="${OUT}/dyna_lam${lam}_${mode}_s${seed}.log"
  local enc_ckpt="${PR_CKPTS}/finetuned_lam${lam}.pt"
  if [[ -s "${out_ckpt}" ]]; then echo "[skip] dyna lam=${lam} mode=${mode} s=${seed}"; return; fi
  if [[ ! -f "${enc_ckpt}" ]]; then echo "[skip-miss] enc lam=${lam}"; return; fi
  echo "[dyna-T] lam=${lam} mode=${mode} seed=${seed}"
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
  for seed in "${EXTRA_SEEDS[@]}"; do
    run_dyna "${lam}" none   ""          "${seed}"
    run_dyna "${lam}" oracle "${ORACLE}" "${seed}"
  done
done
echo "Phase T-extra complete."
