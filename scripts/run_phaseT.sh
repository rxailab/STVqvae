#!/usr/bin/env bash
# Phase T: does Phase R's encoder fix translate to better planning?
#
# For each of the 4 Phase R ckpts (oracle WMs trained on action-aware
# encoders at λ ∈ {0, 0.5, 2.0, 10.0}):
#   (a) run analyze_wm_planning at horizons {1, 3, 5, 10} → rank_corr
#   (b) run train_dyna at imagine_ratio=0.5 (Phase O recipe) × 2 seeds × 3 modes
#
# If λ>0 encoders give higher rank_corr AND Dyna actually helps over vanilla,
# Phase R's encoder fix translates to actionable planning gains. If not,
# the negative claim hardens further.

set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CKPT_DIR="${REPO}/discrete_mbrl/model_free/models/MiniGrid-DoorKey-8x8-v0"
PR_CKPTS="${CKPT_DIR}/phaseR"
OUT="${REPO}/logs/phaseT"
DYNA_CKPTS="${CKPT_DIR}/phaseT"
mkdir -p "${OUT}" "${DYNA_CKPTS}"

LAMBDAS=(0.0 0.5 2.0 10.0)

cd "${REPO}/discrete_mbrl"

# --- Part 1: planning rank_corr per λ (Phase K-style) ---
for lam in "${LAMBDAS[@]}"; do
  ENC="${PR_CKPTS}/finetuned_lam${lam}.pt"
  ORACLE="${PR_CKPTS}/oracle_finetuned_lam${lam}.pt"
  if [[ ! -f "${ORACLE}" ]]; then echo "[skip-miss] oracle lam=${lam}"; continue; fi

  out="${OUT}/phaseT_planning_lam${lam}.json"
  log="${OUT}/phaseT_planning_lam${lam}.log"
  if [[ -s "${out}" ]]; then echo "[skip] planning lam=${lam}"; continue; fi
  echo "[planning] lam=${lam}"
  set +e
  python analyze_wm_planning.py --model_path "${ORACLE}" \
    --n_episodes 300 --max_steps 200 --horizons 1 3 5 10 \
    --stochastic --device cuda --output_json "${out}" > "${log}" 2>&1
  rc=$?; set -e
  if [[ $rc -ne 0 ]]; then echo "[fail] planning lam=${lam}"; rm -f "${out}"; fi
done

# --- Part 2: Dyna comparison per λ (Phase O recipe) ---
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
  for seed in 1 2; do
    run_dyna "${lam}" none   ""          "${seed}"
    run_dyna "${lam}" oracle "${ORACLE}" "${seed}"
  done
done
echo "Phase T complete."
