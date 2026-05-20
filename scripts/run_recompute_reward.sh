#!/usr/bin/env bash
# Phase RR: recompute end-of-training reward for the 12 DK-8 checkpoints whose
# live reward log is not part of the released artifacts (rows shown as "---"
# in Table~\ref{tab:per-checkpoint}). Replaces the dashes with deterministic
# 200-episode mean +- std evaluated post-hoc from the saved checkpoint.
#
# 4 encoder versions (v2, v5, v6, v9) x 3 seeds = 12 ckpts.
# ~200 episodes/ckpt at ~50 steps each => ~2 min/ckpt => ~25 min total.
#
# Idempotent: skip-guard on per-ckpt JSON.

set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="${REPO}/logs/phaseRR"
CKPT_DIR="${REPO}/discrete_mbrl/model_free/models/MiniGrid-DoorKey-8x8-v0"
mkdir -p "${LOG_DIR}"

cd "${REPO}/discrete_mbrl"

run_one() {
  local enc_version="$1" enc_seed="$2"
  local ckpt="${CKPT_DIR}/sweep_doorkey8_${enc_version}_s${enc_seed}_best_model.pt"
  local label="${enc_version}_s${enc_seed}"
  local out="${LOG_DIR}/eval_${label}.json"
  local log="${LOG_DIR}/eval_${label}.log"

  if [[ -s "${out}" ]]; then echo "[skip] ${label}"; return; fi
  if [[ ! -f "${ckpt}" ]]; then echo "[miss] ${label}"; return; fi

  echo "[eval] ${label}"
  set +e
  python eval_model_free.py \
    --model_path "${ckpt}" \
    --env_name MiniGrid-DoorKey-8x8-v0 \
    --ae_model_type vqvae \
    --ae_model_version "${enc_version#v}" \
    --embedding_dim 64 \
    --filter_size 8 \
    --codebook_size 64 \
    --n_episodes 200 \
    --max_steps 500 \
    --device cuda > "${log}" 2>&1
  rc=$?; set -e

  # Parse the printed reward summary; fail-soft.
  if [[ $rc -eq 0 ]]; then
    python - "${log}" "${out}" "${enc_version}" "${enc_seed}" <<'PY'
import json, re, sys
log_path, out_path, enc_v, enc_s = sys.argv[1:]
with open(log_path) as f: txt = f.read()
m_mean = re.search(r'Mean reward[^\d-]*([\-\+]?\d*\.?\d+)', txt)
m_std  = re.search(r'Std[^\d]*([\-\+]?\d*\.?\d+)', txt)
m_succ = re.search(r'Success rate[^\d]*([\-\+]?\d*\.?\d+)', txt)
m_len  = re.search(r'Mean length[^\d]*([\-\+]?\d*\.?\d+)', txt)
out = {
    'encoder_version': enc_v,
    'seed': int(enc_s),
    'log': log_path,
    'mean_reward': float(m_mean.group(1)) if m_mean else None,
    'std_reward':  float(m_std.group(1))  if m_std  else None,
    'success_rate': float(m_succ.group(1)) if m_succ else None,
    'mean_length': float(m_len.group(1)) if m_len else None,
}
with open(out_path, 'w') as f: json.dump(out, f, indent=2)
print(f'Parsed: mean={out["mean_reward"]} std={out["std_reward"]} succ={out["success_rate"]}')
PY
  else
    echo "[fail-eval] ${label} (rc=$rc)"
  fi
}

for v in v2 v5 v6 v9; do
  for s in 1 2 3; do
    run_one "${v}" "${s}"
  done
done

# Aggregate.
python - "${LOG_DIR}" <<'PY'
import csv, glob, json, os, sys
log_dir = sys.argv[1]
rows = []
for path in sorted(glob.glob(os.path.join(log_dir, 'eval_*.json'))):
    with open(path) as f: rows.append(json.load(f))
if rows:
    out_csv = os.path.join(log_dir, 'recompute_reward_summary.csv')
    keys = ['encoder_version','seed','mean_reward','std_reward','success_rate','mean_length']
    with open(out_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction='ignore')
        w.writeheader()
        for r in rows: w.writerow(r)
    print(f'Aggregated {len(rows)} rows -> {out_csv}')
PY

echo "Phase RR complete."
