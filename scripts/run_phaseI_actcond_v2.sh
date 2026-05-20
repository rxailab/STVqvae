#!/usr/bin/env bash
# Phase I (next paper): action-conditioning probe v2 on every phaseHO ckpt.
# Adds per-class action-dependence breakdown + epsilon_action_collapse +
# action_residual_rank to the original phaseS/U metrics.
#
# Output JSON per checkpoint -> logs/phaseI_actcond_v2/<stub>.json
# ~2 min/ckpt on GPU, ~3 h total for 100 ckpts. Idempotent.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="${REPO}/logs/phaseI_actcond_v2"
mkdir -p "${LOG_DIR}"

cd "${REPO}/discrete_mbrl"

mapfile -t CKPT_ROWS < <(awk -F, 'NR>1 {print $1}' \
  "${REPO}/logs/phaseA/phaseA_summary.csv" | sort -u)
echo "Discovered ${#CKPT_ROWS[@]} unique checkpoints in phaseA summary."

run_one() {
  local stub="$1"
  local rel="${stub//__//}"
  local ckpt="${REPO}/${rel}.pt"
  local out="${LOG_DIR}/actcond_${stub}.json"
  local log="${LOG_DIR}/actcond_${stub}.log"

  if [[ -s "${out}" ]]; then echo "[skip] ${stub}"; return; fi
  if [[ ! -f "${ckpt}" ]]; then echo "[miss-ckpt] ${ckpt}"; return; fi

  echo "[actcond-v2] ${stub}"
  set +e
  python analyze_wm_action_cond_v2.py \
    --model_path "${ckpt}" \
    --n_states 2000 \
    --device cuda \
    --output_json "${out}" > "${log}" 2>&1
  rc=$?; set -e
  if [[ $rc -ne 0 ]]; then echo "[fail] ${stub} (rc=$rc)"; rm -f "${out}"; fi
}

for stub in "${CKPT_ROWS[@]}"; do
  run_one "${stub}"
done

# Aggregate to a single CSV for the correlation notebook.
python - "${LOG_DIR}" <<'PY'
import csv, glob, json, os, sys
log_dir = sys.argv[1]
rows = []
for path in sorted(glob.glob(os.path.join(log_dir, 'actcond_*.json'))):
    with open(path) as f:
        d = json.load(f)
    base = os.path.basename(path).replace('actcond_', '').replace('.json', '')
    row = {
        'checkpoint': base,
        'env_name': d.get('env_name'),
        'state_change_norm': d.get('state_change_norm'),
        'action_diff_norm': d.get('action_diff_norm'),
        'action_dep_ratio': d.get('action_dep_ratio'),
        'reward_action_var': d.get('reward_action_var'),
        'gamma_action_var': d.get('gamma_action_var'),
        'effective_action_rank': d.get('effective_action_rank'),
        'action_residual_rank': d.get('action_residual_rank'),
        'epsilon_action_collapse': d.get('epsilon_action_collapse'),
    }
    for cid, name in enumerate(d.get('class_names', [])):
        if str(cid) in d.get('per_class_action_dep_ratio', {}):
            row[f'pc_ratio_{name}'] = d['per_class_action_dep_ratio'][str(cid)]
    rows.append(row)
if not rows:
    print('No JSONs to aggregate.'); sys.exit(0)
keys = list(rows[0].keys()) + sorted({k for r in rows for k in r.keys()} - set(rows[0].keys()))
with open(os.path.join(log_dir, 'actcond_v2_summary.csv'), 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=keys)
    w.writeheader()
    for r in rows: w.writerow(r)
print(f'Aggregated {len(rows)} rows -> actcond_v2_summary.csv')
PY
