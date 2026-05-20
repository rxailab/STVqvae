#!/usr/bin/env bash
# Buffer-coverage analysis on every phaseHO ckpt. ~30s/ckpt on GPU.
# Output JSON per checkpoint -> logs/phaseI_bufcov/<stub>.json
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="${REPO}/logs/phaseI_bufcov"
mkdir -p "${LOG_DIR}"

cd "${REPO}/discrete_mbrl"

mapfile -t CKPT_ROWS < <(awk -F, 'NR>1 {print $1}' \
  "${REPO}/logs/phaseA/phaseA_summary.csv" | sort -u)
echo "Discovered ${#CKPT_ROWS[@]} unique checkpoints."

run_one() {
  local stub="$1"
  local rel="${stub//__//}"
  local ckpt="${REPO}/${rel}.pt"
  local out="${LOG_DIR}/bufcov_${stub}.json"
  local log="${LOG_DIR}/bufcov_${stub}.log"

  if [[ -s "${out}" ]]; then echo "[skip] ${stub}"; return; fi
  if [[ ! -f "${ckpt}" ]]; then echo "[miss-ckpt] ${ckpt}"; return; fi

  echo "[bufcov] ${stub}"
  set +e
  python analyze_buffer_coverage.py \
    --model_path "${ckpt}" \
    --n_steps 10000 \
    --device cuda \
    --output_json "${out}" > "${log}" 2>&1
  rc=$?; set -e
  if [[ $rc -ne 0 ]]; then echo "[fail] ${stub} (rc=$rc)"; rm -f "${out}"; fi
}

for stub in "${CKPT_ROWS[@]}"; do
  run_one "${stub}"
done

# Aggregate ───────────────────────────────────────────────────────────────
python - "${LOG_DIR}" <<'PY'
import csv, glob, json, os, sys
log_dir = sys.argv[1]
CLASSES = {'1': 'empty', '2': 'wall', '4': 'door', '5': 'key', '8': 'goal', '10': 'agent'}
rows = []
for path in sorted(glob.glob(os.path.join(log_dir, 'bufcov_*.json'))):
    with open(path) as f: d = json.load(f)
    base = os.path.basename(path).replace('bufcov_', '').replace('.json', '')
    n_steps = d['n_steps']
    row = {'checkpoint': base, 'env': d['env_name'], 'n_steps': n_steps}
    for cid, cname in CLASSES.items():
        rc = d['random']['changes'].get(cid, 0)
        rt = d['random']['total_positions'].get(cid, 0)
        row[f'rand_{cname}_chg'] = rc
        row[f'rand_{cname}_tot'] = rt
        row[f'rand_{cname}_rate'] = rc / max(rt, 1)
        if d.get('policy'):
            pc = d['policy']['changes'].get(cid, 0)
            pt = d['policy']['total_positions'].get(cid, 0)
            row[f'pol_{cname}_chg'] = pc
            row[f'pol_{cname}_tot'] = pt
            row[f'pol_{cname}_rate'] = pc / max(pt, 1)
            row[f'ratio_{cname}'] = (row[f'pol_{cname}_rate'] / row[f'rand_{cname}_rate']
                                     if row[f'rand_{cname}_rate'] > 0 else float('inf'))
    rows.append(row)
if not rows:
    print('No JSONs to aggregate.'); sys.exit(0)
keys = list(rows[0].keys()) + sorted({k for r in rows for k in r.keys()} - set(rows[0].keys()))
with open(os.path.join(log_dir, 'bufcov_summary.csv'), 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=keys)
    w.writeheader()
    for r in rows: w.writerow(r)
print(f'Aggregated {len(rows)} rows -> bufcov_summary.csv')
PY
