#!/usr/bin/env bash
# Phase HO: held-out probe accuracy for the 100 WM-valid phaseA checkpoints.
# Mirrors the original probe protocol (analyze_wm_multistep::fit_shared_probe)
# but collects 2*n_frames frames and splits 50/50 train/test by temporal index
# (independent rollouts), then reports per-class recall on the held-out split.
#
# Output JSON per checkpoint -> logs/phaseHO/holdout_<ckpt>.json
# Aggregated CSV at the end -> logs/phaseHO/holdout_summary.csv
#
# Each ckpt is fast (~1 min for 40k frames + LR fit). 100 ckpts = ~1.7h.
# Idempotent: skip-guard on per-ckpt JSON.

set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="${REPO}/logs/phaseHO"
mkdir -p "${LOG_DIR}"

cd "${REPO}/discrete_mbrl"

# Discover all unique checkpoint paths from phaseA_summary.csv (col 1).
# Reconstruct the .pt path from the dunder-encoded checkpoint string.
mapfile -t CKPT_ROWS < <(awk -F, 'NR>1 {print $1}' \
  "${REPO}/logs/phaseA/phaseA_summary.csv" | sort -u)

echo "Discovered ${#CKPT_ROWS[@]} unique checkpoints in phaseA summary."

run_one() {
  local stub="$1"
  # Convert "discrete_mbrl__model_free__models__ENV__NAME" to a real path.
  local rel="${stub//__//}"
  local ckpt="${REPO}/${rel}.pt"
  local out="${LOG_DIR}/holdout_${stub}.json"
  local log="${LOG_DIR}/holdout_${stub}.log"

  if [[ -s "${out}" ]]; then echo "[skip] ${stub}"; return; fi
  if [[ ! -f "${ckpt}" ]]; then echo "[miss-ckpt] ${ckpt}"; return; fi

  echo "[probe-HO] ${stub}"
  set +e
  python recompute_holdout_probe.py \
    --model_path "${ckpt}" \
    --n_frames 20000 \
    --output_json "${out}" \
    --device cuda > "${log}" 2>&1
  rc=$?; set -e
  if [[ $rc -ne 0 ]]; then echo "[fail] ${stub} (rc=$rc)"; rm -f "${out}"; fi
}

for stub in "${CKPT_ROWS[@]}"; do
  run_one "${stub}"
done

# Aggregate.
python - "${LOG_DIR}" <<'PY'
import csv, glob, json, os, sys
log_dir = sys.argv[1]
rows = []
for path in sorted(glob.glob(os.path.join(log_dir, 'holdout_*.json'))):
    with open(path) as f:
        d = json.load(f)
    base = os.path.basename(path).replace('holdout_', '').replace('.json','')
    row = {
        'checkpoint': base,
        'env_name':   d.get('env_name'),
        'probe_avg_recall_train': d.get('probe_avg_recall_train'),
        'probe_avg_recall_test':  d.get('probe_avg_recall_test'),
        'generalization_gap':     d.get('generalization_gap'),
    }
    for cid, info in d.get('per_class', {}).items():
        row[f'train_{info["name"]}'] = info['train_acc']
        row[f'test_{info["name"]}']  = info['test_acc']
    rows.append(row)

if not rows:
    print('No holdout JSONs to aggregate.'); sys.exit(0)

# Union of all keys for header.
keys = ['checkpoint','env_name','probe_avg_recall_train','probe_avg_recall_test','generalization_gap']
extra = sorted({k for r in rows for k in r.keys()} - set(keys))
keys += extra
out_csv = os.path.join(log_dir, 'holdout_summary.csv')
with open(out_csv, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=keys)
    w.writeheader()
    for r in rows: w.writerow(r)
print(f'Aggregated {len(rows)} rows -> {out_csv}')
PY

echo "Phase HO complete."
