#!/usr/bin/env bash
# Phase A driver: run analyze_wm_multistep.py on every paper checkpoint and
# emit one JSON per ckpt under logs/phaseA/. Aggregation into CSV is handled
# by scripts/aggregate_phaseA.py.
#
# Args (positional, optional):
#   $1  CKPT_DIR_OR_LIST  comma-separated dirs to scan recursively for *.pt
#                         default: discrete_mbrl/model_free/models/MiniGrid-DoorKey-8x8-v0,
#                                  discrete_mbrl/model_free/models/MiniGrid-DoorKey-16x16-v0
#   $2  OUT_DIR           default: logs/phaseA
#
# Env overrides (optional):
#   PROBE_FRAMES  default 20000
#   N_ROLLOUTS    default 500
#   HORIZONS      default "1 3 5 10"
#   DEVICE        default cuda
#   ONLY_PATTERN  basename glob to filter (e.g. 'sweep_dk8_*_best_model.pt')
#   NO_EXTENDED   set to 1 to skip the E1/E2/E3/E5/E6 metrics

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEFAULT_DIRS="${REPO}/discrete_mbrl/model_free/models/MiniGrid-DoorKey-8x8-v0,${REPO}/discrete_mbrl/model_free/models/MiniGrid-DoorKey-16x16-v0"
CKPT_INPUT="${1:-${DEFAULT_DIRS}}"
OUT_DIR="${2:-${REPO}/logs/phaseA}"

PROBE_FRAMES="${PROBE_FRAMES:-20000}"
N_ROLLOUTS="${N_ROLLOUTS:-500}"
HORIZONS="${HORIZONS:-1 3 5 10}"
DEVICE="${DEVICE:-cuda}"
ONLY_PATTERN="${ONLY_PATTERN:-*.pt}"

mkdir -p "${OUT_DIR}"

IFS=',' read -ra CKPT_DIRS <<< "${CKPT_INPUT}"
CKPTS=()
for d in "${CKPT_DIRS[@]}"; do
  if [[ ! -d "${d}" ]]; then
    echo "warn: ckpt dir not found, skipping: ${d}" >&2
    continue
  fi
  while IFS= read -r f; do CKPTS+=("$f"); done < <(find "${d}" -type f -name "${ONLY_PATTERN}" | sort)
done

if [[ ${#CKPTS[@]} -eq 0 ]]; then
  echo "no .pt files matched under: ${CKPT_INPUT}" >&2
  exit 2
fi

cd "${REPO}/discrete_mbrl"
echo "Phase A: ${#CKPTS[@]} checkpoints"
echo "  probe_frames=${PROBE_FRAMES}  n_rollouts=${N_ROLLOUTS}  horizons=${HORIZONS}  device=${DEVICE}"
[[ -n "${NO_EXTENDED:-}" ]] && echo "  extended metrics: OFF"
echo

EXTRA=()
[[ -n "${NO_EXTENDED:-}" ]] && EXTRA+=("--no_extended")

for ckpt in "${CKPTS[@]}"; do
  rel="${ckpt#${REPO}/}"
  name="${rel//\//__}"
  name="${name%.pt}"
  out="${OUT_DIR}/${name}.json"
  log="${OUT_DIR}/${name}.log"

  if [[ -s "${out}" ]]; then
    echo "[skip] ${name}"
    continue
  fi

  echo "[run]  ${name}"
  set +e
  python analyze_wm_multistep.py \
    --model_path   "${ckpt}" \
    --probe_frames "${PROBE_FRAMES}" \
    --n_rollouts   "${N_ROLLOUTS}" \
    --horizons     ${HORIZONS} \
    --device       "${DEVICE}" \
    --output_json  "${out}" \
    "${EXTRA[@]}" > "${log}" 2>&1
  rc=$?
  set -e
  if [[ $rc -ne 0 ]]; then
    echo "[fail] ${name} (exit ${rc}); see ${log}"
    rm -f "${out}"  # don't leave a partial JSON that 'skip' would honor on retry
  else
    echo "[ok]   ${name}"
  fi
done

echo
echo "All Phase A runs complete. Aggregating..."
python "${REPO}/scripts/aggregate_phaseA.py" "${OUT_DIR}"
