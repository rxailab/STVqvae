#!/usr/bin/env bash
# sync_checkpoints.sh — rsync the 29 paper checkpoints from this Linux box
# to Lancaster HEC (wayland-2022). Requires the Lancaster VPN tunnel to be
# up first; see paper_assets/scripts/README_vpn.md.
#
# Usage:
#   bash sync_checkpoints.sh                   # default: rsync to wayland-2022
#   bash sync_checkpoints.sh --dry-run         # show what would be transferred
#   bash sync_checkpoints.sh --target host:/path  # custom destination
#   bash sync_checkpoints.sh --check           # just verify VPN reachability + counts
#   bash sync_checkpoints.sh --dk8-only        # only DK-8 checkpoints (~8.5 GB)
#   bash sync_checkpoints.sh --dk16-only       # only DK-16 checkpoints (~11 GB)

set -euo pipefail

# ---- defaults ----
SOURCE_ROOT="/home/xiar3/experiments/STVqvae/discrete_mbrl/model_free/models"
DEST_HOST="wayland-2022.hec.lancaster.ac.uk"
DEST_USER="xiar3"
DEST_ROOT="/mmfs1/storage/users/xiar3/exp/STVqvae/discrete_mbrl/model_free/models"

DRY_RUN=""
CHECK_ONLY=0
DK8=1
DK16=1

# ---- arg parsing ----
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)   DRY_RUN="--dry-run"; shift ;;
        --check)     CHECK_ONLY=1; shift ;;
        --dk8-only)  DK16=0; shift ;;
        --dk16-only) DK8=0; shift ;;
        --target)    # format: user@host:/path  (overrides defaults)
            arg="$2"
            DEST_USER="${arg%@*}"
            rest="${arg#*@}"
            DEST_HOST="${rest%%:*}"
            DEST_ROOT="${rest#*:}"
            shift 2 ;;
        --help|-h)
            sed -n '2,15p' "$0"; exit 0 ;;
        *)
            echo "Unknown arg: $1" >&2; exit 2 ;;
    esac
done

DEST="${DEST_USER}@${DEST_HOST}"

# ---- preflight: check VPN reachability ----
echo "==> Checking SSH reachability of ${DEST_HOST}:22 ..."
if timeout 5 bash -c "echo > /dev/tcp/${DEST_HOST}/22" 2>/dev/null; then
    echo "   OK — port 22 reachable."
else
    echo "   FAIL — cannot reach ${DEST_HOST}:22." >&2
    echo "   Bring up the Lancaster VPN first; see paper_assets/scripts/README_vpn.md" >&2
    exit 3
fi

echo "==> Verifying SSH auth (no password prompt) ..."
if ! ssh -o BatchMode=yes -o ConnectTimeout=8 "${DEST}" 'true' 2>/dev/null; then
    echo "   WARN — passwordless SSH not configured. rsync will prompt for the password."
    echo "   Tip: ssh-copy-id ${DEST}  (one-time)"
fi

# ---- count source files ----
DK8_DIR="${SOURCE_ROOT}/MiniGrid-DoorKey-8x8-v0"
DK16_DIR="${SOURCE_ROOT}/MiniGrid-DoorKey-16x16-v0"

# Glob the 29 paper checkpoints precisely (sweep_dk8_*, sweep_doorkey8_*, sweep_doorkey16_*).
# The glob excludes older non-paper ablations (e.g. mf_e2e_semantic_*, rigor_*, doorkey_v6aenc_*).
shopt -s nullglob
DK8_FILES=( "${DK8_DIR}"/sweep_dk8_*_best_model.pt "${DK8_DIR}"/sweep_doorkey8_*_best_model.pt )
DK16_FILES=( "${DK16_DIR}"/sweep_doorkey16_*_best_model.pt )
shopt -u nullglob

DK8_N=${#DK8_FILES[@]}
DK16_N=${#DK16_FILES[@]}
DK8_SIZE=$(du -csh "${DK8_FILES[@]}" 2>/dev/null | tail -1 | cut -f1)
DK16_SIZE=$(du -csh "${DK16_FILES[@]}" 2>/dev/null | tail -1 | cut -f1)

echo "==> Source inventory:"
echo "   DK-8  paper checkpoints: ${DK8_N} files, ${DK8_SIZE}"
echo "   DK-16 paper checkpoints: ${DK16_N} files, ${DK16_SIZE}"

if [[ ${CHECK_ONLY} -eq 1 ]]; then
    echo "==> --check requested; exiting before rsync."
    exit 0
fi

# ---- ensure destination directories exist ----
echo "==> Ensuring destination directories exist on ${DEST_HOST} ..."
ssh "${DEST}" "mkdir -p '${DEST_ROOT}/MiniGrid-DoorKey-8x8-v0' '${DEST_ROOT}/MiniGrid-DoorKey-16x16-v0'" \
    || { echo "Failed to create dest dirs." >&2; exit 4; }

# ---- rsync DK-8 ----
if [[ ${DK8} -eq 1 && ${DK8_N} -gt 0 ]]; then
    echo "==> rsync DK-8 (${DK8_N} files, ${DK8_SIZE}) ..."
    rsync -avh --progress --partial --inplace ${DRY_RUN} \
        "${DK8_FILES[@]}" \
        "${DEST}:${DEST_ROOT}/MiniGrid-DoorKey-8x8-v0/"
fi

# ---- rsync DK-16 ----
if [[ ${DK16} -eq 1 && ${DK16_N} -gt 0 ]]; then
    echo "==> rsync DK-16 (${DK16_N} files, ${DK16_SIZE}) ..."
    rsync -avh --progress --partial --inplace ${DRY_RUN} \
        "${DK16_FILES[@]}" \
        "${DEST}:${DEST_ROOT}/MiniGrid-DoorKey-16x16-v0/"
fi

echo "==> Done."
echo "    Verify on the cluster:"
echo "      ssh ${DEST} 'ls -lh ${DEST_ROOT}/MiniGrid-DoorKey-*x*-v0/sweep_*_best_model.pt | wc -l'"
echo "    Expected: $((DK8_N + DK16_N))"
