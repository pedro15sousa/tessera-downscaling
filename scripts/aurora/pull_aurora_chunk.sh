#!/bin/bash
# Workstation side of the CSD3 Aurora latent extraction: wait for a chunk's
# marker on RDS, pull its latents, check them, and free the chunk on RDS.
#
#   bash scripts/aurora/pull_aurora_chunk.sh --chunk 2010 [--no-clean] [--no-wait] [--poll 300]
#
# Steps (each one idempotent; re-run after any failure):
#   1. poll  ssh <CSD3_HOST> for <RDS_ROOT>/ingest/aurora/chunks/<id>.done (every --poll s)
#   2. rsync <RDS_ROOT>/ingest/aurora/{lead*h/*/latent_*, latent_calibration.json, chunks/}
#            and <RDS_ROOT>/ingest/aurora_verify/ into <LOCAL_ROOT>/ingest/ -- never with
#            --delete: the original forecasts live in the same lead{L}h/<region>/processed
#            directories, and previous chunks' latents are already there
#   3. sha256sum -c the manifest the verify job wrote (<id>.sha256, paths relative to the root)
#   4. scripts/aurora/verify_aurora_latents.py over the chunk WITH the physical drift check
#      (originals under <LOCAL_ROOT>/ingest/aurora/lead{L}h/<region>/processed)
#   5. write <id>.pulled on RDS and run `submit_aurora_latents.sh --mode clean` there
#      (skipped with --no-clean)
#
# Env: CSD3_HOST (ssh alias, default csd3; use ControlMaster so MFA happens once),
#      CSD3_REPO (repo on CSD3, default ~/tessera-downscaling), RDS_ROOT (on CSD3,
#      default as in _csd3.sh), LOCAL_ROOT (default $TESSERA_DATA_ROOT or
#      /data/weather-downscaling), RSYNC_EXTRA (e.g. --bwlimit=200m).

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
CSD3_HOST="${CSD3_HOST:-csd3}"
CSD3_REPO="${CSD3_REPO:-tessera-downscaling}"
RDS_ROOT="${RDS_ROOT:-/rds/project/rds-FIX1FzgwKT0/pmms2/weather-downscaling}"
LOCAL_ROOT="${LOCAL_ROOT:-${TESSERA_DATA_ROOT:-/data/weather-downscaling}}"
RSYNC_EXTRA="${RSYNC_EXTRA:-}"

CHUNK=""; CLEAN=1; WAIT=1; POLL=300
while [ $# -gt 0 ]; do
    case "$1" in
        --chunk) CHUNK="$2"; shift 2 ;;
        --no-clean) CLEAN=0; shift ;;
        --no-wait) WAIT=0; shift ;;
        --poll) POLL="$2"; shift 2 ;;
        -h|--help) sed -n '2,25p' "${BASH_SOURCE[0]}"; exit 0 ;;
        *) echo "unknown option: $1" >&2; exit 2 ;;
    esac
done
[ -n "${CHUNK}" ] || { echo "--chunk is required" >&2; exit 2; }

cd "${REPO_ROOT}"
eval "$(python3 scripts/aurora/chunks.py show "${CHUNK}")"
MARKER_DIR_R="${RDS_ROOT}/${MARKER_DIR}"
MARKER_R="${RDS_ROOT}/${MARKER}"
LOCAL_MARKERS="${LOCAL_ROOT}/ingest/aurora/chunks"
remote() { ssh -o BatchMode=yes "${CSD3_HOST}" "$@"; }

log() { echo "[$(date '+%F %T')] $*"; }

# ---- 1. wait for the marker ------------------------------------------------
# ssh exits 255 when the connection itself fails (BatchMode refuses the MFA
# prompt once the ControlMaster socket is gone); anything else from `test -f`
# means "not there yet". Fail closed on 255 instead of polling a dead link.
wait_for_marker() {
    local rc
    while true; do
        remote "test -f '${MARKER_R}'" && return 0
        rc=$?
        if [ "${rc}" -eq 255 ]; then
            echo "ssh to ${CSD3_HOST} failed (multiplexed connection lost?); re-open it with" \
                 "'ssh -fN ${CSD3_HOST}' (MFA) and re-run this script" >&2
            return 1
        fi
        sleep "${POLL}"
    done
}
if [ "${WAIT}" = "1" ]; then
    log "waiting for ${CSD3_HOST}:${MARKER_R} (poll ${POLL}s)"
    wait_for_marker
fi
remote "test -f '${MARKER_R}'" || { echo "marker ${MARKER_R} not present on ${CSD3_HOST}" >&2; exit 1; }
log "marker present: $(remote "cat '${MARKER_R}'" | tr '\n' ' ')"

# ---- 2. rsync (no --delete) ------------------------------------------------
mkdir -p "${LOCAL_ROOT}/ingest/aurora" "${LOCAL_ROOT}/ingest/aurora_verify" "${LOCAL_MARKERS}"
log "rsync latents + calibration + markers -> ${LOCAL_ROOT}/ingest/aurora"
# shellcheck disable=SC2086
rsync -a --partial --human-readable --info=progress2,stats1 ${RSYNC_EXTRA} \
    --exclude='*.tmp' \
    --include='/latent_calibration.json' \
    --include='/chunks/' --include='/chunks/**' \
    --include='/lead*h/' --include='/lead*h/*/' \
    --include='/lead*h/*/latent_*/' --include='/lead*h/*/latent_*/**' \
    --exclude='*' \
    "${CSD3_HOST}:${RDS_ROOT}/ingest/aurora/" "${LOCAL_ROOT}/ingest/aurora/"
log "rsync decoded-field verification subset -> ${LOCAL_ROOT}/ingest/aurora_verify"
# shellcheck disable=SC2086
rsync -a --partial --human-readable --info=stats1 ${RSYNC_EXTRA} --exclude='*.tmp' \
    "${CSD3_HOST}:${RDS_ROOT}/ingest/aurora_verify/" "${LOCAL_ROOT}/ingest/aurora_verify/"

# ---- 3. checksums ----------------------------------------------------------
manifest="${LOCAL_MARKERS}/${CHUNK}.sha256"
[ -s "${manifest}" ] || { echo "manifest ${manifest} missing/empty after rsync" >&2; exit 1; }
log "sha256sum -c $(wc -l < "${manifest}") files"
(cd "${LOCAL_ROOT}" && sha256sum -c --quiet "${manifest}")
log "checksums ok"

# ---- 4. verifier with the drift check --------------------------------------
log "verify_aurora_latents.py ${VALID_START}..${VALID_END} with drift check"
uv run python scripts/aurora/verify_aurora_latents.py \
    --output-root "${LOCAL_ROOT}/ingest/aurora" \
    --global-dataset "${LOCAL_ROOT}/datasets/dataset_timestamp_global" \
    --leads ${LEADS} --regions ${REGIONS} --split all \
    --start-date "${VALID_START}" --end-date "${VALID_END}" \
    --verify-root "${LOCAL_ROOT}/ingest/aurora_verify" \
    --physical-root "${LOCAL_ROOT}/ingest/aurora" \
    | tee "${LOCAL_MARKERS}/${CHUNK}.verify.workstation.txt"
test "${PIPESTATUS[0]}" -eq 0

# ---- 5. mark pulled, clean on RDS -------------------------------------------
remote "echo 'pulled=$(date -u +%FT%TZ) host=$(hostname)' > '${MARKER_DIR_R}/${CHUNK}.pulled'"
echo "pulled=$(date -u +%FT%TZ)" > "${LOCAL_MARKERS}/${CHUNK}.pulled"
if [ "${CLEAN}" = "1" ]; then
    log "cleaning chunk ${CHUNK} on RDS"
    remote "cd '${CSD3_REPO}' && RDS_ROOT='${RDS_ROOT}' bash scripts/aurora/submit_aurora_latents.sh --chunk '${CHUNK}' --mode clean"
fi
log "chunk ${CHUNK} done"
