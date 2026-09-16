#!/bin/bash
# Workstation driver for the whole Aurora latent extraction: keeps the cluster
# a few chunks ahead, pulls each chunk as it is verified, and stops on the
# first real problem.
#
#   bash scripts/aurora/run_aurora_chunks.sh [--from 2010] [--to 2022]
#                                            [--lookahead 2] [--n-shards 4] [--dry-run]
#
# For each chunk in order it
#   1. makes sure chunks up to <chunk + lookahead> have a pipeline submitted on
#      CSD3 (stage -> shard -> verify), skipping any that are already done or
#      already have jobs queued;
#   2. runs pull_aurora_chunk.sh for the chunk, which waits for the chunk's
#      .done marker, rsyncs, checks the manifest, runs the verifier with the
#      drift check, then cleans that chunk on RDS.
#
# The lookahead is what bounds RDS usage: each chunk not yet pulled holds its
# transient inputs (~430 GB) plus its latents (~230 GB), and the pull's clean
# step is what frees them. Raising it risks filling the project quota.
#
# Resumable: a chunk whose .pulled marker exists locally is skipped, so
# re-running after an interruption continues where it stopped. The transfer is
# the pacing item (~25 MB/s measured, ~2.5 h per chunk), not the GPUs.
#
# Env: CSD3_HOST (default csd3; needs a live ControlMaster so MFA happens once),
#      CSD3_REPO, RDS_ROOT, LOCAL_ROOT, RSYNC_EXTRA -- as in pull_aurora_chunk.sh.

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
CSD3_HOST="${CSD3_HOST:-csd3}"
CSD3_REPO="${CSD3_REPO:-tessera-downscaling}"
RDS_ROOT="${RDS_ROOT:-/rds/project/rds-FIX1FzgwKT0/pmms2/weather-downscaling}"
LOCAL_ROOT="${LOCAL_ROOT:-${TESSERA_DATA_ROOT:-/data/weather-downscaling}}"

FROM=""; TO=""; LOOKAHEAD=2; N_SHARDS=4; DRY=0
while [ $# -gt 0 ]; do
    case "$1" in
        --from) FROM="$2"; shift 2 ;;
        --to) TO="$2"; shift 2 ;;
        --lookahead) LOOKAHEAD="$2"; shift 2 ;;
        --n-shards) N_SHARDS="$2"; shift 2 ;;
        --dry-run) DRY=1; shift ;;
        -h|--help) sed -n '2,30p' "${BASH_SOURCE[0]}"; exit 0 ;;
        *) echo "unknown option: $1" >&2; exit 2 ;;
    esac
done

cd "${REPO_ROOT}"
remote() { ssh -o BatchMode=yes "${CSD3_HOST}" "$@"; }
log() { echo "[$(date '+%F %T')] $*"; }

mapfile -t ALL < <(python3 scripts/aurora/chunks.py list | tail -n +2 | awk '{print $1}')
CHUNKS=()
for c in "${ALL[@]}"; do
    [ -n "${FROM}" ] && [ "${c}" \< "${FROM}" ] && continue
    [ -n "${TO}" ] && [ "${c}" \> "${TO}" ] && continue
    CHUNKS+=("${c}")
done
[ "${#CHUNKS[@]}" -gt 0 ] || { echo "no chunks selected" >&2; exit 2; }
log "chunks: ${CHUNKS[*]}  (lookahead ${LOOKAHEAD}, ${N_SHARDS} shards)"

# Submit the pipeline for one chunk unless it is finished or already queued.
ensure_submitted() {
    local id="$1"
    if remote "test -f '${RDS_ROOT}/ingest/aurora/chunks/${id}.done'"; then
        return 0
    fi
    local queued
    queued=$(remote "squeue --me -h -n aur-stage-${id},aur-stagechk-${id},aur-shard-${id},aur-verify-${id} -o %i 2>/dev/null | wc -l" || echo 0)
    if [ "${queued}" -gt 0 ]; then
        return 0
    fi
    log "submitting the CSD3 pipeline for chunk ${id}"
    if [ "${DRY}" = "1" ]; then
        echo "    would run: submit_aurora_latents.sh --chunk ${id} --mode pipeline --n-shards ${N_SHARDS}"
        return 0
    fi
    remote "cd '${CSD3_REPO}' && RDS_ROOT='${RDS_ROOT}' bash scripts/aurora/submit_aurora_latents.sh --chunk ${id} --mode pipeline --n-shards ${N_SHARDS}"
}

# Wait for a chunk's .done marker, but give up if CSD3 has no jobs for it and
# no marker: the pipeline died, and an unattended run would otherwise wait for
# a marker that is never coming. Two consecutive idle polls (10 min at the
# default) avoid tripping on the gap between submitting and squeue showing it.
wait_for_marker() {
    local id="$1" marker="${RDS_ROOT}/ingest/aurora/chunks/${id}.done" idle=0 queued rc
    log "waiting for chunk ${id} to be verified on ${CSD3_HOST}"
    while true; do
        if remote "test -f '${marker}'"; then
            log "chunk ${id} marker present"
            return 0
        fi
        rc=$?
        if [ "${rc}" -eq 255 ]; then
            log "ssh to ${CSD3_HOST} failed; re-open it with 'ssh -fN ${CSD3_HOST}' (MFA) and re-run"
            return 1
        fi
        queued=$(remote "squeue --me -h -n aur-stage-${id},aur-stagechk-${id},aur-shard-${id},aur-verify-${id} -o %i 2>/dev/null | wc -l" || echo -1)
        if [ "${queued}" = "0" ]; then
            idle=$((idle + 1))
            if [ "${idle}" -ge 2 ]; then
                log "chunk ${id}: no marker and no jobs queued -- the CSD3 pipeline failed."
                log "  check: ssh ${CSD3_HOST} \"cd ${CSD3_REPO} && bash scripts/aurora/submit_aurora_latents.sh --chunk ${id} --mode status\""
                return 1
            fi
        else
            idle=0
        fi
        sleep 300
    done
}

n_chunks="${#CHUNKS[@]}"
for i in "${!CHUNKS[@]}"; do
    chunk="${CHUNKS[$i]}"
    if [ -f "${LOCAL_ROOT}/ingest/aurora/chunks/${chunk}.pulled" ]; then
        log "chunk ${chunk} already pulled; skipping"
        continue
    fi
    # Keep the cluster busy ahead of the pull, quota permitting.
    for ((j = i; j <= i + LOOKAHEAD && j < n_chunks; j++)); do
        ensure_submitted "${CHUNKS[$j]}"
    done
    if [ "${DRY}" = "1" ]; then
        echo "    would wait for the ${chunk} marker, then run: pull_aurora_chunk.sh --chunk ${chunk} --no-wait"
        continue
    fi
    # The driver owns the wait (and its failure policy); the pull script is
    # then given a marker that already exists.
    if ! wait_for_marker "${chunk}"; then
        exit 1
    fi
    log "pulling chunk ${chunk}"
    if ! bash scripts/aurora/pull_aurora_chunk.sh --chunk "${chunk}" --no-wait; then
        log "chunk ${chunk} FAILED in the pull; stopping so it can be looked at"
        exit 1
    fi
    log "chunk ${chunk} complete; $(df -h "${LOCAL_ROOT}" | awk 'NR==2 {print $4" free locally"}')"
done
log "all selected chunks done"
