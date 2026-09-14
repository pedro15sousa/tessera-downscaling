#!/bin/bash
# Stage the 13-level ERA5 inputs of one chunk from WeatherBench2 into
# <RDS_ROOT>/ingest/aurora_inputs (CPU job; submitted by
# scripts/aurora/submit_aurora_latents.sh --mode stage, do not sbatch by hand).
#
# Two steps, selected by STAGE_STEP:
#   download  array task: scripts/data/download_era5_wb2.py --levels aurora over
#             sub-window SLURM_ARRAY_TASK_ID of STAGE_SHARDS of the chunk's staging
#             window. Resume-safe (atomic_completed skips finished files).
#   check     single job after the array: removes any partial file the downloader
#             flagged (<file>-problem.txt) so a resubmit redoes it, then runs the
#             extractor's --dry-run over the chunk, which stats every required
#             (variable, frame) input and the static file. Writes
#             <RDS_ROOT>/<marker_dir>/<chunk>.staged only when that says READY.
#
# Env from the submitter: CHUNK, STAGE_STEP, STAGE_SHARDS, STAGE_PROCESSES, RDS_ROOT.

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
# shellcheck source=scripts/aurora/_csd3.sh
source "${REPO_ROOT}/scripts/aurora/_csd3.sh"
csd3_require_env
cd "${REPO_ROOT}"
export PYTHONUNBUFFERED=1

: "${CHUNK:?CHUNK is required}"
STAGE_STEP="${STAGE_STEP:-download}"
STAGE_SHARDS="${STAGE_SHARDS:-4}"
STAGE_PROCESSES="${STAGE_PROCESSES:-8}"
STAGING_ROOT="${RDS_ROOT}/ingest/aurora_inputs"

echo "[$(date)] stage ${STAGE_STEP} chunk=${CHUNK} host=$(hostname) job=${SLURM_JOB_ID:-?} task=${SLURM_ARRAY_TASK_ID:-single}"

if [ "${STAGE_STEP}" = "download" ]; then
    eval "$(python3 scripts/aurora/chunks.py show "${CHUNK}" \
        --shard "${SLURM_ARRAY_TASK_ID:-0}" --n-shards "${STAGE_SHARDS}" --window staging)"
    echo "downloading ${SHARD_START} .. ${SHARD_END} -> ${STAGING_ROOT} (${STAGE_PROCESSES} processes)"
    "${PY}" scripts/data/download_era5_wb2.py --levels aurora --root "${STAGING_ROOT}" \
        --start "${SHARD_START_ISO}" --end "${SHARD_END_ISO}" --num-processes "${STAGE_PROCESSES}"
    echo "[$(date)] download task done"
    exit 0
fi

if [ "${STAGE_STEP}" = "check" ]; then
    eval "$(python3 scripts/aurora/chunks.py show "${CHUNK}")"
    marker_dir="${RDS_ROOT}/${MARKER_DIR}"
    mkdir -p "${marker_dir}"

    # Partial writes: the downloader leaves <file>-problem.txt next to anything
    # it failed on. Drop both so the next stage run redoes them.
    mapfile -t problems < <(find "${STAGING_ROOT}" -name '*-problem.txt' 2>/dev/null)
    if [ "${#problems[@]}" -gt 0 ]; then
        echo "*** ${#problems[@]} download problem file(s); removing them and their outputs, e.g.:"
        printf '    %s\n' "${problems[@]:0:5}"
        for pf in "${problems[@]}"; do
            rm -f "${pf}" "${pf%-problem.txt}"
        done
        echo "resubmit: bash scripts/aurora/submit_aurora_latents.sh --chunk ${CHUNK} --mode stage"
        exit 1
    fi

    # The extractor's pre-flight stats every (var, frame) the chunk needs.
    # --dtype float16 only fixes the storage estimate: 'auto' would insist on
    # the calibration file, which does not exist before chunk 1 is calibrated.
    set +e
    "${PY}" scripts/aurora/extract_aurora_latents.py \
        --global-metadata "${RDS_ROOT}/datasets/dataset_timestamp_global/metadata.json" \
        --era5-staging-root "${STAGING_ROOT}" \
        --static-file "${RDS_ROOT}/ingest/processed/era5_static/era5_static_0p25_all.nc" \
        --output-root "${RDS_ROOT}/ingest/aurora" \
        --regions ${REGIONS} --leads ${LEADS} --split all \
        --start-date "${VALID_START}" --end-date "${VALID_END}" \
        --dtype float16 --dry-run | tee "${marker_dir}/${CHUNK}.stage-check.txt"
    status=${PIPESTATUS[0]}
    set -e
    if [ "${status}" -ne 0 ] || ! grep -q "PRE-FLIGHT: READY TO RUN" "${marker_dir}/${CHUNK}.stage-check.txt"; then
        echo "*** stage check FAILED for chunk ${CHUNK} (see ${marker_dir}/${CHUNK}.stage-check.txt)"
        exit 1
    fi
    {
        echo "chunk=${CHUNK}"
        echo "staging_window=${STAGING_START}..${STAGING_END}"
        echo "checked=$(date -u +%FT%TZ)"
        echo "job=${SLURM_JOB_ID:-?}"
    } > "${marker_dir}/${CHUNK}.staged"
    echo "[$(date)] chunk ${CHUNK} staged: ${marker_dir}/${CHUNK}.staged"
    exit 0
fi

echo "unknown STAGE_STEP=${STAGE_STEP}" >&2
exit 2
