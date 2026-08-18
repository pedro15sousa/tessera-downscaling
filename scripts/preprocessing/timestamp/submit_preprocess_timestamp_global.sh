#!/usr/bin/env bash
# Submit preprocess_timestamp_global.py as a Slurm job so the rerun
# (after the elevation sentinel filter was added) can complete
# unattended. CPU-bound, no GPU needed.
#
# Reads the same paths the standard invocation in the docstring at the
# top of preprocess_timestamp_global.py uses. Adjust env vars below if
# your cluster paths differ.
#
# Usage (from repo root):
#   bash projects/tessera_downscaling/scripts/preprocessing/timestamp/submit_preprocess_timestamp_global.sh
#
# Dry-run (prints sbatch command, doesn't submit):
#   DRY_RUN=1 bash projects/tessera_downscaling/scripts/preprocessing/timestamp/submit_preprocess_timestamp_global.sh
#
# Pre-flight checks the script does for you:
#   * Refuses to run if --output-dir already exists. The expected
#     workflow is to rename the existing dataset_timestamp_global to
#     dataset_timestamp_global_old first, then run this. That keeps the
#     old data on disk for backup and forces the preprocessor to
#     regenerate everything from scratch (so the new station list is
#     applied consistently to GHCNh files, normalisation stats, and
#     metadata).

set -euo pipefail

# ---- Paths ----
REPO_ROOT="${REPO_ROOT:-/projects/u6do/pmms2/end-to-end-forecasting}"
BASE_DIR="${BASE_DIR:-${REPO_ROOT}/projects/tessera_downscaling/.tmp_output}"

ERA5_DIR="${ERA5_DIR:-${BASE_DIR}/_staging/processed}"
GHCNH_DIR="${GHCNH_DIR:-${BASE_DIR}/_staging/processed/ghcnh/data}"
STATIC_PATH="${STATIC_PATH:-${BASE_DIR}/_staging/processed/era5_static/era5_static_0p25_all.nc}"
STATION_CSV="${STATION_CSV:-${BASE_DIR}/_staging/raw/ghcnh/station_list.csv}"
OUTPUT_DIR="${OUTPUT_DIR:-${BASE_DIR}/dataset_timestamp_global}"

# Date range — keep in sync with the original 14y run.
START_DATE="${START_DATE:-2010-01-01}"
END_DATE="${END_DATE:-2024-01-01}"
REGIONS="${REGIONS:-europe us east_asia australia southern_africa}"

# Slurm settings.
TIME="${TIME:-12:00:00}"   # generous; previous full run took several hours
CPUS="${CPUS:-8}"
MEM="${MEM:-64G}"
PARTITION="${PARTITION:-}"

PREPROCESS_SCRIPT="${REPO_ROOT}/projects/tessera_downscaling/scripts/preprocessing/timestamp/preprocess_timestamp_global.py"
LOG_DIR="${REPO_ROOT}/logs/preprocess"
mkdir -p "${LOG_DIR}"

# ---- Pre-flight ----
if [ ! -f "${PREPROCESS_SCRIPT}" ]; then
    echo "ERROR: preprocess script not found at ${PREPROCESS_SCRIPT}" >&2
    exit 1
fi
if [ ! -d "${ERA5_DIR}" ]; then
    echo "ERROR: --era5-dir does not exist: ${ERA5_DIR}" >&2
    exit 1
fi
if [ ! -d "${GHCNH_DIR}" ]; then
    echo "ERROR: --ghcnh-dir does not exist: ${GHCNH_DIR}" >&2
    exit 1
fi
if [ ! -f "${STATIC_PATH}" ]; then
    echo "ERROR: --static-path does not exist: ${STATIC_PATH}" >&2
    exit 1
fi
if [ ! -f "${STATION_CSV}" ]; then
    echo "ERROR: --station-csv does not exist: ${STATION_CSV}" >&2
    exit 1
fi
if [ -d "${OUTPUT_DIR}" ]; then
    echo "ERROR: ${OUTPUT_DIR} already exists." >&2
    echo "Move the old output aside first:" >&2
    echo "    mv ${OUTPUT_DIR} ${OUTPUT_DIR}_old" >&2
    echo "Then re-run this submit script." >&2
    exit 1
fi

# ---- Build the command ----
PREPROCESS_CMD="${REPO_ROOT}/.venv/bin/python ${PREPROCESS_SCRIPT} \
    --era5-dir ${ERA5_DIR} \
    --ghcnh-dir ${GHCNH_DIR} \
    --static-path ${STATIC_PATH} \
    --station-csv ${STATION_CSV} \
    --output-dir ${OUTPUT_DIR} \
    --start-date ${START_DATE} \
    --end-date ${END_DATE} \
    --regions ${REGIONS}"

JOB_NAME="preprocess_timestamp_global"
SBATCH_CMD="sbatch \
    --job-name=${JOB_NAME} \
    --output=${LOG_DIR}/${JOB_NAME}_%j.out \
    --error=${LOG_DIR}/${JOB_NAME}_%j.err \
    --time=${TIME} \
    --cpus-per-task=${CPUS} \
    --mem=${MEM} \
    ${PARTITION:+--partition=${PARTITION}} \
    --wrap=\"cd ${REPO_ROOT} && ${PREPROCESS_CMD}\""

echo "============================================"
echo "Submitting preprocess_timestamp_global"
echo "============================================"
echo "  output:      ${OUTPUT_DIR}"
echo "  date range:  ${START_DATE} -> ${END_DATE}"
echo "  regions:     ${REGIONS}"
echo "  log dir:     ${LOG_DIR}"
echo ""

if [ "${DRY_RUN:-0}" = "1" ]; then
    echo "DRY RUN — would submit:"
    echo "${SBATCH_CMD}"
else
    JOB_ID=$(eval "${SBATCH_CMD}")
    echo "SUBMITTED: ${JOB_ID}"
    echo ""
    echo "Monitor with:  squeue --me"
    echo "Tail logs:     tail -f ${LOG_DIR}/${JOB_NAME}_*.out"
fi