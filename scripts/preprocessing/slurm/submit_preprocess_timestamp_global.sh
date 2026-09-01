#!/usr/bin/env bash
# Submit preprocess_timestamp_global.py (the dataset_timestamp_global builder)
# as a Slurm job. CPU-bound, no GPU; the full 14-year, five-region build takes
# several hours.
#
# Usage (from anywhere):
#   bash scripts/preprocessing/slurm/submit_preprocess_timestamp_global.sh
#   DRY_RUN=1 bash scripts/preprocessing/slurm/submit_preprocess_timestamp_global.sh   # print only
#   LOCAL=1   bash scripts/preprocessing/slurm/submit_preprocess_timestamp_global.sh   # run here, no sbatch
#
# All inputs default to the data root layout (see DATA.md) and can be
# overridden through the env vars below. Refuses to run if OUTPUT_DIR already
# exists: move the old dataset aside first so the new station list is applied
# consistently to GHCNh files, normalisation stats and metadata.

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
DATA_ROOT="${TESSERA_DATA_ROOT:-/data/weather-downscaling}"
export TESSERA_DATA_ROOT="${DATA_ROOT}"

ERA5_DIR="${ERA5_DIR:-${DATA_ROOT}/ingest/processed}"
GHCNH_DIR="${GHCNH_DIR:-${DATA_ROOT}/ingest/processed/ghcnh/data}"
STATIC_PATH="${STATIC_PATH:-${DATA_ROOT}/ingest/processed/era5_static/era5_static_0p25_all.nc}"
STATION_CSV="${STATION_CSV:-${DATA_ROOT}/ingest/raw/ghcnh/station_list.csv}"
MTPI_CSV="${MTPI_CSV:-${DATA_ROOT}/processed/station_vectors/station_mtpi.csv}"
OUTPUT_DIR="${OUTPUT_DIR:-${DATA_ROOT}/datasets/dataset_timestamp_global}"

# Date range and regions of the paper's dataset.
START_DATE="${START_DATE:-2010-01-01}"
END_DATE="${END_DATE:-2024-01-01}"
REGIONS="${REGIONS:-europe us east_asia australia southern_africa}"

# Slurm settings.
TIME="${TIME:-12:00:00}"
CPUS="${CPUS:-8}"
MEM="${MEM:-64G}"
PARTITION="${PARTITION:-}"

LOG_DIR="${REPO_ROOT}/logs/preprocess"
mkdir -p "${LOG_DIR}"

# ---- Pre-flight ----
for d in "${ERA5_DIR}" "${GHCNH_DIR}"; do
    [ -d "${d}" ] || { echo "ERROR: directory does not exist: ${d}" >&2; exit 1; }
done
for f in "${STATIC_PATH}" "${STATION_CSV}"; do
    [ -f "${f}" ] || { echo "ERROR: file does not exist: ${f}" >&2; exit 1; }
done
if [ -d "${OUTPUT_DIR}" ]; then
    echo "ERROR: ${OUTPUT_DIR} already exists. Move it aside first:" >&2
    echo "    mv ${OUTPUT_DIR} ${OUTPUT_DIR}_old" >&2
    exit 1
fi
MTPI_ARG=""
if [ -f "${MTPI_CSV}" ]; then
    MTPI_ARG="--mtpi-csv ${MTPI_CSV}"
else
    echo "WARNING: ${MTPI_CSV} not found; building without the mtpi column (--use-mtpi runs will not work)." >&2
fi

# ---- Build the command ----
PREPROCESS_CMD="uv run python scripts/preprocessing/preprocess_timestamp_global.py \
    --era5-dir ${ERA5_DIR} \
    --ghcnh-dir ${GHCNH_DIR} \
    --static-path ${STATIC_PATH} \
    --station-csv ${STATION_CSV} \
    --output-dir ${OUTPUT_DIR} \
    --start-date ${START_DATE} \
    --end-date ${END_DATE} \
    --regions ${REGIONS} \
    ${MTPI_ARG}"

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
echo "preprocess_timestamp_global"
echo "============================================"
echo "  data root:   ${DATA_ROOT}"
echo "  output:      ${OUTPUT_DIR}"
echo "  date range:  ${START_DATE} -> ${END_DATE}"
echo "  regions:     ${REGIONS}"
echo "  log dir:     ${LOG_DIR}"
echo ""

if [ "${DRY_RUN:-0}" = "1" ]; then
    echo "DRY RUN — would submit:"
    echo "${SBATCH_CMD}"
elif [ "${LOCAL:-0}" = "1" ]; then
    cd "${REPO_ROOT}" && eval "${PREPROCESS_CMD}"
else
    JOB_ID=$(eval "${SBATCH_CMD}")
    echo "SUBMITTED: ${JOB_ID}"
    echo "Monitor with:  squeue --me"
    echo "Tail logs:     tail -f ${LOG_DIR}/${JOB_NAME}_*.out"
fi
