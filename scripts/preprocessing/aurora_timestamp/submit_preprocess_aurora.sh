#!/bin/bash
#SBATCH --job-name=preprocess_aurora
#SBATCH --output=logs/preprocess_aurora_%j.out
#SBATCH --error=logs/preprocess_aurora_%j.err
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#
# Stage 2 of the Aurora-context pipeline (CPU; no GPU). Crops the global Aurora
# forecasts (Stage 1 output) to EU + East Asia and writes
# dataset_timestamp_aurora_lead{6,24,72}h trees in multi_region_snapshot_v1
# layout. Runs in the TRAINING venv (.venv) -- it needs xarray + the
# preprocessing helpers, not Aurora/torch. Resume-safe: re-running skips
# era5_snapshot files already written.
#
# (Quick local check first, e.g. one lead: append `--leads 72` to ARGS below or
#  run the python directly with .venv/bin/python.)

set -euo pipefail

REPO_ROOT="/projects/u6do/pmms2/end-to-end-forecasting"
BASE_DIR="${REPO_ROOT}/projects/tessera_downscaling/.tmp_output"
VENV="${REPO_ROOT}/.venv"
SCRIPT="${REPO_ROOT}/projects/tessera_downscaling/scripts/preprocessing/aurora_timestamp/preprocess_aurora.py"

LEADS="${LEADS:-6 24 72}"
REGIONS="${REGIONS:-europe east_asia}"

cd "${REPO_ROOT}"
mkdir -p logs

"${VENV}/bin/python" "${SCRIPT}" \
    --global-dataset "${BASE_DIR}/dataset_timestamp_global" \
    --aurora-staging-root "${BASE_DIR}/_staging/aurora" \
    --output-root "${BASE_DIR}" \
    --leads ${LEADS} \
    --regions ${REGIONS}

echo "Stage 2 preprocessing finished at $(date)"