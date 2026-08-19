#!/bin/bash
#SBATCH --job-name=preprocess_aurora
#SBATCH --output=logs/preprocess/preprocess_aurora_%j.out
#SBATCH --error=logs/preprocess/preprocess_aurora_%j.err
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#
# Stage 2 of the Aurora-context pipeline (CPU; no GPU). Turns the per-region
# Aurora forecasts staged by scripts/aurora/generate_aurora_forecasts.py into
# dataset_timestamp_aurora_lead{6,24,72}h trees (multi_region_snapshot_v1
# layout, europe + east_asia). Runs in the normal project env (xarray + the
# preprocessing helpers; no Aurora/torch). Resume-safe: re-running skips
# era5_snapshot files already written.
#
# Usage (from anywhere):
#   TESSERA_DATA_ROOT=/data/weather-downscaling sbatch scripts/preprocessing/slurm/submit_preprocess_aurora.sh
# Env knobs: LEADS (default "6 24 72"), REGIONS ("europe east_asia").
# Quick local check: run the python directly with `--leads 72`.

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
DATA_ROOT="${TESSERA_DATA_ROOT:-/data/weather-downscaling}"
export TESSERA_DATA_ROOT="${DATA_ROOT}"

LEADS="${LEADS:-6 24 72}"
REGIONS="${REGIONS:-europe east_asia}"

cd "${REPO_ROOT}"
mkdir -p logs/preprocess

# shellcheck disable=SC2086  # LEADS / REGIONS are intentionally word-split.
uv run python scripts/preprocessing/preprocess_aurora.py \
    --global-dataset "${DATA_ROOT}/dataset_timestamp_global" \
    --aurora-staging-root "${DATA_ROOT}/_staging/aurora" \
    --output-root "${DATA_ROOT}" \
    --leads ${LEADS} \
    --regions ${REGIONS}

echo "Stage 2 preprocessing finished at $(date)"
