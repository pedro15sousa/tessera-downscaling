#!/bin/bash
#SBATCH --job-name=era5_download
#SBATCH --output=logs/download/era5_%j.out
#SBATCH --error=logs/download/era5_%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

# Download 6-hourly ERA5 (12 variables, 0.25 deg) from WeatherBench2's public
# GCS bucket into <DATA_ROOT>/_staging/processed via
# scripts/data/download_era5_wb2.py. Runs as a Slurm job so it survives
# login-session disconnects; the downloader is resume-safe (atomic_completed
# skips finished files), so re-submitting picks up where it stopped.
#
# Resource sizing: ~200k files at ~4/s with 8 workers is well inside 24 h; each
# worker holds one (var, time) slice (~250 MB peak for 3-level fields).
#
# Usage (from anywhere):
#   TESSERA_DATA_ROOT=/data/weather-downscaling sbatch scripts/data/slurm/download_era5.sh
# Env knobs: START (default 2010-01-01), END (2023-01-10T18:00:00), NUM_PROCESSES (8).

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
DATA_ROOT="${TESSERA_DATA_ROOT:-/data/weather-downscaling}"
export TESSERA_DATA_ROOT="${DATA_ROOT}"

cd "${REPO_ROOT}"
mkdir -p logs/download

echo "ERA5 download starting on $(hostname) at $(date)"
echo "Job ID: ${SLURM_JOB_ID:-unknown} | data root: ${DATA_ROOT}"

uv run python scripts/data/download_era5_wb2.py \
    --start "${START:-2010-01-01}" \
    --end "${END:-2023-01-10T18:00:00}" \
    --num-processes "${NUM_PROCESSES:-8}"

echo "ERA5 download finished at $(date)"
