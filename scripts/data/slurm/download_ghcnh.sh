#!/bin/bash
#SBATCH --job-name=ghcnh_download
#SBATCH --output=logs/download/ghcnh_%j.out
#SBATCH --error=logs/download/ghcnh_%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=256G

# Download GHCNh station observations for a year range and bin them into
# 6-hourly NetCDFs under <DATA_ROOT>/_staging via scripts/data/download_ghcnh.py.
# Two phases per year: ~38k HTTP GETs to NOAA (I/O-bound), then a parse +
# groupby + NetCDF write that holds one year's merged dataframe in RAM (the
# reason for the large --mem). Resume-safe; re-submitting skips finished files.
#
# Usage (from anywhere):
#   TESSERA_DATA_ROOT=/data/weather-downscaling sbatch scripts/data/slurm/download_ghcnh.sh
# Env knobs: YEARS (default "2010 2023"), NUM_PROCESSES (4).

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
DATA_ROOT="${TESSERA_DATA_ROOT:-/data/weather-downscaling}"
export TESSERA_DATA_ROOT="${DATA_ROOT}"

cd "${REPO_ROOT}"
mkdir -p logs/download

echo "GHCNh download starting on $(hostname) at $(date)"
echo "Job ID: ${SLURM_JOB_ID:-unknown} | data root: ${DATA_ROOT}"

# shellcheck disable=SC2086  # YEARS is intentionally word-split into two ints.
uv run python scripts/data/download_ghcnh.py \
    --years ${YEARS:-2010 2023} \
    --num-processes "${NUM_PROCESSES:-4}"

echo "GHCNh download finished at $(date)"
