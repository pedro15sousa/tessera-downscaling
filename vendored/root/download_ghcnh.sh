#!/bin/bash
#SBATCH --job-name=ghcnh_download
#SBATCH --output=download_logs/ghcnh_%j.out
#SBATCH --error=download_logs/ghcnh_%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=256G

# Download GHCNh station observations for the year range specified
# inside projects/dataprocessing/scripts/ghcnh/ghcnh.py (currently
# range(2010, 2024)).
#
# Runs as a Slurm job to survive login-session disconnects. The script
# has two phases per year:
#   1. PSV download: ~38k HTTP GETs to NOAA (heavily I/O-bound, benefits
#      from --cpus-per-task=8 since parallel_foreach() uses 8 workers).
#   2. Parse + groupby + NetCDF write: ~10 min per year (peak memory
#      usage is the full year's dataframe in memory during groupby).
#
# Resource sizing:
#   - 24h wall time: 14 years × (~30 min download + 10 min process) ≈
#     9h expected. 24h leaves headroom for 404-storms or NOAA slowdowns.
#   - 8 CPUs: ghcnh.py hardcodes num_processes=8 for the download phase.
#   - 8G memory: parse phase keeps one year's merged dataframe in RAM
#     (~38k stations × ~8k hourly rows × ~10 cols). Peak usage is
#     typically 3-4 GB; 8 GB gives comfortable headroom.
#
# Usage (from repo root):
#   sbatch download_ghcnh.sh
#
# Monitor:
#   squeue --me
#   tail -f download_logs/ghcnh_<JOB_ID>.out

set -euo pipefail

cd /projects/u6do/pmms2/end-to-end-forecasting

echo "GHCNh download starting on $(hostname) at $(date)"
echo "Job ID: ${SLURM_JOB_ID:-unknown}"

uv run --project projects/dataprocessing python \
    projects/dataprocessing/scripts/ghcnh/ghcnh.py

echo "GHCNh download finished at $(date)"