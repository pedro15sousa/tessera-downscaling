#!/bin/bash
#SBATCH --job-name=era5_download
#SBATCH --output=download_logs/era5_%j.out
#SBATCH --error=download_logs/era5_%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

# Download ERA5 data from WeatherBench2's public GCS bucket for the
# date range specified inside projects/dataprocessing/scripts/era5/
# weatherbench2.py (currently 2010-01-01 → 2023-01-10).
#
# Runs as a Slurm job rather than a foreground/nohup process so it
# survives login-session disconnects. The script is resume-safe — if
# the job is killed mid-run, relaunching picks up where it left off
# (atomic_completed() skip check in the downloader).
#
# Resource sizing:
#   - 48h wall time: conservative upper bound for ~204k new file
#     downloads at ~4/sec with 2 workers. Real run is more like
#     15-20h; 48h leaves headroom.
#   - 4 CPUs: 2 are used by parallel_foreach() for actual downloads,
#     the other 2 give xarray + h5netcdf + gcsfs breathing room for
#     I/O orchestration.
#   - 16G memory: each xr.open_zarr() slice keeps the selected
#     (var, time, lat, lon) in memory before writing. 20 variables
#     with 3 pressure levels at 1440×721 are ~250 MB peak per
#     worker, plus xarray/gcsfs overhead. 16G is comfortable.
#
# Usage (from repo root):
#   sbatch download_era5.sh
#
# Monitor:
#   squeue --me
#   tail -f download_logs/era5_<JOB_ID>.out

set -euo pipefail

cd /projects/u6do/pmms2/end-to-end-forecasting

echo "ERA5 download starting on $(hostname) at $(date)"
echo "Job ID: ${SLURM_JOB_ID:-unknown}"

uv run --project projects/dataprocessing python \
    projects/dataprocessing/scripts/era5/weatherbench2.py

echo "ERA5 download finished at $(date)"