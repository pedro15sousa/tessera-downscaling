#!/bin/bash
#SBATCH --job-name=aurora_forecast
#SBATCH --output=logs/aurora/aurora_forecast_%j.out
#SBATCH --error=logs/aurora/aurora_forecast_%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --mem=64G
#
# Stage 1 of the Aurora-context pipeline. Runs the pretrained Aurora 0.25 deg
# model from ERA5 initial conditions and writes region-cropped forecasts
# (4 surface + 5 atmos x 13 levels, fp32) in ERA5-staging layout under
# <DATA_ROOT>/ingest/aurora/lead{6,24,72}h/<region>/processed/. Resume-safe:
# re-submitting skips (lead, valid) frames already written.
#
# Environment: Aurora is an optional extra of this project --
#     uv sync --extra aurora
# -- kept out of the default env because microsoft-aurora pins its own
# torch/timm. The first model load downloads the checkpoint from HuggingFace
# and caches it under ~/.cache/huggingface. If the resolved torch was built for
# a newer CUDA than the node driver supports, pin torch to a matching wheel
# (see pyproject's pytorch-cu126 index).
#
# Inputs: 13-level ERA5 staging from scripts/data/download_era5_wb2.py --levels aurora
# (ERA5_STAGING, default <DATA_ROOT>/ingest/aurora_inputs) and the ERA5 static
# file (z / lsm / slt).
#
# RECOMMENDED SEQUENCE
#   1) Dry run (login node, no GPU -- verifies every required ERA5 input frame
#      and the static file are present, prints rollout/storage accounting):
#        DRY_RUN=1 bash scripts/aurora/submit_aurora_forecasts.sh
#   2) Smoke test (GPU node, 3 inits, small model):
#        MODEL=small LIMIT=3 sbatch scripts/aurora/submit_aurora_forecasts.sh
#   3) Full run:
#        SPLIT=all sbatch scripts/aurora/submit_aurora_forecasts.sh
#
# Monitor:  squeue --me ; tail -f logs/aurora/aurora_forecast_<JOB_ID>.out

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
DATA_ROOT="${TESSERA_DATA_ROOT:-/data/weather-downscaling}"
export TESSERA_DATA_ROOT="${DATA_ROOT}"

DATASET_META="${DATASET_META:-${DATA_ROOT}/datasets/dataset_timestamp_global/metadata.json}"
ERA5_STAGING="${ERA5_STAGING:-${DATA_ROOT}/ingest/aurora_inputs}"
STATIC_FILE="${STATIC_FILE:-${DATA_ROOT}/ingest/processed/era5_static/era5_static_0p25_all.nc}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${DATA_ROOT}/ingest/aurora}"

# ---- Knobs (override via env) ----
MODEL="${MODEL:-pretrained}"       # pretrained | small
LIMIT="${LIMIT:-0}"                # 0 = all inits; >0 = first N (smoke test)
LEADS="${LEADS:-6 24 72}"          # lead times in hours
DTYPE="${DTYPE:-float32}"
SPLIT="${SPLIT:-test}"             # test | trainval | all (cross-lead training needs all)
REGIONS="${REGIONS:-europe east_asia}"

# shellcheck disable=SC2206  # LEADS / REGIONS are intentionally word-split.
ARGS=(
    --global-metadata "${DATASET_META}"
    --era5-staging-root "${ERA5_STAGING}"
    --static-file "${STATIC_FILE}"
    --output-root "${OUTPUT_ROOT}"
    --leads ${LEADS}
    --regions ${REGIONS}
    --split "${SPLIT}"
    --model "${MODEL}"
    --dtype "${DTYPE}"
)
if [ "${LIMIT}" != "0" ]; then
    ARGS+=(--limit "${LIMIT}")
fi

cd "${REPO_ROOT}"
mkdir -p logs/aurora "${OUTPUT_ROOT}"

# ---- Dry run path (no GPU; safe on the login node) ----
if [ "${DRY_RUN:-0}" = "1" ]; then
    echo "DRY RUN -- no model load, no GPU"
    uv run python scripts/aurora/generate_aurora_forecasts.py "${ARGS[@]}" --dry-run
    exit 0
fi

echo "Aurora forecast generation starting on $(hostname) at $(date)"
echo "Job ID: ${SLURM_JOB_ID:-unknown} | model=${MODEL} | limit=${LIMIT} | leads=${LEADS} | split=${SPLIT} | dtype=${DTYPE}"
nvidia-smi || true

uv run --extra aurora python scripts/aurora/generate_aurora_forecasts.py "${ARGS[@]}"

echo "Aurora forecast generation finished at $(date)"
