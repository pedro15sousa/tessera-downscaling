#!/bin/bash
#SBATCH --job-name=aurora_forecast
#SBATCH --output=logs/aurora_forecast_%j.out
#SBATCH --error=logs/aurora_forecast_%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --mem=64G
#
# Stage 1 of the Aurora-context pipeline. Runs the pretrained Aurora 0.25 deg
# model from ERA5 initial conditions over the test window and writes global
# forecasts (4 surface + 5 atmos x 13 levels, fp32) in ERA5-staging layout, per
# lead {6, 24, 72}h. Resume-safe: re-submitting skips (lead, valid) frames
# already written.
#
# ---------------------------------------------------------------------------
# ONE-TIME SETUP (login node -- has internet; uv venvs ship without pip, so
# install via `uv pip --python`):
#   cd /projects/u6do/pmms2/end-to-end-forecasting
#   uv venv .venv-aurora --python 3.11
#   uv pip install --python .venv-aurora/bin/python \
#       microsoft-aurora xarray h5netcdf h5py netcdf4 pandas tqdm
#   # NOTE: h5py is required as h5netcdf's HDF5 backend for writing. Also, if a
#   # fresh install pulls a torch built for a newer CUDA than the node driver
#   # (e.g. cu130 vs the 12.7 driver), mirror your training .venv's torch:
#   #   uv pip install --python .venv-aurora/bin/python "torch==<VER>" torchvision \
#   #       --index-url https://download.pytorch.org/whl/cu126
#   # The Aurora package is kept in its OWN venv so its torch/timm pins cannot
#   # perturb the training env (.venv). The first model load (in the job)
#   # downloads the checkpoint from HuggingFace and caches it under
#   # ~/.cache/huggingface for subsequent runs.
# ---------------------------------------------------------------------------
#
# RECOMMENDED SEQUENCE
#   1) Dry run (login node, no GPU -- verifies every required ERA5 input frame
#      and the static file are present, prints rollout/storage accounting):
#        DRY_RUN=1 bash projects/tessera_downscaling/scripts/aurora/submit_aurora_forecasts.sh
#   2) Smoke test (GPU node, 3 inits, small model -- shakes out the Aurora API,
#      static load, write path, and gives a real per-step time):
#        MODEL=small LIMIT=3 sbatch projects/tessera_downscaling/scripts/aurora/submit_aurora_forecasts.sh
#   3) Full run:
#        sbatch projects/tessera_downscaling/scripts/aurora/submit_aurora_forecasts.sh
#
# Monitor:  squeue --me ; tail -f logs/aurora_forecast_<JOB_ID>.out

set -euo pipefail

# ---- Paths ----
REPO_ROOT="/projects/u6do/pmms2/end-to-end-forecasting"
BASE_DIR="${REPO_ROOT}/projects/tessera_downscaling/.tmp_output"
VENV="${REPO_ROOT}/.venv-aurora"
SCRIPT="${REPO_ROOT}/projects/tessera_downscaling/scripts/aurora/generate_aurora_forecasts.py"

DATASET_META="${BASE_DIR}/dataset_timestamp_global/metadata.json"
ERA5_STAGING="${BASE_DIR}/_staging/processed"
STATIC_FILE="${ERA5_STAGING}/era5_static/era5_static_0p25_all.nc"
OUTPUT_ROOT="${BASE_DIR}/_staging/aurora"

# ---- Knobs (override via env) ----
MODEL="${MODEL:-pretrained}"   # pretrained | small
LIMIT="${LIMIT:-0}"            # 0 = all inits; >0 = first N (smoke test)
LEADS="${LEADS:-6 24 72}"      # lead times in hours
DTYPE="${DTYPE:-float32}"

# ---- Guard: venv exists ----
if [ ! -x "${VENV}/bin/python" ]; then
    echo "ERROR: ${VENV} not found. Run the ONE-TIME SETUP block in this script's header first." >&2
    exit 1
fi

ARGS=(
    --global-metadata "${DATASET_META}"
    --era5-staging-root "${ERA5_STAGING}"
    --static-file "${STATIC_FILE}"
    --output-root "${OUTPUT_ROOT}"
    --leads ${LEADS}
    --model "${MODEL}"
    --dtype "${DTYPE}"
)
if [ "${LIMIT}" != "0" ]; then
    ARGS+=(--limit "${LIMIT}")
fi

cd "${REPO_ROOT}"
mkdir -p logs "${OUTPUT_ROOT}"

# ---- Dry run path (no GPU; safe on the login node) ----
if [ "${DRY_RUN:-0}" = "1" ]; then
    echo "DRY RUN -- no model load, no GPU"
    "${VENV}/bin/python" "${SCRIPT}" "${ARGS[@]}" --dry-run
    exit 0
fi

echo "Aurora forecast generation starting on $(hostname) at $(date)"
echo "Job ID: ${SLURM_JOB_ID:-unknown} | model=${MODEL} | limit=${LIMIT} | leads=${LEADS} | dtype=${DTYPE}"
nvidia-smi || true

"${VENV}/bin/python" "${SCRIPT}" "${ARGS[@]}"

echo "Aurora forecast generation finished at $(date)"