#!/bin/bash
# GPU job of the CSD3 Aurora latent extraction: one MI355X, one chunk (or one
# shard of it). Submitted by scripts/aurora/submit_aurora_latents.sh, which
# sets the Slurm resources; do not sbatch by hand.
#
# EXTRACT_MODE selects what scripts/aurora/extract_aurora_latents.py does:
#   smoke      small model, 3 inits, float16, --verify-decoder, physical subset
#              of every init -- into a SEPARATE tree (ingest/aurora_smoke*) so
#              the small model's latents never sit in the real one, where the
#              resume logic would take them for finished files.
#   calibrate  pretrained model, --calibrate CALIBRATE_N inits of the chunk in
#              fp32 -> <RDS_ROOT>/ingest/aurora/latent_calibration.json (no latents).
#   shard      pretrained model over sub-window SLURM_ARRAY_TASK_ID of N_SHARDS
#              of the chunk's valid times, --dtype auto (reads the calibration
#              file), physical subset every 200th init. Resume-safe: files
#              already present are skipped, so a failed array task is simply
#              resubmitted.
#
# Env from the submitter: CHUNK, EXTRACT_MODE, N_SHARDS, CALIBRATE_N, RDS_ROOT.

set -euo pipefail

# sbatch runs a COPY of this script from Slurm's spool directory, so
# BASH_SOURCE cannot locate the repo: take the submitter's exported REPO_ROOT,
# then the submission directory (the submitter cd's to the repo root).
REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}}"
# shellcheck source=scripts/aurora/_csd3.sh
source "${REPO_ROOT}/scripts/aurora/_csd3.sh"
csd3_require_env
cd "${REPO_ROOT}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"  # checkpoints are prefetched by setup_csd3_env.sh
# MIOpen (ROCm's convolution library) keeps a per-GPU-arch sqlite database of
# tuned kernels; by default under /tmp of the node, shared by every user. A
# task died with "Cannot open database file: /tmp/gfx950100.ukdb" on a node
# where that file belonged to someone else. Give each job its own directory.
export MIOPEN_USER_DB_PATH="${TMPDIR:-/tmp}/${USER}/miopen-${SLURM_JOB_ID:-$$}"
export MIOPEN_CUSTOM_CACHE_DIR="${MIOPEN_USER_DB_PATH}"
mkdir -p "${MIOPEN_USER_DB_PATH}"

: "${CHUNK:?CHUNK is required}"
EXTRACT_MODE="${EXTRACT_MODE:-shard}"
N_SHARDS="${N_SHARDS:-4}"
CALIBRATE_N="${CALIBRATE_N:-50}"

OUTPUT_ROOT="${RDS_ROOT}/ingest/aurora"
VERIFY_ROOT="${RDS_ROOT}/ingest/aurora_verify"
if [ "${EXTRACT_MODE}" = "smoke" ]; then
    OUTPUT_ROOT="${RDS_ROOT}/ingest/aurora_smoke"
    VERIFY_ROOT="${RDS_ROOT}/ingest/aurora_smoke_verify"
fi

echo "[$(date)] extract ${EXTRACT_MODE} chunk=${CHUNK} host=$(hostname) job=${SLURM_JOB_ID:-?} task=${SLURM_ARRAY_TASK_ID:-single}"
echo "ROCR_VISIBLE_DEVICES=${ROCR_VISIBLE_DEVICES:-} torch: $("${PY}" -c 'import torch; print(torch.__version__, "cuda_available", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")')"

if [ "${EXTRACT_MODE}" = "shard" ]; then
    eval "$(python3 scripts/aurora/chunks.py show "${CHUNK}" --shard "${SLURM_ARRAY_TASK_ID:-0}" --n-shards "${N_SHARDS}")"
    START="${SHARD_START}"; END="${SHARD_END}"
else
    eval "$(python3 scripts/aurora/chunks.py show "${CHUNK}")"
    START="${VALID_START}"; END="${VALID_END}"
fi

ARGS=(
    --global-metadata "${RDS_ROOT}/datasets/dataset_timestamp_global/metadata.json"
    --era5-staging-root "${RDS_ROOT}/ingest/aurora_inputs"
    --static-file "${RDS_ROOT}/ingest/processed/era5_static/era5_static_0p25_all.nc"
    --output-root "${OUTPUT_ROOT}"
    --verify-root "${VERIFY_ROOT}"
    --regions ${REGIONS} --leads ${LEADS} --split all
    --start-date "${START}" --end-date "${END}"
    --device cuda
)
case "${EXTRACT_MODE}" in
    smoke)
        ARGS+=(--model small --limit 3 --dtype float16 --verify-decoder --verify-physical-every 1)
        ;;
    calibrate)
        if [ -f "${OUTPUT_ROOT}/latent_calibration.json" ] && [ "${FORCE:-0}" != "1" ]; then
            echo "calibration file exists: ${OUTPUT_ROOT}/latent_calibration.json (FORCE=1 to redo)"
            exit 0
        fi
        ARGS+=(--model pretrained --calibrate "${CALIBRATE_N}")
        ;;
    shard)
        if [ ! -f "${OUTPUT_ROOT}/latent_calibration.json" ]; then
            echo "*** no calibration file at ${OUTPUT_ROOT}/latent_calibration.json; run --mode calibrate first" >&2
            exit 1
        fi
        ARGS+=(--model pretrained --dtype auto --verify-physical-every 200)
        ;;
    *)
        echo "unknown EXTRACT_MODE=${EXTRACT_MODE}" >&2
        exit 2
        ;;
esac

echo "window ${START} .. ${END}; output ${OUTPUT_ROOT}"
"${PY}" scripts/aurora/extract_aurora_latents.py "${ARGS[@]}"
echo "[$(date)] extract ${EXTRACT_MODE} done"
