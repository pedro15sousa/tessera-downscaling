#!/usr/bin/env bash
# Submit one eval_vae.py job per trained run (GPU, a few hours each).
#
# Each run's eval/station_latents.npy is what the downscaler consumes; copy the
# chosen one into processed/vae_tessera_1B-M/ afterwards and record it in that
# folder's provenance.txt.
#
# Usage:
#   bash scripts/patch_encoder/slurm/submit_eval.sh <run_dir> [<run_dir> ...]
#   bash scripts/patch_encoder/slurm/submit_eval.sh \
#       /data/weather-downscaling/tessera_patch_encoder/outputs/vae/p128_2017_*
#   DRY_RUN=1 bash scripts/patch_encoder/slurm/submit_eval.sh <run_dir>

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
DATA_ROOT="${TESSERA_DATA_ROOT:-/data/weather-downscaling}"
export TESSERA_DATA_ROOT="${DATA_ROOT}"

LOG_DIR="${REPO_ROOT}/logs/patch_encoder"
mkdir -p "${LOG_DIR}"

TIME="${TIME:-03:00:00}"
PARTITION="${PARTITION:-}"

if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <run_dir> [<run_dir> ...]" >&2
    exit 1
fi

for run_dir in "$@"; do
    abs_run_dir="$(cd "${run_dir}" && pwd)"
    run_name="$(basename "${abs_run_dir}")"

    SBATCH_CMD="sbatch \
        --job-name=eval-${run_name} \
        --gpus=1 \
        --time=${TIME} \
        --output=${LOG_DIR}/eval_${run_name}_%j.log \
        --error=${LOG_DIR}/eval_${run_name}_%j.log \
        ${PARTITION:+--partition=${PARTITION}} \
        --wrap=\"cd ${REPO_ROOT} && uv run python scripts/patch_encoder/eval_vae.py ${abs_run_dir}\""

    if [ "${DRY_RUN:-0}" = "1" ]; then
        echo "DRY RUN:   ${run_name}"
        echo "  ${SBATCH_CMD}"
    else
        JOB_ID=$(eval "${SBATCH_CMD}" | awk '{print $NF}')
        echo "SUBMITTED: eval-${run_name} -> ${JOB_ID}"
    fi
done

echo ""
echo "Logs: ${LOG_DIR}/eval_<run_name>_<jobid>.log. Monitor: squeue --me"
