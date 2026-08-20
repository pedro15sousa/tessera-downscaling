#!/usr/bin/env bash
# Submit the foundation-model patch extractions (AlphaEarth / OlmoEarth).
#
# Three targets, in the order the OlmoEarth arm needs them:
#   alphaearth    AlphaEarth embedding patches straight from GCS (CPU, network)
#   olmo-imagery  Sentinel-2 monthly composites from the Planetary Computer
#                 (CPU, network) -- input of the next target
#   olmo-embed    the OlmoEarth encoder over that imagery (one GPU per year)
#
# Each target first runs --init-only inline, which pre-allocates the output
# mmaps so the sharded jobs never race on file creation, and then submits one
# job per (year, shard). Every stage resumes: re-submitting after a timeout or
# a failure picks up from the progress files.
#
# Usage:
#   bash scripts/patch_encoder/slurm/submit_extract.sh alphaearth   [n_shards=4]
#   bash scripts/patch_encoder/slurm/submit_extract.sh olmo-imagery [n_shards=8]
#   bash scripts/patch_encoder/slurm/submit_extract.sh olmo-embed
#   DRY_RUN=1 bash scripts/patch_encoder/slurm/submit_extract.sh alphaearth
#   YEARS=2017 bash scripts/patch_encoder/slurm/submit_extract.sh alphaearth

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
DATA_ROOT="${TESSERA_DATA_ROOT:-/data/weather-downscaling}"
export TESSERA_DATA_ROOT="${DATA_ROOT}"

EXTRACT_DIR="scripts/patch_encoder/extract"
LOG_DIR="${REPO_ROOT}/logs/patch_encoder"
mkdir -p "${LOG_DIR}"

TARGET="${1:?usage: submit_extract.sh alphaearth|olmo-imagery|olmo-embed [n_shards]}"
N_SHARDS="${2:-4}"
YEARS="${YEARS:-2017 2024}"
TIME="${TIME:-12:00:00}"
PARTITION="${PARTITION:-}"

init() {  # init <command...> -- pre-allocate the output mmaps before sharding
    if [ "${DRY_RUN:-0}" = "1" ]; then
        echo "DRY RUN:   init"
        echo "  $*"
    else
        eval "$*"
    fi
}

submit() {  # submit <job_name> <gpus> <command...>
    local name="$1" gpus="$2"
    shift 2
    local sbatch_cmd="sbatch \
        --job-name=${name} \
        --time=${TIME} \
        ${gpus:+--gpus=${gpus}} \
        ${PARTITION:+--partition=${PARTITION}} \
        --output=${LOG_DIR}/${name}_%j.log \
        --error=${LOG_DIR}/${name}_%j.log \
        --wrap=\"cd ${REPO_ROOT} && $*\""
    if [ "${DRY_RUN:-0}" = "1" ]; then
        echo "DRY RUN:   ${name}"
        echo "  ${sbatch_cmd}"
    else
        JOB_ID=$(eval "${sbatch_cmd}" | awk '{print $NF}')
        echo "SUBMITTED: ${name} -> ${JOB_ID}"
    fi
}

cd "${REPO_ROOT}"

case "${TARGET}" in
  alphaearth)
    # 24 download threads per job: at 64 the node ran out of TIME_WAIT sockets
    # and the truncated zstd streams corrupted patches.
    init "uv run python ${EXTRACT_DIR}/extract_alphaearth.py --years ${YEARS} --init-only"
    for year in ${YEARS}; do
        for ((k = 0; k < N_SHARDS; k++)); do
            submit "aef_${year}_s${k}" "" \
                "uv run python ${EXTRACT_DIR}/extract_alphaearth.py \
                    --years ${year} --shard ${k} ${N_SHARDS} --workers 24"
        done
    done
    echo ""
    echo "After every shard finishes, record the nonzero counts:"
    echo "  uv run python ${EXTRACT_DIR}/extract_alphaearth.py --years ${YEARS} --report"
    ;;
  olmo-imagery)
    init "uv run python ${EXTRACT_DIR}/extract_olmoearth_imagery.py --years ${YEARS} --init-only"
    for year in ${YEARS}; do
        for ((k = 0; k < N_SHARDS; k++)); do
            submit "olmoimg_${year}_s${k}" "" \
                "uv run python ${EXTRACT_DIR}/extract_olmoearth_imagery.py \
                    --years ${year} --shard ${k} ${N_SHARDS} --workers 32"
        done
    done
    ;;
  olmo-embed)
    for year in ${YEARS}; do
        submit "olmoemb_${year}" 1 \
            "uv run python ${EXTRACT_DIR}/extract_olmoearth_embed.py \
                --years ${year} --model BASE"
    done
    ;;
  *)
    echo "Unknown target: ${TARGET} (alphaearth | olmo-imagery | olmo-embed)" >&2
    exit 1
    ;;
esac

echo ""
echo "Logs: ${LOG_DIR}/<name>_<jobid>.log. Monitor: squeue --me"
