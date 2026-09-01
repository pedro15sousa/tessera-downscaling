#!/usr/bin/env bash
# Submit patch-encoder training runs as one Slurm job each (one GPU per job).
#
# The sweep spans year x crop size x latent dim x auxiliary heads; the paper's
# descriptors come from its 2017 / crop 64 / latent 16 corner (the settings of
# scripts/patch_encoder/vae.yaml), and the siblings are the ablations in
# processed/vae_tessera_1B-M/. Run names are the directory names under
# <data root>/tessera_patch_encoder/outputs/vae/.
#
# A run with last.pt is complete and gets skipped; a run with periodic
# checkpoints but no last.pt is resumed from its newest one, so a wall-clock
# timeout costs at most the epochs since that checkpoint. Build the dataset
# caches first (scripts/patch_encoder/prebuild_cache.py), otherwise every job
# scans the same 326 GB patch file at once.
#
# Usage:
#   bash scripts/patch_encoder/slurm/submit_vae_sweep.sh
#   DRY_RUN=1 bash scripts/patch_encoder/slurm/submit_vae_sweep.sh   # print only
#   YEARS=2017 CROP_SIZES=64 LATENT_DIMS=16 AUX_MODES=on bash ...    # one run

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
DATA_ROOT="${TESSERA_DATA_ROOT:-/data/weather-downscaling}"
export TESSERA_DATA_ROOT="${DATA_ROOT}"

PATCH_DIR="${DATA_ROOT}/processed/tessera_station_patches"
OUTPUT_ROOT="${DATA_ROOT}/tessera_patch_encoder/outputs/vae"
LOG_DIR="${REPO_ROOT}/logs/patch_encoder"
mkdir -p "${LOG_DIR}"

# Sweep axes (space-separated; override from the environment).
YEARS="${YEARS:-2017 2024}"
CROP_SIZES="${CROP_SIZES:-64 128}"
LATENT_DIMS="${LATENT_DIMS:-16 32}"
AUX_MODES="${AUX_MODES:-on off}"
GRADIENT_WEIGHT="${GRADIENT_WEIGHT:-0.5}"

# Slurm settings. TIME must be explicit: a partition default of a few hours
# silently kills 200-epoch runs mid-training.
TIME="${TIME:-24:00:00}"
PARTITION="${PARTITION:-}"

JOB_COUNT=0
for year in ${YEARS}; do
    patches="${PATCH_DIR}/patch_embeddings_${year}_p128.npy"
    if [ ! -f "${patches}" ]; then
        echo "MISSING:   ${patches} (skipping year ${year})" >&2
        continue
    fi
    for crop in ${CROP_SIZES}; do
        for latent in ${LATENT_DIMS}; do
            for aux in ${AUX_MODES}; do
                name="p128_${year}_crop${crop}_lat${latent}_grad${GRADIENT_WEIGHT}_aux${aux}"
                run_dir="${OUTPUT_ROOT}/${name}"

                if [ -f "${run_dir}/last.pt" ]; then
                    echo "SKIP:      ${name} (already complete)"
                    continue
                fi

                # Continue from the newest periodic checkpoint if there is one.
                # The `|| true` matters: without it the failing `ls` of a run
                # directory that does not exist yet would abort the whole sweep
                # under `set -e -o pipefail`, i.e. on every first submission.
                resume_arg=""
                latest_ckpt=$(ls -1 "${run_dir}"/checkpoint_epoch*.pt 2>/dev/null \
                    | sed -E 's/.*epoch([0-9]+)\.pt$/\1 &/' \
                    | sort -n -k1,1 | tail -1 | cut -d' ' -f2- || true)
                if [ -n "${latest_ckpt}" ]; then
                    resume_arg="--resume ${latest_ckpt}"
                    echo "RESUME:    ${name} <- $(basename "${latest_ckpt}")"
                fi

                TRAIN_CMD="uv run python scripts/patch_encoder/train_vae.py \
                    --outdir ${run_dir} \
                    --patches-path ${patches} \
                    --stations-path ${PATCH_DIR}/station_list_filtered.csv \
                    --crop-size ${crop} \
                    --latent-dim ${latent} \
                    --gradient-weight ${GRADIENT_WEIGHT} \
                    --aux ${aux} ${resume_arg}"

                SBATCH_CMD="sbatch \
                    --job-name=${name} \
                    --gpus=1 \
                    --time=${TIME} \
                    --output=${LOG_DIR}/${name}_%j.log \
                    --error=${LOG_DIR}/${name}_%j.log \
                    ${PARTITION:+--partition=${PARTITION}} \
                    --wrap=\"cd ${REPO_ROOT} && ${TRAIN_CMD}\""

                if [ "${DRY_RUN:-0}" = "1" ]; then
                    echo "DRY RUN:   ${name}"
                    echo "  ${SBATCH_CMD}"
                else
                    JOB_ID=$(eval "${SBATCH_CMD}" | awk '{print $NF}')
                    echo "SUBMITTED: ${name} -> ${JOB_ID}"
                fi
                JOB_COUNT=$((JOB_COUNT + 1))
            done
        done
    done
done

echo ""
echo "Queued ${JOB_COUNT} jobs. Monitor: squeue --me. Logs: ${LOG_DIR}/<name>_<jobid>.log"
