#!/usr/bin/env bash
# Submit the foundation-model benchmark sweep: one Slurm job (one GPU) per run.
#
#   {alphaearth, olmoearth} x {2017, 2024} x {lat16, lat32} x {aux on, off}
#
# = 16 runs, all on the paper's recipe (beta 5e-4, gradient weight 0.5, 200
# epochs, seed 42) so that only the surface embedding differs from the TESSERA
# arm in scripts/patch_encoder/slurm/submit_vae_sweep.sh.
#
# "crop64" in the run name is the 640 m station-centred window: a runtime
# 128 -> 64 centre crop for AlphaEarth, and the whole native 16x16 token grid
# (same footprint) for OlmoEarth, which is therefore not cropped at all.
#
# Run layout, matching the TESSERA sweep one level deeper:
#   <data root>/tessera_patch_encoder/outputs/vae/<source>/<year>/crop64_lat<L>_aux<on|off>/
# eval_vae.py then writes eval/station_latents.npy inside each.
#
# A run with last.pt is complete and gets skipped; a run with periodic
# checkpoints but no last.pt is resumed from its newest one. Build the dataset
# caches first (scripts/patch_encoder/prebuild_cache.py --patches ...),
# otherwise every job scans the same patch file at once.
#
# Usage:
#   bash scripts/patch_encoder/slurm/submit_fm_sweep.sh
#   DRY_RUN=1 bash scripts/patch_encoder/slurm/submit_fm_sweep.sh
#   SOURCES=olmoearth YEARS=2017 LATENT_DIMS=16 AUX_MODES=on bash ...   # one run

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
DATA_ROOT="${TESSERA_DATA_ROOT:-/data/weather-downscaling}"
export TESSERA_DATA_ROOT="${DATA_ROOT}"

CONFIG_DIR="scripts/patch_encoder"
OUTPUT_ROOT="${DATA_ROOT}/tessera_patch_encoder/outputs/vae"
LOG_DIR="${REPO_ROOT}/logs/patch_encoder"
mkdir -p "${LOG_DIR}"

# Sweep axes (space-separated; override from the environment).
SOURCES="${SOURCES:-alphaearth olmoearth}"
YEARS="${YEARS:-2017 2024}"
LATENT_DIMS="${LATENT_DIMS:-16 32}"
AUX_MODES="${AUX_MODES:-on off}"

# 12 h is the right wall for these geometries: the AlphaEarth latent-32 runs
# hit an 8 h limit mid-training and had to be resumed.
TIME="${TIME:-12:00:00}"
PARTITION="${PARTITION:-}"

# The patch file and the runtime crop of each source. AlphaEarth stores 128 px
# rasters and is cropped to 64; OlmoEarth's 16x16 token grid is used whole.
patches_for() {  # patches_for <source> <year>
    case "$1" in
        alphaearth)
            echo "${DATA_ROOT}/processed/alphaearth_station_patches/patch_embeddings_alphaearth_$2_p128.npy"
            ;;
        olmoearth)
            echo "${DATA_ROOT}/processed/olmoearth_station_patches/patch_embeddings_olmoearth_$2_g16.npy"
            ;;
        *)
            echo "Unknown source: $1" >&2
            exit 1
            ;;
    esac
}

STATIONS="${DATA_ROOT}/processed/tessera_station_patches/station_list_filtered.csv"

JOB_COUNT=0
for source in ${SOURCES}; do
    for year in ${YEARS}; do
        patches="$(patches_for "${source}" "${year}")"
        if [ ! -f "${patches}" ]; then
            echo "MISSING:   ${patches} (skipping ${source} ${year})" >&2
            continue
        fi
        crop_arg=""
        [ "${source}" = "alphaearth" ] && crop_arg="--crop-size 64"

        for latent in ${LATENT_DIMS}; do
            for aux in ${AUX_MODES}; do
                name="crop64_lat${latent}_aux${aux}"
                run_dir="${OUTPUT_ROOT}/${source}/${year}/${name}"
                tag="${source}_${year}_${name}"

                if [ -f "${run_dir}/last.pt" ]; then
                    echo "SKIP:      ${tag} (already complete)"
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
                    echo "RESUME:    ${tag} <- $(basename "${latest_ckpt}")"
                fi

                TRAIN_CMD="uv run python scripts/patch_encoder/train_vae.py \
                    --config ${CONFIG_DIR}/vae_${source}.yaml \
                    --outdir ${run_dir} \
                    --patches-path ${patches} \
                    --stations-path ${STATIONS} \
                    ${crop_arg} \
                    --latent-dim ${latent} \
                    --aux ${aux} ${resume_arg}"

                SBATCH_CMD="sbatch \
                    --job-name=${tag} \
                    --gpus=1 \
                    --time=${TIME} \
                    --output=${LOG_DIR}/${tag}_%j.log \
                    --error=${LOG_DIR}/${tag}_%j.log \
                    ${PARTITION:+--partition=${PARTITION}} \
                    --wrap=\"cd ${REPO_ROOT} && ${TRAIN_CMD}\""

                if [ "${DRY_RUN:-0}" = "1" ]; then
                    echo "DRY RUN:   ${tag}"
                    echo "  ${SBATCH_CMD}"
                else
                    JOB_ID=$(eval "${SBATCH_CMD}" | awk '{print $NF}')
                    echo "SUBMITTED: ${tag} -> ${JOB_ID}"
                fi
                JOB_COUNT=$((JOB_COUNT + 1))
            done
        done
    done
done

echo ""
echo "Queued ${JOB_COUNT} jobs. Monitor: squeue --me. Logs: ${LOG_DIR}/<name>_<jobid>.log"
echo "Then evaluate each run: bash scripts/patch_encoder/slurm/submit_eval.sh ${OUTPUT_ROOT}/<source>/<year>/*"
