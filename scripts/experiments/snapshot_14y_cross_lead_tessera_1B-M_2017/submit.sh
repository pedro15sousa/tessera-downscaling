#!/usr/bin/env bash
# =========================================================================
# snapshot_14y_cross_lead_tessera_1B-M_2017 -- lead-conditioned downscaling,
# the paper's TESSERA arm (1B-M 2017 crop64_lat16_auxon latents)
# =========================================================================
# Same job structure as snapshot_14y_cross_lead/submit.sh (train once on the
# lead mix, then evaluate per lead into eval_lead{0,6,24,72}h/); the only
# difference is VAE_LATENTS_PATH. The no-TESSERA baseline and the ERA5-interp
# references are latents-independent and are NOT re-run here.
#
# Runs: <data root>/training_runs/snapshot_14y_cross_lead_tessera_1B-M_2017/<region>/<name>_seed<S>
# Usage (from anywhere; data root from $TESSERA_DATA_ROOT):
#   bash scripts/experiments/snapshot_14y_cross_lead_tessera_1B-M_2017/submit.sh
#   DRY_RUN=1 ... (print only)   LOCAL=1 ... (run in this shell)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FOLDER="$(basename "${SCRIPT_DIR}")"
BATCH_SIZE="${BATCH_SIZE:-4}"
source "${SCRIPT_DIR}/../_lib.sh"

export TESSERA_VARIANT="1B-M"
export EMBED_YEAR="2017"
export VAE_LATENTS_PATH="${VAE_LATENTS_PATH_1BM}"

LEAD_DIRS=(
    "0:${DATASET_DIR}"
    "6:${BASE_DIR}/datasets/dataset_timestamp_aurora_lead6h"
    "24:${BASE_DIR}/datasets/dataset_timestamp_aurora_lead24h"
    "72:${BASE_DIR}/datasets/dataset_timestamp_aurora_lead72h"
)
LEAD_DATASETS="${LEAD_DIRS[*]}"
REGIONS=(europe east_asia)

load_flat_experiments
for spec in "${LEAD_DIRS[@]}"; do
    d="${spec#*:}"
    if [ ! -f "${d}/metadata.json" ]; then
        echo "ERROR: ${d}/metadata.json does not exist (preprocess_timestamp_global.py / preprocess_aurora.py)." >&2
        [ "${DRY_RUN}" = "1" ] || exit 1
    fi
done
announce
echo "Regions:     ${REGIONS[*]}"
echo "Leads:       ${LEAD_DATASETS}"
echo "Latents:     ${VAE_LATENTS_PATH}"
echo ""

for REGION in "${REGIONS[@]}"; do
    REGION_OUT="${OUTPUT_ROOT}/${REGION}"
    ensure_dir "${REGION_OUT}"
    COMMON_ARGS="--lead-datasets ${LEAD_DATASETS} --tessera-path ${TESSERA_PATH} \
--tessera-station-csv ${TESSERA_CSV} --use-mtpi --train-regions ${REGION} \
--drop-context-channels total_precipitation_sum ${MODEL_ARGS}"

    for experiment in "${EXPERIMENTS[@]}"; do
        IFS='|' read -r name target_args extra_args _ <<< "${experiment}"
        for seed in "${SEEDS[@]}"; do
            run_dir="${REGION_OUT}/${name}_seed${seed}"
            job="xl1BM17_${REGION}_${name}_s${seed}"
            if [ -f "${run_dir}/eval_lead72h/test_summary.json" ]; then
                echo "SKIP: ${job} (already complete)"; SKIP_COUNT=$((SKIP_COUNT + 1)); continue
            fi
            train="${TRAIN_CMD} ${COMMON_ARGS} ${target_args} ${extra_args} --seed ${seed} --output-dir ${run_dir}"
            eval_base="${EVAL_CMD} --checkpoint ${run_dir}/best_model.pt --batch-size ${BATCH_SIZE} --num-workers ${NUM_WORKERS}"
            cmd="([ -f ${run_dir}/best_model.pt ] || ${train})"
            for spec in "${LEAD_DIRS[@]}"; do
                lead="${spec%%:*}"; d="${spec#*:}"
                cmd="${cmd} && ([ -f ${run_dir}/eval_lead${lead}h/test_summary.json ] || \
${eval_base} --dataset-dir ${d} --lead-hours ${lead} --output-dir ${run_dir}/eval_lead${lead}h)"
            done
            run_job "${job}" gpu "${cmd}"
        done
    done
done
summarise
