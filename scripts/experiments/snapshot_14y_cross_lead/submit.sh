#!/usr/bin/env bash
# =========================================================================
# snapshot_14y_cross_lead -- lead-conditioned downscaling, baseline arms
# =========================================================================
# For each (region, entry, seed) this dispatches ONE GPU job that:
#   1. TRAINS a single lead-conditioned model on a MIX of leads via
#      --lead-datasets (ERA5 analysis lead-0 + Aurora +6/+24/+72h). Each lead
#      carries a normalised lead/72 context channel; precip is dropped so ERA5
#      (20ch) lines up with Aurora (19ch). One epoch sees every episode at
#      every lead (batch size 4 keeps the per-epoch step count comparable to
#      the single-lead runs).
#   2. then runs FOUR per-lead evals of that checkpoint, each telling the
#      dataset its lead via --lead-hours so the lead channel matches training:
#        eval_lead0h/   dataset_timestamp_global         --lead-hours 0
#        eval_lead6h/   dataset_timestamp_aurora_lead6h  --lead-hours 6
#        eval_lead24h/  dataset_timestamp_aurora_lead24h --lead-hours 24
#        eval_lead72h/  dataset_timestamp_aurora_lead72h --lead-hours 72
#      Each step is guarded so a resubmit only re-runs the missing pieces.
# baseline_kind entries (the per-lead ERA5-interp references, Fig 6's black
# line) run tessera-baselines once per lead into the same eval_lead{L}h/
# layout, on the matched station set (TESSERA filter + non-NaN v1 latents),
# seed 42 only.
#
# --use-mtpi is in COMMON_ARGS, so every lead's stations.csv needs an mtpi
# column (scripts/data/backfill_station_mtpi.py on the Aurora datasets).
#
# Runs: <data root>/training_runs/snapshot_14y_cross_lead/<region>/<name>_seed<S>
# Usage (from anywhere; data root from $TESSERA_DATA_ROOT):
#   bash scripts/experiments/snapshot_14y_cross_lead/submit.sh
#   DRY_RUN=1 ... (print only)   LOCAL=1 ... (run in this shell)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FOLDER="$(basename "${SCRIPT_DIR}")"
BATCH_SIZE="${BATCH_SIZE:-4}"
source "${SCRIPT_DIR}/../_lib.sh"

# The v1 TESSERA arm of this folder reads the v1 lat16 latents; the matched
# interp references use the same file as their station filter.
export VAE_LATENTS_PATH="${VAE_LATENTS_PATH_V1}"

# ---- The four leads (mixed at train time, evaluated separately) ----
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
echo ""

for REGION in "${REGIONS[@]}"; do
    REGION_OUT="${OUTPUT_ROOT}/${REGION}"
    ensure_dir "${REGION_OUT}"

    # --tessera-path/--tessera-station-csv filter EVERY trained run (baseline
    # included) to the TESSERA-valid station set (filter-only; no patches
    # loaded). --drop-context-channels lines the 20-channel ERA5 lead up with
    # the 19-channel Aurora leads and is stored in config.json so evaluation
    # rebuilds the model identically.
    COMMON_ARGS="--lead-datasets ${LEAD_DATASETS} --tessera-path ${TESSERA_PATH} \
--tessera-station-csv ${TESSERA_CSV} --use-mtpi --train-regions ${REGION} \
--drop-context-channels total_precipitation_sum ${MODEL_ARGS}"

    for experiment in "${EXPERIMENTS[@]}"; do
        IFS='|' read -r name target_args extra_args baseline_kind <<< "${experiment}"

        if [ -n "${baseline_kind}" ]; then
            # ---- per-lead no-model reference, matched station set, seed 42 ----
            run_dir="${REGION_OUT}/${name}_seed42"
            job="xl_${REGION}_${name}"
            if [ -f "${run_dir}/eval_lead72h/test_summary.json" ]; then
                echo "SKIP: ${job} (already complete)"; SKIP_COUNT=$((SKIP_COUNT + 1)); continue
            fi
            cmd="true"
            for spec in "${LEAD_DIRS[@]}"; do
                lead="${spec%%:*}"; d="${spec#*:}"
                cmd="${cmd} && ([ -f ${run_dir}/eval_lead${lead}h/test_summary.json ] || \
${BASELINES_CMD} --baseline ${baseline_kind} --dataset-dir ${d} ${target_args} \
--train-regions ${REGION} --tessera-path ${TESSERA_PATH} --tessera-station-csv ${TESSERA_CSV} \
--min-tessera-patch-coverage 0.5 \
--vae-latents-path ${VAE_LATENTS_PATH} --vae-latents-station-csv ${VAE_LATENTS_CSV} \
${extra_args} --output-dir ${run_dir}/eval_lead${lead}h --seed 42)"
            done
            run_job "${job}" cpu "${cmd}"
            continue
        fi

        for seed in "${SEEDS[@]}"; do
            run_dir="${REGION_OUT}/${name}_seed${seed}"
            job="xl_${REGION}_${name}_s${seed}"
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
