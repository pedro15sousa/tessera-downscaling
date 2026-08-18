#!/usr/bin/env bash
# =========================================================================
# snapshot_14y_aurora_zeroshot — paper §3.3.3 P1
# "Aurora forecast as coarse-grid context" (zero-shot)
# =========================================================================
# For each (region, seed, config) this submits ONE sbatch job that:
#   1. TRAINS on the ERA5 dataset (dataset_timestamp_global) with precip
#      dropped (--drop-context-channels total_precipitation_sum -> 19ch),
#   2. then runs FOUR evals of that single checkpoint:
#        eval_era5/             on dataset_timestamp_global   (precip dropped,
#                                                              in-distribution)
#        eval_aurora_lead6h/    on dataset_timestamp_aurora_lead6h
#        eval_aurora_lead24h/   on dataset_timestamp_aurora_lead24h
#        eval_aurora_lead72h/   on dataset_timestamp_aurora_lead72h
#      The Aurora datasets are native 19-channel (no precip); the same
#      checkpoint loads on all four because eval resolves the dropped channel
#      leniently (precip dropped from ERA5, skipped where already absent).
#
# Matrix: 6 configs x {europe, east_asia} x seeds {42,123,456} = 36 jobs,
#         each producing 4 test_summary.json files.
#
# Env knobs: DRY_RUN=1 prints the sbatch lines without submitting.
set -euo pipefail

REPO_ROOT="/projects/u6do/pmms2/end-to-end-forecasting"
BASE_DIR="${REPO_ROOT}/projects/tessera_downscaling/.tmp_output"

# ---- Datasets ----
# Training + the in-distribution ERA5 eval both use the global dataset; the
# precip channel is dropped at train AND eval time (config-driven, lenient).
DATASET_DIR="${BASE_DIR}/dataset_timestamp_global"
AURORA_LEAD6="${BASE_DIR}/dataset_timestamp_aurora_lead6h"
AURORA_LEAD24="${BASE_DIR}/dataset_timestamp_aurora_lead24h"
AURORA_LEAD72="${BASE_DIR}/dataset_timestamp_aurora_lead72h"

# ---- TESSERA / VAE latents (GLOBAL — identical for both regions) ----
TESSERA_PATH="${BASE_DIR}/processed/tessera_global/patch_embeddings_2024.npy"
TESSERA_CSV="${BASE_DIR}/processed/tessera_global/station_list_filtered.csv"
export VAE_LATENTS_PATH="${BASE_DIR}/processed/station_latents_lat16_grad0.5.npy"
export VAE_LATENTS_CSV="${BASE_DIR}/processed/tessera_global/station_list_filtered.csv"

OUTPUT_BASE="${BASE_DIR}/training_runs_snapshot_14y_aurora_zeroshot"
TRAIN_SCRIPT="projects/tessera_downscaling/scripts/train.py"
EVAL_SCRIPT="projects/tessera_downscaling/scripts/evaluate.py"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPERIMENTS_YAML="${SCRIPT_DIR}/experiments.yaml"

# ---- Experiment matrix ----
REGIONS=(europe east_asia)
NORMALISATION_POLICY="per_region"
DROP_CHANNELS="total_precipitation_sum"   # -> 19-channel context

# ---- Slurm settings ----
TIME="24:00:00"
PARTITION=""  # leave empty for default

# ---- Hyperparameters (match snapshot_14y_eu) ----
SEEDS=(42 123 456)
BATCH_SIZE=1
EPOCHS=100
PATIENCE=10
LR="2.5e-5"
LR_WARMUP_PCT="0.05"
CNN_HIDDEN=128
CNN_LAYERS=7
MLP_HIDDEN=128
MLP_N_HIDDEN=3
NUM_WORKERS=4

echo "Pre-syncing environment..."
cd ${REPO_ROOT}
uv sync --group core

# ---- Load experiments from YAML ----
# Emits one pipe-delimited entry per experiment: name|target_args|extra_args.
# ${VAE_LATENTS_*} refs in extra_args are expanded from the environment.
if [ ! -f "${EXPERIMENTS_YAML}" ]; then
    echo "ERROR: Experiments YAML not found at ${EXPERIMENTS_YAML}" >&2
    exit 1
fi

mapfile -t EXPERIMENTS < <(python3 <<PYEOF
import os, yaml
with open("${EXPERIMENTS_YAML}") as f:
    for e in yaml.safe_load(f):
        tv_arg = "--target-variables " + " ".join(e["target_variables"])
        extra = os.path.expandvars(e["extra_args"])
        print(f"{e['name']}|{tv_arg}|{extra}")
PYEOF
)

if [ "${#EXPERIMENTS[@]}" -eq 0 ]; then
    echo "ERROR: No experiments loaded from ${EXPERIMENTS_YAML}" >&2
    exit 1
fi

# ---- Preflight: all four datasets must exist ----
for D in "${DATASET_DIR}" "${AURORA_LEAD6}" "${AURORA_LEAD24}" "${AURORA_LEAD72}"; do
    if [ ! -f "${D}/metadata.json" ]; then
        echo "ERROR: ${D}/metadata.json does not exist." >&2
        echo "Run preprocess_timestamp_global.py / preprocess_aurora.py first." >&2
        exit 1
    fi
done

mkdir -p "${REPO_ROOT}/logs"

# ---- Submit jobs ----
JOB_COUNT=0
for REGION in "${REGIONS[@]}"; do
    OUTPUT_ROOT="${OUTPUT_BASE}/${REGION}"

    # Shared training args for this region. --drop-context-channels is what
    # makes the model 19-channel; it is recorded in the checkpoint config and
    # re-read by evaluate.py so every eval matches without extra flags.
    COMMON_ARGS="--dataset-dir ${DATASET_DIR} \
        --tessera-path ${TESSERA_PATH} \
        --tessera-station-csv ${TESSERA_CSV} \
        --train-regions ${REGION} \
        --normalisation-policy ${NORMALISATION_POLICY} \
        --drop-context-channels ${DROP_CHANNELS} \
        --batch-size ${BATCH_SIZE} \
        --epochs ${EPOCHS} \
        --patience ${PATIENCE} \
        --lr ${LR} \
        --lr-warmup-pct ${LR_WARMUP_PCT} \
        --cnn-hidden ${CNN_HIDDEN} \
        --cnn-layers ${CNN_LAYERS} \
        --mlp-hidden ${MLP_HIDDEN} \
        --mlp-n-hidden ${MLP_N_HIDDEN} \
        --num-workers ${NUM_WORKERS}"

    for experiment in "${EXPERIMENTS[@]}"; do
        IFS='|' read -r name target_args extra_args <<< "${experiment}"

        for seed in "${SEEDS[@]}"; do
            run_dir="${OUTPUT_ROOT}/${name}_seed${seed}"
            job_name="az_${REGION}_${name}_s${seed}"

            # Skip the whole job only if the LAST eval finished.
            if [ -f "${run_dir}/eval_aurora_lead72h/test_summary.json" ]; then
                echo "SKIP: ${job_name} (already complete)"
                continue
            fi

            TRAIN_CMD="${REPO_ROOT}/.venv/bin/python ${TRAIN_SCRIPT} ${COMMON_ARGS} ${target_args} ${extra_args} --seed ${seed} --output-dir ${run_dir}"

            # Eval defaults: test the training region; the dropped channel and
            # normalisation policy come from the checkpoint config.
            EVAL_BASE="${REPO_ROOT}/.venv/bin/python ${EVAL_SCRIPT} --checkpoint ${run_dir}/best_model.pt --batch-size ${BATCH_SIZE} --num-workers ${NUM_WORKERS} --test-regions ${REGION}"
            EVAL_ERA5="${EVAL_BASE} --dataset-dir ${DATASET_DIR} --output-dir ${run_dir}/eval_era5"
            EVAL_A6="${EVAL_BASE} --dataset-dir ${AURORA_LEAD6} --output-dir ${run_dir}/eval_aurora_lead6h"
            EVAL_A24="${EVAL_BASE} --dataset-dir ${AURORA_LEAD24} --output-dir ${run_dir}/eval_aurora_lead24h"
            EVAL_A72="${EVAL_BASE} --dataset-dir ${AURORA_LEAD72} --output-dir ${run_dir}/eval_aurora_lead72h"

            # Each step is guarded so a resubmit after a partial run only
            # re-runs the missing pieces (train is the expensive one).
            WRAP="cd ${REPO_ROOT} \
                && ([ -f ${run_dir}/best_model.pt ] || ${TRAIN_CMD}) \
                && ([ -f ${run_dir}/eval_era5/test_summary.json ] || ${EVAL_ERA5}) \
                && ([ -f ${run_dir}/eval_aurora_lead6h/test_summary.json ] || ${EVAL_A6}) \
                && ([ -f ${run_dir}/eval_aurora_lead24h/test_summary.json ] || ${EVAL_A24}) \
                && ([ -f ${run_dir}/eval_aurora_lead72h/test_summary.json ] || ${EVAL_A72})"

            SBATCH_CMD="sbatch --job-name=${job_name} --output=${REPO_ROOT}/logs/${job_name}_%j.out --error=${REPO_ROOT}/logs/${job_name}_%j.err --gpus=1 --time=${TIME} ${PARTITION:+--partition=${PARTITION}} --wrap=\"${WRAP}\""

            if [ "${DRY_RUN:-0}" = "1" ]; then
                echo "DRY RUN: ${job_name}"
                echo "  ${SBATCH_CMD}"
                echo ""
            else
                JOB_ID=$(eval "${SBATCH_CMD}")
                echo "SUBMITTED: ${job_name} -> ${JOB_ID}"
            fi
            JOB_COUNT=$((JOB_COUNT + 1))
        done
    done
done

echo ""
echo "============================================"
echo "Submitted ${JOB_COUNT} sbatch jobs (snapshot_14y_aurora_zeroshot)"
echo "============================================"
echo ""
echo "Monitor with:  squeue --me"
echo "Cancel all:    scancel --me"
echo "Results in:    ${OUTPUT_BASE}/{europe,east_asia}/"