#!/usr/bin/env bash
# =========================================================================
# snapshot_14y_cross_lead — lead-conditioned downscaling
# =========================================================================
# For each (region, seed, config) this submits ONE sbatch job that:
#   1. TRAINS a single lead-conditioned model on a MIX of leads via
#      --lead-datasets (ERA5 analysis lead-0 + Aurora +6/+24/+72h). Each lead
#      carries a normalised lead/72 context channel; precip is dropped
#      automatically so ERA5 (20ch) lines up with Aurora (19ch). One epoch sees
#      every episode at every lead.
#   2. then runs FOUR per-lead evals of that single checkpoint, each telling the
#      dataset its lead via --lead-hours so the lead channel matches training:
#        eval_lead0h/   dataset_timestamp_global        --lead-hours 0
#        eval_lead6h/   dataset_timestamp_aurora_lead6h  --lead-hours 6
#        eval_lead24h/  dataset_timestamp_aurora_lead24h --lead-hours 24
#        eval_lead72h/  dataset_timestamp_aurora_lead72h --lead-hours 72
#      Each produces a test_summary.json -> per-lead RMSE / CRPS / calibration,
#      i.e. the σ(lead) curve the experiment is built to measure. The dropped
#      precip channel + static-field setting come from the checkpoint config, so
#      every eval reconstructs the right context automatically.
#
# Matrix: 6 configs x {europe, east_asia} x seeds {42,123,456} = 36 jobs,
#         each producing 4 test_summary.json files.
#
# Env knobs: DRY_RUN=1 prints the sbatch lines without submitting.
set -euo pipefail

REPO_ROOT="/projects/u6do/pmms2/end-to-end-forecasting"
BASE_DIR="${REPO_ROOT}/projects/tessera_downscaling/.tmp_output"

# ---- Datasets (the four leads mixed at train time, evaluated separately) ----
DATASET_DIR="${BASE_DIR}/dataset_timestamp_global"          # lead 0 (ERA5 analysis)
AURORA_LEAD6="${BASE_DIR}/dataset_timestamp_aurora_lead6h"
AURORA_LEAD24="${BASE_DIR}/dataset_timestamp_aurora_lead24h"
AURORA_LEAD72="${BASE_DIR}/dataset_timestamp_aurora_lead72h"

# Lead spec consumed by train.py. Add/remove a lead just by editing this line
# (e.g. drop '0:...' to train without the ERA5 analysis).
LEAD_DATASETS="0:${DATASET_DIR} 6:${AURORA_LEAD6} 24:${AURORA_LEAD24} 72:${AURORA_LEAD72}"

# ---- TESSERA / VAE latents (GLOBAL — identical for both regions) ----
# Exported so experiments.yaml can reference them in the concat/FiLM configs.
# TESSERA_PATH/CSV are ALSO applied to every run (baseline included) via
# COMMON_ARGS as a station filter — no patches are loaded for the baseline, it
# is filter-only, so all configs land on the same TESSERA-valid station set.
# Only the VAE configs reference VAE_LATENTS_* (the latents ARE loaded there).
export TESSERA_PATH="${BASE_DIR}/processed/tessera_global/patch_embeddings_2024.npy"
export TESSERA_CSV="${BASE_DIR}/processed/tessera_global/station_list_filtered.csv"
export VAE_LATENTS_PATH="${BASE_DIR}/processed/station_latents_lat16_grad0.5.npy"
export VAE_LATENTS_CSV="${BASE_DIR}/processed/tessera_global/station_list_filtered.csv"

OUTPUT_BASE="${BASE_DIR}/training_runs_snapshot_14y_cross_lead"
TRAIN_SCRIPT="projects/tessera_downscaling/scripts/train.py"
EVAL_SCRIPT="projects/tessera_downscaling/scripts/evaluate.py"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPERIMENTS_YAML="${SCRIPT_DIR}/experiments.yaml"

# ---- Experiment matrix ----
REGIONS=(europe east_asia)
NORMALISATION_POLICY="per_region"

# ---- Slurm settings ----
TIME="24:00:00"
PARTITION=""  # leave empty for default

# ---- Hyperparameters (match snapshot_14y_aurora_zeroshot) ----
# NOTE: batch size 4 (not 1) because the cross-lead training set is 4x larger
# (every episode x 4 leads); this keeps the per-epoch step count and wall time
# comparable to the single-lead runs. Validated in the cross-lead smoke.
SEEDS=(42 123 456)
BATCH_SIZE=4
EPOCHS=100
PATIENCE=10
LR="2.5e-5"
LR_WARMUP_PCT="0.05"
CNN_HIDDEN=128
CNN_LAYERS=7
MLP_HIDDEN=128
MLP_N_HIDDEN=3
NUM_WORKERS=4

cd ${REPO_ROOT}
# Skip the env sync on a dry run — DRY_RUN should have no side effects on the
# shared .venv (it only prints the sbatch commands it would submit).
if [ "${DRY_RUN:-0}" = "1" ]; then
    echo "DRY_RUN=1 — skipping 'uv sync --group core' (no env changes)."
else
    echo "Pre-syncing environment..."
    uv sync --group core
fi

# ---- Load experiments from YAML ----
# Emits one pipe-delimited entry per experiment: name|target_args|extra_args.
# ${TESSERA_*}/${VAE_LATENTS_*} refs in extra_args are expanded from the env.
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

# ---- Preflight: all four lead datasets must exist ----
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

    # Shared training args for this region. --lead-datasets supplies the per-lead
    # dirs; the precip drop that lines up the 20-channel ERA5 lead-0 dataset with
    # the 19-channel Aurora leads is passed EXPLICITLY (same as the zero-shot
    # submit), so it lands in config.json and evaluate.py rebuilds the model
    # identically. The drop is by name and lenient by default: it hits the ERA5
    # lead-0 dataset and no-ops the Aurora leads where precip is already absent.
    #
    # --tessera-path/--tessera-station-csv are in COMMON_ARGS (mirroring the
    # zero-shot submit) so EVERY trained run — baseline included — filters
    # stations to those with valid TESSERA patches. Without this the no-TESSERA
    # bilinear baseline trained+evaluated on the full spatial-test set (europe
    # t2m 947 / wind 598) while the VAE variants used the TESSERA∩latent subset
    # (898 / 550), so the headline rows sat on different station sets. The VAE
    # configs add only --vae-latents-path on top; they no longer repeat
    # --tessera-path in their extra_args (see experiments.yaml).
    #
    # --use-mtpi is in COMMON_ARGS so EVERY cross-lead run (baseline + VAE) trains
    # with mTPI as a 3rd per-station feature (n_elev_features=3). This requires an
    # `mtpi` column in EVERY lead's stations.csv — the lead-0 (global) dataset was
    # backfilled 2026-06-25 and the three Aurora lead datasets on 2026-07-06 (all
    # via backfill_station_mtpi.py with processed/station_mtpi.csv), so the four
    # leads carry identical, row-aligned mTPI (full coverage, missing→0.0, no NaN
    # → no station dropped). Because it's uniform across leads, MultiLeadDataset
    # serves target_mtpi for every episode and the batch is homogeneous.
    COMMON_ARGS="--lead-datasets ${LEAD_DATASETS} \
        --tessera-path ${TESSERA_PATH} \
        --tessera-station-csv ${TESSERA_CSV} \
        --use-mtpi \
        --train-regions ${REGION} \
        --drop-context-channels total_precipitation_sum \
        --normalisation-policy ${NORMALISATION_POLICY} \
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
            job_name="xl_${REGION}_${name}_s${seed}"

            # Skip the whole job only if the LAST eval finished.
            if [ -f "${run_dir}/eval_lead72h/test_summary.json" ]; then
                echo "SKIP: ${job_name} (already complete)"
                continue
            fi

            TRAIN_CMD="${REPO_ROOT}/.venv/bin/python ${TRAIN_SCRIPT} ${COMMON_ARGS} ${target_args} ${extra_args} --seed ${seed} --output-dir ${run_dir}"

            # Per-lead eval: same checkpoint, each dataset tagged with its lead so
            # the lead channel matches training. Dropped channel / static setting
            # come from the checkpoint config.
            EVAL_BASE="${REPO_ROOT}/.venv/bin/python ${EVAL_SCRIPT} --checkpoint ${run_dir}/best_model.pt --batch-size ${BATCH_SIZE} --num-workers ${NUM_WORKERS} --test-regions ${REGION}"
            EVAL_L0="${EVAL_BASE} --dataset-dir ${DATASET_DIR}  --lead-hours 0  --output-dir ${run_dir}/eval_lead0h"
            EVAL_L6="${EVAL_BASE} --dataset-dir ${AURORA_LEAD6}  --lead-hours 6  --output-dir ${run_dir}/eval_lead6h"
            EVAL_L24="${EVAL_BASE} --dataset-dir ${AURORA_LEAD24} --lead-hours 24 --output-dir ${run_dir}/eval_lead24h"
            EVAL_L72="${EVAL_BASE} --dataset-dir ${AURORA_LEAD72} --lead-hours 72 --output-dir ${run_dir}/eval_lead72h"

            # Each step is guarded so a resubmit after a partial run only
            # re-runs the missing pieces (train is the expensive one).
            WRAP="cd ${REPO_ROOT} \
                && ([ -f ${run_dir}/best_model.pt ] || ${TRAIN_CMD}) \
                && ([ -f ${run_dir}/eval_lead0h/test_summary.json ]  || ${EVAL_L0}) \
                && ([ -f ${run_dir}/eval_lead6h/test_summary.json ]  || ${EVAL_L6}) \
                && ([ -f ${run_dir}/eval_lead24h/test_summary.json ] || ${EVAL_L24}) \
                && ([ -f ${run_dir}/eval_lead72h/test_summary.json ] || ${EVAL_L72})"

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
echo "Submitted ${JOB_COUNT} sbatch jobs (snapshot_14y_cross_lead)"
echo "============================================"
echo ""
echo "Monitor with:  squeue --me"
echo "Cancel all:    scancel --me"
echo "Results in:    ${OUTPUT_BASE}/{europe,east_asia}/"