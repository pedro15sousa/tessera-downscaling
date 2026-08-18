#!/bin/bash
# Submit the flat-EU (single-region Europe) experiments on the 6-year
# (2017-2023) snapshot dataset.
#
# Reads the 28-config matrix from
# projects/tessera_downscaling/scripts/experiments/snapshot_6y_eu/experiments.yaml
# — identical config list to the 14y variants; the only difference
# across the 6y vs 14y sibling runs is which dataset gets pointed at.
#
# Output dir: .tmp_output/training_runs_snapshot_6y_eu/
#
# Why this script exists separately from submit_snapshot_14y_eu.sh:
#   The 6y dataset has layout_version="snapshot_v1" (flat, single-
#   region Europe only) and is served by SnapshotDownscalingDataset.
#   The 14y dataset has layout_version="multi_region_snapshot_v1" and
#   is served by MultiRegionSnapshotDownscalingDataset. train.py
#   auto-detects both from metadata.json, but the 6y version doesn't
#   accept --train-regions or --normalisation-policy flags, so this
#   script omits them.
#
# Usage (from repo root):
#   bash projects/tessera_downscaling/scripts/experiments/snapshot_6y_eu/submit.sh
#
# Dry run:
#   DRY_RUN=1 bash projects/tessera_downscaling/scripts/experiments/snapshot_6y_eu/submit.sh
set -euo pipefail

# ---- Paths ----
REPO_ROOT="/projects/u6do/pmms2/end-to-end-forecasting"
BASE_DIR="${REPO_ROOT}/projects/tessera_downscaling/.tmp_output"

# 6y flat EU dataset. Produced by preprocess_timestamp.py (not the
# _global variant).
DATASET_DIR="${BASE_DIR}/dataset_timestamp"

# 6y-EU was built before the global TESSERA extraction, and uses the
# smaller EU-only TESSERA patches (~12,600 stations) to match. We keep
# this here so the 6y results are numerically comparable to previous
# weekly reports. If that compatibility wasn't needed, we could point
# at tessera_global/ instead.
TESSERA_PATH="${BASE_DIR}/processed/tessera/patch16_embeddings_2024.npy"
TESSERA_CSV="${BASE_DIR}/processed/tessera/station_list_filtered.csv"

# VAE latents — exported so os.path.expandvars in the YAML loader
# can resolve ${VAE_LATENTS_PATH...} refs. The VAE CSV references
# the global TESSERA station set (VAE was trained on all regions'
# patches); this is fine because the VAE latent loader filters to
# the subset present in the current dataset's station list.
export VAE_LATENTS_PATH="${BASE_DIR}/processed/station_latents_lat16_grad0.5.npy"
export VAE_LATENTS_PATH_LAT64="${BASE_DIR}/processed/station_latents_lat64_l1.npy"
export VAE_LATENTS_CSV="${BASE_DIR}/processed/tessera_global/station_list_filtered.csv"

OUTPUT_ROOT="${BASE_DIR}/training_runs_snapshot_6y_eu"
TRAIN_SCRIPT="projects/tessera_downscaling/scripts/train.py"
EVAL_SCRIPT="projects/tessera_downscaling/scripts/evaluate.py"

# Resolve YAML next to this script — robust to directory renames.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPERIMENTS_YAML="${SCRIPT_DIR}/experiments.yaml"

# ---- Slurm settings ----
TIME="24:00:00"
PARTITION=""

echo "Pre-syncing environment..."
cd ${REPO_ROOT}
uv sync --group core

# ---- Hyperparameters ----
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

# ---- Load experiments from YAML ----
if [ ! -f "${EXPERIMENTS_YAML}" ]; then
    echo "ERROR: Experiments YAML not found at ${EXPERIMENTS_YAML}" >&2
    exit 1
fi

mapfile -t EXPERIMENTS < <(python3 <<PYEOF
import os, yaml
with open("${EXPERIMENTS_YAML}") as f:
    for e in yaml.safe_load(f):
        # Assemble the target-variables arg. Multi-task entries get
        # multiple variables separated by spaces.
        tv_arg = "--target-variables " + " ".join(e["target_variables"])
        # Expand \${VAE_LATENTS_PATH...} refs from the environment.
        extra = os.path.expandvars(e["extra_args"])
        # Optional baseline_kind field — present only for simple-baseline
        # entries (era5_interp, persistence). Empty for trained runs.
        kind = e.get("baseline_kind", "")
        # Pipe-separated so bash can split without worrying about spaces.
        print(f"{e['name']}|{tv_arg}|{extra}|{kind}")
PYEOF
)

if [ "${#EXPERIMENTS[@]}" -eq 0 ]; then
    echo "ERROR: No experiments loaded from ${EXPERIMENTS_YAML}" >&2
    exit 1
fi

# ---- Preflight: dataset sanity ----
if [ ! -f "${DATASET_DIR}/metadata.json" ]; then
    echo "ERROR: ${DATASET_DIR}/metadata.json does not exist." >&2
    echo "Run preprocess_timestamp.py (NOT _global) before submitting." >&2
    exit 1
fi

LAYOUT_VERSION=$(
    python3 -c "import json; print(json.load(open('${DATASET_DIR}/metadata.json')).get('layout_version', ''))"
)
if [ "${LAYOUT_VERSION}" != "snapshot_v1" ]; then
    echo "ERROR: ${DATASET_DIR}/metadata.json has layout_version='${LAYOUT_VERSION}', expected 'snapshot_v1'." >&2
    echo "If you see 'multi_region_snapshot_v1', you may have the 14y dataset here — use submit_snapshot_14y_eu.sh instead." >&2
    exit 1
fi

# ---- Create log + output directories ----
mkdir -p "${REPO_ROOT}/logs"
mkdir -p "${OUTPUT_ROOT}"

# ---- Announce ----
echo "============================================"
echo "Submitting $(basename "${OUTPUT_ROOT}") experiments"
echo "============================================"
echo "Dataset:    ${DATASET_DIR}"
echo "Layout:     ${LAYOUT_VERSION} (flat EU 6y)"
echo "Output:     ${OUTPUT_ROOT}"
echo "YAML:       ${EXPERIMENTS_YAML}"
echo "Seeds:      ${SEEDS[*]}"
echo "Configs:    ${#EXPERIMENTS[@]}"
echo "Total jobs: $(( ${#EXPERIMENTS[@]} * ${#SEEDS[@]} ))"
echo ""

# Shared training args. Note: no --train-regions, no
# --normalisation-policy — those are multi-region-only flags. The flat
# snapshot dataset always uses its single built-in normalisation.
COMMON_ARGS="--dataset-dir ${DATASET_DIR} \
    --tessera-path ${TESSERA_PATH} \
    --tessera-station-csv ${TESSERA_CSV} \
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

# ---- Submit jobs ----
# Simple baseline entries (with `baseline_kind`) bypass sbatch entirely
# and run scripts/baselines/evaluate_simple_baselines.py directly. They
# are CPU-only, finish in seconds, and seed-deterministic — running 3
# seeds is just for analyzer compatibility (it produces 3 identical
# test_summary.json files).
SIMPLE_BASELINES_SCRIPT="projects/tessera_downscaling/scripts/baselines/evaluate_simple_baselines.py"

# Build the region/normalisation args that simple baselines need for
# multi-region datasets. Falls through to empty for the flat
# single-region snapshot_6y_eu folder where REGION isn't defined.
SIMPLE_BASELINE_REGION_ARGS=""
if [ -n "${REGION:-}" ]; then
    SIMPLE_BASELINE_REGION_ARGS="--train-regions ${REGION}"
fi
if [ -n "${NORMALISATION_POLICY:-}" ]; then
    SIMPLE_BASELINE_REGION_ARGS="${SIMPLE_BASELINE_REGION_ARGS} --normalisation-policy ${NORMALISATION_POLICY}"
fi

JOB_COUNT=0
SIMPLE_BASELINE_COUNT=0
for experiment in "${EXPERIMENTS[@]}"; do
    IFS='|' read -r name target_args extra_args baseline_kind <<< "${experiment}"

    for seed in "${SEEDS[@]}"; do
        run_dir="${OUTPUT_ROOT}/${name}_seed${seed}"
        job_name="${name}_s${seed}"

        # Skip if already completed.
        if [ -f "${run_dir}/test_summary.json" ]; then
            echo "SKIP: ${job_name} (already complete)"
            continue
        fi

        if [ -n "${baseline_kind}" ]; then
            # ---- Simple baseline path: run directly, no sbatch ----
            # target_args looks like "--target-variables t2m" — pass through.
            # Pass the SAME TESSERA station filter as COMMON_ARGS so simple
            # baselines are evaluated on the identical (TESSERA-valid) station
            # set as the trained ConvCNPs they're compared against. Without
            # this they ran on every spatial-test station, leaving the
            # headline table comparing rows on different station sets.
            BASELINE_CMD="${REPO_ROOT}/.venv/bin/python ${SIMPLE_BASELINES_SCRIPT} \
                --baseline ${baseline_kind} \
                --dataset-dir ${DATASET_DIR} \
                ${target_args} \
                ${SIMPLE_BASELINE_REGION_ARGS} \
                --tessera-path ${TESSERA_PATH} \
                --tessera-station-csv ${TESSERA_CSV} \
                --min-tessera-patch-coverage 0.5 \
                --output-dir ${run_dir} \
                --seed ${seed}"

            if [ "${DRY_RUN:-0}" = "1" ]; then
                echo "DRY RUN (simple baseline): ${job_name}"
                echo "  ${BASELINE_CMD}"
                echo ""
            else
                echo "RUNNING (simple baseline): ${job_name}"
                cd ${REPO_ROOT} && eval "${BASELINE_CMD}"
                if [ $? -eq 0 ]; then
                    echo "DONE: ${job_name}"
                else
                    echo "FAILED: ${job_name}" >&2
                fi
            fi
            SIMPLE_BASELINE_COUNT=$((SIMPLE_BASELINE_COUNT + 1))
        else
            # ---- Trained-model path: existing sbatch pipeline ----
            # evaluate.py defaults to the training region when --test-regions
            # is omitted, so no explicit test-region flag needed for these
            # single-region runs. Normalisation policy is stored in the
            # training checkpoint config and re-read by evaluate.py.
            TRAIN_CMD="${REPO_ROOT}/.venv/bin/python ${TRAIN_SCRIPT} ${COMMON_ARGS} ${target_args} ${extra_args} --seed ${seed} --output-dir ${run_dir}"
            EVAL_CMD="${REPO_ROOT}/.venv/bin/python ${EVAL_SCRIPT} --checkpoint ${run_dir}/best_model.pt --batch-size ${BATCH_SIZE} --num-workers ${NUM_WORKERS}"

            SBATCH_CMD="sbatch --job-name=${job_name} --output=${REPO_ROOT}/logs/${job_name}_%j.out --error=${REPO_ROOT}/logs/${job_name}_%j.err --gpus=1 --time=${TIME} ${PARTITION:+--partition=${PARTITION}} --wrap=\"cd ${REPO_ROOT} && ${TRAIN_CMD} && ${EVAL_CMD}\""

            if [ "${DRY_RUN:-0}" = "1" ]; then
                echo "DRY RUN: ${job_name}"
                echo "  ${SBATCH_CMD}"
                echo ""
            else
                JOB_ID=$(eval "${SBATCH_CMD}")
                echo "SUBMITTED: ${job_name} -> ${JOB_ID}"
            fi
            JOB_COUNT=$((JOB_COUNT + 1))
        fi
    done
done

echo ""
echo "============================================"
echo "Submitted ${JOB_COUNT} sbatch jobs ($(basename "${OUTPUT_ROOT}"))"
echo "Ran ${SIMPLE_BASELINE_COUNT} simple-baseline jobs locally"
echo "============================================"
echo ""
echo "Monitor with:  squeue --me"
echo "Cancel all:    scancel --me"
echo "Results in:    ${OUTPUT_ROOT}/"

# === SIMPLE_BASELINES_PATCH_APPLIED ===
