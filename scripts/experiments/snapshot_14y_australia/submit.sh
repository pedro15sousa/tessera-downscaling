#!/bin/bash
# Submit the single-region US experiments on the 14-year (2010-2023)
# multi-region snapshot dataset.
#
# Reads the 28-config matrix from
# projects/tessera_downscaling/scripts/experiments/snapshot_14y_australia/experiments.yaml
# — the same YAML that the notebook's EXPERIMENT_DEFS will eventually read,
# so bash and Python don't duplicate the experiment list.
#
# Output dir: .tmp_output/training_runs_snapshot_14y_australia/
#
# Why single-region-from-multi-region-dataset:
#   This script points --dataset-dir at dataset_timestamp_global (the
#   14y multi-region layout) and passes --train-regions us to train.py,
#   which is detected as multi_region_snapshot_v1 and dispatched to
#   MultiRegionSnapshotDownscalingDataset with regions=["us"]. With
#   --normalisation-policy per_region, this loads
#   regions/us/normalisation_stats_no_static.npz — so the region-specific
#   ERA5 z-score scale is preserved. Equivalent to if we had pointed at
#   regions/us/ as a flat snapshot dataset, except no extra preprocessing
#   needed.
#
# Usage (from repo root):
#   bash projects/tessera_downscaling/scripts/experiments/snapshot_14y_australia/submit.sh
#
# Dry run (prints sbatch commands without submitting):
#   DRY_RUN=1 bash projects/tessera_downscaling/scripts/experiments/snapshot_14y_australia/submit.sh
set -euo pipefail

# ---- Paths ----
REPO_ROOT="/projects/u6do/pmms2/end-to-end-forecasting"
BASE_DIR="${REPO_ROOT}/projects/tessera_downscaling/.tmp_output"

# 14y multi-region dataset. Everything downstream detects the layout
# from its metadata.json (layout_version="multi_region_snapshot_v1") —
# no need to pass a separate flag.
DATASET_DIR="${BASE_DIR}/dataset_timestamp_global"

# TESSERA and VAE latents. Use the global TESSERA extraction so the
# station-set filter covers all 5 regions; the VAE latents were trained
# on those same stations.
TESSERA_PATH="${BASE_DIR}/processed/tessera_global/patch_embeddings_2024.npy"
TESSERA_CSV="${BASE_DIR}/processed/tessera_global/station_list_filtered.csv"
export VAE_LATENTS_PATH="${BASE_DIR}/processed/station_latents_lat16_grad0.5.npy"
export VAE_LATENTS_PATH_LAT64="${BASE_DIR}/processed/station_latents_lat64_l1.npy"
export VAE_LATENTS_CSV="${BASE_DIR}/processed/tessera_global/station_list_filtered.csv"

# Hand-crafted extra surface descriptors (Bakketun et al. 2026 style) — see
# snapshot_14y_eu/submit.sh for details. Combined with --vae-latents-path,
# train.py concatenates internally (derived cache pre-built globally).
export EXTRA_DESCRIPTORS_PATH="${BASE_DIR}/processed/extra_descriptors.npy"
export VAE_LATENTS_PATH_SHUFFLED="${BASE_DIR}/processed/station_latents_lat16_grad0.5_shuffle_seed0.npy"
export VAE_LATENTS_PATH_STATS_DIM16="${BASE_DIR}/processed/station_summary_stats_dim16.npy"

OUTPUT_ROOT="${BASE_DIR}/training_runs_snapshot_14y_australia"
TRAIN_SCRIPT="projects/tessera_downscaling/scripts/train.py"
EVAL_SCRIPT="projects/tessera_downscaling/scripts/evaluate.py"

# Where this script gets its experiment list from. Same YAML read by
# the notebook — single source of truth for what experiments belong
# to this output dir.
# Resolve YAML next to this script — robust to directory renames.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPERIMENTS_YAML="${SCRIPT_DIR}/experiments.yaml"

# The region this script trains+tests on. Threaded through as
# --train-regions and implicitly into evaluate (evaluate defaults to
# the training region when --test-regions is omitted).
REGION="australia"
NORMALISATION_POLICY="per_region"

# ---- Slurm settings ----
TIME="24:00:00"
PARTITION=""  # leave empty for default, or set e.g. "workq"

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
# Inline Python that emits one pipe-delimited entry per experiment,
# expanding ${VAE_LATENTS_PATH...} refs from the environment. The
# fields are: name, target_args, extra_args. Keeping it simple —
# no eval_extra since single-region runs default to the training region.
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
    echo "Run preprocess_timestamp_global.py before submitting jobs." >&2
    exit 1
fi

LAYOUT_VERSION=$(
    python3 -c "import json; print(json.load(open('${DATASET_DIR}/metadata.json')).get('layout_version', ''))"
)
if [ "${LAYOUT_VERSION}" != "multi_region_snapshot_v1" ]; then
    echo "ERROR: ${DATASET_DIR}/metadata.json has layout_version='${LAYOUT_VERSION}', expected 'multi_region_snapshot_v1'." >&2
    echo "This submit script expects the multi-region snapshot layout." >&2
    exit 1
fi

# Confirm the region exists in the dataset. Cheaper to fail fast here
# than to queue 84 jobs that will all error at dataset construction.
REGION_PRESENT=$(
    python3 -c "
import json
md = json.load(open('${DATASET_DIR}/metadata.json'))
print('yes' if '${REGION}' in md.get('regions', {}) else 'no')
"
)
if [ "${REGION_PRESENT}" != "yes" ]; then
    echo "ERROR: region '${REGION}' not present in ${DATASET_DIR}/metadata.json" >&2
    exit 1
fi

# Confirm the per-region normalisation stats exist (without-static
# variant — no-static is the default for the 28-config matrix apart
# from the baselines, but baselines also need it since they use
# with-static which requires the 'normalisation_stats.npz' file).
# Both files should have been written by the preprocessor.
for stats_name in normalisation_stats_no_static.npz normalisation_stats.npz; do
    stats_path="${DATASET_DIR}/regions/${REGION}/${stats_name}"
    if [ ! -f "${stats_path}" ]; then
        echo "ERROR: ${stats_path} missing. Re-run preprocess_timestamp_global.py." >&2
        exit 1
    fi
done

# ---- Create log + output directories ----
mkdir -p "${REPO_ROOT}/logs"
mkdir -p "${OUTPUT_ROOT}"

# ---- Announce ----
echo "============================================"
echo "Submitting $(basename "${OUTPUT_ROOT}") experiments"
echo "============================================"
echo "Dataset:       ${DATASET_DIR}"
echo "Region:        ${REGION}"
echo "Normalisation: ${NORMALISATION_POLICY}"
echo "Output:        ${OUTPUT_ROOT}"
echo "YAML:          ${EXPERIMENTS_YAML}"
echo "Seeds:         ${SEEDS[*]}"
echo "Configs:       ${#EXPERIMENTS[@]}"
echo "Total jobs:    $(( ${#EXPERIMENTS[@]} * ${#SEEDS[@]} ))"
echo ""

# Shared training args. --train-regions selects the single region, and
# --normalisation-policy per_region loads regions/${REGION}/normalisation_stats.
COMMON_ARGS="--dataset-dir ${DATASET_DIR} \
    --tessera-path ${TESSERA_PATH} \
    --tessera-station-csv ${TESSERA_CSV} \
    --train-regions ${REGION} \
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
                ${extra_args} \
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
