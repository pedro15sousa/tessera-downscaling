#!/bin/bash
# Submit the east_asia TESSERA-1B-M VAE-latent sweep (embedding year
# 2017) on the 14-year (2010-2023) multi-region snapshot dataset.
#
# Sweep axes live at two levels:
#   - THIS FOLDER fixes the tessera variant (1B-M) and embedding year
#     (2017), exported below as TESSERA_VARIANT / EMBED_YEAR and
#     expanded inside experiments.yaml — the yaml is byte-identical across
#     all snapshot_14y_<region>_tessera_1B-M_<year> folders.
#   - experiments.yaml sweeps crop {64,128} x latent dim {16,32,64} x
#     aux {on,off}, paper config only (direct concat, +mTPI, with elev,
#     no static fields, wd=1e-4).
#
# Latents: .tmp_output/processed/vae_tessera_1B-M/ (see provenance.txt
# there). Station filtering is unchanged from the original region sweeps:
# the old v1 TESSERA_PATH patch filter stays in COMMON_ARGS, so the
# effective station set is old-patch-valid ∩ new-latent-valid.
#
# Output dir: .tmp_output/training_runs_snapshot_14y_east_asia_tessera_1B-M_2017_shuffled/
#
# Usage (from repo root) — DRY RUN IS THE DEFAULT for this folder:
#   bash projects/tessera_downscaling/scripts/experiments/snapshot_14y_east_asia_tessera_1B-M_2017_shuffled/submit.sh
# Real submission:
#   DRY_RUN=0 bash projects/tessera_downscaling/scripts/experiments/snapshot_14y_east_asia_tessera_1B-M_2017_shuffled/submit.sh
set -euo pipefail

# Dry run by default — this sweep must be submitted deliberately.
DRY_RUN="${DRY_RUN:-1}"

# ---- Paths ----
REPO_ROOT="/projects/u6do/pmms2/end-to-end-forecasting"
BASE_DIR="${REPO_ROOT}/projects/tessera_downscaling/.tmp_output"

# 14y multi-region dataset. Everything downstream detects the layout
# from its metadata.json (layout_version="multi_region_snapshot_v1").
DATASET_DIR="${BASE_DIR}/dataset_timestamp_global"

# TESSERA station filter — unchanged from the original region sweeps
# (old v1 global extraction) so station sets stay comparable.
TESSERA_PATH="${BASE_DIR}/processed/tessera_global/patch_embeddings_2024.npy"
TESSERA_CSV="${BASE_DIR}/processed/tessera_global/station_list_filtered.csv"

# Folder-level sweep axes, expanded inside experiments.yaml.
export TESSERA_VARIANT="1B-M"
export EMBED_YEAR="2017"
export VAE_LATENTS_DIR="${BASE_DIR}/processed/vae_tessera_${TESSERA_VARIANT}"
export VAE_LATENTS_CSV="${BASE_DIR}/processed/tessera_global/station_list_filtered.csv"

OUTPUT_ROOT="${BASE_DIR}/training_runs_snapshot_14y_east_asia_tessera_1B-M_2017_shuffled"
TRAIN_SCRIPT="projects/tessera_downscaling/scripts/train.py"
EVAL_SCRIPT="projects/tessera_downscaling/scripts/evaluate.py"

# Short tag prefixed to slurm job names so the four sibling folders are
# distinguishable in squeue (yaml entry names deliberately omit
# region/variant/year).
JOB_TAG="ea1BM17sh"

# Resolve YAML next to this script — robust to directory renames.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPERIMENTS_YAML="${SCRIPT_DIR}/experiments.yaml"

# The region this script trains+tests on.
REGION="east_asia"
NORMALISATION_POLICY="per_region"

# ---- Slurm settings ----
TIME="24:00:00"
PARTITION=""  # leave empty for default, or set e.g. "workq"

echo "Pre-syncing environment..."
cd ${REPO_ROOT}
uv sync --group core

# ---- Hyperparameters (identical to the original region sweeps) ----
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
        tv_arg = "--target-variables " + " ".join(e["target_variables"])
        extra = os.path.expandvars(e["extra_args"])
        print(f"{e['name']}|{tv_arg}|{extra}")
PYEOF
)

if [ "${#EXPERIMENTS[@]}" -eq 0 ]; then
    echo "ERROR: No experiments loaded from ${EXPERIMENTS_YAML}" >&2
    exit 1
fi

# ---- Preflight: dataset sanity ----
if [ ! -f "${DATASET_DIR}/metadata.json" ]; then
    echo "ERROR: ${DATASET_DIR}/metadata.json does not exist." >&2
    exit 1
fi

LAYOUT_VERSION=$(
    python3 -c "import json; print(json.load(open('${DATASET_DIR}/metadata.json')).get('layout_version', ''))"
)
if [ "${LAYOUT_VERSION}" != "multi_region_snapshot_v1" ]; then
    echo "ERROR: layout_version='${LAYOUT_VERSION}', expected 'multi_region_snapshot_v1'." >&2
    exit 1
fi

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
echo "Tessera:       ${TESSERA_VARIANT}, embeddings ${EMBED_YEAR}"
echo "Latents dir:   ${VAE_LATENTS_DIR}"
echo "Normalisation: ${NORMALISATION_POLICY}"
echo "Output:        ${OUTPUT_ROOT}"
echo "YAML:          ${EXPERIMENTS_YAML}"
echo "Seeds:         ${SEEDS[*]}"
echo "Configs:       ${#EXPERIMENTS[@]}"
echo "Total jobs:    $(( ${#EXPERIMENTS[@]} * ${#SEEDS[@]} ))"
echo ""

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
JOB_COUNT=0
SKIP_COUNT=0
for experiment in "${EXPERIMENTS[@]}"; do
    IFS='|' read -r name target_args extra_args <<< "${experiment}"

    # Skip entries whose latents file doesn't exist yet (e.g. a VAE
    # variant still training when this sweep was first submitted).
    # Re-running this script later picks them up automatically.
    latents_path=$(sed -n 's/.*--vae-latents-path \([^ ]*\).*/\1/p' <<< "${extra_args}")
    if [ -n "${latents_path}" ] && [ ! -f "${latents_path}" ]; then
        echo "SKIP: ${name} (latents not yet available: $(basename "${latents_path}"))"
        SKIP_COUNT=$((SKIP_COUNT + 1))
        continue
    fi

    for seed in "${SEEDS[@]}"; do
        run_dir="${OUTPUT_ROOT}/${name}_seed${seed}"
        job_name="${JOB_TAG}_${name}_s${seed}"

        # Skip if already completed.
        if [ -f "${run_dir}/test_summary.json" ]; then
            echo "SKIP: ${job_name} (already complete)"
            continue
        fi

        TRAIN_CMD="${REPO_ROOT}/.venv/bin/python ${TRAIN_SCRIPT} ${COMMON_ARGS} ${target_args} ${extra_args} --seed ${seed} --output-dir ${run_dir}"
        EVAL_CMD="${REPO_ROOT}/.venv/bin/python ${EVAL_SCRIPT} --checkpoint ${run_dir}/best_model.pt --batch-size ${BATCH_SIZE} --num-workers ${NUM_WORKERS}"

        SBATCH_CMD="sbatch --job-name=${job_name} --output=${REPO_ROOT}/logs/${job_name}_%j.out --error=${REPO_ROOT}/logs/${job_name}_%j.err --gpus=1 --time=${TIME} ${PARTITION:+--partition=${PARTITION}} --wrap=\"cd ${REPO_ROOT} && ${TRAIN_CMD} && ${EVAL_CMD}\""

        if [ "${DRY_RUN}" = "1" ]; then
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

echo ""
echo "============================================"
if [ "${DRY_RUN}" = "1" ]; then
    echo "DRY RUN: ${JOB_COUNT} jobs would be submitted ($(basename "${OUTPUT_ROOT}"))"
else
    echo "Submitted ${JOB_COUNT} sbatch jobs ($(basename "${OUTPUT_ROOT}"))"
fi
if [ "${SKIP_COUNT}" -gt 0 ]; then
    echo "Skipped ${SKIP_COUNT} configs with missing latents — re-run once the VAE eval lands"
fi
echo "============================================"
echo ""
echo "Monitor with:  squeue --me"
echo "Results in:    ${OUTPUT_ROOT}/"
