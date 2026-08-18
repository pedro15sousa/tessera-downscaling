#!/bin/bash
# JEPA latents A/B: mirror the 4 flat-EU VAE-lat16 experiments using
# JEPA-trained latents (station_latents_jepa_lat16_lam1.0_mask0.5.npy)
# instead of VAE latents. Same architecture, same dataset, same hyper-
# parameters — only the latent representation differs.
#
# The goal: direct A/B between VAE and JEPA as frozen pre-trained
# representations for the downstream ConvCNP task. Both produce 16-d
# station-level latents from TESSERA patches, so they're architecturally
# interchangeable.
#
# Experiment matrix: 4 configs × 3 seeds = 12 jobs.
#   - t2m × {concat, film} × with_elev
#   - wind × {concat, film} × with_elev
#
# All experiments named with `jepa_` prefix so they don't collide with
# the existing VAE results in training_runs_snapshot/. 'concat' variants
# are the direct A/B against the existing VAE-lat16 winners
# (t2m_snap_vae_lat16_concat ... MAE 1.226; wind ... MAE 1.264).
#
# Usage (from repo root):
#   bash projects/tessera_downscaling/scripts/submit_snapshot_jepa.sh
#
# Dry run:
#   DRY_RUN=1 bash projects/tessera_downscaling/scripts/submit_snapshot_jepa.sh
set -euo pipefail

# ---- Paths (identical to submit_snapshot.sh for flat-EU) ----
REPO_ROOT="/projects/u6do/pmms2/end-to-end-forecasting"
BASE_DIR="${REPO_ROOT}/projects/tessera_downscaling/.tmp_output"

DATASET_DIR_SNAPSHOT="${BASE_DIR}/dataset_timestamp"

TESSERA_PATH="${BASE_DIR}/processed/tessera/patch16_embeddings_2024.npy"
TESSERA_CSV="${BASE_DIR}/processed/tessera/station_list_filtered.csv"

# JEPA latents — trained in a separate repo via JEPA objective, same
# 38,870 TESSERA-station rows as the VAE latents. Shape (38870, 16),
# row-aligned with VAE_LATENTS_CSV.
JEPA_LATENTS_PATH="${BASE_DIR}/processed/station_latents_jepa_lat16_lam1.0_mask0.5.npy"
VAE_LATENTS_CSV="${BASE_DIR}/processed/tessera_global/station_list_filtered.csv"

OUTPUT_ROOT="${BASE_DIR}/training_runs_snapshot"
TRAIN_SCRIPT="projects/tessera_downscaling/scripts/train.py"
EVAL_SCRIPT="projects/tessera_downscaling/scripts/evaluate.py"

# ---- Slurm settings ----
TIME="24:00:00"
PARTITION=""

echo "Pre-syncing environment..."
cd ${REPO_ROOT}
uv sync --group core

# ---- Experiment matrix ----
SEEDS=(42 123 456)

BATCH_SIZE=1
EPOCHS=100
PATIENCE=10
LR="2.5e-5"
CNN_HIDDEN=128
CNN_LAYERS=7
MLP_HIDDEN=128
MLP_N_HIDDEN=3
NUM_WORKERS=4

# LR warmup on to match the recent experiment pipeline. Helps
# convergence for models with new injection heads. Not applied to
# the original VAE flat-EU baselines, but since we're starting
# fresh here, apply it from the start.
COMMON_ARGS="--dataset-dir ${DATASET_DIR_SNAPSHOT} --tessera-path ${TESSERA_PATH} --tessera-station-csv ${TESSERA_CSV} --batch-size ${BATCH_SIZE} --epochs ${EPOCHS} --patience ${PATIENCE} --lr ${LR} --cnn-hidden ${CNN_HIDDEN} --cnn-layers ${CNN_LAYERS} --mlp-hidden ${MLP_HIDDEN} --mlp-n-hidden ${MLP_N_HIDDEN} --num-workers ${NUM_WORKERS} --lr-warmup-pct 0.05"

# ---- Experiments ----
# Format: "name|extra_args"
#
# 4 experiments × 3 seeds = 12 jobs.
EXPERIMENTS=(
    # ==================================================================
    # t2m × JEPA latents × {concat, film} × with elev, no static
    # ==================================================================
    # Direct A/B against VAE-lat16 winner (t2m_snap_vae_lat16_concat_no_static_wd_drop0, MAE 1.226)
    "t2m_snap_jepa_lat16_concat_with_elev_no_static_wd|--interpolation bilinear --tessera-injection concat --vae-latents-path ${JEPA_LATENTS_PATH} --vae-latents-station-csv ${VAE_LATENTS_CSV} --vae-latents-drop-prob 0.0 --no-static-fields --weight-decay 1e-4 --target-variables t2m"
    # FiLM variant — not in the VAE lat16 baseline set, but worth testing
    "t2m_snap_jepa_lat16_film_with_elev_no_static_wd|--interpolation bilinear --tessera-injection film --vae-latents-path ${JEPA_LATENTS_PATH} --vae-latents-station-csv ${VAE_LATENTS_CSV} --vae-latents-drop-prob 0.0 --no-static-fields --weight-decay 1e-4 --target-variables t2m"

    # ==================================================================
    # wind × JEPA latents × {concat, film} × with elev, no static
    # ==================================================================
    # Direct A/B against VAE-lat16 winner (wind_snap_vae_lat16_concat_no_static_wd_drop0, MAE 1.264)
    "wind_snap_jepa_lat16_concat_with_elev_no_static_wd|--interpolation bilinear --tessera-injection concat --vae-latents-path ${JEPA_LATENTS_PATH} --vae-latents-station-csv ${VAE_LATENTS_CSV} --vae-latents-drop-prob 0.0 --no-static-fields --weight-decay 1e-4 --target-variables wind"
    # FiLM variant
    "wind_snap_jepa_lat16_film_with_elev_no_static_wd|--interpolation bilinear --tessera-injection film --vae-latents-path ${JEPA_LATENTS_PATH} --vae-latents-station-csv ${VAE_LATENTS_CSV} --vae-latents-drop-prob 0.0 --no-static-fields --weight-decay 1e-4 --target-variables wind"
)

echo "============================================"
echo "Submitting JEPA flat-EU A/B jobs"
echo "============================================"
echo ""
echo "Dataset:    ${DATASET_DIR_SNAPSHOT}"
echo "Latents:    ${JEPA_LATENTS_PATH}"
echo "Output:     ${OUTPUT_ROOT}"
echo "Batch size: ${BATCH_SIZE}"
echo "Epochs:     ${EPOCHS}"
echo "Patience:   ${PATIENCE}"
echo "Seeds:      ${SEEDS[*]}"
echo "Configs:    ${#EXPERIMENTS[@]}"
echo "Total jobs: $(( ${#EXPERIMENTS[@]} * ${#SEEDS[@]} ))"
echo ""

# --- Preflight: confirm the JEPA latents file exists ---
if [ ! -f "${JEPA_LATENTS_PATH}" ]; then
    echo "ERROR: ${JEPA_LATENTS_PATH} does not exist." >&2
    echo "Copy the JEPA latents file into place before submitting." >&2
    exit 1
fi

# Confirm the latents file is 16-d (row count can vary — dataset filters).
LATENT_DIM=$(
    python3 -c "import numpy as np; print(np.load('${JEPA_LATENTS_PATH}').shape[1])"
)
if [ "${LATENT_DIM}" != "16" ]; then
    echo "WARNING: ${JEPA_LATENTS_PATH} has latent_dim=${LATENT_DIM}, expected 16." >&2
    echo "If this is intentional (e.g. different-sized JEPA variant)," >&2
    echo "you may need to add --vae-latents-proj-dim to the experiments." >&2
fi

if [ ! -f "${DATASET_DIR_SNAPSHOT}/metadata.json" ]; then
    echo "ERROR: ${DATASET_DIR_SNAPSHOT}/metadata.json does not exist." >&2
    echo "Run preprocess_timestamp.py before submitting these jobs." >&2
    exit 1
fi

JOB_COUNT=0
for experiment in "${EXPERIMENTS[@]}"; do
    IFS='|' read -r name extra_args <<< "${experiment}"

    for seed in "${SEEDS[@]}"; do
        run_dir="${OUTPUT_ROOT}/${name}_seed${seed}"
        job_name="${name}_s${seed}"

        if [ -f "${run_dir}/test_summary.json" ]; then
            echo "SKIP: ${job_name} (already complete)"
            continue
        fi

        mkdir -p "${run_dir}"

        TRAIN_CMD="${REPO_ROOT}/.venv/bin/python ${TRAIN_SCRIPT} ${COMMON_ARGS} ${extra_args} --seed ${seed} --output-dir ${run_dir}"
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
    done
done

echo ""
echo "============================================"
echo "Submitted ${JOB_COUNT} JEPA A/B jobs"
echo "============================================"
echo ""
echo "Monitor with:  squeue --me"
echo "Cancel all:    scancel --me"
echo "Results in:    ${OUTPUT_ROOT}/ (look for *_jepa_*)"