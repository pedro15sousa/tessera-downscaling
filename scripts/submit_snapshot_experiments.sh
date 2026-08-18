#!/bin/bash
# Submit snapshot-cadence (timestamp) experiments × seeds as parallel Slurm jobs.
#
# Companion to submit_parallel.sh, kept separate because:
#   * Target-variable names differ (t2m/wind vs tmax/wind_mean). The
#     SnapshotDownscalingDataset refuses daily-cadence names, so the
#     configs can't be unified without a bigger rewrite.
#   * Output directory is different (training_runs_snapshot vs
#     training_runs), so daily and snapshot results stay separated
#     for clean analysis.
#
# Usage (from repo root):
#   bash projects/tessera_downscaling/scripts/submit_snapshot.sh
#
# Dry run (prints commands without submitting):
#   DRY_RUN=1 bash projects/tessera_downscaling/scripts/submit_snapshot.sh
set -euo pipefail

# ---- Paths ----
REPO_ROOT="/projects/u6do/pmms2/end-to-end-forecasting"
BASE_DIR="${REPO_ROOT}/projects/tessera_downscaling/.tmp_output"

# Snapshot-cadence dataset produced by preprocess_timestamp.py (Europe only
# in this release; multi-region snapshot will come with step 2).
DATASET_DIR_SNAPSHOT="${BASE_DIR}/dataset_timestamp"

# TESSERA patches + station CSV are shared across daily and snapshot — the
# snapshot dataset class filters stations by the same valid-patch rule.
TESSERA_PATH="${BASE_DIR}/processed/tessera/patch16_embeddings_2024.npy"
TESSERA_CSV="${BASE_DIR}/processed/tessera/station_list_filtered.csv"

# Pre-computed VAE latents. Two variants are available:
#   VAE_LATENTS_PATH      — 16-d (original). Used for lat16 configs.
#   VAE_LATENTS_PATH_LAT64 — 64-d (larger, l1-regularised). Used for the
#                            lat64+proj configs (proj8, proj16 etc.).
# Both share the same station CSV — they're computed from the same
# TESSERA rows in the same order, just with different latent sizes.
VAE_LATENTS_PATH="${BASE_DIR}/processed/station_latents_lat16_grad0.5.npy"
VAE_LATENTS_PATH_LAT64="${BASE_DIR}/processed/station_latents_lat64_l1.npy"
VAE_LATENTS_CSV="${BASE_DIR}/processed/tessera_global/station_list_filtered.csv"

OUTPUT_ROOT="${BASE_DIR}/training_runs_snapshot"
TRAIN_SCRIPT="projects/tessera_downscaling/scripts/train.py"
EVAL_SCRIPT="projects/tessera_downscaling/scripts/evaluate.py"

# ---- Slurm settings ----
TIME="24:00:00"
PARTITION=""  # leave empty for default, or set e.g. "workq"

echo "Pre-syncing environment..."
cd ${REPO_ROOT}
uv sync --group core

# ---- Experiment matrix ----
SEEDS=(42 123 456)

# Shared hyperparameters. Kept identical to submit_parallel.sh so
# snapshot-vs-daily comparisons aren't confounded by training knobs.
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

# Experiment definitions: "name|extra_args"
#
# Matrix over (target ∈ {t2m, wind}) × (projection ∈ {none=lat16,
# proj8, proj16, proj16-MLP, proj32}) × (injection ∈ {concat, FiLM}) ×
# (elev ∈ {with, no}). We don't run every cell — we run enough to
# isolate each axis while re-using the daily winners as anchors.
EXPERIMENTS=(
    # =================================================================
    # Baselines (no TESSERA, with-elev + with-static + wd)
    # =================================================================

    # 1. Wind snapshot baseline: bilinear + elev + static + wd, no
    # TESSERA. Reference point for every other wind snapshot experiment.
    "wind_snap_bilinear_baseline_wd|--interpolation bilinear --weight-decay 1e-4 --target-variables wind"

    # 2. t2m snapshot baseline: bilinear + elev + static + wd.
    "t2m_snap_bilinear_baseline_wd|--interpolation bilinear --weight-decay 1e-4 --target-variables t2m"

    # =================================================================
    # lat16 + concat  —  both elev settings, both targets
    # =================================================================
    # Simplest VAE integration: 16-d latent fed directly into the MLP
    # via concatenation, no projection head, no FiLM machinery. Tests
    # whether the extra architecture is earning its keep.

    # 3. Wind, lat16 + concat, WITH elev, no static, wd, drop=0.
    # Current flat-EU wind winner.
    "wind_snap_vae_lat16_concat_with_elev_no_static_wd_drop0|--interpolation bilinear --tessera-injection concat --vae-latents-path ${VAE_LATENTS_PATH} --vae-latents-station-csv ${VAE_LATENTS_CSV} --vae-latents-drop-prob 0.0 --no-static-fields --weight-decay 1e-4 --target-variables wind"

    # 4. Wind, lat16 + concat, NO elev, no static, wd, drop=0.
    # Counterpart to #3. Tests whether the VAE-lat16 latent already
    # carries enough elev-correlated signal (via the VAE's aux
    # elev-reconstruction loss) to make explicit elevation redundant
    # — the way it appears to be for the lat64+proj16+FiLM variants.
    "wind_snap_vae_lat16_concat_no_elev_no_static_wd_drop0|--interpolation bilinear --tessera-injection concat --vae-latents-path ${VAE_LATENTS_PATH} --vae-latents-station-csv ${VAE_LATENTS_CSV} --vae-latents-drop-prob 0.0 --no-elevation --no-static-fields --weight-decay 1e-4 --target-variables wind"

    # 5. t2m, lat16 + concat, WITH elev, no static, wd, drop=0.
    # Current flat-EU t2m winner.
    "t2m_snap_vae_lat16_concat_with_elev_no_static_wd_drop0|--interpolation bilinear --tessera-injection concat --vae-latents-path ${VAE_LATENTS_PATH} --vae-latents-station-csv ${VAE_LATENTS_CSV} --vae-latents-drop-prob 0.0 --no-static-fields --weight-decay 1e-4 --target-variables t2m"

    # 6. t2m, lat16 + concat, NO elev, no static, wd, drop=0.
    # Counterpart to #5. Same hypothesis as #4 for t2m.
    "t2m_snap_vae_lat16_concat_no_elev_no_static_wd_drop0|--interpolation bilinear --tessera-injection concat --vae-latents-path ${VAE_LATENTS_PATH} --vae-latents-station-csv ${VAE_LATENTS_CSV} --vae-latents-drop-prob 0.0 --no-elevation --no-static-fields --weight-decay 1e-4 --target-variables t2m"

    # =================================================================
    # lat64 + proj8 + FiLM  —  both elev settings, both targets
    # =================================================================
    # The daily wind frontier was tied between proj8 and proj16; the
    # daily tmax winner was lat64+proj8+FiLM with elev. Worth both
    # elev settings on both targets at snapshot cadence.

    # 7. Wind, lat64+proj8+FiLM, WITH elev.
    "wind_snap_vae_lat64_proj8_film_with_elev_no_static_wd_drop0|--interpolation bilinear --tessera-injection film --vae-latents-path ${VAE_LATENTS_PATH_LAT64} --vae-latents-station-csv ${VAE_LATENTS_CSV} --vae-latents-drop-prob 0.0 --vae-latents-proj-dim 8 --no-static-fields --weight-decay 1e-4 --target-variables wind"

    # 8. Wind, lat64+proj8+FiLM, NO elev. Daily proj8 was tied with
    # proj16 for wind at MAE 0.998 ± 0.009.
    "wind_snap_vae_lat64_proj8_film_no_elev_no_static_wd_drop0|--interpolation bilinear --tessera-injection film --vae-latents-path ${VAE_LATENTS_PATH_LAT64} --vae-latents-station-csv ${VAE_LATENTS_CSV} --vae-latents-drop-prob 0.0 --vae-latents-proj-dim 8 --no-elevation --no-static-fields --weight-decay 1e-4 --target-variables wind"

    # 9. t2m, lat64+proj8+FiLM, WITH elev. Daily tmax winner shape —
    # MAE 1.042 ± 0.004, RMSE 1.440 ± 0.001.
    "t2m_snap_vae_lat64_proj8_film_with_elev_no_static_wd_drop0|--interpolation bilinear --tessera-injection film --vae-latents-path ${VAE_LATENTS_PATH_LAT64} --vae-latents-station-csv ${VAE_LATENTS_CSV} --vae-latents-drop-prob 0.0 --vae-latents-proj-dim 8 --no-static-fields --weight-decay 1e-4 --target-variables t2m"

    # 10. t2m, lat64+proj8+FiLM, NO elev. Counterpart to #9.
    "t2m_snap_vae_lat64_proj8_film_no_elev_no_static_wd_drop0|--interpolation bilinear --tessera-injection film --vae-latents-path ${VAE_LATENTS_PATH_LAT64} --vae-latents-station-csv ${VAE_LATENTS_CSV} --vae-latents-drop-prob 0.0 --vae-latents-proj-dim 8 --no-elevation --no-static-fields --weight-decay 1e-4 --target-variables t2m"

    # =================================================================
    # lat64 + proj16 + FiLM  —  both elev settings, both targets
    # =================================================================
    # The daily wind frontier. Keep both elev settings on both targets
    # so we have a clean FiLM vs concat × with-elev vs no-elev 2×2 for
    # proj16.

    # 11. Wind, lat64+proj16+FiLM, WITH elev.
    "wind_snap_vae_lat64_proj16_film_with_elev_no_static_wd_drop0|--interpolation bilinear --tessera-injection film --vae-latents-path ${VAE_LATENTS_PATH_LAT64} --vae-latents-station-csv ${VAE_LATENTS_CSV} --vae-latents-drop-prob 0.0 --vae-latents-proj-dim 16 --no-static-fields --weight-decay 1e-4 --target-variables wind"

    # 12. Wind, lat64+proj16+FiLM, NO elev. Daily wind overall frontier.
    "wind_snap_vae_lat64_proj16_film_no_elev_no_static_wd_drop0|--interpolation bilinear --tessera-injection film --vae-latents-path ${VAE_LATENTS_PATH_LAT64} --vae-latents-station-csv ${VAE_LATENTS_CSV} --vae-latents-drop-prob 0.0 --vae-latents-proj-dim 16 --no-elevation --no-static-fields --weight-decay 1e-4 --target-variables wind"

    # 13. t2m, lat64+proj16+FiLM, WITH elev.
    "t2m_snap_vae_lat64_proj16_film_with_elev_no_static_wd_drop0|--interpolation bilinear --tessera-injection film --vae-latents-path ${VAE_LATENTS_PATH_LAT64} --vae-latents-station-csv ${VAE_LATENTS_CSV} --vae-latents-drop-prob 0.0 --vae-latents-proj-dim 16 --no-static-fields --weight-decay 1e-4 --target-variables t2m"

    # 14. t2m, lat64+proj16+FiLM, NO elev. If t2m behaves like tmax,
    # this will lose to #13; if instantaneous t2m changes the
    # elev/proj-size tradeoff, it could flip.
    "t2m_snap_vae_lat64_proj16_film_no_elev_no_static_wd_drop0|--interpolation bilinear --tessera-injection film --vae-latents-path ${VAE_LATENTS_PATH_LAT64} --vae-latents-station-csv ${VAE_LATENTS_CSV} --vae-latents-drop-prob 0.0 --vae-latents-proj-dim 16 --no-elevation --no-static-fields --weight-decay 1e-4 --target-variables t2m"

    # =================================================================
    # lat64 + proj16 + concat (linear)  —  both elev settings, both targets
    # =================================================================
    # Tests whether the FiLM machinery earns its keep when the VAE
    # latent is already projected down to 16-d: concat treats those 16
    # dims as "just more MLP inputs", while FiLM uses them to generate
    # per-station scale/shift parameters for the decoder's hidden
    # layers. If concat closes most of the gap with FiLM at proj=16,
    # FiLM's expressiveness benefit is small and concat wins on
    # simplicity.

    # 15. Wind, lat64+proj16+concat, WITH elev.
    "wind_snap_vae_lat64_proj16_concat_with_elev_no_static_wd|--interpolation bilinear --tessera-injection concat --vae-latents-path ${VAE_LATENTS_PATH_LAT64} --vae-latents-station-csv ${VAE_LATENTS_CSV} --vae-latents-drop-prob 0.0 --vae-latents-proj-dim 16 --no-static-fields --weight-decay 1e-4 --target-variables wind"

    # 16. Wind, lat64+proj16+concat, NO elev.
    "wind_snap_vae_lat64_proj16_concat_no_elev_no_static_wd|--interpolation bilinear --tessera-injection concat --vae-latents-path ${VAE_LATENTS_PATH_LAT64} --vae-latents-station-csv ${VAE_LATENTS_CSV} --vae-latents-drop-prob 0.0 --vae-latents-proj-dim 16 --no-elevation --no-static-fields --weight-decay 1e-4 --target-variables wind"

    # 17. t2m, lat64+proj16+concat, WITH elev.
    "t2m_snap_vae_lat64_proj16_concat_with_elev_no_static_wd|--interpolation bilinear --tessera-injection concat --vae-latents-path ${VAE_LATENTS_PATH_LAT64} --vae-latents-station-csv ${VAE_LATENTS_CSV} --vae-latents-drop-prob 0.0 --vae-latents-proj-dim 16 --no-static-fields --weight-decay 1e-4 --target-variables t2m"

    # 18. t2m, lat64+proj16+concat, NO elev.
    "t2m_snap_vae_lat64_proj16_concat_no_elev_no_static_wd|--interpolation bilinear --tessera-injection concat --vae-latents-path ${VAE_LATENTS_PATH_LAT64} --vae-latents-station-csv ${VAE_LATENTS_CSV} --vae-latents-drop-prob 0.0 --vae-latents-proj-dim 16 --no-elevation --no-static-fields --weight-decay 1e-4 --target-variables t2m"

    # =================================================================
    # lat64 + proj16 MLP + concat  —  both elev settings, both targets
    # =================================================================
    # Non-linear projection head (Linear(64,32) → ReLU → Linear(32,16))
    # instead of pure Linear. Tests whether a small MLP head extracts
    # more task-useful info from the VAE latent than a pure linear
    # projection does.

    # 19. Wind, proj16-MLP+concat, WITH elev.
    "wind_snap_vae_lat64_proj16_mlp_concat_with_elev_no_static_wd|--interpolation bilinear --tessera-injection concat --vae-latents-path ${VAE_LATENTS_PATH_LAT64} --vae-latents-station-csv ${VAE_LATENTS_CSV} --vae-latents-drop-prob 0.0 --vae-latents-proj-dim 16 --vae-latents-proj-mlp --no-static-fields --weight-decay 1e-4 --target-variables wind"

    # 20. Wind, proj16-MLP+concat, NO elev.
    "wind_snap_vae_lat64_proj16_mlp_concat_no_elev_no_static_wd|--interpolation bilinear --tessera-injection concat --vae-latents-path ${VAE_LATENTS_PATH_LAT64} --vae-latents-station-csv ${VAE_LATENTS_CSV} --vae-latents-drop-prob 0.0 --vae-latents-proj-dim 16 --vae-latents-proj-mlp --no-elevation --no-static-fields --weight-decay 1e-4 --target-variables wind"

    # 21. t2m, proj16-MLP+concat, WITH elev.
    "t2m_snap_vae_lat64_proj16_mlp_concat_with_elev_no_static_wd|--interpolation bilinear --tessera-injection concat --vae-latents-path ${VAE_LATENTS_PATH_LAT64} --vae-latents-station-csv ${VAE_LATENTS_CSV} --vae-latents-drop-prob 0.0 --vae-latents-proj-dim 16 --vae-latents-proj-mlp --no-static-fields --weight-decay 1e-4 --target-variables t2m"

    # 22. t2m, proj16-MLP+concat, NO elev.
    "t2m_snap_vae_lat64_proj16_mlp_concat_no_elev_no_static_wd|--interpolation bilinear --tessera-injection concat --vae-latents-path ${VAE_LATENTS_PATH_LAT64} --vae-latents-station-csv ${VAE_LATENTS_CSV} --vae-latents-drop-prob 0.0 --vae-latents-proj-dim 16 --vae-latents-proj-mlp --no-elevation --no-static-fields --weight-decay 1e-4 --target-variables t2m"

    # =================================================================
    # lat64 + proj32 + concat (linear)  —  both elev settings, both targets
    # =================================================================
    # Wider linear projection instead of non-linearity. Tests whether
    # the bottleneck is capacity or expressivity: if proj32+linear
    # matches or beats the MLP variant with proj16, it was capacity.

    # 23. Wind, proj32+concat, WITH elev.
    "wind_snap_vae_lat64_proj32_concat_with_elev_no_static_wd|--interpolation bilinear --tessera-injection concat --vae-latents-path ${VAE_LATENTS_PATH_LAT64} --vae-latents-station-csv ${VAE_LATENTS_CSV} --vae-latents-drop-prob 0.0 --vae-latents-proj-dim 32 --no-static-fields --weight-decay 1e-4 --target-variables wind"

    # 24. Wind, proj32+concat, NO elev.
    "wind_snap_vae_lat64_proj32_concat_no_elev_no_static_wd|--interpolation bilinear --tessera-injection concat --vae-latents-path ${VAE_LATENTS_PATH_LAT64} --vae-latents-station-csv ${VAE_LATENTS_CSV} --vae-latents-drop-prob 0.0 --vae-latents-proj-dim 32 --no-elevation --no-static-fields --weight-decay 1e-4 --target-variables wind"

    # 25. t2m, proj32+concat, WITH elev.
    "t2m_snap_vae_lat64_proj32_concat_with_elev_no_static_wd|--interpolation bilinear --tessera-injection concat --vae-latents-path ${VAE_LATENTS_PATH_LAT64} --vae-latents-station-csv ${VAE_LATENTS_CSV} --vae-latents-drop-prob 0.0 --vae-latents-proj-dim 32 --no-static-fields --weight-decay 1e-4 --target-variables t2m"

    # 26. t2m, proj32+concat, NO elev.
    "t2m_snap_vae_lat64_proj32_concat_no_elev_no_static_wd|--interpolation bilinear --tessera-injection concat --vae-latents-path ${VAE_LATENTS_PATH_LAT64} --vae-latents-station-csv ${VAE_LATENTS_CSV} --vae-latents-drop-prob 0.0 --vae-latents-proj-dim 32 --no-elevation --no-static-fields --weight-decay 1e-4 --target-variables t2m"

    # =================================================================
    # Multi-task — predict t2m AND wind jointly
    # =================================================================
    # The multi-task win on daily was narrow. At snapshot cadence
    # there's more data per task, so the training signal dilution that
    # sometimes hurts multi-task may be less of a concern — worth
    # re-checking.

    # 27. Multi-task snapshot baseline (no TESSERA).
    "multitask_snap_bilinear_baseline_wd|--interpolation bilinear --weight-decay 1e-4 --target-variables t2m wind"

    # 28. Multi-task, lat64+proj16+FiLM, NO elev. Matches daily winner.
    # Using proj16 (not proj8) because with two targets the projection
    # head has to compress lat64 → k dims useful for both, and proj16
    # gives it more room.
    "multitask_snap_vae_lat64_proj16_film_no_elev_no_static_wd_drop0|--interpolation bilinear --tessera-injection film --vae-latents-path ${VAE_LATENTS_PATH_LAT64} --vae-latents-station-csv ${VAE_LATENTS_CSV} --vae-latents-drop-prob 0.0 --vae-latents-proj-dim 16 --no-elevation --no-static-fields --weight-decay 1e-4 --target-variables t2m wind"
)

# ---- Create log directory ----
mkdir -p "${REPO_ROOT}/logs"
mkdir -p "${OUTPUT_ROOT}"

# ---- Submit jobs ----
echo "============================================"
echo "Submitting SNAPSHOT experiment jobs"
echo "============================================"
echo ""
echo "Dataset:    ${DATASET_DIR_SNAPSHOT}"
echo "Output:     ${OUTPUT_ROOT}"
echo "Batch size: ${BATCH_SIZE}"
echo "Epochs:     ${EPOCHS}"
echo "Patience:   ${PATIENCE}"
echo "LR warmup:  ${LR_WARMUP_PCT}"
echo "Seeds:      ${SEEDS[*]}"
echo "Configs:    ${#EXPERIMENTS[@]}"
echo "Total jobs: $(( ${#EXPERIMENTS[@]} * ${#SEEDS[@]} ))"
echo ""

# --- Preflight: confirm the snapshot dataset exists before submitting ---
# Nothing downstream checks this, so better to fail fast than to queue 30
# jobs that'll all error immediately at dataset construction.
if [ ! -f "${DATASET_DIR_SNAPSHOT}/metadata.json" ]; then
    echo "ERROR: ${DATASET_DIR_SNAPSHOT}/metadata.json does not exist." >&2
    echo "Run preprocess_timestamp.py before submitting snapshot jobs." >&2
    exit 1
fi

# Sanity-check the layout version so a stale daily dataset at this path
# doesn't silently get picked up.
LAYOUT_VERSION=$(
    python3 -c "import json; print(json.load(open('${DATASET_DIR_SNAPSHOT}/metadata.json')).get('layout_version', ''))"
)
if [ "${LAYOUT_VERSION}" != "snapshot_v1" ]; then
    echo "ERROR: ${DATASET_DIR_SNAPSHOT}/metadata.json has layout_version='${LAYOUT_VERSION}', expected 'snapshot_v1'." >&2
    echo "Is that actually a snapshot preprocessing output?" >&2
    exit 1
fi

# --lr-warmup-pct applied here once rather than per-experiment so every
# run gets the same LR schedule. Individual experiments can still
# override it by passing --lr-warmup-pct in their extra_args (argparse
# last-flag-wins).
COMMON_ARGS="--dataset-dir ${DATASET_DIR_SNAPSHOT} --tessera-path ${TESSERA_PATH} --tessera-station-csv ${TESSERA_CSV} --batch-size ${BATCH_SIZE} --epochs ${EPOCHS} --patience ${PATIENCE} --lr ${LR} --lr-warmup-pct ${LR_WARMUP_PCT} --cnn-hidden ${CNN_HIDDEN} --cnn-layers ${CNN_LAYERS} --mlp-hidden ${MLP_HIDDEN} --mlp-n-hidden ${MLP_N_HIDDEN} --num-workers ${NUM_WORKERS}"

JOB_COUNT=0
for experiment in "${EXPERIMENTS[@]}"; do
    # Experiment entry format: "name|train_extra_args" or
    # "name|train_extra_args|eval_extra_args". Snapshot is single-region
    # in this release, so no --test-regions ever needs passing; the
    # third field is kept for future-compatibility with the daily script.
    IFS='|' read -r name extra_args eval_extra <<< "${experiment}"

    for seed in "${SEEDS[@]}"; do
        run_dir="${OUTPUT_ROOT}/${name}_seed${seed}"
        job_name="${name}_s${seed}"

        # Skip if already completed
        if [ -f "${run_dir}/test_summary.json" ]; then
            echo "SKIP: ${job_name} (already complete)"
            continue
        fi

        SBATCH_CMD="sbatch --job-name=${job_name} --output=${REPO_ROOT}/logs/${job_name}_%j.out --error=${REPO_ROOT}/logs/${job_name}_%j.err --gpus=1 --time=${TIME} ${PARTITION:+--partition=${PARTITION}} --wrap=\"cd ${REPO_ROOT} && ${REPO_ROOT}/.venv/bin/python ${TRAIN_SCRIPT} ${COMMON_ARGS} ${extra_args} --seed ${seed} --output-dir ${run_dir} && ${REPO_ROOT}/.venv/bin/python ${EVAL_SCRIPT} --checkpoint ${run_dir}/best_model.pt --batch-size ${BATCH_SIZE} --num-workers ${NUM_WORKERS} ${eval_extra}\""

        if [ "${DRY_RUN:-0}" = "1" ]; then
            echo "DRY RUN: ${job_name}"
            echo "  ${SBATCH_CMD}"
            echo ""
        else
            # Actually submit
            JOB_ID=$(eval "${SBATCH_CMD}")
            echo "SUBMITTED: ${job_name} -> ${JOB_ID}"
        fi

        JOB_COUNT=$((JOB_COUNT + 1))
    done
done

echo ""
echo "============================================"
echo "Submitted ${JOB_COUNT} snapshot jobs"
echo "============================================"
echo ""
echo "Monitor with:  squeue --me"
echo "Cancel all:    scancel --me"
echo "Results in:    ${OUTPUT_ROOT}/"