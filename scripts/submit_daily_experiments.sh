#!/bin/bash
# Submit all experiment configurations × seeds as independent parallel Slurm jobs.
#
# Each job requests 1 GPU and runs a single training run. All jobs can
# run simultaneously on different nodes.
#
# Usage (from repo root):
#   bash projects/tessera_downscaling/scripts/submit_parallel.sh
#
# Dry run (prints commands without submitting):
#   DRY_RUN=1 bash projects/tessera_downscaling/scripts/submit_parallel.sh
set -euo pipefail

# ---- Paths ----
REPO_ROOT="/projects/u6do/pmms2/end-to-end-forecasting"
BASE_DIR="${REPO_ROOT}/projects/tessera_downscaling/.tmp_output"
DATASET_DIR="${BASE_DIR}/dataset_daily"
DATASET_DIR_GLOBAL="${BASE_DIR}/dataset_daily_global"
TESSERA_PATH="${BASE_DIR}/processed/tessera/patch16_embeddings_2024.npy"
TESSERA_CSV="${BASE_DIR}/processed/tessera/station_list_filtered.csv"
# For multi-region runs we use the global TESSERA extraction (row-aligned
# with the full 38,870 station set — the multi-region dataset class does
# station-ID-based lookup, so supplying the global patch file works for
# any subset of regions).
TESSERA_PATH_GLOBAL="${BASE_DIR}/processed/tessera_global/patch_embeddings_2024.npy"
TESSERA_CSV_GLOBAL="${BASE_DIR}/processed/tessera_global/station_list_filtered.csv"
# Pre-computed VAE latents (global, row-aligned with tessera_global CSV).
# Two variants are available: 16-d (original) and 64-d (larger, l1-regularised).
# Both share the same station CSV — they're computed from the same TESSERA
# rows in the same order, just with different latent sizes.
VAE_LATENTS_PATH="${BASE_DIR}/processed/station_latents_lat16_grad0.5.npy"
VAE_LATENTS_PATH_LAT64="${BASE_DIR}/processed/station_latents_lat64_l1.npy"
VAE_LATENTS_CSV="${BASE_DIR}/processed/tessera_global/station_list_filtered.csv"
OUTPUT_ROOT="${BASE_DIR}/training_runs"
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

# Shared hyperparameters
BATCH_SIZE=1
EPOCHS=100
PATIENCE=10
LR="2.5e-5"
CNN_HIDDEN=128
CNN_LAYERS=7
MLP_HIDDEN=128
MLP_N_HIDDEN=3
NUM_WORKERS=4

# Experiment definitions: "name|extra_args"
EXPERIMENTS=(
    # # Experiment 1: Tmax only, TESSERA CNN + embedding dropout 0.3, no elevation
    # "tmax_tessera_cnn_embdrop03_no_elev|--tessera-method cnn --tessera-output-dim 16 --tessera-drop-prob 0.3 --no-elevation --target-variables tmax"

    # # Experiment 2: Multi-task (tmax + wind), TESSERA CNN + embedding dropout 0.3, no elevation
    # "multitask_tessera_cnn_embdrop03_no_elev|--tessera-method cnn --tessera-output-dim 16 --tessera-drop-prob 0.3 --no-elevation --target-variables tmax wind_mean"

    # # Experiment 3: Multi-task baseline (no TESSERA), with elevation (reference)
    # "multitask_baseline|--target-variables tmax wind_mean"

    # # Experiment 4: Wind only, TESSERA CNN + embedding dropout 0.3, no elevation
    # "wind_tessera_cnn_embdrop03_no_elev|--tessera-method cnn --tessera-output-dim 16 --tessera-drop-prob 0.3 --no-elevation --target-variables wind_mean"

    # # Experiment 5: Wind baseline (no TESSERA), with elevation
    # "wind_baseline|--target-variables wind_mean"

    # # Experiment 6: Tmax baseline (no TESSERA), with elevation
    # "tmax_baseline|--target-variables tmax"

    # # Experiment 7: Tmax TESSERA + dropout WITH elevation
    # "tmax_tessera_cnn_embdrop03_with_elev|--tessera-method cnn --tessera-output-dim 16 --tessera-drop-prob 0.3 --target-variables tmax"

    # # Experiment 8: Wind TESSERA + dropout WITH elevation
    # "wind_tessera_cnn_embdrop03_with_elev|--tessera-method cnn --tessera-output-dim 16 --tessera-drop-prob 0.3 --target-variables wind_mean"

    # # Experiment 9: Tmax TESSERA WITHOUT dropout, no elevation
    # "tmax_tessera_cnn_no_drop_no_elev|--tessera-method cnn --tessera-output-dim 16 --no-elevation --target-variables tmax"

    # # Experiment 10: Wind TESSERA WITHOUT dropout, no elevation
    # "wind_tessera_cnn_no_drop_no_elev|--tessera-method cnn --tessera-output-dim 16 --no-elevation --target-variables wind_mean"

    # # Experiment: Tmax TESSERA no dropout, with elevation
    # "tmax_tessera_cnn_no_drop_with_elev|--tessera-method cnn --tessera-output-dim 16 --target-variables tmax"

    # # Experiment: Wind TESSERA no dropout, with elevation
    # "wind_tessera_cnn_no_drop_with_elev|--tessera-method cnn --tessera-output-dim 16 --target-variables wind_mean"

    # # Experiment 11: Tmax bilinear + TESSERA dropout, no elevation
    # "tmax_bilinear_tessera_embdrop03_no_elev|--interpolation bilinear --tessera-method cnn --tessera-output-dim 16 --tessera-drop-prob 0.3 --no-elevation --target-variables tmax"

    # # Experiment 12: Wind bilinear + TESSERA dropout, no elevation
    # "wind_bilinear_tessera_embdrop03_no_elev|--interpolation bilinear --tessera-method cnn --tessera-output-dim 16 --tessera-drop-prob 0.3 --no-elevation --target-variables wind_mean"

    # # Experiment 13: Multi-task bilinear + TESSERA dropout, no elevation
    # "multitask_bilinear_tessera_embdrop03_no_elev|--interpolation bilinear --tessera-method cnn --tessera-output-dim 16 --tessera-drop-prob 0.3 --no-elevation --target-variables tmax wind_mean"

    # # Experiment 14: Tmax bilinear baseline (no TESSERA), with elevation
    # "tmax_bilinear_baseline|--interpolation bilinear --target-variables tmax"

    # # Experiment 15: Wind bilinear baseline (no TESSERA), with elevation
    # "wind_bilinear_baseline|--interpolation bilinear --target-variables wind_mean"

    # # # # Bilinear, no elevation, no static fields, TESSERA+dropout
    # "tmax_bilinear_tessera_embdrop03_no_elev_no_static|--interpolation bilinear --tessera-method cnn --tessera-output-dim 16 --tessera-drop-prob 0.3 --no-elevation --no-static-fields --target-variables tmax"

    # "wind_bilinear_tessera_embdrop03_no_elev_no_static|--interpolation bilinear --tessera-method cnn --tessera-output-dim 16 --tessera-drop-prob 0.3 --no-elevation --no-static-fields --target-variables wind_mean"

    # # # Also need the corresponding baseline (no TESSERA, no static, no elev) to measure the gap
    # "tmax_bilinear_no_elev_no_static|--interpolation bilinear --no-elevation --no-static-fields --target-variables tmax"

    # "wind_bilinear_no_elev_no_static|--interpolation bilinear --no-elevation --no-static-fields --target-variables wind_mean"

    # "tmax_bilinear_no_elev|--interpolation bilinear --no-elevation --target-variables tmax"
    # "wind_bilinear_no_elev|--interpolation bilinear --no-elevation --target-variables wind_mean"

    # # FiLM experiments:
    # "tmax_bilinear_film_tessera_embdrop03_no_elev|--interpolation bilinear --tessera-method cnn --tessera-output-dim 16 --tessera-drop-prob 0.3 --tessera-injection film --no-elevation --target-variables tmax"

    # "wind_bilinear_film_tessera_embdrop03_no_elev|--interpolation bilinear --tessera-method cnn --tessera-output-dim 16 --tessera-drop-prob 0.3 --tessera-injection film --no-elevation --target-variables wind_mean"

    # "tmax_bilinear_film_tessera_embdrop03_with_elev|--interpolation bilinear --tessera-method cnn --tessera-output-dim 16 --tessera-drop-prob 0.3 --tessera-injection film --target-variables tmax"

    # "wind_bilinear_film_tessera_embdrop03_with_elev|--interpolation bilinear --tessera-method cnn --tessera-output-dim 16 --tessera-drop-prob 0.3 --tessera-injection film --target-variables wind_mean"

    # "tmax_bilinear_film_tessera_embdrop03_no_elev_no_static|--interpolation bilinear --tessera-method cnn --tessera-output-dim 16 --tessera-drop-prob 0.3 --tessera-injection film --no-elevation --no-static-fields --target-variables tmax"

    # "wind_bilinear_film_tessera_embdrop03_no_elev_no_static|--interpolation bilinear --tessera-method cnn --tessera-output-dim 16 --tessera-drop-prob 0.3 --tessera-injection film --no-elevation --no-static-fields --target-variables wind_mean"

    # "tmax_bilinear_film_tessera_no_drop_no_elev|--interpolation bilinear --tessera-method cnn --tessera-output-dim 16 --tessera-injection film --no-elevation --target-variables tmax"
    # "wind_bilinear_film_tessera_no_drop_no_elev|--interpolation bilinear --tessera-method cnn --tessera-output-dim 16 --tessera-injection film --no-elevation --target-variables wind_mean"

    # # SetConv + no static + TESSERA (to compare with bilinear no-static)
    # "wind_tessera_cnn_embdrop03_no_elev_no_static|--tessera-method cnn --tessera-output-dim 16 --tessera-drop-prob 0.3 --no-elevation --no-static-fields --target-variables wind_mean"
    # "tmax_tessera_cnn_embdrop03_no_elev_no_static|--tessera-method cnn --tessera-output-dim 16 --tessera-drop-prob 0.3 --no-elevation --no-static-fields --target-variables tmax"

    # # SetConv baselines no static no elev (to measure the gap)
    # "wind_no_elev_no_static|--no-elevation --no-static-fields --target-variables wind_mean"
    # "tmax_no_elev_no_static|--no-elevation --no-static-fields --target-variables tmax"

    # # TESSERA with elev + no static (potentially best combo)
    # "wind_bilinear_tessera_embdrop03_with_elev_no_static|--interpolation bilinear --tessera-method cnn --tessera-output-dim 16 --tessera-drop-prob 0.3 --no-static-fields --target-variables wind_mean"
    # "tmax_bilinear_tessera_embdrop03_with_elev_no_static|--interpolation bilinear --tessera-method cnn --tessera-output-dim 16 --tessera-drop-prob 0.3 --no-static-fields --target-variables tmax"

    # # FiLM with elev + no static
    # "wind_bilinear_film_tessera_embdrop03_with_elev_no_static|--interpolation bilinear --tessera-method cnn --tessera-output-dim 16 --tessera-drop-prob 0.3 --tessera-injection film --no-static-fields --target-variables wind_mean"
    # "tmax_bilinear_film_tessera_embdrop03_with_elev_no_static|--interpolation bilinear --tessera-method cnn --tessera-output-dim 16 --tessera-drop-prob 0.3 --tessera-injection film --no-static-fields --target-variables tmax"
    
    # "wind_bilinear_film_tessera_no_drop_with_elev|--interpolation bilinear --tessera-method cnn --tessera-output-dim 16 --tessera-injection film --target-variables wind_mean"
    # Hypernet WITHOUT dropout
    # "wind_bilinear_hypernet_tessera_no_drop_no_elev|--interpolation bilinear --tessera-method cnn --tessera-output-dim 16 --tessera-injection hypernet --no-elevation --target-variables wind_mean"

    # "wind_bilinear_tessera_embdrop03_no_elev_no_static_wd|--interpolation bilinear --tessera-method cnn --tessera-output-dim 16 --tessera-drop-prob 0.3 --no-elevation --no-static-fields --weight-decay 1e-4 --target-variables wind_mean"

    # Best config with elevation + no static (currently 1.078)
    # "wind_bilinear_tessera_embdrop03_with_elev_no_static_wd|--interpolation bilinear --tessera-method cnn --tessera-output-dim 16 --tessera-drop-prob 0.3 --no-static-fields --weight-decay 1e-4 --target-variables wind_mean"

    # FiLM with elevation (currently 1.084)
    # "wind_bilinear_film_tessera_embdrop03_with_elev_wd|--interpolation bilinear --tessera-method cnn --tessera-output-dim 16 --tessera-drop-prob 0.3 --tessera-injection film --weight-decay 1e-4 --target-variables wind_mean"

    # Also tmax best TESSERA config (with elev, no static, currently 1.119)
    # "tmax_bilinear_tessera_embdrop03_with_elev_no_static_wd|--interpolation bilinear --tessera-method cnn --tessera-output-dim 16 --tessera-drop-prob 0.3 --no-static-fields --weight-decay 1e-4 --target-variables tmax"

    # "wind_baseline_ls025|--setconv-length-scale 0.25 --target-variables wind_mean"
    # "wind_tessera_cnn_embdrop03_no_elev_no_static_wd_ls025|--tessera-method cnn --tessera-output-dim 16 --tessera-drop-prob 0.3 --no-static-fields --no-elevation --weight-decay 1e-4 --setconv-length-scale 0.25 --target-variables wind_mean"

    # -----------------------------------------------------------------
    # VAE latent experiments (frozen pre-trained 16-d latents per station).
    # The VAE was trained with an auxiliary elevation loss, so elevation
    # is already baked into the latents. Still we include the explicit
    # elevation MLP features to keep the comparison with TESSERA CNN
    # baselines fair (those configs used elevation too).
    #
    # Structure: {wind,tmax} × {concat,film} × {drop0, drop03}
    # => 8 configs × 3 seeds = 24 runs.
    # All use bilinear interpolation + weight decay 1e-4 (best-TESSERA
    # recipe from last week). Weather static fields are kept in (we do
    # NOT pass --no-static-fields) because the VAE latents already encode
    # surface info and we want to test whether they can add on top of
    # the existing static surface features rather than replace them.
    # -----------------------------------------------------------------
    "wind_bilinear_vae_lat16_concat_wd_drop0|--interpolation bilinear --tessera-injection concat --vae-latents-path ${VAE_LATENTS_PATH} --vae-latents-station-csv ${VAE_LATENTS_CSV} --vae-latents-drop-prob 0.0 --weight-decay 1e-4 --target-variables wind_mean"

    "wind_bilinear_vae_lat16_concat_wd_drop03|--interpolation bilinear --tessera-injection concat --vae-latents-path ${VAE_LATENTS_PATH} --vae-latents-station-csv ${VAE_LATENTS_CSV} --vae-latents-drop-prob 0.3 --weight-decay 1e-4 --target-variables wind_mean"

    "wind_bilinear_vae_lat16_film_wd_drop0|--interpolation bilinear --tessera-injection film --vae-latents-path ${VAE_LATENTS_PATH} --vae-latents-station-csv ${VAE_LATENTS_CSV} --vae-latents-drop-prob 0.0 --weight-decay 1e-4 --target-variables wind_mean"

    "wind_bilinear_vae_lat16_film_wd_drop03|--interpolation bilinear --tessera-injection film --vae-latents-path ${VAE_LATENTS_PATH} --vae-latents-station-csv ${VAE_LATENTS_CSV} --vae-latents-drop-prob 0.3 --weight-decay 1e-4 --target-variables wind_mean"

    "tmax_bilinear_vae_lat16_concat_wd_drop0|--interpolation bilinear --tessera-injection concat --vae-latents-path ${VAE_LATENTS_PATH} --vae-latents-station-csv ${VAE_LATENTS_CSV} --vae-latents-drop-prob 0.0 --weight-decay 1e-4 --target-variables tmax"

    "tmax_bilinear_vae_lat16_concat_wd_drop03|--interpolation bilinear --tessera-injection concat --vae-latents-path ${VAE_LATENTS_PATH} --vae-latents-station-csv ${VAE_LATENTS_CSV} --vae-latents-drop-prob 0.3 --weight-decay 1e-4 --target-variables tmax"

    "tmax_bilinear_vae_lat16_film_wd_drop0|--interpolation bilinear --tessera-injection film --vae-latents-path ${VAE_LATENTS_PATH} --vae-latents-station-csv ${VAE_LATENTS_CSV} --vae-latents-drop-prob 0.0 --weight-decay 1e-4 --target-variables tmax"

    "tmax_bilinear_vae_lat16_film_wd_drop03|--interpolation bilinear --tessera-injection film --vae-latents-path ${VAE_LATENTS_PATH} --vae-latents-station-csv ${VAE_LATENTS_CSV} --vae-latents-drop-prob 0.3 --weight-decay 1e-4 --target-variables tmax"

    # --- Wind extensions: no_elev variants (with static) ---
    "wind_bilinear_vae_lat16_concat_no_elev_wd_drop0|--interpolation bilinear --tessera-injection concat --vae-latents-path ${VAE_LATENTS_PATH} --vae-latents-station-csv ${VAE_LATENTS_CSV} --vae-latents-drop-prob 0.0 --no-elevation --weight-decay 1e-4 --target-variables wind_mean"

    "wind_bilinear_vae_lat16_concat_no_elev_wd_drop03|--interpolation bilinear --tessera-injection concat --vae-latents-path ${VAE_LATENTS_PATH} --vae-latents-station-csv ${VAE_LATENTS_CSV} --vae-latents-drop-prob 0.3 --no-elevation --weight-decay 1e-4 --target-variables wind_mean"

    "wind_bilinear_vae_lat16_film_no_elev_wd_drop0|--interpolation bilinear --tessera-injection film --vae-latents-path ${VAE_LATENTS_PATH} --vae-latents-station-csv ${VAE_LATENTS_CSV} --vae-latents-drop-prob 0.0 --no-elevation --weight-decay 1e-4 --target-variables wind_mean"

    "wind_bilinear_vae_lat16_film_no_elev_wd_drop03|--interpolation bilinear --tessera-injection film --vae-latents-path ${VAE_LATENTS_PATH} --vae-latents-station-csv ${VAE_LATENTS_CSV} --vae-latents-drop-prob 0.3 --no-elevation --weight-decay 1e-4 --target-variables wind_mean"

    # --- Wind extensions: no_static variants (with elev) ---
    "wind_bilinear_vae_lat16_concat_no_static_wd_drop0|--interpolation bilinear --tessera-injection concat --vae-latents-path ${VAE_LATENTS_PATH} --vae-latents-station-csv ${VAE_LATENTS_CSV} --vae-latents-drop-prob 0.0 --no-static-fields --weight-decay 1e-4 --target-variables wind_mean"

    "wind_bilinear_vae_lat16_concat_no_static_wd_drop03|--interpolation bilinear --tessera-injection concat --vae-latents-path ${VAE_LATENTS_PATH} --vae-latents-station-csv ${VAE_LATENTS_CSV} --vae-latents-drop-prob 0.3 --no-static-fields --weight-decay 1e-4 --target-variables wind_mean"

    "wind_bilinear_vae_lat16_film_no_static_wd_drop0|--interpolation bilinear --tessera-injection film --vae-latents-path ${VAE_LATENTS_PATH} --vae-latents-station-csv ${VAE_LATENTS_CSV} --vae-latents-drop-prob 0.0 --no-static-fields --weight-decay 1e-4 --target-variables wind_mean"

    "wind_bilinear_vae_lat16_film_no_static_wd_drop03|--interpolation bilinear --tessera-injection film --vae-latents-path ${VAE_LATENTS_PATH} --vae-latents-station-csv ${VAE_LATENTS_CSV} --vae-latents-drop-prob 0.3 --no-static-fields --weight-decay 1e-4 --target-variables wind_mean"

    # --- Wind extensions: no_elev + no_static variants ---
    "wind_bilinear_vae_lat16_concat_no_elev_no_static_wd_drop0|--interpolation bilinear --tessera-injection concat --vae-latents-path ${VAE_LATENTS_PATH} --vae-latents-station-csv ${VAE_LATENTS_CSV} --vae-latents-drop-prob 0.0 --no-elevation --no-static-fields --weight-decay 1e-4 --target-variables wind_mean"

    "wind_bilinear_vae_lat16_concat_no_elev_no_static_wd_drop03|--interpolation bilinear --tessera-injection concat --vae-latents-path ${VAE_LATENTS_PATH} --vae-latents-station-csv ${VAE_LATENTS_CSV} --vae-latents-drop-prob 0.3 --no-elevation --no-static-fields --weight-decay 1e-4 --target-variables wind_mean"

    "wind_bilinear_vae_lat16_film_no_elev_no_static_wd_drop0|--interpolation bilinear --tessera-injection film --vae-latents-path ${VAE_LATENTS_PATH} --vae-latents-station-csv ${VAE_LATENTS_CSV} --vae-latents-drop-prob 0.0 --no-elevation --no-static-fields --weight-decay 1e-4 --target-variables wind_mean"

    "wind_bilinear_vae_lat16_film_no_elev_no_static_wd_drop03|--interpolation bilinear --tessera-injection film --vae-latents-path ${VAE_LATENTS_PATH} --vae-latents-station-csv ${VAE_LATENTS_CSV} --vae-latents-drop-prob 0.3 --no-elevation --no-static-fields --weight-decay 1e-4 --target-variables wind_mean"

    # -----------------------------------------------------------------
    # Lat64 VAE + hypernet study. Europe-only, same splits as above so
    # results are directly comparable to the lat16 runs.
    #   (a) lat64 + FiLM on wind: does larger latent beat lat16 at the
    #       winning config (FiLM + with elev + no static + wd + drop=0)?
    #   (b) lat64 + FiLM on tmax: tmax lat16 had a wider NLL gap (~3.3 vs
    #       TESSERA CNN's ~2.4); larger latent may sharpen the mean.
    #   (c) hypernet at lat16: architectural comparison at established
    #       latent size — isolates "hypernet vs FiLM" independently of
    #       the latent-size change.
    #   (d) hypernet at lat64: the full-bandwidth case. Hypernet body is
    #       ~0.5M params either way; only the first-layer input dim
    #       changes. If hypernet helps at all, it should help most here.
    # -----------------------------------------------------------------
    "wind_bilinear_vae_lat64_film_no_static_wd_drop0|--interpolation bilinear --tessera-injection film --vae-latents-path ${VAE_LATENTS_PATH_LAT64} --vae-latents-station-csv ${VAE_LATENTS_CSV} --vae-latents-drop-prob 0.0 --no-static-fields --weight-decay 1e-4 --target-variables wind_mean"

    "tmax_bilinear_vae_lat64_film_no_static_wd_drop0|--interpolation bilinear --tessera-injection film --vae-latents-path ${VAE_LATENTS_PATH_LAT64} --vae-latents-station-csv ${VAE_LATENTS_CSV} --vae-latents-drop-prob 0.0 --no-static-fields --weight-decay 1e-4 --target-variables tmax"

    "wind_bilinear_vae_lat16_hypernet_no_static_wd_drop0|--interpolation bilinear --tessera-injection hypernet --vae-latents-path ${VAE_LATENTS_PATH} --vae-latents-station-csv ${VAE_LATENTS_CSV} --vae-latents-drop-prob 0.0 --no-static-fields --weight-decay 1e-4 --target-variables wind_mean"

    "wind_bilinear_vae_lat64_hypernet_no_static_wd_drop0|--interpolation bilinear --tessera-injection hypernet --vae-latents-path ${VAE_LATENTS_PATH_LAT64} --vae-latents-station-csv ${VAE_LATENTS_CSV} --vae-latents-drop-prob 0.0 --no-static-fields --weight-decay 1e-4 --target-variables wind_mean"

    # --- Additional lat64 variants: no_elev counterparts + tmax hypernet.
    # The lat16 story has FiLM+no_elev ≈ FiLM+with_elev on wind; re-test at
    # lat64 to see if the larger latent fully encodes elevation. No hypernet
    # had been run without elevation or on tmax, so these fill those gaps.
    "wind_bilinear_vae_lat64_film_no_elev_no_static_wd_drop0|--interpolation bilinear --tessera-injection film --vae-latents-path ${VAE_LATENTS_PATH_LAT64} --vae-latents-station-csv ${VAE_LATENTS_CSV} --vae-latents-drop-prob 0.0 --no-elevation --no-static-fields --weight-decay 1e-4 --target-variables wind_mean"

    "wind_bilinear_vae_lat64_hypernet_no_elev_no_static_wd_drop0|--interpolation bilinear --tessera-injection hypernet --vae-latents-path ${VAE_LATENTS_PATH_LAT64} --vae-latents-station-csv ${VAE_LATENTS_CSV} --vae-latents-drop-prob 0.0 --no-elevation --no-static-fields --weight-decay 1e-4 --target-variables wind_mean"

    "tmax_bilinear_vae_lat64_hypernet_no_static_wd_drop0|--interpolation bilinear --tessera-injection hypernet --vae-latents-path ${VAE_LATENTS_PATH_LAT64} --vae-latents-station-csv ${VAE_LATENTS_CSV} --vae-latents-drop-prob 0.0 --no-static-fields --weight-decay 1e-4 --target-variables tmax"

    # -----------------------------------------------------------------
    # Lat64 + learnable linear projection variants. The hypothesis from
    # the lat64+proj16 win is "fat encoder + thin task projection lets
    # the FiLM head consume only the bandwidth it can actually use, while
    # leaving the VAE itself with full reconstruction capacity". proj16
    # already won decisively over plain lat16/lat64; proj8 tests the
    # floor — does the task-relevant signal collapse into 8 dims, or do
    # we lose information?
    # -----------------------------------------------------------------
    "wind_bilinear_vae_lat64_proj16_film_no_elev_no_static_wd_drop0|--interpolation bilinear --tessera-injection film --vae-latents-path ${VAE_LATENTS_PATH_LAT64} --vae-latents-station-csv ${VAE_LATENTS_CSV} --vae-latents-drop-prob 0.0 --vae-latents-proj-dim 16 --no-elevation --no-static-fields --weight-decay 1e-4 --target-variables wind_mean"

    "wind_bilinear_vae_lat64_proj8_film_no_elev_no_static_wd_drop0|--interpolation bilinear --tessera-injection film --vae-latents-path ${VAE_LATENTS_PATH_LAT64} --vae-latents-station-csv ${VAE_LATENTS_CSV} --vae-latents-drop-prob 0.0 --vae-latents-proj-dim 8 --no-elevation --no-static-fields --weight-decay 1e-4 --target-variables wind_mean"

    # proj4: pushes the wind floor further. proj16 and proj8 tied at MAE
    # 1.000 / 0.998, so the task signal might be even lower-dimensional.
    # If proj4 holds up, the bottleneck is genuinely tiny (4 dims of
    # task-relevant surface info).
    "wind_bilinear_vae_lat64_proj4_film_no_elev_no_static_wd_drop0|--interpolation bilinear --tessera-injection film --vae-latents-path ${VAE_LATENTS_PATH_LAT64} --vae-latents-station-csv ${VAE_LATENTS_CSV} --vae-latents-drop-prob 0.0 --vae-latents-proj-dim 4 --no-elevation --no-static-fields --weight-decay 1e-4 --target-variables wind_mean"

    # tmax counterpart of the proj16 winner. Tmax had a wider NLL gap
    # against TESSERA CNN in earlier results, so this checks whether the
    # "fat encoder + thin task projection" pattern that helped wind also
    # helps tmax.
    "tmax_bilinear_vae_lat64_proj16_film_no_static_wd_drop0|--interpolation bilinear --tessera-injection film --vae-latents-path ${VAE_LATENTS_PATH_LAT64} --vae-latents-station-csv ${VAE_LATENTS_CSV} --vae-latents-drop-prob 0.0 --vae-latents-proj-dim 16 --no-static-fields --weight-decay 1e-4 --target-variables tmax"

    # tmax no-elev counterpart of the proj16 winner. The "elevation is
    # free under FiLM+VAE" pattern was established for wind; for tmax
    # (where lapse rate makes elevation a strong predictor) we don't
    # know if the proj16 architecture absorbs that signal as efficiently.
    # If MAE matches the with-elev variant, the VAE latents are encoding
    # elevation information sufficient for tmax prediction.
    "tmax_bilinear_vae_lat64_proj16_film_no_elev_no_static_wd_drop0|--interpolation bilinear --tessera-injection film --vae-latents-path ${VAE_LATENTS_PATH_LAT64} --vae-latents-station-csv ${VAE_LATENTS_CSV} --vae-latents-drop-prob 0.0 --vae-latents-proj-dim 16 --no-elevation --no-static-fields --weight-decay 1e-4 --target-variables tmax"

    # tmax projection floor: confirms the tmax pattern at proj8 and proj4.
    # If tmax follows wind's pattern (proj16 ≈ proj8 ≈ proj4), the
    # bottleneck claim generalises across target variables.
    "tmax_bilinear_vae_lat64_proj8_film_no_static_wd_drop0|--interpolation bilinear --tessera-injection film --vae-latents-path ${VAE_LATENTS_PATH_LAT64} --vae-latents-station-csv ${VAE_LATENTS_CSV} --vae-latents-drop-prob 0.0 --vae-latents-proj-dim 8 --no-static-fields --weight-decay 1e-4 --target-variables tmax"

    "tmax_bilinear_vae_lat64_proj4_film_no_static_wd_drop0|--interpolation bilinear --tessera-injection film --vae-latents-path ${VAE_LATENTS_PATH_LAT64} --vae-latents-station-csv ${VAE_LATENTS_CSV} --vae-latents-drop-prob 0.0 --vae-latents-proj-dim 4 --no-static-fields --weight-decay 1e-4 --target-variables tmax"

    "tmax_bilinear_vae_lat64_proj4_film_no_elev_no_static_wd_drop0|--interpolation bilinear --tessera-injection film --vae-latents-path ${VAE_LATENTS_PATH_LAT64} --vae-latents-station-csv ${VAE_LATENTS_CSV} --vae-latents-drop-prob 0.0 --vae-latents-proj-dim 4 --no-elevation --no-static-fields --weight-decay 1e-4 --target-variables tmax"

    # -----------------------------------------------------------------
    # Multi-task experiments (predict tmax AND wind_mean jointly).
    # Earlier multi-task attempts (TESSERA-CNN with concat injection) were
    # mediocre and got cut. Worth revisiting now that lat64+proj16+FiLM is
    # the clear winner: in multi-task mode, the projection has to compress
    # lat64 → 16 dims useful for both targets simultaneously, which can
    # regularise the representation and help each task. Multi-task mode
    # is auto-detected by train.py from multiple --target-variables; the
    # learned per-task loss weighting handles the scale mismatch.
    #
    # Elevation kept in: tmax has strong lapse-rate dependence and
    # benefits substantially; wind is roughly neutral on elevation.
    # -----------------------------------------------------------------
    "multitask_bilinear_baseline_wd|--interpolation bilinear --weight-decay 1e-4 --target-variables tmax wind_mean"

    "multitask_bilinear_vae_lat64_proj16_film_no_static_wd_drop0|--interpolation bilinear --tessera-injection film --vae-latents-path ${VAE_LATENTS_PATH_LAT64} --vae-latents-station-csv ${VAE_LATENTS_CSV} --vae-latents-drop-prob 0.0 --vae-latents-proj-dim 16 --no-static-fields --weight-decay 1e-4 --target-variables tmax wind_mean"

    # Multi-task no-elev counterpart. If wind benefits from no-elev (or is
    # neutral) and tmax benefits from with-elev, the multi-task no-elev
    # variant tests whether the VAE latents carry enough elevation signal
    # to compensate for tmax. If yes, no-elev becomes the simpler default.
    "multitask_bilinear_vae_lat64_proj16_film_no_elev_no_static_wd_drop0|--interpolation bilinear --tessera-injection film --vae-latents-path ${VAE_LATENTS_PATH_LAT64} --vae-latents-station-csv ${VAE_LATENTS_CSV} --vae-latents-drop-prob 0.0 --vae-latents-proj-dim 16 --no-elevation --no-static-fields --weight-decay 1e-4 --target-variables tmax wind_mean"

    # -----------------------------------------------------------------
    # Multi-region experiments (train on US, test on Europe).
    # Override --dataset-dir to point at dataset_daily_global, and use
    # the global TESSERA extraction files. --train-regions us restricts
    # training to US stations/grid, --test-regions europe (passed to
    # evaluate.py via the third field) evaluates cross-continent transfer.
    # -----------------------------------------------------------------
    # Best baseline (no TESSERA, no VAE): keeps static fields and elevation,
    # since the baseline's job is to be the strongest non-TESSERA model
    # achievable — anything else would inflate the apparent TESSERA gain.
    "wind_us_to_eu_baseline_wd|--dataset-dir ${DATASET_DIR_GLOBAL} --tessera-path ${TESSERA_PATH_GLOBAL} --tessera-station-csv ${TESSERA_CSV_GLOBAL} --train-regions us --val-regions us --interpolation bilinear --weight-decay 1e-4 --target-variables wind_mean|--test-regions europe"

    # Winning VAE config: bilinear + FiLM + VAE-lat16 + with elev + no static
    # + wd=1e-4 + drop=0 (European-only MAE was 1.018, best overall).
    "wind_us_to_eu_vae_lat16_film_no_static_wd_drop0|--dataset-dir ${DATASET_DIR_GLOBAL} --tessera-path ${TESSERA_PATH_GLOBAL} --tessera-station-csv ${TESSERA_CSV_GLOBAL} --train-regions us --val-regions us --interpolation bilinear --tessera-injection film --vae-latents-path ${VAE_LATENTS_PATH} --vae-latents-station-csv ${VAE_LATENTS_CSV} --vae-latents-drop-prob 0.0 --no-static-fields --weight-decay 1e-4 --target-variables wind_mean|--test-regions europe"

    # Elevation-free transfer test: elevation distributions differ between
    # US and Europe (different median, different range). Training with
    # --no-elevation removes the risk that the model latched onto US-specific
    # elevation statistics and generalises poorly to Europe because of it.
    "wind_us_to_eu_vae_lat16_film_no_elev_no_static_wd_drop0|--dataset-dir ${DATASET_DIR_GLOBAL} --tessera-path ${TESSERA_PATH_GLOBAL} --tessera-station-csv ${TESSERA_CSV_GLOBAL} --train-regions us --val-regions us --interpolation bilinear --tessera-injection film --vae-latents-path ${VAE_LATENTS_PATH} --vae-latents-station-csv ${VAE_LATENTS_CSV} --vae-latents-drop-prob 0.0 --no-elevation --no-static-fields --weight-decay 1e-4 --target-variables wind_mean|--test-regions europe"

    # Larger latent for transfer: does 64-d VAE help or hurt US→EU?
    # Plausible yes (more universal surface detail) or no (overfits US
    # quirks). Only one way to find out.
    "wind_us_to_eu_vae_lat64_film_no_static_wd_drop0|--dataset-dir ${DATASET_DIR_GLOBAL} --tessera-path ${TESSERA_PATH_GLOBAL} --tessera-station-csv ${TESSERA_CSV_GLOBAL} --train-regions us --val-regions us --interpolation bilinear --tessera-injection film --vae-latents-path ${VAE_LATENTS_PATH_LAT64} --vae-latents-station-csv ${VAE_LATENTS_CSV} --vae-latents-drop-prob 0.0 --no-static-fields --weight-decay 1e-4 --target-variables wind_mean|--test-regions europe"

    # Winning architecture (proj16) for US→EU transfer. Adds the missing
    # row so US→EU is comparable to the other 3 transfer directions
    # (EU→US, US→Asia, EU→Asia), all of which have proj16 variants.
    "wind_us_to_eu_vae_lat64_proj16_film_no_elev_no_static_wd_drop0|--dataset-dir ${DATASET_DIR_GLOBAL} --tessera-path ${TESSERA_PATH_GLOBAL} --tessera-station-csv ${TESSERA_CSV_GLOBAL} --train-regions us --val-regions us --interpolation bilinear --tessera-injection film --vae-latents-path ${VAE_LATENTS_PATH_LAT64} --vae-latents-station-csv ${VAE_LATENTS_CSV} --vae-latents-drop-prob 0.0 --vae-latents-proj-dim 16 --no-elevation --no-static-fields --weight-decay 1e-4 --target-variables wind_mean|--test-regions europe"

    # End-to-end TESSERA CNN on multi-region transfer: confirms (or
    # refutes) that VAE > end-to-end holds in the transfer setting too.
    # "wind_us_to_eu_tessera_cnn_no_elev_no_static_wd|--dataset-dir ${DATASET_DIR_GLOBAL} --tessera-path ${TESSERA_PATH_GLOBAL} --tessera-station-csv ${TESSERA_CSV_GLOBAL} --train-regions us --val-regions us --interpolation bilinear --tessera-method cnn --tessera-output-dim 16 --tessera-drop-prob 0.3 --no-elevation --no-static-fields --weight-decay 1e-4 --target-variables wind_mean|--test-regions europe"

    # tmax transfer check: tmax climatology is more latitude/elevation-
    # driven than wind, so ERA5 captures it better. Expect smaller
    # TESSERA gains but need a data point.
    "tmax_us_to_eu_baseline_wd|--dataset-dir ${DATASET_DIR_GLOBAL} --tessera-path ${TESSERA_PATH_GLOBAL} --tessera-station-csv ${TESSERA_CSV_GLOBAL} --train-regions us --val-regions us --interpolation bilinear --weight-decay 1e-4 --target-variables tmax|--test-regions europe"

    "tmax_us_to_eu_vae_lat16_film_no_static_wd_drop0|--dataset-dir ${DATASET_DIR_GLOBAL} --tessera-path ${TESSERA_PATH_GLOBAL} --tessera-station-csv ${TESSERA_CSV_GLOBAL} --train-regions us --val-regions us --interpolation bilinear --tessera-injection film --vae-latents-path ${VAE_LATENTS_PATH} --vae-latents-station-csv ${VAE_LATENTS_CSV} --vae-latents-drop-prob 0.0 --no-static-fields --weight-decay 1e-4 --target-variables tmax|--test-regions europe"

    # tmax US→EU no-elev counterpart: same elevation-distribution-mismatch
    # rationale as the wind transfer experiments. Tests whether removing
    # US-specific elevation patterns helps tmax transfer to Europe even
    # though tmax depends strongly on elevation in absolute terms.
    "tmax_us_to_eu_vae_lat16_film_no_elev_no_static_wd_drop0|--dataset-dir ${DATASET_DIR_GLOBAL} --tessera-path ${TESSERA_PATH_GLOBAL} --tessera-station-csv ${TESSERA_CSV_GLOBAL} --train-regions us --val-regions us --interpolation bilinear --tessera-injection film --vae-latents-path ${VAE_LATENTS_PATH} --vae-latents-station-csv ${VAE_LATENTS_CSV} --vae-latents-drop-prob 0.0 --no-elevation --no-static-fields --weight-decay 1e-4 --target-variables tmax|--test-regions europe"

    # -----------------------------------------------------------------
    # Additional multi-region transfer directions. Each direction has
    # two variants: a bilinear baseline (necessary anchor for the gain
    # comparison) and the lat64+proj16+FiLM winner (currently the best
    # architecture per Europe-only results). We use --no-elevation in
    # both because elevation distributions differ across continents and
    # we don't want the model latching onto continent-specific stats.
    #
    # Naming convention: <var>_<src>_to_<tgt>_<config>. 'us', 'eu', 'as'
    # are short tokens. eval_stragglers.sh detects '<src>_to_<tgt>' and
    # auto-passes --test-regions <tgt-full-name>.
    # -----------------------------------------------------------------

    # EU → US: inverts the direction. Tests transfer symmetry. Training
    # on dense Europe and testing on sparse US should — by the "TESSERA
    # helps more for sparse test sets" hypothesis — show a bigger
    # baseline-vs-VAE gap than US→EU did.
    "wind_eu_to_us_baseline_wd|--dataset-dir ${DATASET_DIR_GLOBAL} --tessera-path ${TESSERA_PATH_GLOBAL} --tessera-station-csv ${TESSERA_CSV_GLOBAL} --train-regions europe --val-regions europe --interpolation bilinear --weight-decay 1e-4 --target-variables wind_mean|--test-regions us"

    "wind_eu_to_us_vae_lat64_proj16_film_no_elev_no_static_wd_drop0|--dataset-dir ${DATASET_DIR_GLOBAL} --tessera-path ${TESSERA_PATH_GLOBAL} --tessera-station-csv ${TESSERA_CSV_GLOBAL} --train-regions europe --val-regions europe --interpolation bilinear --tessera-injection film --vae-latents-path ${VAE_LATENTS_PATH_LAT64} --vae-latents-station-csv ${VAE_LATENTS_CSV} --vae-latents-drop-prob 0.0 --vae-latents-proj-dim 16 --no-elevation --no-static-fields --weight-decay 1e-4 --target-variables wind_mean|--test-regions us"

    # tmax EU → US: completes the tmax transfer story with one inverted
    # direction. Tmax climatology is mostly latitude+elevation-driven, so
    # transfer should be more straightforward than wind. Useful as a
    # sanity check that "transfer works" is generic, not wind-specific.
    "tmax_eu_to_us_baseline_wd|--dataset-dir ${DATASET_DIR_GLOBAL} --tessera-path ${TESSERA_PATH_GLOBAL} --tessera-station-csv ${TESSERA_CSV_GLOBAL} --train-regions europe --val-regions europe --interpolation bilinear --weight-decay 1e-4 --target-variables tmax|--test-regions us"

    "tmax_eu_to_us_vae_lat64_proj16_film_no_static_wd_drop0|--dataset-dir ${DATASET_DIR_GLOBAL} --tessera-path ${TESSERA_PATH_GLOBAL} --tessera-station-csv ${TESSERA_CSV_GLOBAL} --train-regions europe --val-regions europe --interpolation bilinear --tessera-injection film --vae-latents-path ${VAE_LATENTS_PATH_LAT64} --vae-latents-station-csv ${VAE_LATENTS_CSV} --vae-latents-drop-prob 0.0 --vae-latents-proj-dim 16 --no-static-fields --weight-decay 1e-4 --target-variables tmax|--test-regions us"

    # tmax EU→US no-elev counterpart. EU and US elevation distributions
    # differ significantly (US has more high-elevation stations); removing
    # elevation lets the model rely on VAE-encoded surface info instead
    # of potentially-misleading raw elevation values.
    "tmax_eu_to_us_vae_lat64_proj16_film_no_elev_no_static_wd_drop0|--dataset-dir ${DATASET_DIR_GLOBAL} --tessera-path ${TESSERA_PATH_GLOBAL} --tessera-station-csv ${TESSERA_CSV_GLOBAL} --train-regions europe --val-regions europe --interpolation bilinear --tessera-injection film --vae-latents-path ${VAE_LATENTS_PATH_LAT64} --vae-latents-station-csv ${VAE_LATENTS_CSV} --vae-latents-drop-prob 0.0 --vae-latents-proj-dim 16 --no-elevation --no-static-fields --weight-decay 1e-4 --target-variables tmax|--test-regions us"


    # -----------------------------------------------------------------
    # Multi-source training: trains on US AND EU jointly, then evaluates
    # on a third (held-out) region. Tests the "training-source diversity
    # matters" hypothesis from the apparent asymmetry where US-trained
    # models transfer well to Asia but EU-trained models don't. If joint
    # US+EU training matches or beats best-single-source, the limiting
    # factor was source breadth and joint training is strictly better.
    # -----------------------------------------------------------------

    # US+EU joint training, tested on Asia. Asia is fully held out from
    # both training sources, so no contamination.
    "wind_useu_to_as_baseline_wd|--dataset-dir ${DATASET_DIR_GLOBAL} --tessera-path ${TESSERA_PATH_GLOBAL} --tessera-station-csv ${TESSERA_CSV_GLOBAL} --train-regions us europe --val-regions us europe --interpolation bilinear --weight-decay 1e-4 --target-variables wind_mean|--test-regions east_asia"

    "wind_useu_to_as_vae_lat64_proj16_film_no_elev_no_static_wd_drop0|--dataset-dir ${DATASET_DIR_GLOBAL} --tessera-path ${TESSERA_PATH_GLOBAL} --tessera-station-csv ${TESSERA_CSV_GLOBAL} --train-regions us europe --val-regions us europe --interpolation bilinear --tessera-injection film --vae-latents-path ${VAE_LATENTS_PATH_LAT64} --vae-latents-station-csv ${VAE_LATENTS_CSV} --vae-latents-drop-prob 0.0 --vae-latents-proj-dim 16 --no-elevation --no-static-fields --weight-decay 1e-4 --target-variables wind_mean|--test-regions east_asia"

    # US → East Asia: tests universality of US-trained features against
    # very different terrain (Japan, Korea, Taiwan — mountainous archipelagos,
    # different surface composition than US continental interior).
    "wind_us_to_as_baseline_wd|--dataset-dir ${DATASET_DIR_GLOBAL} --tessera-path ${TESSERA_PATH_GLOBAL} --tessera-station-csv ${TESSERA_CSV_GLOBAL} --train-regions us --val-regions us --interpolation bilinear --weight-decay 1e-4 --target-variables wind_mean|--test-regions east_asia"

    "wind_us_to_as_vae_lat64_proj16_film_no_elev_no_static_wd_drop0|--dataset-dir ${DATASET_DIR_GLOBAL} --tessera-path ${TESSERA_PATH_GLOBAL} --tessera-station-csv ${TESSERA_CSV_GLOBAL} --train-regions us --val-regions us --interpolation bilinear --tessera-injection film --vae-latents-path ${VAE_LATENTS_PATH_LAT64} --vae-latents-station-csv ${VAE_LATENTS_CSV} --vae-latents-drop-prob 0.0 --vae-latents-proj-dim 16 --no-elevation --no-static-fields --weight-decay 1e-4 --target-variables wind_mean|--test-regions east_asia"

    # EU → East Asia: complement to US→Asia. If both EU- and US-trained
    # models transfer comparably to Asia, the limiting factor is the
    # Asian terrain itself (most novel surfaces). If one transfers much
    # better, the source distribution matters more than I'd guess.
    "wind_eu_to_as_baseline_wd|--dataset-dir ${DATASET_DIR_GLOBAL} --tessera-path ${TESSERA_PATH_GLOBAL} --tessera-station-csv ${TESSERA_CSV_GLOBAL} --train-regions europe --val-regions europe --interpolation bilinear --weight-decay 1e-4 --target-variables wind_mean|--test-regions east_asia"

    "wind_eu_to_as_vae_lat64_proj16_film_no_elev_no_static_wd_drop0|--dataset-dir ${DATASET_DIR_GLOBAL} --tessera-path ${TESSERA_PATH_GLOBAL} --tessera-station-csv ${TESSERA_CSV_GLOBAL} --train-regions europe --val-regions europe --interpolation bilinear --tessera-injection film --vae-latents-path ${VAE_LATENTS_PATH_LAT64} --vae-latents-station-csv ${VAE_LATENTS_CSV} --vae-latents-drop-prob 0.0 --vae-latents-proj-dim 16 --no-elevation --no-static-fields --weight-decay 1e-4 --target-variables wind_mean|--test-regions east_asia"
)

# ---- Create log directory ----
mkdir -p "${REPO_ROOT}/logs"

# ---- Submit jobs ----
echo "============================================"
echo "Submitting experiment jobs"
echo "============================================"
echo ""
echo "Dataset:    ${DATASET_DIR}"
echo "Output:     ${OUTPUT_ROOT}"
echo "Batch size: ${BATCH_SIZE}"
echo "Epochs:     ${EPOCHS}"
echo "Seeds:      ${SEEDS[*]}"
echo "Configs:    ${#EXPERIMENTS[@]}"
echo "Total jobs: $(( ${#EXPERIMENTS[@]} * ${#SEEDS[@]} ))"
echo ""

COMMON_ARGS="--dataset-dir ${DATASET_DIR} --tessera-path ${TESSERA_PATH} --tessera-station-csv ${TESSERA_CSV} --batch-size ${BATCH_SIZE} --epochs ${EPOCHS} --patience ${PATIENCE} --lr ${LR} --cnn-hidden ${CNN_HIDDEN} --cnn-layers ${CNN_LAYERS} --mlp-hidden ${MLP_HIDDEN} --mlp-n-hidden ${MLP_N_HIDDEN} --num-workers ${NUM_WORKERS}"

JOB_COUNT=0
for experiment in "${EXPERIMENTS[@]}"; do
    # Experiment entry format: "name|train_extra_args" or
    # "name|train_extra_args|eval_extra_args". The third field is optional
    # and passed only to the evaluate.py invocation (e.g. --test-regions).
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
echo "Submitted ${JOB_COUNT} jobs"
echo "============================================"
echo ""
echo "Monitor with:  squeue --me"
echo "Cancel all:    scancel --me"
echo "Results in:    ${OUTPUT_ROOT}/"