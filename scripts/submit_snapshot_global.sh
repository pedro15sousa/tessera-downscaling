#!/bin/bash
# Submit MULTI-REGION snapshot experiments × seeds as parallel Slurm jobs.
#
# Companion to submit_snapshot.sh. The flat snapshot script handles single-
# region Europe experiments against dataset_timestamp/; this script handles
# multi-region experiments against dataset_timestamp_global/, using:
#   * --region-specs-train / --val / --test JSON dicts for held-out-within-
#     training and transfer configurations
#   * SnapshotDownscalingDataset(region=) for single-region Asia-only baselines
#     (reads from the global dataset rather than needing a separate flat one)
#
# The 30 experiments in this script answer three hypotheses per target
# variable × target region (EU, Asia):
#
#   (a) Same-region joint source BEATS single-source? e.g. does training
#       on {EU train + all of US} improve EU-test over training on EU train
#       alone (from submit_snapshot.sh)?
#   (b) Is transfer strong? e.g. how well does {US only} → EU test fare?
#   (c) Does concat+proj do as well as FiLM+proj? Tests whether the FiLM
#       machinery earns its keep when the projection head is free to do
#       most of the work.
#
# Each of the 6 (variable × target × model) cells gets baseline + FiLM +
# concat, repeated across (transfer, joint, same-source) where applicable.
#
# Usage (from repo root):
#   bash projects/tessera_downscaling/scripts/submit_snapshot_global.sh
#
# Dry run (prints commands without submitting):
#   DRY_RUN=1 bash projects/tessera_downscaling/scripts/submit_snapshot_global.sh
set -euo pipefail

# ---- Paths ----
REPO_ROOT="/projects/u6do/pmms2/end-to-end-forecasting"
BASE_DIR="${REPO_ROOT}/projects/tessera_downscaling/.tmp_output"

# Multi-region snapshot dataset produced by preprocess_timestamp_global.py.
# Has layout_version="multi_region_snapshot_v1". Every experiment below
# reads from this one tree — single-region experiments via the dataset
# class's region= kwarg rather than from a separate flat dataset.
DATASET_DIR="${BASE_DIR}/dataset_timestamp_global"

# TESSERA and VAE latents — same files as the daily multi-region
# pipeline. `tessera_global/` has all 38,870 stations (used by the VAE
# latents), whereas `tessera/` is the 12,622-station EU-only subset
# used by `submit_snapshot.sh` (flat-EU experiments). We point at
# `tessera_global` here so the TESSERA station-set filter matches
# the VAE-latent coverage and non-EU regions aren't dropped.
TESSERA_PATH="${BASE_DIR}/processed/tessera_global/patch_embeddings_2024.npy"
TESSERA_CSV="${BASE_DIR}/processed/tessera_global/station_list_filtered.csv"
VAE_LATENTS_PATH_LAT64="${BASE_DIR}/processed/station_latents_lat64_l1.npy"
VAE_LATENTS_PATH_LAT16="${BASE_DIR}/processed/station_latents_lat16_grad0.5.npy"
VAE_LATENTS_CSV="${BASE_DIR}/processed/tessera_global/station_list_filtered.csv"

OUTPUT_ROOT="${BASE_DIR}/training_runs_snapshot_global"
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

# Shared hyperparameters. Match submit_snapshot.sh for comparability.
BATCH_SIZE=1
EPOCHS=100
PATIENCE=10
LR="2.5e-5"
CNN_HIDDEN=128
CNN_LAYERS=7
MLP_HIDDEN=128
MLP_N_HIDDEN=3
NUM_WORKERS=4

# Region-specs JSON values. These are now plain JSON strings — no
# shell-quoting gymnastics required because they are written to disk
# as files and passed to the training / eval scripts as file paths.
# Training region-specs for each experiment pattern:
SPEC_EU_US_JOINT='{"europe":"train","us":"all"}'    # train on EU train + all of US
SPEC_AS_US_JOINT='{"east_asia":"train","us":"all"}' # train on Asia train + all of US
SPEC_US_ONLY='{"us":"train"}'                       # transfer: US only
SPEC_AS_ONLY='{"east_asia":"train"}'                # single-region Asia baseline
SPEC_EU_ONLY='{"europe":"train"}'                   # single-region EU (reproducibility check vs submit_snapshot.sh)

# Test region-specs:
SPEC_EU_TEST='{"europe":"test"}'
SPEC_AS_TEST='{"east_asia":"test"}'

# ---- Experiments ----
# Format: "name|train_specs_json|test_specs_json|extra_args"
# train_specs_json and test_specs_json are plain JSON strings. They
# are written to files in the per-experiment output dir and the
# file paths are passed via --region-specs-train-file /
# --region-specs-test-file, so the JSON never needs shell quoting.
# The train pipeline derives --region-specs-val automatically
# (swaps "train"→"test", keeps "all" as-is).
#
# 30 experiments × 3 seeds = 90 jobs.
EXPERIMENTS=(
    # ==================================================================
    # WIND — test on EU
    # ==================================================================
    # Joint-source: train on {EU train + all US}, test on EU test.
    # Answers: "does US data help European wind predictions?"
    "wind_snap_euus2eu_baseline_wd|${SPEC_EU_US_JOINT}|${SPEC_EU_TEST}|--interpolation bilinear --weight-decay 1e-4 --target-variables wind"
    "wind_snap_euus2eu_vae_lat64_proj16_film_no_elev_no_static_wd|${SPEC_EU_US_JOINT}|${SPEC_EU_TEST}|--interpolation bilinear --tessera-injection film --vae-latents-path ${VAE_LATENTS_PATH_LAT64} --vae-latents-station-csv ${VAE_LATENTS_CSV} --vae-latents-drop-prob 0.0 --vae-latents-proj-dim 16 --no-elevation --no-static-fields --weight-decay 1e-4 --target-variables wind"
    "wind_snap_euus2eu_vae_lat64_proj16_concat_no_elev_no_static_wd|${SPEC_EU_US_JOINT}|${SPEC_EU_TEST}|--interpolation bilinear --tessera-injection concat --vae-latents-path ${VAE_LATENTS_PATH_LAT64} --vae-latents-station-csv ${VAE_LATENTS_CSV} --vae-latents-drop-prob 0.0 --vae-latents-proj-dim 16 --no-elevation --no-static-fields --weight-decay 1e-4 --target-variables wind"

    # Single-source transfer: train on US only, test on EU.
    # Answers: "how well does US training alone generalise to EU?"
    "wind_snap_us2eu_baseline_wd|${SPEC_US_ONLY}|${SPEC_EU_TEST}|--interpolation bilinear --weight-decay 1e-4 --target-variables wind"
    "wind_snap_us2eu_vae_lat64_proj16_film_no_elev_no_static_wd|${SPEC_US_ONLY}|${SPEC_EU_TEST}|--interpolation bilinear --tessera-injection film --vae-latents-path ${VAE_LATENTS_PATH_LAT64} --vae-latents-station-csv ${VAE_LATENTS_CSV} --vae-latents-drop-prob 0.0 --vae-latents-proj-dim 16 --no-elevation --no-static-fields --weight-decay 1e-4 --target-variables wind"
    "wind_snap_us2eu_vae_lat64_proj16_concat_no_elev_no_static_wd|${SPEC_US_ONLY}|${SPEC_EU_TEST}|--interpolation bilinear --tessera-injection concat --vae-latents-path ${VAE_LATENTS_PATH_LAT64} --vae-latents-station-csv ${VAE_LATENTS_CSV} --vae-latents-drop-prob 0.0 --vae-latents-proj-dim 16 --no-elevation --no-static-fields --weight-decay 1e-4 --target-variables wind"

    # ==================================================================
    # WIND — test on Asia
    # ==================================================================
    # Joint-source Asia: train on {Asia train + all US}, test on Asia.
    "wind_snap_asus2as_baseline_wd|${SPEC_AS_US_JOINT}|${SPEC_AS_TEST}|--interpolation bilinear --weight-decay 1e-4 --target-variables wind"
    "wind_snap_asus2as_vae_lat64_proj16_film_no_elev_no_static_wd|${SPEC_AS_US_JOINT}|${SPEC_AS_TEST}|--interpolation bilinear --tessera-injection film --vae-latents-path ${VAE_LATENTS_PATH_LAT64} --vae-latents-station-csv ${VAE_LATENTS_CSV} --vae-latents-drop-prob 0.0 --vae-latents-proj-dim 16 --no-elevation --no-static-fields --weight-decay 1e-4 --target-variables wind"
    "wind_snap_asus2as_vae_lat64_proj16_concat_no_elev_no_static_wd|${SPEC_AS_US_JOINT}|${SPEC_AS_TEST}|--interpolation bilinear --tessera-injection concat --vae-latents-path ${VAE_LATENTS_PATH_LAT64} --vae-latents-station-csv ${VAE_LATENTS_CSV} --vae-latents-drop-prob 0.0 --vae-latents-proj-dim 16 --no-elevation --no-static-fields --weight-decay 1e-4 --target-variables wind"

    # Single-source transfer to Asia: train on US only, test on Asia.
    "wind_snap_us2as_baseline_wd|${SPEC_US_ONLY}|${SPEC_AS_TEST}|--interpolation bilinear --weight-decay 1e-4 --target-variables wind"
    "wind_snap_us2as_vae_lat64_proj16_film_no_elev_no_static_wd|${SPEC_US_ONLY}|${SPEC_AS_TEST}|--interpolation bilinear --tessera-injection film --vae-latents-path ${VAE_LATENTS_PATH_LAT64} --vae-latents-station-csv ${VAE_LATENTS_CSV} --vae-latents-drop-prob 0.0 --vae-latents-proj-dim 16 --no-elevation --no-static-fields --weight-decay 1e-4 --target-variables wind"
    "wind_snap_us2as_vae_lat64_proj16_concat_no_elev_no_static_wd|${SPEC_US_ONLY}|${SPEC_AS_TEST}|--interpolation bilinear --tessera-injection concat --vae-latents-path ${VAE_LATENTS_PATH_LAT64} --vae-latents-station-csv ${VAE_LATENTS_CSV} --vae-latents-drop-prob 0.0 --vae-latents-proj-dim 16 --no-elevation --no-static-fields --weight-decay 1e-4 --target-variables wind"

    # Same-region Asia single-source: train on Asia only, test on Asia.
    # Asia's flat counterpart of what submit_snapshot.sh does for EU.
    # Completes the 3-way comparison row for Asia.
    "wind_snap_as_baseline_wd|${SPEC_AS_ONLY}|${SPEC_AS_TEST}|--interpolation bilinear --weight-decay 1e-4 --target-variables wind"
    "wind_snap_as_vae_lat64_proj16_film_no_elev_no_static_wd|${SPEC_AS_ONLY}|${SPEC_AS_TEST}|--interpolation bilinear --tessera-injection film --vae-latents-path ${VAE_LATENTS_PATH_LAT64} --vae-latents-station-csv ${VAE_LATENTS_CSV} --vae-latents-drop-prob 0.0 --vae-latents-proj-dim 16 --no-elevation --no-static-fields --weight-decay 1e-4 --target-variables wind"
    "wind_snap_as_vae_lat64_proj16_concat_no_elev_no_static_wd|${SPEC_AS_ONLY}|${SPEC_AS_TEST}|--interpolation bilinear --tessera-injection concat --vae-latents-path ${VAE_LATENTS_PATH_LAT64} --vae-latents-station-csv ${VAE_LATENTS_CSV} --vae-latents-drop-prob 0.0 --vae-latents-proj-dim 16 --no-elevation --no-static-fields --weight-decay 1e-4 --target-variables wind"

    # ==================================================================
    # T2M — test on EU
    # ==================================================================
    # NOTE: t2m VAE variants use proj16 + WITH elev + NO static (per
    # user specification for multi-region runs). This differs from the
    # same-region EU t2m config in submit_snapshot.sh which uses proj8.
    "t2m_snap_euus2eu_baseline_wd|${SPEC_EU_US_JOINT}|${SPEC_EU_TEST}|--interpolation bilinear --weight-decay 1e-4 --target-variables t2m"
    "t2m_snap_euus2eu_vae_lat64_proj16_film_with_elev_no_static_wd|${SPEC_EU_US_JOINT}|${SPEC_EU_TEST}|--interpolation bilinear --tessera-injection film --vae-latents-path ${VAE_LATENTS_PATH_LAT64} --vae-latents-station-csv ${VAE_LATENTS_CSV} --vae-latents-drop-prob 0.0 --vae-latents-proj-dim 16 --no-static-fields --weight-decay 1e-4 --target-variables t2m"
    "t2m_snap_euus2eu_vae_lat64_proj16_concat_with_elev_no_static_wd|${SPEC_EU_US_JOINT}|${SPEC_EU_TEST}|--interpolation bilinear --tessera-injection concat --vae-latents-path ${VAE_LATENTS_PATH_LAT64} --vae-latents-station-csv ${VAE_LATENTS_CSV} --vae-latents-drop-prob 0.0 --vae-latents-proj-dim 16 --no-static-fields --weight-decay 1e-4 --target-variables t2m"

    # US-to-EU transfer for t2m.
    "t2m_snap_us2eu_baseline_wd|${SPEC_US_ONLY}|${SPEC_EU_TEST}|--interpolation bilinear --weight-decay 1e-4 --target-variables t2m"
    "t2m_snap_us2eu_vae_lat64_proj16_film_with_elev_no_static_wd|${SPEC_US_ONLY}|${SPEC_EU_TEST}|--interpolation bilinear --tessera-injection film --vae-latents-path ${VAE_LATENTS_PATH_LAT64} --vae-latents-station-csv ${VAE_LATENTS_CSV} --vae-latents-drop-prob 0.0 --vae-latents-proj-dim 16 --no-static-fields --weight-decay 1e-4 --target-variables t2m"
    "t2m_snap_us2eu_vae_lat64_proj16_concat_with_elev_no_static_wd|${SPEC_US_ONLY}|${SPEC_EU_TEST}|--interpolation bilinear --tessera-injection concat --vae-latents-path ${VAE_LATENTS_PATH_LAT64} --vae-latents-station-csv ${VAE_LATENTS_CSV} --vae-latents-drop-prob 0.0 --vae-latents-proj-dim 16 --no-static-fields --weight-decay 1e-4 --target-variables t2m"

    # ==================================================================
    # T2M — test on Asia
    # ==================================================================
    # Joint-source t2m for Asia.
    "t2m_snap_asus2as_baseline_wd|${SPEC_AS_US_JOINT}|${SPEC_AS_TEST}|--interpolation bilinear --weight-decay 1e-4 --target-variables t2m"
    "t2m_snap_asus2as_vae_lat64_proj16_film_with_elev_no_static_wd|${SPEC_AS_US_JOINT}|${SPEC_AS_TEST}|--interpolation bilinear --tessera-injection film --vae-latents-path ${VAE_LATENTS_PATH_LAT64} --vae-latents-station-csv ${VAE_LATENTS_CSV} --vae-latents-drop-prob 0.0 --vae-latents-proj-dim 16 --no-static-fields --weight-decay 1e-4 --target-variables t2m"
    "t2m_snap_asus2as_vae_lat64_proj16_concat_with_elev_no_static_wd|${SPEC_AS_US_JOINT}|${SPEC_AS_TEST}|--interpolation bilinear --tessera-injection concat --vae-latents-path ${VAE_LATENTS_PATH_LAT64} --vae-latents-station-csv ${VAE_LATENTS_CSV} --vae-latents-drop-prob 0.0 --vae-latents-proj-dim 16 --no-static-fields --weight-decay 1e-4 --target-variables t2m"

    # US-to-Asia transfer for t2m.
    "t2m_snap_us2as_baseline_wd|${SPEC_US_ONLY}|${SPEC_AS_TEST}|--interpolation bilinear --weight-decay 1e-4 --target-variables t2m"
    "t2m_snap_us2as_vae_lat64_proj16_film_with_elev_no_static_wd|${SPEC_US_ONLY}|${SPEC_AS_TEST}|--interpolation bilinear --tessera-injection film --vae-latents-path ${VAE_LATENTS_PATH_LAT64} --vae-latents-station-csv ${VAE_LATENTS_CSV} --vae-latents-drop-prob 0.0 --vae-latents-proj-dim 16 --no-static-fields --weight-decay 1e-4 --target-variables t2m"
    "t2m_snap_us2as_vae_lat64_proj16_concat_with_elev_no_static_wd|${SPEC_US_ONLY}|${SPEC_AS_TEST}|--interpolation bilinear --tessera-injection concat --vae-latents-path ${VAE_LATENTS_PATH_LAT64} --vae-latents-station-csv ${VAE_LATENTS_CSV} --vae-latents-drop-prob 0.0 --vae-latents-proj-dim 16 --no-static-fields --weight-decay 1e-4 --target-variables t2m"

    # Same-region Asia single-source for t2m.
    "t2m_snap_as_baseline_wd|${SPEC_AS_ONLY}|${SPEC_AS_TEST}|--interpolation bilinear --weight-decay 1e-4 --target-variables t2m"
    "t2m_snap_as_vae_lat64_proj16_film_with_elev_no_static_wd|${SPEC_AS_ONLY}|${SPEC_AS_TEST}|--interpolation bilinear --tessera-injection film --vae-latents-path ${VAE_LATENTS_PATH_LAT64} --vae-latents-station-csv ${VAE_LATENTS_CSV} --vae-latents-drop-prob 0.0 --vae-latents-proj-dim 16 --no-static-fields --weight-decay 1e-4 --target-variables t2m"
    "t2m_snap_as_vae_lat64_proj16_concat_with_elev_no_static_wd|${SPEC_AS_ONLY}|${SPEC_AS_TEST}|--interpolation bilinear --tessera-injection concat --vae-latents-path ${VAE_LATENTS_PATH_LAT64} --vae-latents-station-csv ${VAE_LATENTS_CSV} --vae-latents-drop-prob 0.0 --vae-latents-proj-dim 16 --no-static-fields --weight-decay 1e-4 --target-variables t2m"
    # ##################################################################
    # BATCH 2 EXTENSIONS (added 2026-04-20)
    # ##################################################################

    # ==================================================================
    # GROUP A: EU→EU baseline reproducibility check (with elev + static,
    # ==================================================================
    # matches submit_snapshot.sh baselines so we can verify global dataset
    # produces numerically-equivalent results to the flat-EU pipeline.
    # Expected: t2m≈1.241, wind≈1.365 (from flat-EU).

    "wind_snap_eu2eu_baseline_repro_wd|${SPEC_EU_ONLY}|${SPEC_EU_TEST}|--interpolation bilinear --weight-decay 1e-4 --target-variables wind"
    "t2m_snap_eu2eu_baseline_repro_wd|${SPEC_EU_ONLY}|${SPEC_EU_TEST}|--interpolation bilinear --weight-decay 1e-4 --target-variables t2m"

    # ==================================================================
    # GROUP B: Elev-axis swap of existing lat64+proj16 VAE variants.
    # ==================================================================
    # Existing wind variants are no-elev; add with-elev. Existing t2m
    # variants are with-elev; add no-elev. Tests whether elevation helps
    # across regions / transfer scenarios at the lat64+proj16 scale.

    "wind_snap_euus2eu_vae_lat64_proj16_film_with_elev_no_static_wd|${SPEC_EU_US_JOINT}|${SPEC_EU_TEST}|--interpolation bilinear --tessera-injection film --vae-latents-path ${VAE_LATENTS_PATH_LAT64} --vae-latents-station-csv ${VAE_LATENTS_CSV} --vae-latents-drop-prob 0.0 --vae-latents-proj-dim 16 --no-static-fields --weight-decay 1e-4 --target-variables wind"
    "wind_snap_euus2eu_vae_lat64_proj16_concat_with_elev_no_static_wd|${SPEC_EU_US_JOINT}|${SPEC_EU_TEST}|--interpolation bilinear --tessera-injection concat --vae-latents-path ${VAE_LATENTS_PATH_LAT64} --vae-latents-station-csv ${VAE_LATENTS_CSV} --vae-latents-drop-prob 0.0 --vae-latents-proj-dim 16 --no-static-fields --weight-decay 1e-4 --target-variables wind"
    "t2m_snap_euus2eu_vae_lat64_proj16_film_no_elev_no_static_wd|${SPEC_EU_US_JOINT}|${SPEC_EU_TEST}|--interpolation bilinear --tessera-injection film --vae-latents-path ${VAE_LATENTS_PATH_LAT64} --vae-latents-station-csv ${VAE_LATENTS_CSV} --vae-latents-drop-prob 0.0 --vae-latents-proj-dim 16 --no-elevation --no-static-fields --weight-decay 1e-4 --target-variables t2m"
    "t2m_snap_euus2eu_vae_lat64_proj16_concat_no_elev_no_static_wd|${SPEC_EU_US_JOINT}|${SPEC_EU_TEST}|--interpolation bilinear --tessera-injection concat --vae-latents-path ${VAE_LATENTS_PATH_LAT64} --vae-latents-station-csv ${VAE_LATENTS_CSV} --vae-latents-drop-prob 0.0 --vae-latents-proj-dim 16 --no-elevation --no-static-fields --weight-decay 1e-4 --target-variables t2m"
    "wind_snap_us2eu_vae_lat64_proj16_film_with_elev_no_static_wd|${SPEC_US_ONLY}|${SPEC_EU_TEST}|--interpolation bilinear --tessera-injection film --vae-latents-path ${VAE_LATENTS_PATH_LAT64} --vae-latents-station-csv ${VAE_LATENTS_CSV} --vae-latents-drop-prob 0.0 --vae-latents-proj-dim 16 --no-static-fields --weight-decay 1e-4 --target-variables wind"
    "wind_snap_us2eu_vae_lat64_proj16_concat_with_elev_no_static_wd|${SPEC_US_ONLY}|${SPEC_EU_TEST}|--interpolation bilinear --tessera-injection concat --vae-latents-path ${VAE_LATENTS_PATH_LAT64} --vae-latents-station-csv ${VAE_LATENTS_CSV} --vae-latents-drop-prob 0.0 --vae-latents-proj-dim 16 --no-static-fields --weight-decay 1e-4 --target-variables wind"
    "t2m_snap_us2eu_vae_lat64_proj16_film_no_elev_no_static_wd|${SPEC_US_ONLY}|${SPEC_EU_TEST}|--interpolation bilinear --tessera-injection film --vae-latents-path ${VAE_LATENTS_PATH_LAT64} --vae-latents-station-csv ${VAE_LATENTS_CSV} --vae-latents-drop-prob 0.0 --vae-latents-proj-dim 16 --no-elevation --no-static-fields --weight-decay 1e-4 --target-variables t2m"
    "t2m_snap_us2eu_vae_lat64_proj16_concat_no_elev_no_static_wd|${SPEC_US_ONLY}|${SPEC_EU_TEST}|--interpolation bilinear --tessera-injection concat --vae-latents-path ${VAE_LATENTS_PATH_LAT64} --vae-latents-station-csv ${VAE_LATENTS_CSV} --vae-latents-drop-prob 0.0 --vae-latents-proj-dim 16 --no-elevation --no-static-fields --weight-decay 1e-4 --target-variables t2m"
    "wind_snap_asus2as_vae_lat64_proj16_film_with_elev_no_static_wd|${SPEC_AS_US_JOINT}|${SPEC_AS_TEST}|--interpolation bilinear --tessera-injection film --vae-latents-path ${VAE_LATENTS_PATH_LAT64} --vae-latents-station-csv ${VAE_LATENTS_CSV} --vae-latents-drop-prob 0.0 --vae-latents-proj-dim 16 --no-static-fields --weight-decay 1e-4 --target-variables wind"
    "wind_snap_asus2as_vae_lat64_proj16_concat_with_elev_no_static_wd|${SPEC_AS_US_JOINT}|${SPEC_AS_TEST}|--interpolation bilinear --tessera-injection concat --vae-latents-path ${VAE_LATENTS_PATH_LAT64} --vae-latents-station-csv ${VAE_LATENTS_CSV} --vae-latents-drop-prob 0.0 --vae-latents-proj-dim 16 --no-static-fields --weight-decay 1e-4 --target-variables wind"
    "t2m_snap_asus2as_vae_lat64_proj16_film_no_elev_no_static_wd|${SPEC_AS_US_JOINT}|${SPEC_AS_TEST}|--interpolation bilinear --tessera-injection film --vae-latents-path ${VAE_LATENTS_PATH_LAT64} --vae-latents-station-csv ${VAE_LATENTS_CSV} --vae-latents-drop-prob 0.0 --vae-latents-proj-dim 16 --no-elevation --no-static-fields --weight-decay 1e-4 --target-variables t2m"
    "t2m_snap_asus2as_vae_lat64_proj16_concat_no_elev_no_static_wd|${SPEC_AS_US_JOINT}|${SPEC_AS_TEST}|--interpolation bilinear --tessera-injection concat --vae-latents-path ${VAE_LATENTS_PATH_LAT64} --vae-latents-station-csv ${VAE_LATENTS_CSV} --vae-latents-drop-prob 0.0 --vae-latents-proj-dim 16 --no-elevation --no-static-fields --weight-decay 1e-4 --target-variables t2m"
    "wind_snap_us2as_vae_lat64_proj16_film_with_elev_no_static_wd|${SPEC_US_ONLY}|${SPEC_AS_TEST}|--interpolation bilinear --tessera-injection film --vae-latents-path ${VAE_LATENTS_PATH_LAT64} --vae-latents-station-csv ${VAE_LATENTS_CSV} --vae-latents-drop-prob 0.0 --vae-latents-proj-dim 16 --no-static-fields --weight-decay 1e-4 --target-variables wind"
    "wind_snap_us2as_vae_lat64_proj16_concat_with_elev_no_static_wd|${SPEC_US_ONLY}|${SPEC_AS_TEST}|--interpolation bilinear --tessera-injection concat --vae-latents-path ${VAE_LATENTS_PATH_LAT64} --vae-latents-station-csv ${VAE_LATENTS_CSV} --vae-latents-drop-prob 0.0 --vae-latents-proj-dim 16 --no-static-fields --weight-decay 1e-4 --target-variables wind"
    "t2m_snap_us2as_vae_lat64_proj16_film_no_elev_no_static_wd|${SPEC_US_ONLY}|${SPEC_AS_TEST}|--interpolation bilinear --tessera-injection film --vae-latents-path ${VAE_LATENTS_PATH_LAT64} --vae-latents-station-csv ${VAE_LATENTS_CSV} --vae-latents-drop-prob 0.0 --vae-latents-proj-dim 16 --no-elevation --no-static-fields --weight-decay 1e-4 --target-variables t2m"
    "t2m_snap_us2as_vae_lat64_proj16_concat_no_elev_no_static_wd|${SPEC_US_ONLY}|${SPEC_AS_TEST}|--interpolation bilinear --tessera-injection concat --vae-latents-path ${VAE_LATENTS_PATH_LAT64} --vae-latents-station-csv ${VAE_LATENTS_CSV} --vae-latents-drop-prob 0.0 --vae-latents-proj-dim 16 --no-elevation --no-static-fields --weight-decay 1e-4 --target-variables t2m"
    "wind_snap_as_vae_lat64_proj16_film_with_elev_no_static_wd|${SPEC_AS_ONLY}|${SPEC_AS_TEST}|--interpolation bilinear --tessera-injection film --vae-latents-path ${VAE_LATENTS_PATH_LAT64} --vae-latents-station-csv ${VAE_LATENTS_CSV} --vae-latents-drop-prob 0.0 --vae-latents-proj-dim 16 --no-static-fields --weight-decay 1e-4 --target-variables wind"
    "wind_snap_as_vae_lat64_proj16_concat_with_elev_no_static_wd|${SPEC_AS_ONLY}|${SPEC_AS_TEST}|--interpolation bilinear --tessera-injection concat --vae-latents-path ${VAE_LATENTS_PATH_LAT64} --vae-latents-station-csv ${VAE_LATENTS_CSV} --vae-latents-drop-prob 0.0 --vae-latents-proj-dim 16 --no-static-fields --weight-decay 1e-4 --target-variables wind"
    "t2m_snap_as_vae_lat64_proj16_film_no_elev_no_static_wd|${SPEC_AS_ONLY}|${SPEC_AS_TEST}|--interpolation bilinear --tessera-injection film --vae-latents-path ${VAE_LATENTS_PATH_LAT64} --vae-latents-station-csv ${VAE_LATENTS_CSV} --vae-latents-drop-prob 0.0 --vae-latents-proj-dim 16 --no-elevation --no-static-fields --weight-decay 1e-4 --target-variables t2m"
    "t2m_snap_as_vae_lat64_proj16_concat_no_elev_no_static_wd|${SPEC_AS_ONLY}|${SPEC_AS_TEST}|--interpolation bilinear --tessera-injection concat --vae-latents-path ${VAE_LATENTS_PATH_LAT64} --vae-latents-station-csv ${VAE_LATENTS_CSV} --vae-latents-drop-prob 0.0 --vae-latents-proj-dim 16 --no-elevation --no-static-fields --weight-decay 1e-4 --target-variables t2m"

    # ==================================================================
    # GROUP C: lat16 variants (both elev directions × both injections).
    # ==================================================================
    # Tests whether the current flat-EU wind winner (lat16+concat+with_elev,
    # MAE 1.264) holds up in multi-region. 5 pairs × 4 configs × 2 targets.
    # Uses raw 16-d VAE latent (no projection head) since lat16 is already
    # small enough that projection isn't obviously useful.

    "wind_snap_euus2eu_vae_lat16_film_with_elev_no_static_wd|${SPEC_EU_US_JOINT}|${SPEC_EU_TEST}|--interpolation bilinear --tessera-injection film --vae-latents-path ${VAE_LATENTS_PATH_LAT16} --vae-latents-station-csv ${VAE_LATENTS_CSV} --vae-latents-drop-prob 0.0 --no-static-fields --weight-decay 1e-4 --target-variables wind"
    "wind_snap_euus2eu_vae_lat16_film_no_elev_no_static_wd|${SPEC_EU_US_JOINT}|${SPEC_EU_TEST}|--interpolation bilinear --tessera-injection film --vae-latents-path ${VAE_LATENTS_PATH_LAT16} --vae-latents-station-csv ${VAE_LATENTS_CSV} --vae-latents-drop-prob 0.0 --no-elevation --no-static-fields --weight-decay 1e-4 --target-variables wind"
    "wind_snap_euus2eu_vae_lat16_concat_with_elev_no_static_wd|${SPEC_EU_US_JOINT}|${SPEC_EU_TEST}|--interpolation bilinear --tessera-injection concat --vae-latents-path ${VAE_LATENTS_PATH_LAT16} --vae-latents-station-csv ${VAE_LATENTS_CSV} --vae-latents-drop-prob 0.0 --no-static-fields --weight-decay 1e-4 --target-variables wind"
    "wind_snap_euus2eu_vae_lat16_concat_no_elev_no_static_wd|${SPEC_EU_US_JOINT}|${SPEC_EU_TEST}|--interpolation bilinear --tessera-injection concat --vae-latents-path ${VAE_LATENTS_PATH_LAT16} --vae-latents-station-csv ${VAE_LATENTS_CSV} --vae-latents-drop-prob 0.0 --no-elevation --no-static-fields --weight-decay 1e-4 --target-variables wind"
    "t2m_snap_euus2eu_vae_lat16_film_with_elev_no_static_wd|${SPEC_EU_US_JOINT}|${SPEC_EU_TEST}|--interpolation bilinear --tessera-injection film --vae-latents-path ${VAE_LATENTS_PATH_LAT16} --vae-latents-station-csv ${VAE_LATENTS_CSV} --vae-latents-drop-prob 0.0 --no-static-fields --weight-decay 1e-4 --target-variables t2m"
    "t2m_snap_euus2eu_vae_lat16_film_no_elev_no_static_wd|${SPEC_EU_US_JOINT}|${SPEC_EU_TEST}|--interpolation bilinear --tessera-injection film --vae-latents-path ${VAE_LATENTS_PATH_LAT16} --vae-latents-station-csv ${VAE_LATENTS_CSV} --vae-latents-drop-prob 0.0 --no-elevation --no-static-fields --weight-decay 1e-4 --target-variables t2m"
    "t2m_snap_euus2eu_vae_lat16_concat_with_elev_no_static_wd|${SPEC_EU_US_JOINT}|${SPEC_EU_TEST}|--interpolation bilinear --tessera-injection concat --vae-latents-path ${VAE_LATENTS_PATH_LAT16} --vae-latents-station-csv ${VAE_LATENTS_CSV} --vae-latents-drop-prob 0.0 --no-static-fields --weight-decay 1e-4 --target-variables t2m"
    "t2m_snap_euus2eu_vae_lat16_concat_no_elev_no_static_wd|${SPEC_EU_US_JOINT}|${SPEC_EU_TEST}|--interpolation bilinear --tessera-injection concat --vae-latents-path ${VAE_LATENTS_PATH_LAT16} --vae-latents-station-csv ${VAE_LATENTS_CSV} --vae-latents-drop-prob 0.0 --no-elevation --no-static-fields --weight-decay 1e-4 --target-variables t2m"
    "wind_snap_us2eu_vae_lat16_film_with_elev_no_static_wd|${SPEC_US_ONLY}|${SPEC_EU_TEST}|--interpolation bilinear --tessera-injection film --vae-latents-path ${VAE_LATENTS_PATH_LAT16} --vae-latents-station-csv ${VAE_LATENTS_CSV} --vae-latents-drop-prob 0.0 --no-static-fields --weight-decay 1e-4 --target-variables wind"
    "wind_snap_us2eu_vae_lat16_film_no_elev_no_static_wd|${SPEC_US_ONLY}|${SPEC_EU_TEST}|--interpolation bilinear --tessera-injection film --vae-latents-path ${VAE_LATENTS_PATH_LAT16} --vae-latents-station-csv ${VAE_LATENTS_CSV} --vae-latents-drop-prob 0.0 --no-elevation --no-static-fields --weight-decay 1e-4 --target-variables wind"
    "wind_snap_us2eu_vae_lat16_concat_with_elev_no_static_wd|${SPEC_US_ONLY}|${SPEC_EU_TEST}|--interpolation bilinear --tessera-injection concat --vae-latents-path ${VAE_LATENTS_PATH_LAT16} --vae-latents-station-csv ${VAE_LATENTS_CSV} --vae-latents-drop-prob 0.0 --no-static-fields --weight-decay 1e-4 --target-variables wind"
    "wind_snap_us2eu_vae_lat16_concat_no_elev_no_static_wd|${SPEC_US_ONLY}|${SPEC_EU_TEST}|--interpolation bilinear --tessera-injection concat --vae-latents-path ${VAE_LATENTS_PATH_LAT16} --vae-latents-station-csv ${VAE_LATENTS_CSV} --vae-latents-drop-prob 0.0 --no-elevation --no-static-fields --weight-decay 1e-4 --target-variables wind"
    "t2m_snap_us2eu_vae_lat16_film_with_elev_no_static_wd|${SPEC_US_ONLY}|${SPEC_EU_TEST}|--interpolation bilinear --tessera-injection film --vae-latents-path ${VAE_LATENTS_PATH_LAT16} --vae-latents-station-csv ${VAE_LATENTS_CSV} --vae-latents-drop-prob 0.0 --no-static-fields --weight-decay 1e-4 --target-variables t2m"
    "t2m_snap_us2eu_vae_lat16_film_no_elev_no_static_wd|${SPEC_US_ONLY}|${SPEC_EU_TEST}|--interpolation bilinear --tessera-injection film --vae-latents-path ${VAE_LATENTS_PATH_LAT16} --vae-latents-station-csv ${VAE_LATENTS_CSV} --vae-latents-drop-prob 0.0 --no-elevation --no-static-fields --weight-decay 1e-4 --target-variables t2m"
    "t2m_snap_us2eu_vae_lat16_concat_with_elev_no_static_wd|${SPEC_US_ONLY}|${SPEC_EU_TEST}|--interpolation bilinear --tessera-injection concat --vae-latents-path ${VAE_LATENTS_PATH_LAT16} --vae-latents-station-csv ${VAE_LATENTS_CSV} --vae-latents-drop-prob 0.0 --no-static-fields --weight-decay 1e-4 --target-variables t2m"
    "t2m_snap_us2eu_vae_lat16_concat_no_elev_no_static_wd|${SPEC_US_ONLY}|${SPEC_EU_TEST}|--interpolation bilinear --tessera-injection concat --vae-latents-path ${VAE_LATENTS_PATH_LAT16} --vae-latents-station-csv ${VAE_LATENTS_CSV} --vae-latents-drop-prob 0.0 --no-elevation --no-static-fields --weight-decay 1e-4 --target-variables t2m"
    "wind_snap_asus2as_vae_lat16_film_with_elev_no_static_wd|${SPEC_AS_US_JOINT}|${SPEC_AS_TEST}|--interpolation bilinear --tessera-injection film --vae-latents-path ${VAE_LATENTS_PATH_LAT16} --vae-latents-station-csv ${VAE_LATENTS_CSV} --vae-latents-drop-prob 0.0 --no-static-fields --weight-decay 1e-4 --target-variables wind"
    "wind_snap_asus2as_vae_lat16_film_no_elev_no_static_wd|${SPEC_AS_US_JOINT}|${SPEC_AS_TEST}|--interpolation bilinear --tessera-injection film --vae-latents-path ${VAE_LATENTS_PATH_LAT16} --vae-latents-station-csv ${VAE_LATENTS_CSV} --vae-latents-drop-prob 0.0 --no-elevation --no-static-fields --weight-decay 1e-4 --target-variables wind"
    "wind_snap_asus2as_vae_lat16_concat_with_elev_no_static_wd|${SPEC_AS_US_JOINT}|${SPEC_AS_TEST}|--interpolation bilinear --tessera-injection concat --vae-latents-path ${VAE_LATENTS_PATH_LAT16} --vae-latents-station-csv ${VAE_LATENTS_CSV} --vae-latents-drop-prob 0.0 --no-static-fields --weight-decay 1e-4 --target-variables wind"
    "wind_snap_asus2as_vae_lat16_concat_no_elev_no_static_wd|${SPEC_AS_US_JOINT}|${SPEC_AS_TEST}|--interpolation bilinear --tessera-injection concat --vae-latents-path ${VAE_LATENTS_PATH_LAT16} --vae-latents-station-csv ${VAE_LATENTS_CSV} --vae-latents-drop-prob 0.0 --no-elevation --no-static-fields --weight-decay 1e-4 --target-variables wind"
    "t2m_snap_asus2as_vae_lat16_film_with_elev_no_static_wd|${SPEC_AS_US_JOINT}|${SPEC_AS_TEST}|--interpolation bilinear --tessera-injection film --vae-latents-path ${VAE_LATENTS_PATH_LAT16} --vae-latents-station-csv ${VAE_LATENTS_CSV} --vae-latents-drop-prob 0.0 --no-static-fields --weight-decay 1e-4 --target-variables t2m"
    "t2m_snap_asus2as_vae_lat16_film_no_elev_no_static_wd|${SPEC_AS_US_JOINT}|${SPEC_AS_TEST}|--interpolation bilinear --tessera-injection film --vae-latents-path ${VAE_LATENTS_PATH_LAT16} --vae-latents-station-csv ${VAE_LATENTS_CSV} --vae-latents-drop-prob 0.0 --no-elevation --no-static-fields --weight-decay 1e-4 --target-variables t2m"
    "t2m_snap_asus2as_vae_lat16_concat_with_elev_no_static_wd|${SPEC_AS_US_JOINT}|${SPEC_AS_TEST}|--interpolation bilinear --tessera-injection concat --vae-latents-path ${VAE_LATENTS_PATH_LAT16} --vae-latents-station-csv ${VAE_LATENTS_CSV} --vae-latents-drop-prob 0.0 --no-static-fields --weight-decay 1e-4 --target-variables t2m"
    "t2m_snap_asus2as_vae_lat16_concat_no_elev_no_static_wd|${SPEC_AS_US_JOINT}|${SPEC_AS_TEST}|--interpolation bilinear --tessera-injection concat --vae-latents-path ${VAE_LATENTS_PATH_LAT16} --vae-latents-station-csv ${VAE_LATENTS_CSV} --vae-latents-drop-prob 0.0 --no-elevation --no-static-fields --weight-decay 1e-4 --target-variables t2m"
    "wind_snap_us2as_vae_lat16_film_with_elev_no_static_wd|${SPEC_US_ONLY}|${SPEC_AS_TEST}|--interpolation bilinear --tessera-injection film --vae-latents-path ${VAE_LATENTS_PATH_LAT16} --vae-latents-station-csv ${VAE_LATENTS_CSV} --vae-latents-drop-prob 0.0 --no-static-fields --weight-decay 1e-4 --target-variables wind"
    "wind_snap_us2as_vae_lat16_film_no_elev_no_static_wd|${SPEC_US_ONLY}|${SPEC_AS_TEST}|--interpolation bilinear --tessera-injection film --vae-latents-path ${VAE_LATENTS_PATH_LAT16} --vae-latents-station-csv ${VAE_LATENTS_CSV} --vae-latents-drop-prob 0.0 --no-elevation --no-static-fields --weight-decay 1e-4 --target-variables wind"
    "wind_snap_us2as_vae_lat16_concat_with_elev_no_static_wd|${SPEC_US_ONLY}|${SPEC_AS_TEST}|--interpolation bilinear --tessera-injection concat --vae-latents-path ${VAE_LATENTS_PATH_LAT16} --vae-latents-station-csv ${VAE_LATENTS_CSV} --vae-latents-drop-prob 0.0 --no-static-fields --weight-decay 1e-4 --target-variables wind"
    "wind_snap_us2as_vae_lat16_concat_no_elev_no_static_wd|${SPEC_US_ONLY}|${SPEC_AS_TEST}|--interpolation bilinear --tessera-injection concat --vae-latents-path ${VAE_LATENTS_PATH_LAT16} --vae-latents-station-csv ${VAE_LATENTS_CSV} --vae-latents-drop-prob 0.0 --no-elevation --no-static-fields --weight-decay 1e-4 --target-variables wind"
    "t2m_snap_us2as_vae_lat16_film_with_elev_no_static_wd|${SPEC_US_ONLY}|${SPEC_AS_TEST}|--interpolation bilinear --tessera-injection film --vae-latents-path ${VAE_LATENTS_PATH_LAT16} --vae-latents-station-csv ${VAE_LATENTS_CSV} --vae-latents-drop-prob 0.0 --no-static-fields --weight-decay 1e-4 --target-variables t2m"
    "t2m_snap_us2as_vae_lat16_film_no_elev_no_static_wd|${SPEC_US_ONLY}|${SPEC_AS_TEST}|--interpolation bilinear --tessera-injection film --vae-latents-path ${VAE_LATENTS_PATH_LAT16} --vae-latents-station-csv ${VAE_LATENTS_CSV} --vae-latents-drop-prob 0.0 --no-elevation --no-static-fields --weight-decay 1e-4 --target-variables t2m"
    "t2m_snap_us2as_vae_lat16_concat_with_elev_no_static_wd|${SPEC_US_ONLY}|${SPEC_AS_TEST}|--interpolation bilinear --tessera-injection concat --vae-latents-path ${VAE_LATENTS_PATH_LAT16} --vae-latents-station-csv ${VAE_LATENTS_CSV} --vae-latents-drop-prob 0.0 --no-static-fields --weight-decay 1e-4 --target-variables t2m"
    "t2m_snap_us2as_vae_lat16_concat_no_elev_no_static_wd|${SPEC_US_ONLY}|${SPEC_AS_TEST}|--interpolation bilinear --tessera-injection concat --vae-latents-path ${VAE_LATENTS_PATH_LAT16} --vae-latents-station-csv ${VAE_LATENTS_CSV} --vae-latents-drop-prob 0.0 --no-elevation --no-static-fields --weight-decay 1e-4 --target-variables t2m"
    "wind_snap_as_vae_lat16_film_with_elev_no_static_wd|${SPEC_AS_ONLY}|${SPEC_AS_TEST}|--interpolation bilinear --tessera-injection film --vae-latents-path ${VAE_LATENTS_PATH_LAT16} --vae-latents-station-csv ${VAE_LATENTS_CSV} --vae-latents-drop-prob 0.0 --no-static-fields --weight-decay 1e-4 --target-variables wind"
    "wind_snap_as_vae_lat16_film_no_elev_no_static_wd|${SPEC_AS_ONLY}|${SPEC_AS_TEST}|--interpolation bilinear --tessera-injection film --vae-latents-path ${VAE_LATENTS_PATH_LAT16} --vae-latents-station-csv ${VAE_LATENTS_CSV} --vae-latents-drop-prob 0.0 --no-elevation --no-static-fields --weight-decay 1e-4 --target-variables wind"
    "wind_snap_as_vae_lat16_concat_with_elev_no_static_wd|${SPEC_AS_ONLY}|${SPEC_AS_TEST}|--interpolation bilinear --tessera-injection concat --vae-latents-path ${VAE_LATENTS_PATH_LAT16} --vae-latents-station-csv ${VAE_LATENTS_CSV} --vae-latents-drop-prob 0.0 --no-static-fields --weight-decay 1e-4 --target-variables wind"
    "wind_snap_as_vae_lat16_concat_no_elev_no_static_wd|${SPEC_AS_ONLY}|${SPEC_AS_TEST}|--interpolation bilinear --tessera-injection concat --vae-latents-path ${VAE_LATENTS_PATH_LAT16} --vae-latents-station-csv ${VAE_LATENTS_CSV} --vae-latents-drop-prob 0.0 --no-elevation --no-static-fields --weight-decay 1e-4 --target-variables wind"
    "t2m_snap_as_vae_lat16_film_with_elev_no_static_wd|${SPEC_AS_ONLY}|${SPEC_AS_TEST}|--interpolation bilinear --tessera-injection film --vae-latents-path ${VAE_LATENTS_PATH_LAT16} --vae-latents-station-csv ${VAE_LATENTS_CSV} --vae-latents-drop-prob 0.0 --no-static-fields --weight-decay 1e-4 --target-variables t2m"
    "t2m_snap_as_vae_lat16_film_no_elev_no_static_wd|${SPEC_AS_ONLY}|${SPEC_AS_TEST}|--interpolation bilinear --tessera-injection film --vae-latents-path ${VAE_LATENTS_PATH_LAT16} --vae-latents-station-csv ${VAE_LATENTS_CSV} --vae-latents-drop-prob 0.0 --no-elevation --no-static-fields --weight-decay 1e-4 --target-variables t2m"
    "t2m_snap_as_vae_lat16_concat_with_elev_no_static_wd|${SPEC_AS_ONLY}|${SPEC_AS_TEST}|--interpolation bilinear --tessera-injection concat --vae-latents-path ${VAE_LATENTS_PATH_LAT16} --vae-latents-station-csv ${VAE_LATENTS_CSV} --vae-latents-drop-prob 0.0 --no-static-fields --weight-decay 1e-4 --target-variables t2m"
    "t2m_snap_as_vae_lat16_concat_no_elev_no_static_wd|${SPEC_AS_ONLY}|${SPEC_AS_TEST}|--interpolation bilinear --tessera-injection concat --vae-latents-path ${VAE_LATENTS_PATH_LAT16} --vae-latents-station-csv ${VAE_LATENTS_CSV} --vae-latents-drop-prob 0.0 --no-elevation --no-static-fields --weight-decay 1e-4 --target-variables t2m"
)

# ---- Create log directory ----
mkdir -p "${REPO_ROOT}/logs"
mkdir -p "${OUTPUT_ROOT}"

# ---- Submit jobs ----
echo "============================================"
echo "Submitting MULTI-REGION SNAPSHOT experiment jobs"
echo "============================================"
echo ""
echo "Dataset:    ${DATASET_DIR}"
echo "Output:     ${OUTPUT_ROOT}"
echo "Batch size: ${BATCH_SIZE}"
echo "Epochs:     ${EPOCHS}"
echo "Patience:   ${PATIENCE}"
echo "Seeds:      ${SEEDS[*]}"
echo "Configs:    ${#EXPERIMENTS[@]}"
echo "Total jobs: $(( ${#EXPERIMENTS[@]} * ${#SEEDS[@]} ))"
echo ""

# --- Preflight: confirm the multi-region snapshot dataset exists ---
if [ ! -f "${DATASET_DIR}/metadata.json" ]; then
    echo "ERROR: ${DATASET_DIR}/metadata.json does not exist." >&2
    echo "Run preprocess_timestamp_global.py before submitting these jobs." >&2
    exit 1
fi

LAYOUT_VERSION=$(
    python3 -c "import json; print(json.load(open('${DATASET_DIR}/metadata.json')).get('layout_version', ''))"
)
if [ "${LAYOUT_VERSION}" != "multi_region_snapshot_v1" ]; then
    echo "ERROR: ${DATASET_DIR}/metadata.json has layout_version='${LAYOUT_VERSION}'," >&2
    echo "expected 'multi_region_snapshot_v1'. Is that actually a" >&2
    echo "multi-region snapshot preprocessing output?" >&2
    exit 1
fi

COMMON_ARGS="--dataset-dir ${DATASET_DIR} --tessera-path ${TESSERA_PATH} --tessera-station-csv ${TESSERA_CSV} --batch-size ${BATCH_SIZE} --epochs ${EPOCHS} --patience ${PATIENCE} --lr ${LR} --cnn-hidden ${CNN_HIDDEN} --cnn-layers ${CNN_LAYERS} --mlp-hidden ${MLP_HIDDEN} --mlp-n-hidden ${MLP_N_HIDDEN} --num-workers ${NUM_WORKERS} --normalisation-policy global --lr-warmup-pct 0.05"

JOB_COUNT=0
for experiment in "${EXPERIMENTS[@]}"; do
    # Entry format:
    #   name | train_specs_json | test_specs_json | other_args
    # Splitting on '|' keeps each field intact.
    IFS='|' read -r name train_specs_json test_specs_json extra_args <<< "${experiment}"

    for seed in "${SEEDS[@]}"; do
        run_dir="${OUTPUT_ROOT}/${name}_seed${seed}"
        job_name="${name}_s${seed}"

        # Skip if already completed.
        if [ -f "${run_dir}/test_summary.json" ]; then
            echo "SKIP: ${job_name} (already complete)"
            continue
        fi

        # Write the region-specs JSONs to files in the run dir (the
        # run dir is created now if needed — training will also
        # create it but that's a no-op). Using files avoids shell
        # quoting gymnastics through sbatch --wrap, which was a
        # source of persistent bugs with inline JSON.
        mkdir -p "${run_dir}"
        printf '%s' "${train_specs_json}" > "${run_dir}/region_specs_train.json"
        printf '%s' "${test_specs_json}" > "${run_dir}/region_specs_test.json"

        TRAIN_CMD="${REPO_ROOT}/.venv/bin/python ${TRAIN_SCRIPT} ${COMMON_ARGS} --region-specs-train-file ${run_dir}/region_specs_train.json ${extra_args} --seed ${seed} --output-dir ${run_dir}"
        EVAL_CMD="${REPO_ROOT}/.venv/bin/python ${EVAL_SCRIPT} --checkpoint ${run_dir}/best_model.pt --batch-size ${BATCH_SIZE} --num-workers ${NUM_WORKERS} --region-specs-test-file ${run_dir}/region_specs_test.json"

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
echo "Submitted ${JOB_COUNT} multi-region snapshot jobs"
echo "============================================"
echo ""
echo "Monitor with:  squeue --me"
echo "Cancel all:    scancel --me"
echo "Results in:    ${OUTPUT_ROOT}/"