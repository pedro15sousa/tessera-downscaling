#!/usr/bin/env bash
# Submit the Norway station-network rollout (Fig 7 of the preprint / Fig 9 of
# the AMS draft), baseline arms: the no-TESSERA ConvCNP (+mTPI), the v1 TESSERA
# arm, and the ERA5-interp references -- see experiments.yaml. The paper's
# TESSERA arm with the 1B-M 2017 latents is the sibling folder
# snapshot_14y_eu_temporal_rollout_norway_tessera_1B-M_2017 (same schedule).
#
# Every (architecture, sweep point, seed) is one GPU job: tessera-train on
# Europe with the 1,505 probe stations hidden before their activation date
# (probe_active_from_<sweep>.json) and the training window cut at the sweep
# point (train_end_overrides.json), then tessera-evaluate on all European
# stations (region_specs_test.json). Both sidecars are materialised from
# rollout_schedule.json on every run; the references run once (seed 42).
# Runs land in
# <data root>/training_runs/snapshot_14y_eu_temporal_rollout_norway_lat16_mtpi/
# <arch>_<sweep>_seed<S> and <reference>_seed42.
#
# Sidecars (built once, committed): probe_station_ids.json from
# scripts/experiments/pick_probe_set.py and rollout_schedule.json from
# scripts/experiments/build_rollout_schedule.py.
#
# Usage (from anywhere; data root from $TESSERA_DATA_ROOT):
#   bash scripts/experiments/snapshot_14y_eu_temporal_rollout_norway_lat16_mtpi/submit.sh
#   DRY_RUN=1 ... (print the commands only)   LOCAL=1 ... (run in this shell)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FOLDER="$(basename "${SCRIPT_DIR}")"
source "${SCRIPT_DIR}/../_lib.sh"

JOB_TAG="roll_"
# The v1 TESSERA arm of this folder reads the v1 lat16 latents.
export VAE_LATENTS_PATH="${VAE_LATENTS_PATH_V1}"

run_rollout_matrix
