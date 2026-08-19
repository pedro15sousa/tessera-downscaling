#!/usr/bin/env bash
# Submit the paper's TESSERA arm (1B-M 2017 crop64_lat16_auxon latents) of the
# Norway station-network rollout. Identical design and schedule files to
# snapshot_14y_eu_temporal_rollout_norway_lat16_mtpi (same probe set,
# byte-identical rollout_schedule.json, same architectures and
# hyperparameters); only VAE_LATENTS_PATH differs. The no-TESSERA baseline and
# the ERA5-interp references are latents-independent and are NOT re-run here.
# Runs land in
# <data root>/training_runs_snapshot_14y_eu_temporal_rollout_norway_tessera_1B-M_2017/
# <arch>_<sweep>_seed<S>.
#
# Usage (from anywhere; data root from $TESSERA_DATA_ROOT):
#   bash scripts/experiments/snapshot_14y_eu_temporal_rollout_norway_tessera_1B-M_2017/submit.sh
#   DRY_RUN=1 ... (print the commands only)   LOCAL=1 ... (run in this shell)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FOLDER="$(basename "${SCRIPT_DIR}")"
source "${SCRIPT_DIR}/../_lib.sh"

JOB_TAG="roll1BM17_"
export TESSERA_VARIANT="1B-M"
export EMBED_YEAR="2017"
export VAE_LATENTS_PATH="${VAE_LATENTS_PATH_1BM}"

run_rollout_matrix
