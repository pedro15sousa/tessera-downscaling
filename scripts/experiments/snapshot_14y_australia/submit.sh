#!/usr/bin/env bash
# Submit the baseline arms for region "australia" of the paper: no-model references
# (persistence, ERA5 interpolation), the no-TESSERA ConvCNP (+mTPI), the same
# with hand-crafted extra descriptors, and the v1 TESSERA arm -- see
# experiments.yaml. Trained on dataset_timestamp_global with
# --train-regions australia (per-region normalisation stats); runs land in
# <data root>/training_runs_snapshot_14y_australia/<name>_seed<S>.
#
# Usage (from anywhere; data root from $TESSERA_DATA_ROOT):
#   bash scripts/experiments/snapshot_14y_australia/submit.sh              # sbatch one job per (entry, seed)
#   DRY_RUN=1 bash scripts/experiments/snapshot_14y_australia/submit.sh    # print the commands only
#   LOCAL=1   bash scripts/experiments/snapshot_14y_australia/submit.sh    # run sequentially in this shell
# Trained entries: one GPU job = tessera-train then tessera-evaluate on the same
# region. baseline_kind entries: one CPU job running tessera-baselines on the
# same TESSERA-filtered station set. Completed runs are skipped.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FOLDER="$(basename "${SCRIPT_DIR}")"
source "${SCRIPT_DIR}/../_lib.sh"

REGION="australia"
JOB_TAG="au_"
# The v1 TESSERA arm of this folder reads the v1 lat16 latents.
export VAE_LATENTS_PATH="${VAE_LATENTS_PATH_V1}"

run_single_region_matrix "${REGION}"
