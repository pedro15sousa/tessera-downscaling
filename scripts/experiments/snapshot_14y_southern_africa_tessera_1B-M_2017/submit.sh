#!/usr/bin/env bash
# Submit the TESSERA 1B-M (2017 embeddings) arm for region "southern_africa": the
# paper's canonical TESSERA runs (crop64_lat16_auxon) plus the VAE-variant
# model-selection sweep and the extradesc / summary-stats controls -- see
# experiments.yaml. Same dataset, region, normalisation and hyperparameters as
# the baseline folder snapshot_14y_southern_africa; runs land in
# <data root>/training_runs/snapshot_14y_southern_africa_tessera_1B-M_2017/<name>_seed<S>.
#
# Usage (from anywhere; data root from $TESSERA_DATA_ROOT):
#   bash scripts/experiments/snapshot_14y_southern_africa_tessera_1B-M_2017/submit.sh
#   DRY_RUN=1 bash scripts/experiments/snapshot_14y_southern_africa_tessera_1B-M_2017/submit.sh    # print the commands only
#   LOCAL=1   bash scripts/experiments/snapshot_14y_southern_africa_tessera_1B-M_2017/submit.sh    # run sequentially in this shell
# Entries whose latents file is not on disk yet are skipped (re-run later).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FOLDER="$(basename "${SCRIPT_DIR}")"
source "${SCRIPT_DIR}/../_lib.sh"

REGION="southern_africa"
JOB_TAG="sa1BM17_"
# Folder-level axes, expanded inside experiments.yaml (entry names carry only
# crop / latent dim / aux).
export TESSERA_VARIANT="1B-M"
export EMBED_YEAR="2017"
export VAE_LATENTS_DIR="${BASE_DIR}/processed/vae_tessera_${TESSERA_VARIANT}"
# Patch summary-statistics control (Appendix A): 16 hand-crafted stats over the
# same crop64 window of the same patches (scripts/data/build_summary_stats_latents.py).
export SUMMARY_STATS_PATH="${BASE_DIR}/processed/station_vectors/station_summary_stats_${TESSERA_VARIANT}_p128_${EMBED_YEAR}_crop64_dim16.npy"

run_single_region_matrix "${REGION}"
