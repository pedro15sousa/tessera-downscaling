#!/usr/bin/env bash
# Submit the shuffled-latent control for region "europe" (Table 4 / App. C):
# the canonical 1B-M 2017 arm with the station->latent assignment permuted
# (scripts/data/shuffle_latents.py --seed 0). Compare against
# training_runs/snapshot_14y_eu_tessera_1B-M_2017; runs land in
# <data root>/training_runs/snapshot_14y_eu_tessera_1B-M_2017_shuffled/<name>_seed<S>.
#
# Usage (from anywhere; data root from $TESSERA_DATA_ROOT):
#   bash scripts/experiments/snapshot_14y_eu_tessera_1B-M_2017_shuffled/submit.sh
#   DRY_RUN=1 bash scripts/experiments/snapshot_14y_eu_tessera_1B-M_2017_shuffled/submit.sh    # print the commands only
#   LOCAL=1   bash scripts/experiments/snapshot_14y_eu_tessera_1B-M_2017_shuffled/submit.sh    # run sequentially in this shell
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FOLDER="$(basename "${SCRIPT_DIR}")"
source "${SCRIPT_DIR}/../_lib.sh"

REGION="europe"
JOB_TAG="eu1BM17sh_"
export TESSERA_VARIANT="1B-M"
export EMBED_YEAR="2017"
export VAE_LATENTS_DIR="${BASE_DIR}/processed/vae_tessera_${TESSERA_VARIANT}"

run_single_region_matrix "${REGION}"
