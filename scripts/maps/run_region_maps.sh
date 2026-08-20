#!/usr/bin/env bash
# Build the dense-downscaling figure inputs for one region (maps, per-station
# evaluation, summary tables) plus the region-overview figure.
#
#   REGION=norway bash scripts/maps/run_region_maps.sh
#
# Dates/paths/grid come from regions.py (per-region `dates`, picked by
# select_dates.py). Outputs go to
#   ${TESSERA_MAPS_OUT:-$DATA_ROOT/paper_figure_outputs/maps_outputs}/<region>/<var>_<ts>/
# which is where scripts/paper/make_paper_figures.py reads them from.
# Needs internet for the SRTM DEM (first run only; cached under processed/dem_cache).
#
# Prerequisite: the dense TESSERA latent grid
#   $DATA_ROOT/processed/dense/<region>/<region>_0.05deg_2024.npz
# If it is missing, the raw patches are extracted here (extract_dense_grid_patches.py)
# and the script stops: encoding them into that npz is a separate step
# (uv run python scripts/patch_encoder/encode_dense_grid.py). See generate_maps.py for
# the provenance of the paper's grids and runs.
set -euo pipefail

REGION="${REGION:?set REGION=<name> (see regions.py REGIONS)}"
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
DATA_ROOT="${TESSERA_DATA_ROOT:-/data/weather-downscaling}"
export TESSERA_DATA_ROOT="$DATA_ROOT"
M="$REPO_ROOT/scripts/maps"
DENSE_NPZ="$DATA_ROOT/processed/dense/$REGION/${REGION}_0.05deg_2024.npz"
cd "$REPO_ROOT"

run() { echo "### [$REGION] $1"; shift; REGION="$REGION" "$@"; }

if [[ ! -f "$DENSE_NPZ" ]]; then
  run "extract_dense_grid_patches — raw TESSERA patches" \
    uv run python "$M/extract_dense_grid_patches.py" --region "$REGION" --resolution 0.05 --year 2024
  echo "raw patches extracted; encode them with the VAE into $DENSE_NPZ, then re-run." >&2
  exit 1
fi

run "DEM fetch (cached if present)"       uv run python "$M/fetch_dem.py"
for VAR in t2m wind; do
  run "generate $VAR — proxy elevation"    env MAPS_NO_DEM=1 MAPS_VARS="$VAR" uv run python "$M/generate_maps.py"
  run "generate $VAR — high-res DEM"       env MAPS_VARS="$VAR" uv run python "$M/generate_maps.py"
done
run "station_eval — per-station preds"    uv run python "$M/station_eval.py"
run "summary_table — per-variant tables"  uv run python "$M/summary_table.py"
run "plot_region_overview — Fig 1"        uv run python "$M/plot_region_overview.py"

echo "### [$REGION] DONE -> ${TESSERA_MAPS_OUT:-$DATA_ROOT/paper_figure_outputs/maps_outputs}/$REGION/<var>_<ts>/"
