#!/usr/bin/env bash
# Build the full DEM-based dense-downscaling figure set for one region.
#
#   REGION=norway bash projects/tessera_downscaling/scripts/maps/run_region_maps.sh
#
# Dates/paths/grid come from regions.py (per-region `dates`, picked by
# select_dates.py). Outputs go to outputs/<region>/<var>_<ts>/. Needs internet
# for the SRTM DEM (first run only; cached) and the ESRI terrain tiles.
set -euo pipefail

REGION="${REGION:?set REGION=<name> (see regions.py REGIONS)}"
REPO=/lus/lfs1aip2/projects/u6do/pmms2/end-to-end-forecasting
PY="$REPO/.venv/bin/python"
M="$REPO/projects/tessera_downscaling/scripts/maps"
cd "$REPO"

run() { echo "### [$REGION] $1"; shift; REGION="$REGION" "$@"; }

run "DEM fetch (cached if present)"      "$PY" "$M/fetch_dem.py"
run "generate — proxy elevation"         env MAPS_NO_DEM=1 "$PY" "$M/generate_maps.py"
run "generate — high-res DEM elevation"  "$PY" "$M/generate_maps.py"
run "analyze — tessera contribution"     "$PY" "$M/analyze_maps.py"
run "station_eval — per-station preds"   "$PY" "$M/station_eval.py"
run "dem_plots — dem contribution + field/terrain" "$PY" "$M/dem_plots.py"
run "interpret_plots — field/terrain"    "$PY" "$M/interpret_plots.py"
run "station_error_maps — station errors" "$PY" "$M/station_error_maps.py"
run "summary_table — per-variant tables"  "$PY" "$M/summary_table.py"

echo "### [$REGION] DONE -> outputs/$REGION/<var>_<ts>/"
