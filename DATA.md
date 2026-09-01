# Data: sources, pipeline, layout

Everything the code reads or writes outside this repository lives under one
directory, the *data root*: `$TESSERA_DATA_ROOT`, default
`/data/weather-downscaling` (`src/tessera_downscaling/paths.py`; shell scripts
use `DATA_ROOT="${TESSERA_DATA_ROOT:-/data/weather-downscaling}"`). Point it
anywhere with enough disk and every script below reads and writes inside it.
This document lists the external data sources, the pipeline that turns them
into the training data, and the layout that pipeline produces.

Timestamps are 6-hourly UTC (`00/06/12/18`), written `YYYY-MM-DD-HH`.

## 1. External sources

| source | what | access | ingested by |
|---|---|---|---|
| WeatherBench2 ERA5 | 0.25 deg ERA5 reanalysis, public GCS zarr (to 2023-01-10) | none (public bucket) | `scripts/data/download_era5_wb2.py` |
| ARCO-ERA5 | same variables for later dates | none (public bucket) | `scripts/data/download_era5_arco.py` |
| Copernicus CDS | the 13 ERA5 invariant fields, one file (`era5_static_0p25_all.nc`) | CDS account | one-off manual download (no script); place under `ingest/processed/era5_static/` |
| NOAA GHCNh | hourly station observations, per-station-year PSV files + `station_list.csv` | none (public) | `scripts/data/download_ghcnh.py` |
| TESSERA v2 embeddings | 10 m embedding tiles (distributed through the GeoTessera library) | local tile tree (`$TESSERA_V2_MOUNT`, default `/tessera/v2/global_0.1_degree_representation`, or `--mount-dir`) | `scripts/data/extract_tessera_patches_local.py` |
| Google Earth Engine | mTPI; WorldCover v200, ETH canopy height, SoilGrids, GLO-30 (the hand-crafted descriptors) | Earth Engine account | `scripts/data/fetch_station_mtpi.py`, `fetch_station_extra_descriptors.py` |
| Aurora checkpoint | pretrained 0.25 deg Aurora weights | none (HuggingFace, cached on first load) | `scripts/aurora/generate_aurora_forecasts.py` |
| SRTM 1-arcsec DEM | terrain tiles for the map figures | none (AWS open data) | `scripts/maps/fetch_dem.py` |

Storage to plan for: the raw + intermediate inputs are ~6 TB, the extracted
TESSERA patches ~3 TB, the datasets ~440 GB, the trained runs ~60 GB.

## 2. Pipeline

Install with `uv sync --extra ingest` (steps 1-4) and `--extra aurora`
(step 6); Slurm wrappers for the long steps sit next to each script.

1. **Download ERA5 + GHCNh** -> `ingest/`:
   `scripts/data/download_era5_wb2.py` (+ `download_era5_arco.py` for dates
   after 2023-01-10) and `download_ghcnh.py`; place the CDS static file under
   `ingest/processed/era5_static/`.
2. **Station descriptors** (Earth Engine) -> `processed/station_vectors/`:
   `scripts/data/fetch_station_mtpi.py`, then
   `fetch_station_extra_descriptors.py` + `build_extra_descriptors.py`.
3. **Build the dataset** -> `datasets/dataset_timestamp_global/`:
   `scripts/preprocessing/preprocess_timestamp_global.py
   --mtpi-csv processed/station_vectors/station_mtpi.csv`.
4. **Extract TESSERA patches** -> `processed/tessera_station_patches/`:
   `scripts/data/shortlist_tessera_tiles.py` (which tiles are needed), then
   `extract_tessera_patches_local.py --out-dir
   processed/tessera_station_patches` (landmasks are cached under
   `<data root>/_cache/geotessera/`).
5. **Train the patch encoder and export latents** ->
   `processed/vae_tessera_1B-M/`:
   `scripts/patch_encoder/{prebuild_cache,train_vae,eval_vae}.py` with config
   `scripts/patch_encoder/vae.yaml`; copy `eval/station_latents.npy` into
   `processed/vae_tessera_1B-M/` and record it in `provenance.txt`. Controls:
   `scripts/data/shuffle_latents.py --seed 0`,
   `build_summary_stats_latents.py`, `concat_station_vectors.py`.
6. **Aurora forecast context** -> `datasets/dataset_timestamp_aurora_lead{6,24,72}h/`:
   `download_era5_wb2.py --levels aurora` ->
   `scripts/aurora/generate_aurora_forecasts.py` ->
   `scripts/preprocessing/preprocess_aurora.py` ->
   `scripts/data/backfill_station_mtpi.py` on each aurora `stations.csv` ->
   `validate_aurora_datasets.py`.
7. **Train and evaluate** -> `training_runs/<folder>/`:
   `bash scripts/experiments/<folder>/submit.sh` for the 19 folders
   (`scripts/experiments/README.md`); the Norway rollout sidecars come from
   `pick_probe_set.py` and `build_rollout_schedule.py`. Then
   `scripts/reeval_train_stations.sh` and `scripts/reeval_truncated_normal.sh`.

## 3. Layout the pipeline produces

```
<data root>/
  ingest/                    raw + intermediate inputs (steps 1, 6)
    processed/era5_wb2_quarter_<var>/data/   one NetCDF per (variable, timestamp), 0.25 deg global
    processed/era5_static/                   the 13 invariant fields (CDS, one-off)
    processed/ghcnh/data/                    GHCNh binned to the synoptic hours
    raw/ghcnh/                               raw PSVs + NOAA station_list.csv
    aurora/lead{6,24,72}h/<region>/...       Aurora forecasts in the ERA5 staging layout
  processed/                 patches, latents, descriptors (steps 2, 4, 5, 8)
    tessera_global/          64x64 patches used as the station-validity filter + the row-alignment CSV
    tessera_station_patches/ 128x128 TESSERA v2 patches (the encoder's input)
    vae_tessera_1B-M/        exported per-station latents + provenance.txt
    station_vectors/         loose per-station vectors (v1 latents, summary stats,
                             extra_descriptors, station_mtpi.csv)
    dense/  tessera_dense_grid/  dem_cache/  overview_cache/   map inputs
  datasets/                  the training / evaluation datasets (steps 3, 6)
    dataset_timestamp_global/                the paper's dataset (multi_region_snapshot_v1)
    dataset_timestamp_aurora_lead{6,24,72}h/ Aurora-context datasets
  training_runs/<folder>/    one directory per scripts/experiments/ folder (step 7)
  tessera_patch_encoder/     encoder runs and dataset caches (step 5)
  paper_figure_outputs/maps_outputs/   cached map-figure inputs (step 8)
```

`datasets/dataset_timestamp_global/` holds 6-hourly episodes 2010-2023 for
five regions (`europe`, `us`, `east_asia`, `australia`, `southern_africa`):
23,766 stations (`stations.csv`; 85/15 spatial split, seed 42), shared
`ghcnh_snapshot/` targets, and per-region `era5_snapshot/` context (20 dynamic
channels), `static_fields.npy` and normalisation stats. Only `t2m` and `wind`
are modelled; `precip` is stored unused. The aurora datasets mirror it with 19
dynamic channels (no precipitation).

A `training_runs/<folder>/<name>_seed<S>/` run holds `config.json`,
`best_model.pt` / `latest_model.pt`, `training_curves.npz`,
`training_summary.json` and the evaluation outputs (`test_summary.json` /
`test_results.json`, `test_predictions.npz`, `test_station_errors.npz`; plus
`eval_train_stations/`, `eval_lead{0,6,24,72}h/` or `station_crps_cache.npz`
where applicable). No-model references (`*_era5_interp*`, `*_persistence*`)
carry only `config.json` + the `test_*` files.
