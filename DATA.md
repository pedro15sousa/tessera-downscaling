# The data root

Everything the code reads or writes outside this repository lives under one
directory, the *data root*: `$TESSERA_DATA_ROOT`, default
`/data/weather-downscaling` (`src/tessera_downscaling/paths.py`; shell scripts
use `DATA_ROOT="${TESSERA_DATA_ROOT:-/data/weather-downscaling}"`). The paper's
runs were produced on Isambard, where the same tree lived under a project-local
`.tmp_output/`; `paths.resolve()` rewrites those prefixes, which still appear
inside every stored `config.json` and checkpoint, onto the current root.

Sizes below are approximate. Timestamps are 6-hourly UTC (`00/06/12/18`) and
written as `YYYY-MM-DD-HH`.

```
<data root>/
  _staging/                  raw and intermediate inputs (ERA5, GHCNh, Aurora)       ~5.9 TB
  processed/                 station lists, descriptors, latents, patches, caches    ~3 TB
  dataset_timestamp_global/  THE paper's dataset (multi_region_snapshot_v1)           ~176 GB
  dataset_timestamp_aurora_lead{6,24,72}h/   Aurora-forecast context datasets        3 x 88 GB
  training_runs_<experiment_folder>/          trained runs and baselines             ~60 GB
  paper_figure_outputs/maps_outputs/          cached inputs of the map figures       ~97 MB
  tessera_patch_encoder/     VAE patch-encoder outputs (checkpoints, latents)         ~60 GB
```

## 1. `_staging/`

### `_staging/processed/era5_wb2_quarter_<variable>/data/YYYY-MM-DD-HH.nc`

One NetCDF per (variable, timestamp) on the global 0.25 deg WeatherBench2 grid
(721 x 1440; longitudes 0..359.75), from `scripts/data/download_era5_wb2.py`
(WeatherBench2 zarr, ends 2023-01-10) and `scripts/data/download_era5_arco.py`
(ARCO-ERA5, same layout, for later dates). The 12 variables the paper uses
(`tessera_downscaling.io_utils.SURFACE_VARIABLES` / `ATMOS_VARIABLES`):

| kind | variables | file content |
|---|---|---|
| surface (5) | `2m_temperature`, `10m_u_component_of_wind`, `10m_v_component_of_wind`, `mean_sea_level_pressure`, `total_precipitation_6hr` | `(latitude, longitude)` |
| pressure-level (5) | `temperature`, `u_component_of_wind`, `v_component_of_wind`, `specific_humidity`, `geopotential` | `(level, latitude, longitude)`, `level = [500, 700, 850]` hPa |

Coverage on disk: 2009-12-28 to 2026-05-01 (~23,860 files per variable).
`era5_wb2_quarter_{100m_u,100m_v}_component_of_wind`, `surface_pressure` and
`boundary_layer_height` belong to the separate wind-energy project, not to this
repository.

### `_staging/processed/era5_static/era5_static_0p25_all.nc`

The 13 ERA5 invariant fields on the same grid (`slt cvh cvl tvh tvl anor isor z
lsm slor sdfor cl dl`), dims `(valid_time=1, latitude, longitude)`. A one-off
CDS download **provided with the data root as-is**; there is no script for it.
Read by the preprocessing (`static_fields.npy` of every region) and by the
Aurora driver (`z`, `lsm`, `slt`).

### `_staging/processed/ghcnh/data/YYYY-MM-DD-HH.nc`

GHCNh station observations binned to the 6-hourly synoptic hours by
`scripts/data/download_ghcnh.py` (2010-01-01 to 2024-01-01, 20,453 files).
Flat per-row tables: `STATION`, `time`, `LATITUDE`, `LONGITUDE`, `Elevation`,
`temperature`, `dew_point_temperature`, `station_level_pressure`,
`wind_direction`, `wind_speed`, `precipitation`, `precipitation_6_hour`. The raw
per-station-year PSV files the downloader fetched sit in `_staging/raw/ghcnh/`
(544k files, 1.7 TB, regenerable) next to NOAA's master
`_staging/raw/ghcnh/station_list.csv`. `ghcnh/data.legacy/` is an earlier
binning and is unused.

### `_staging/aurora/lead{6,24,72}h/<region>/processed/era5_wb2_quarter_<variable>/data/YYYY-MM-DD-HH.nc`

Aurora 0.25 deg forecasts at +6/+24/+72 h, region-cropped and written in the
ERA5 staging layout by `scripts/aurora/generate_aurora_forecasts.py` (9
variables: the 12 above minus precipitation, and no 100 m winds; 19,032 valid
times 2010-01-01..2023-01-10). `europe` and `east_asia` are the ones the paper
uses; `us`, `australia`, `southern_africa`, `uk` were generated but never
distilled into datasets. The 13-level ERA5 initial conditions Aurora needs
(`scripts/data/download_era5_wb2_aurora_levels.py`, `_staging/aurora_inputs/`)
are not kept.

## 2. `processed/`

Every per-station vector file is **row-aligned with
`processed/tessera_global/station_list_filtered.csv`** (38,870 rows: NOAA's
GHCNh station list after the elevation filter; `station_id, latitude,
longitude, elevation, ...`). Rows are NaN where a station has no valid value;
the dataset loader drops those stations.

| path | shape / content | produced by |
|---|---|---|
| `tessera_global -> tessera_global_v1_p64_2024_outdated/` | symlink; `patch_embeddings_2024.npy` (38870, 64, 64, 128) f32, 81.5 GB, v1 TESSERA 2024 patches; `station_list_filtered.csv`; `README.txt` | April-2026 extraction (see trap 1 below) |
| `tessera_station_patches/patch_embeddings_2017_p128.npy` | (38870, 128, 128, 128) f32, 326 GB, TESSERA v2 (1B-M) 2017 patches -- the VAE's input; `patch_embeddings_2024_p128.npy` likewise for 2024; `station_list_filtered.csv` (identical); `landmasks/` (41,771 GeoTIFFs); `extraction_metadata.json` | `scripts/data/extract_tessera_patches_local.py` |
| `vae_tessera_1B-M/station_latents_1B-M_p128_<year>_crop<64,128>_lat<16,32,64>_grad0.5_aux<on,off>.npy` | (38870, d) f32 VAE latents, 2702 NaN rows; `*_global_stats.npz {mean, std}` z-score caches; `provenance.txt` maps each file to its encoder run. **Paper arm: `..._2017_crop64_lat16_grad0.5_auxon.npy`**; `*_shuffle_seed0.npy` (+`.perm.npy`) is the shuffled control, `*_plus_extra_descriptors.npy` the concatenation cache | patch encoder (`tessera_patch_encoder/`), `scripts/data/shuffle_latents.py`, `scripts/data/concat_station_vectors.py` |
| `station_latents_lat16_grad0.5.npy` (+ `_shuffle_seed0`, `_plus_extra_descriptors`, `station_latents_lat64_l1.npy`) | (38870, 16/64) v1-generation latents (2024 p64 patches); still read by the regional baseline folders, the `_lat16_mtpi` rollout folder, the cross-lead baseline folder and the dense maps | v1 patch encoder |
| `extra_descriptors.npy`, `extra_descriptors_names.json`, `extra_descriptors_global_stats.npz` | (38870, 17) f32 hand-crafted surface descriptors (7 land-cover fractions, tree height, clay/sand, 5 DEM statistics, 2 slopes); built from `station_extra_descriptors.csv` | `scripts/data/fetch_station_extra_descriptors.py` (Earth Engine) then `scripts/data/build_extra_descriptors.py` |
| `station_mtpi.csv` | `station_id, mtpi` for the 23,766 dataset stations | `scripts/data/fetch_station_mtpi.py` (Earth Engine); joined into `stations.csv` by the preprocessing or `scripts/data/backfill_station_mtpi.py` |
| `station_summary_stats_1B-M_p128_2017_crop64_dim16.npy` (+ `_meta.json`, `_global_stats.npz`) | (38870, 16) patch summary statistics (mean/std/p10/p90 of 4 channels) -- the Appendix A control | `scripts/data/build_summary_stats_latents.py` |
| `dense/{norway,iberia}/<r>_0.05deg_2024.{Z.npy,npz}`, `<r>_0.05deg_dem.npy` | dense 0.05 deg grids of v1 latents (62,901 / 39,411 points x 16; `npz` adds `valid_mask`, `coords`, provenance) + DEM for the map figures | `scripts/maps/extract_dense_grid_patches.py` -> `tessera_dense_grid/<r>_0.05deg_2024/patch_embeddings.npy` ((N, 64, 64, 128), 132 GB for Norway) encoded by the v1 VAE (`tessera_patch_encoder/`); `scripts/maps/fetch_dem.py` |
| `dem_cache/*.hgt` | 198 SRTM 1-arcsec tiles (5 GB) for the map figures | `scripts/maps/fetch_dem.py` |
| `overview_cache/{obs_counts_2022,obs_counts_test_split,patch_valid_2024}.npz` | per-station observation counts and the TESSERA validity mask (`valid, coverage, centre` over the 38,870 rows) | `scripts/maps/plot_region_overview.py` |

`alphaearth_station_patches/`, `olmoearth_*`, `station_latents_jepa_*` and
`tessera_patch_encoder/outputs/{jepa,vae/alphaearth,...}` are benchmark
experiments outside this paper.

## 3. Datasets

### `dataset_timestamp_global/` (the paper's dataset, `layout_version = multi_region_snapshot_v1`)

Built by `scripts/preprocessing/preprocess_timestamp_global.py` from `_staging`:
6-hourly episodes 2010-01-01..2023-12-31 for five non-overlapping regions
(`europe`, `us`, `east_asia`, `australia`, `southern_africa`; boxes in the
script). Temporal split train <= 2020-12-31, val 2021, test 2022-; spatial split
85/15 per region, seed 42.

```
metadata.json                      layout_version, era5_dynamic_channels (20), regions{bbox, grid_shape},
                                   valid_timestamps, temporal_split, spatial_split counts
stations.csv                       the 23,766 dataset stations: station_id, latitude, longitude, elevation,
                                   region, delta_elevation, n_valid_episodes_{t2m,wind,precip}, spatial_split, mtpi
valid_station_indices.npy          (23766,) int64 -- rows of stations.csv with valid obs
ghcnh_snapshot/YYYY-MM-DD-HH.npz   shared across regions: {t2m, wind, precip} (n_stations,) f32 NaN-padded,
                                   obs_count (n_stations,) int32
regions/<region>/
  era5_snapshot/YYYY-MM-DD-HH.npy  (20, H, W) f32 region crop, channel order era5_snapshot_channel_names()
                                   (5 surface incl. total_precipitation_sum, then 5 vars x 3 levels)
  lats.npy, lons.npy               (H,), (W,)   e.g. europe 161 x 257
  static_fields.npy                (13, H, W) the ERA5 invariant fields
  normalisation_stats.npz          era5_mean/era5_std (35,) = 20 dynamic + 13 static + lat + lon,
                                   from training timestamps of this region
  normalisation_stats_no_static.npz  (22,) = 20 dynamic + lat + lon
  region_metadata.json             bbox, grid_shape, static_channels
```

Only `t2m` and `wind` are used by the paper (`precip` is stored but no model
trains on it). The existing copy also carries `normalisation_stats*_global.npz`
at the top level, written for a global-normalisation option that no longer
exists; nothing reads them.

### `dataset_timestamp_aurora_lead{6,24,72}h/`

Same layout, regions `europe` and `east_asia` only, built by
`scripts/preprocessing/preprocess_aurora.py` from `_staging/aurora` (19
dynamic channels: no precipitation; `metadata.json` carries `source: aurora`,
`lead_hours`). Checked by `scripts/preprocessing/validate_aurora_datasets.py`.
Their `ghcnh_snapshot` is a **symlink into
`dataset_timestamp_global/ghcnh_snapshot`** (trap 2); `stations.csv` is a copy
with the `mtpi` column back-filled.

`dataset_timestamp/` (flat single-region EU 2017-23) and
`dataset_timestamp_aurora_lead6h_test/` are superseded precursors.

## 4. `training_runs_<experiment_folder>/`

One directory per folder of `scripts/experiments/` (`submit.sh` writes
`<data root>/training_runs_<folder>/`), one run per `<name>_seed<S>`
(seeds 42/123/456; cross-lead adds a `<region>/` level, the Norway rollout uses
`<arch>_<sweep>_seed<S>`):

| file | content |
|---|---|
| `config.json` | the full `tessera-train` argument namespace (paths as stored on Isambard; resolve with `paths.resolve`) |
| `best_model.pt`, `latest_model.pt` | `{model_state_dict, optimizer_state_dict, config, epoch, val_loss, val_maes}`; `best_model.pt` is what `tessera-evaluate` scores |
| `training_curves.npz` | `train_losses, val_losses, val_maes, nonfinite_skips_per_epoch, attempted_steps_per_epoch` |
| `training_summary.json` | epochs run, best/final losses, non-finite step counts, lr / wd / seed |
| `test_summary.json` | held-out metrics per target (`<var>_{nll,crps,mae,rmse,bias,correlation,pit_chi2_*,within_1sigma,...}`, station/prediction counts, `head_spec`) |
| `test_results.json` | same content as `test_summary.json` (written by the same evaluation) |
| `test_predictions.npz` | per prediction: `<var>_param_*` (head parameters), `<var>_targets`, `<var>_station_indices` |
| `test_station_errors.npz` | per station: ids, lat/lon/elev/delta_elev, `<var>_station_{mae,rmse,bias,count}` |
| `eval_train_stations/` | the four `test_*` files re-evaluated on the *training* stations at held-out times (`scripts/reeval_train_stations.sh`) |
| `station_crps_cache.npz` | rollout runs only: `<var>_station_crps` per evaluated European station |
| `eval_lead{0,6,24,72}h/` | cross-lead runs only: the four `test_*` files per lead (`eval_lead0h` on ERA5, the others on the Aurora datasets) |

No-model references (`*_era5_interp*`, `*_persistence*`) have only the
`config.json` + `test_*` files. `training_runs_*_tessera_1B-M_2024` (the
2024-embedding arm, model-selection provenance), the empty `training_runs_*`
stubs and `training_runs_snapshot_14y_southern_africa_old` are not read by
anything in this repository.

## 5. `paper_figure_outputs/maps_outputs/`

Cached inputs of the figure scripts (`paths.paper_figure_inputs_dir()`):
`overview/` (region boxes, station counts, `region_overview.pdf`),
`{norway,iberia}/<var>_<timestamp>/` (dense-map `.npz` grids, DEM grids and
previews for Figs 3-4) and `<region>_summary.csv`. Written by
`scripts/maps/*.py`, consumed by `scripts/paper/make_paper_figures.py`.

## 6. `tessera_patch_encoder/`

Outputs of the VAE patch encoder, whose code now lives in this repository
(`src/tessera_downscaling/patch_encoder/` + `scripts/patch_encoder/`; `repo/`
here is the archived original research tree): `outputs/vae/<run>/` with `best.pt`,
`checkpoint_epoch*.pt`, `eval/{station_latents.npy, latents.npz,
reconstruction_metrics.npz, probe_*}` and per-run Slurm logs. The runs named in
`processed/vae_tessera_1B-M/provenance.txt` (`p128_2017_*`, `p128_2024_*`) and
the v1 `lat16_beta0.0005_grad0.5_e200/` (dense-map latents) are the ones that
matter; `data/` holds the station lists the encoder was pointed at.

## 7. Rebuilding everything from scratch

1. **Download** (`uv sync --extra ingest`): `scripts/data/download_era5_wb2.py`
   (+ `download_era5_arco.py` for dates after 2023-01-10), `download_ghcnh.py`
   -- Slurm wrappers in `scripts/data/slurm/`. Put `era5_static_0p25_all.nc`
   under `_staging/processed/era5_static/` (CDS, one-off).
2. **Station descriptors** (Earth Engine): `scripts/data/fetch_station_mtpi.py`
   -> `processed/station_mtpi.csv`; `fetch_station_extra_descriptors.py` then
   `build_extra_descriptors.py` -> `processed/extra_descriptors.npy`.
3. **Dataset**: `scripts/preprocessing/preprocess_timestamp_global.py
   --mtpi-csv processed/station_mtpi.csv` (Slurm:
   `scripts/preprocessing/slurm/submit_preprocess_timestamp_global.sh`) ->
   `dataset_timestamp_global/`.
4. **TESSERA patches**: `scripts/data/shortlist_tessera_tiles.py` (which tiles
   the embedding pipeline must provide), then
   `scripts/data/extract_tessera_patches_local.py --out-dir
   processed/tessera_station_patches` from the TESSERA v2 mount ->
   `patch_embeddings_2017_p128.npy`. The v1 2024 p64 file behind
   `processed/tessera_global` is only the station-validity filter (trap 1).
5. **Latents**: `scripts/patch_encoder/{prebuild_cache,train_vae,eval_vae}.py`
   (config `scripts/patch_encoder/vae.yaml` = the paper's run: crop 64,
   latent 16, gradient loss 0.5, aux heads on) and copy
   `eval/station_latents.npy` to `processed/vae_tessera_1B-M/...` (record it
   in `provenance.txt`). Controls: `scripts/data/shuffle_latents.py --seed 0`,
   `scripts/data/build_summary_stats_latents.py`,
   `scripts/data/concat_station_vectors.py`.
6. **Aurora context** (`uv sync --extra aurora`):
   `scripts/data/download_era5_wb2_aurora_levels.py` ->
   `scripts/aurora/generate_aurora_forecasts.py` (`submit_aurora_forecasts.sh`)
   -> `scripts/preprocessing/preprocess_aurora.py`
   (`slurm/submit_preprocess_aurora.sh`) -> `scripts/data/backfill_station_mtpi.py`
   on each `dataset_timestamp_aurora_lead*h/stations.csv` ->
   `scripts/preprocessing/validate_aurora_datasets.py`.
7. **Train / evaluate**: `bash scripts/experiments/<folder>/submit.sh` for the
   19 folders (`scripts/experiments/README.md`); the Norway rollout sidecars
   come from `scripts/experiments/pick_probe_set.py` and
   `build_rollout_schedule.py`. Then `scripts/reeval_train_stations.sh` and
   `scripts/reeval_truncated_normal.sh`.
8. **Maps and figures**: `scripts/maps/extract_dense_grid_patches.py` ->
   `processed/tessera_dense_grid/`, encode with
   `scripts/patch_encoder/encode_dense_grid.py` -> `processed/dense/<region>/`,
   then `REGION=<region> bash scripts/maps/run_region_maps.sh` (DEM via
   `fetch_dem.py` -> `processed/dem_cache/`; map inference, station evaluation
   and overview -> `paper_figure_outputs/maps_outputs/`) and finally
   `scripts/paper/make_paper_figures.py` and `make_paper_tables.py`.

## Two traps

1. `processed/tessera_global` is a symlink to
   `tessera_global_v1_p64_2024_outdated/`. Every run -- baselines included --
   passes its 81.5 GB `patch_embeddings_2024.npy` as `--tessera-path` purely as
   the **station-validity filter** (centre pixel non-zero and >= 50 % patch
   coverage); no patch is ever fed to a model. Deleting or swapping that file
   silently changes the station set of every experiment (the mask is also
   cached in `processed/overview_cache/patch_valid_2024.npz`).
2. `dataset_timestamp_aurora_lead{6,24,72}h/ghcnh_snapshot` are symlinks into
   `dataset_timestamp_global/ghcnh_snapshot` (the targets are identical across
   leads). On a copy of the tree they must point at the sibling dataset
   (`../dataset_timestamp_global/ghcnh_snapshot`); on `/data/weather-downscaling`
   they still carry the absolute Isambard path and dangle until
   `dataset_timestamp_global/` is complete and they are re-linked.
