# The data root

Everything the code reads or writes outside this repository lives under one
directory, the *data root*: `$TESSERA_DATA_ROOT`, default
`/data/weather-downscaling` (`src/tessera_downscaling/paths.py`; shell scripts
use `DATA_ROOT="${TESSERA_DATA_ROOT:-/data/weather-downscaling}"`). Runs record
absolute paths in their `config.json` and checkpoints; stored paths from the
HPC the paper's runs were trained on -- and from the data root's older layout
-- are mapped onto the current root and layout by `paths.resolve()`
(`paths.LEGACY_ROOT_PREFIXES`, `paths.RELOCATIONS`), and experiment sidecar
JSONs fall back to the copies committed under `scripts/experiments/`
(`evaluate.resolve_sidecar_path`). Fresh runs need none of this: their configs
record wherever the data root and `--output-dir` put them.

Timestamps are 6-hourly UTC (`00/06/12/18`), written `YYYY-MM-DD-HH`.
Sizes are approximate.

```
<data root>/
  ingest/          downloaded raw + intermediate inputs (ERA5, GHCNh, Aurora)    ~5.9 TB
  processed/       patches, latents, descriptors, dense grids, caches            ~3 TB
  datasets/        the training / evaluation datasets                            ~440 GB
  training_runs/   one directory per experiment folder of scripts/experiments/   ~60 GB
  tessera_patch_encoder/               VAE patch-encoder runs and caches         ~60 GB
  paper_figure_outputs/maps_outputs/   cached inputs of the map figures          ~97 MB
```

## 1. `ingest/` -- source data

Re-downloadable in principle, but terabytes -- not scratch.

* `ingest/processed/era5_wb2_quarter_<variable>/data/YYYY-MM-DD-HH.nc` -- one
  NetCDF per (variable, timestamp) on the global 0.25 deg WeatherBench2 grid,
  for the 10 variables the paper uses (5 surface + 5 atmospheric at
  500/700/850 hPa; `tessera_downscaling.io_utils`). From
  `scripts/data/download_era5_wb2.py` (+ `download_era5_arco.py` for dates
  after 2023-01-10; scratch under `ingest/tmp/`).
* `ingest/processed/era5_static/era5_static_0p25_all.nc` -- the 13 ERA5
  invariant fields; a one-off CDS download provided as-is (no script).
* `ingest/processed/ghcnh/data/YYYY-MM-DD-HH.nc` -- GHCNh station observations
  binned to the synoptic hours by `scripts/data/download_ghcnh.py`; the raw
  per-station-year PSV files and NOAA's `station_list.csv` sit under
  `ingest/raw/ghcnh/`.
* `ingest/aurora/lead{6,24,72}h/<region>/processed/...` -- Aurora 0.25 deg
  forecasts in the same layout (`scripts/aurora/generate_aurora_forecasts.py`;
  9 variables: the 10 above minus precipitation). The 13-level initial
  conditions (`download_era5_wb2.py --levels aurora`,
  `ingest/aurora_inputs/`) are not kept.

## 2. `processed/` -- patches and descriptors

Every per-station vector file is **row-aligned with
`processed/tessera_global/station_list_filtered.csv`** (38,870 rows; NaN rows
are dropped by the dataset loader).

* `tessera_global/` -- symlink (trap 1 below) to the v1 2024 64x64 patches
  that every run uses as its station-validity filter, plus the row-alignment
  CSV.
* `tessera_station_patches/` -- TESSERA v2 (1B-M) 128x128 patches for 2017 and
  2024, the patch encoder's input
  (`scripts/data/extract_tessera_patches_local.py`).
* `vae_tessera_1B-M/` -- the patch encoder's per-station latents, one file per
  variant; `provenance.txt` maps each file to its encoder run. The paper's
  TESSERA arm reads
  `station_latents_1B-M_p128_2017_crop64_lat16_grad0.5_auxon.npy`;
  `*_shuffle_seed0.npy` are the shuffled controls and `*_global_stats.npz` the
  z-score caches.
* `station_vectors/` -- the loose per-station vectors: the v1-generation
  latents (`station_latents_lat16_grad0.5*.npy`, read by the v1 arms declared
  in the regional and cross-lead `experiments.yaml`, the `_lat16_mtpi` rollout
  folder and the dense maps), the patch summary-statistics control
  (`station_summary_stats_*`), the 17 hand-crafted surface descriptors
  (`extra_descriptors.npy` + names/stats, built from
  `station_extra_descriptors.csv`) and `station_mtpi.csv`.
* `alphaearth_station_patches/`, `olmoearth_imagery/`,
  `olmoearth_station_patches/` -- the foundation-model benchmark arms
  (follow-up ablations; `scripts/patch_encoder/extract/`).
* `dense/`, `tessera_dense_grid/`, `dem_cache/`, `overview_cache/` -- dense-map
  latents and patches, SRTM tiles and region-overview caches
  (`scripts/maps/`).

## 3. `datasets/`

* `datasets/dataset_timestamp_global/` -- **the paper's dataset**
  (`layout_version = multi_region_snapshot_v1`), built by
  `scripts/preprocessing/preprocess_timestamp_global.py`: 6-hourly episodes
  2010-2023 for five regions (`europe`, `us`, `east_asia`, `australia`,
  `southern_africa`), 23,766 stations (`stations.csv`; 85/15 spatial split,
  seed 42), shared `ghcnh_snapshot/` targets, and per-region `era5_snapshot/`
  context (20 dynamic channels), `static_fields.npy` and normalisation stats.
  Only `t2m` and `wind` are modelled; `precip` is stored unused.
* `datasets/dataset_timestamp_aurora_lead{6,24,72}h/` -- the same layout for
  the Aurora forecast context, regions `europe` and `east_asia`, 19 dynamic
  channels (no precipitation); built by `preprocess_aurora.py`, checked by
  `validate_aurora_datasets.py`. Their `ghcnh_snapshot` is a relative symlink
  into the global dataset (trap 2).
* `datasets/dataset_timestamp/` (an early flat single-region layout) and
  `datasets/dataset_timestamp_aurora_lead6h_test/` are superseded precursors,
  kept as history; nothing reads them.

## 4. `training_runs/`

One directory per folder of `scripts/experiments/` (`submit.sh` writes
`<data root>/training_runs/<folder>/`), one run per `<name>_seed<S>` (seeds
42/123/456; the cross-lead folders add a `<region>/` level). A run holds
`config.json` (the full `tessera-train` argument namespace; paths as stored at
training time, resolved by `paths.resolve`), `best_model.pt` /
`latest_model.pt`, `training_curves.npz`, `training_summary.json` and the
evaluation outputs: `test_summary.json` / `test_results.json`,
`test_predictions.npz`, `test_station_errors.npz`, plus
`eval_train_stations/` (`scripts/reeval_train_stations.sh`),
`eval_lead{0,6,24,72}h/` (cross-lead runs) and `station_crps_cache.npz`
(rollout runs) where applicable. No-model references (`*_era5_interp*`,
`*_persistence*`) carry only `config.json` + the `test_*` files.

Folders without a matching `scripts/experiments/` entry
(`snapshot_*_tessera_1B-M_2024`, `snapshot_14y_southern_africa_old`,
`snapshot_6y_eu`, `snapshot_global_*`, empty stubs) are archived provenance;
nothing in this repository reads them.

## 5. `tessera_patch_encoder/` and `paper_figure_outputs/`

`tessera_patch_encoder/outputs/vae/<run>/` holds the patch-encoder runs
(`best.pt`, `eval/station_latents.npy`, ...). The runs named in
`processed/vae_tessera_1B-M/provenance.txt` and the v1
`lat16_beta0.0005_grad0.5_e200/` (dense-map latents) are the paper's;
`outputs/vae/{alphaearth,olmoearth}/...` are the benchmark arms.
`outputs/dataset_cache/` holds the per-patch-file validity / normalisation
caches; `data/` the station lists the encoder was pointed at; `repo/` is an
archived copy of the original research tree.

`paper_figure_outputs/maps_outputs/` (`paths.paper_figure_inputs_dir()`;
redirect with `$TESSERA_MAPS_OUT`) caches the map-figure inputs consumed by
`scripts/paper/make_paper_figures.py`.

## 6. Rebuilding from scratch

1. **Download** (`uv sync --extra ingest`): `scripts/data/download_era5_wb2.py`
   (+ `download_era5_arco.py` for later dates) and `download_ghcnh.py`, Slurm
   wrappers in `scripts/data/slurm/`. Put `era5_static_0p25_all.nc` under
   `ingest/processed/era5_static/` (CDS, one-off).
2. **Station descriptors** (Earth Engine): `scripts/data/fetch_station_mtpi.py`,
   then `fetch_station_extra_descriptors.py` + `build_extra_descriptors.py`
   -> `processed/station_vectors/`.
3. **Dataset**: `scripts/preprocessing/preprocess_timestamp_global.py
   --mtpi-csv processed/station_vectors/station_mtpi.csv` (Slurm wrapper in
   `scripts/preprocessing/slurm/`) -> `datasets/dataset_timestamp_global/`.
4. **TESSERA patches**: `scripts/data/shortlist_tessera_tiles.py`, then
   `extract_tessera_patches_local.py --out-dir processed/tessera_station_patches`
   reading the TESSERA v2 embedding tiles (`$TESSERA_V2_MOUNT`, default
   `/tessera/v2/global_0.1_degree_representation`, or `--mount-dir`; landmasks
   are cached under `<data root>/_cache/geotessera/`).
5. **Latents**: `scripts/patch_encoder/{prebuild_cache,train_vae,eval_vae}.py`
   (config `scripts/patch_encoder/vae.yaml`); copy `eval/station_latents.npy`
   to `processed/vae_tessera_1B-M/` and record it in `provenance.txt`.
   Controls: `scripts/data/shuffle_latents.py --seed 0`,
   `build_summary_stats_latents.py`, `concat_station_vectors.py`.
6. **Aurora context** (`uv sync --extra aurora`):
   `download_era5_wb2.py --levels aurora` ->
   `scripts/aurora/generate_aurora_forecasts.py` ->
   `scripts/preprocessing/preprocess_aurora.py` ->
   `scripts/data/backfill_station_mtpi.py` on each aurora `stations.csv` ->
   `validate_aurora_datasets.py`.
7. **Train / evaluate**: `bash scripts/experiments/<folder>/submit.sh` for the
   19 folders (`scripts/experiments/README.md`); the Norway rollout sidecars
   come from `pick_probe_set.py` and `build_rollout_schedule.py`. Then
   `scripts/reeval_train_stations.sh` and `scripts/reeval_truncated_normal.sh`.
8. **Maps and figures**: `scripts/maps/extract_dense_grid_patches.py` ->
   `scripts/patch_encoder/encode_dense_grid.py` -> `REGION=<region> bash
   scripts/maps/run_region_maps.sh`, then `make figures tables`. Caveat: the
   paper's map figures come from v1-generation runs whose stems are pinned in
   `scripts/maps/regions.py` and are not trained by step 7 (see the provenance
   note atop `scripts/maps/generate_maps.py`); the published figures rebuild
   from the cached `paper_figure_outputs/` inputs without rerunning the maps.

## Two traps

1. `processed/tessera_global` is a symlink to
   `tessera_global_v1_p64_2024_outdated/`. Every run -- baselines included --
   passes its `patch_embeddings_2024.npy` as `--tessera-path` purely as the
   **station-validity filter** (centre pixel non-zero and >= 50 % patch
   coverage); no patch is ever fed to a model. Deleting or swapping that file
   silently changes the station set of every experiment.
2. `datasets/dataset_timestamp_aurora_lead{6,24,72}h/ghcnh_snapshot` are
   symlinks into `datasets/dataset_timestamp_global/ghcnh_snapshot` (identical
   targets across leads, so the ~50 GB of observations exist once).
   `preprocess_aurora.py` creates them relative, which survives copying or
   re-rooting the tree; if a copy carries absolute links from another machine,
   re-link them once the global dataset is in place.
