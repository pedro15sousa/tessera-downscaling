# Data and training pipeline

Everything the code reads or writes outside this repository lives under one
directory, the *data root*: `$TESSERA_DATA_ROOT`, default
`/data/weather-downscaling` (`src/tessera_downscaling/paths.py`). Point it
anywhere with enough disk and every script below reads and writes inside it.
Relative paths given to any CLI are interpreted under the data root.

This document covers the main line: fetch the external data, train the patch
encoder, train a downscaler. The paper's experiments are variants of this
line, driven by the folders of `scripts/experiments/` (see its README);
follow-on pipelines (Aurora forecast context, dense maps and figures) are
documented in the headers of `scripts/aurora/`, `scripts/maps/` and
`scripts/paper/`.

## 1. External data

Required for the main line (`uv sync --extra ingest`):

| source | what | access | fetched by |
|---|---|---|---|
| WeatherBench2 ERA5 | 0.25 deg ERA5 reanalysis, public GCS zarr (to 2023-01-10) | none (public bucket) | `scripts/data/download_era5_wb2.py` |
| ARCO-ERA5 | same variables for later dates | none (public bucket) | `scripts/data/download_era5_arco.py` |
| Copernicus CDS | the 13 ERA5 invariant fields, one file (`era5_static_0p25_all.nc`) | CDS account | one-off manual download (no script); place under `ingest/processed/era5_static/` |
| NOAA GHCNh | hourly station observations | none (public) | `scripts/data/download_ghcnh.py` |
| TESSERA embeddings | 10 m embedding tiles on the global 0.1 deg grid | GeoTessera library (public releases) | `scripts/data/prefetch_tessera_tiles.py` -> local tile tree -> `extract_tessera_patches_local.py` |
| Google Earth Engine | mTPI per station | Earth Engine account | `scripts/data/fetch_station_mtpi.py` |

Variant experiments additionally use: the Aurora checkpoint (forecast-context
experiment; HuggingFace, fetched on first load), further Earth Engine surface
products (hand-crafted-descriptor baseline;
`scripts/data/fetch_station_extra_descriptors.py`) and SRTM terrain tiles
(map figures; `scripts/maps/fetch_dem.py`).

Storage to plan for: raw + intermediate inputs ~6 TB, extracted TESSERA
patches ~3 TB, the dataset ~170 GB.

## 2. Building the training data

1. **Download ERA5 + GHCNh** -> `ingest/`:
   `scripts/data/download_era5_wb2.py` (+ `download_era5_arco.py` for dates
   after 2023-01-10) and `download_ghcnh.py`; place the CDS static file under
   `ingest/processed/era5_static/`. Slurm wrappers in `scripts/data/slurm/`.
2. **Fetch mTPI** -> `processed/station_vectors/station_mtpi.csv`:
   `scripts/data/fetch_station_mtpi.py`.
3. **Build the dataset** -> `datasets/dataset_timestamp_global/`:
   `scripts/preprocessing/preprocess_timestamp_global.py
   --mtpi-csv processed/station_vectors/station_mtpi.csv`. This produces the
   6-hourly episodes 2010-2023 for the five regions (`europe`, `us`,
   `east_asia`, `australia`, `southern_africa`): `stations.csv` with the
   85/15 spatial split, shared `ghcnh_snapshot/` station targets, and
   per-region ERA5 context snapshots, static fields and normalisation stats.
4. **Extract TESSERA station patches** ->
   `processed/tessera_station_patches/`: first populate a local tile tree
   with `scripts/data/prefetch_tessera_tiles.py --station-csv
   ingest/raw/ghcnh/station_list.csv --tiles-dir <tile tree>` -- it computes
   the exact set of tiles the stations' patches touch (using the extractor's
   own patch geometry, so a patch straddling a tile edge pulls in all of its
   tiles) and downloads them through the GeoTessera library, skipping tiles
   already present -- then run
   `extract_tessera_patches_local.py --mount-dir <tile tree> --out-dir
   processed/tessera_station_patches`. This writes the patch array
   (`patch_embeddings_<year>_p128.npy`) and `station_list_filtered.csv` --
   the station list that **every per-station artefact from here on is
   row-aligned with**. The extraction is byte-identical to geotessera's own
   mosaicking, so any tile source with the library's layout works.

   *Unreleased embedding versions:* if the version you want is not yet
   released through the library, run
   `scripts/data/shortlist_tessera_tiles.py` to emit the same tile set as a
   hand-off list for the TESSERA team, and place the generated tiles in the
   tile tree the extractor reads from.

## 3. Training the patch encoder (VAE)

Inputs: the station patches and station list from step 2.4.

1. `scripts/patch_encoder/prebuild_cache.py` -- one-off validity /
   normalisation cache for the patch file.
2. `scripts/patch_encoder/train_vae.py` with config
   `scripts/patch_encoder/vae.yaml` (the paper's encoder settings) ->
   `tessera_patch_encoder/outputs/vae/<run>/`.
3. `scripts/patch_encoder/eval_vae.py <run_dir>` -> the artefact the
   downscaler consumes: `<run_dir>/eval/station_latents.npy`, one latent per
   row of `station_list_filtered.csv` (NaN where the station has no valid
   patch).

Store the exported latents wherever suits your embedding source -- the
downscaler takes the `.npy` path directly (`--vae-latents-path`), so no
particular directory or file name is required; keeping a `provenance.txt`
mapping each latents file to the encoder run that produced it is
recommended. The paper's instantiation stores its latents under
`processed/vae_tessera_1B-M/`, which is what the defaults in
`scripts/experiments/` point at.

## 4. Training a downscaler

Input dependencies:

* `datasets/dataset_timestamp_global/` -- context grids + station targets
  (step 2.3).
* A patch file + station CSV for the **station-validity filter**
  (`--tessera-path` / `--tessera-station-csv`): passed to *every* run,
  baselines included, so all arms train and evaluate on identical station
  sets. No patch is ever fed to the model, and swapping the file changes the
  station set of every experiment. On a fresh build this is naturally the
  patch file from step 2.4; the paper's instantiation used the patch file at
  `processed/tessera_global/`.
* For the TESSERA arm: the latents file from step 3
  (`--vae-latents-path` / `--vae-latents-station-csv`).

Train and evaluate (the full flag set is documented in
`tessera-train --help` and the module docstring of
`src/tessera_downscaling/train.py`):

```bash
uv run tessera-train --dataset-dir datasets/dataset_timestamp_global \
    --train-regions europe \
    --tessera-path processed/tessera_station_patches/patch_embeddings_<year>_p128.npy \
    --tessera-station-csv processed/tessera_station_patches/station_list_filtered.csv \
    --vae-latents-path <path to your station_latents.npy> \
    --vae-latents-station-csv processed/tessera_station_patches/station_list_filtered.csv \
    --interpolation bilinear --tessera-injection concat \
    --no-static-fields --use-mtpi --weight-decay 1e-4 \
    --seed 42 --output-dir training_runs/<folder>/<run_name>

uv run tessera-evaluate --checkpoint training_runs/<folder>/<run_name>/best_model.pt
```

The ERA5-only baseline is the same command without the `--vae-latents-*`
flags and without `--no-static-fields`. `tessera-baselines` produces the
no-model references (ERA5 interpolation, persistence).

This workflow is embedding-agnostic: which embedding version, latents file
and filter patches to use are choices made at the command line. The paper is
one specific instantiation of it -- its exact choices, together with the full
experiment matrix (regions, controls, ablations, the Aurora forecast context
and the Norway rollout), are pinned in
`scripts/experiments/<folder>/experiments.yaml` + `submit.sh`.
