# tessera-downscaling

Probabilistic, off-grid weather downscaling with a convolutional conditional
neural process (ConvCNP) whose decoder is conditioned on a learned surface
descriptor: a 16-dimensional VAE compression of a 64 × 64-pixel patch of
[TESSERA](https://github.com/ucam-eo/tessera) Earth-observation embeddings
around each target location.

Code for *Earth observation embeddings are effective sub-grid descriptors for
probabilistic weather downscaling* (Sousa, Tebbutt, Jaffer, Young,
Madhavapeddy, Turner, 2026).

## What the model does

```
ERA5 0.25° regional grid  ──►  residual CNN  ──►  grid→point interpolation  ──►  MLP  ──►  p(y | x)
(20 dynamic [+13 static]        (128 ch, 7 layers)   (bilinear; vanilla SetConv     ↑
 + coordinates + time)                                optional)                      │
                                        topography e = (elevation, Δelevation, mTPI) ┘
                                        Tessera descriptor z_T (16-d VAE latent)     ┘
```

Targets are instantaneous 2 m temperature (Gaussian head) and 10 m wind speed
(truncated-normal head) at GHCNh stations, at 6-hourly UTC snapshots, in five
regions (Europe, United States, East Asia, Southern Africa, Australia).
Training uses 2010–2020, validation 2021, test 2022, with 15 % of stations held
out entirely. Everything is scored on stations the model never saw, on dates it
never trained on.

## Installation

```bash
uv sync --group dev              # core + tests   (make setup-env adds the ingest extra)
uv sync --extra ingest           # + GCS / Earth Engine clients for scripts/data
uv sync --extra aurora           # + microsoft-aurora for scripts/aurora
uv run pytest                    # make test
uv run pre-commit install        # ruff (python only), nbstripout, hygiene hooks
```

Python ≥ 3.12; CUDA wheels for `torch==2.9.0` come from the pinned
`pytorch-cu126` index. All data lives under one directory, the *data root*
(`$TESSERA_DATA_ROOT`, default `/data/weather-downscaling`); see
[`DATA.md`](DATA.md) for its layout and how to rebuild it from scratch.

## Repository layout

```
src/tessera_downscaling/
  data/          datasets (multi-region 6-hourly snapshots, cross-lead), episode assembly,
                 station filters, VAE-latent loading, dense-grid patch extraction
  model/         ConvCNP (residual grid CNN, bilinear / SetConv interpolation, MLP decoder)
                 and the Gaussian / truncated-normal likelihood heads
  train.py       tessera-train      -- train one configuration
  evaluate.py    tessera-evaluate   -- score a checkpoint on held-out stations
  baselines.py   tessera-baselines  -- persistence, ERA5 interpolation, fitted lapse rate
  patch_encoder/ the TESSERA patch VAE that produces the 16-d surface descriptor
  preprocessing/ shared preprocessing helpers (crops, Δelevation, mTPI lookup)
  paths.py       the data root and legacy-path resolution
  io_utils.py    atomic NetCDF writes, filename convention, parallel map
scripts/
  data/          download ERA5 (WeatherBench2 / ARCO) and GHCNh; Earth-Engine station descriptors
                 (mTPI, hand-crafted surface features); Tessera patch extraction; latent controls
                 (shuffling, summary statistics, descriptor concatenation); slurm/ wrappers
  preprocessing/ build the multi-region snapshot dataset and the Aurora-lead datasets
  aurora/        generate the Aurora forecasts used as forecast-driven context
  experiments/   one folder per experiment: experiments.yaml (the runs) + submit.sh (shared _lib.sh:
                 DRY_RUN / LOCAL / sbatch); Norway rollout schedules; see experiments/README.md
  patch_encoder/ train / evaluate the VAE, encode stations and dense grids (vae.yaml = the paper's run)
  maps/          dense 0.05° map inference over Iberia and Norway (Figs 3–4, 9)
  analysis/      descriptor-space analyses (residual probe; Norway reachability / separability)
  paper/         make_paper_figures.py, make_paper_tables.py -- regenerate every figure and table
  reeval_*.sh    re-evaluation sweeps (train-station scores; truncated-normal point metrics)
notebooks/       the analysis notebooks behind the paper's figures/tables (outputs stripped; numbers
                 live in notebooks/*_analysis_outputs/ and paper/tables/) -- see notebooks/README.md
paper/figures/   the 21 delivered figures;  paper/tables/  the regenerated tables (md / tex / csv)
tests/           unit tests + checkpoint-compatibility tests against the data root
```

## Reproducing the paper

The trained runs, latents, descriptors and figure inputs are stored under the
data root; regenerating every figure and table from them needs no GPU:

```bash
uv run python scripts/paper/make_paper_figures.py      # → paper/figures/*.pdf, with numeric cross-checks
uv run python scripts/paper/make_paper_tables.py       # → paper/tables/, Tables 1, 4/C1, 6/B1
```

To retrain, each experiment folder under `scripts/experiments/` lists its runs
in `experiments.yaml` and submits them with `submit.sh` (Slurm, or `LOCAL=1`).
The canonical Tessera arm is

```
tessera-train --interpolation bilinear --tessera-injection concat \
  --vae-latents-path processed/vae_tessera_1B-M/station_latents_1B-M_p128_2017_crop64_lat16_grad0.5_auxon.npy \
  --vae-latents-station-csv processed/tessera_global/station_list_filtered.csv \
  --no-static-fields --use-mtpi --weight-decay 1e-4 [--likelihood wind=truncated_normal] ...
```

and the no-Tessera baseline drops the latent flags and keeps the ERA5 static
fields. See `scripts/experiments/README.md` for the run-name grammar and
`DATA.md` for the end-to-end data pipeline (download → preprocess →
descriptors → latents → train → evaluate → figures).

The surface descriptor itself is reproducible in-repo: the VAE in
`src/tessera_downscaling/patch_encoder/` compresses each station's 64 × 64
TESSERA patch to the 16-d latent (recon + gradient + KL loss, plus auxiliary
elevation/lat/lon heads), trained and applied via
`scripts/patch_encoder/{prebuild_cache,train_vae,eval_vae,encode_dense_grid}.py`;
the paper's checkpoint loads into this code strict=True and reproduces its
published latents to ~2e-6.

## Provenance

Imported from `cambridge-mlg/end-to-end-forecasting@68e54b0`
(`projects/tessera_downscaling`) on 2026-08-18; see [`HISTORY.md`](HISTORY.md)
and [`MIGRATION_PLAN.md`](MIGRATION_PLAN.md).

## Citation

```bibtex
@article{sousa2026tessera,
  title   = {Earth observation embeddings are effective sub-grid descriptors for probabilistic weather downscaling},
  author  = {Sousa, Pedro and Tebbutt, Will and Jaffer, Sadiq and Young, Robin and Madhavapeddy, Anil and Turner, Richard E.},
  year    = {2026},
}
```

## License

MIT (see `LICENSE`).
