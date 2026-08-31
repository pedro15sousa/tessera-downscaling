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
(20 dynamic [+13 static]        (128 ch, 7 layers)   (SetConv)                    ↑
 + coordinates + time)                                                            |
                                     topography e = (elevation, Δelevation, mTPI) ┘
                                     Tessera descriptor z_T (16-d VAE latent)     ┘
```

Targets are instantaneous 2 m temperature (Gaussian head) and 10 m wind speed
(truncated-normal head) at GHCNh stations, at 6-hourly UTC snapshots, in five
regions (Europe, United States, East Asia, Southern Africa, Australia).
Training uses 2010–2020, validation 2021, test 2022, with 15 % of stations held
out entirely.

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
