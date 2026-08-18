# tessera-downscaling

Probabilistic off-grid weather downscaling with a convolutional conditional neural
process (ConvCNP) whose decoder is conditioned on a learned surface descriptor
compressed from TESSERA Earth-observation embeddings.

Code for *Earth observation embeddings are effective sub-grid descriptors for
probabilistic weather downscaling* (Sousa, Tebbutt, Jaffer, Young, Madhavapeddy,
Turner; 2026).

Imported from `cambridge-mlg/end-to-end-forecasting@68e54b0`
(`projects/tessera_downscaling`); see `MIGRATION_PLAN.md` for the migration
status and `HISTORY.md` for provenance.

```
uv sync --extra ingest        # + --extra aurora for scripts/aurora
uv run pytest
```
