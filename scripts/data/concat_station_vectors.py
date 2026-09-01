"""Concatenate two row-aligned per-station vector files into one npy.

Pre-builds the derived cache file that ``train.py`` uses when
``--vae-latents-path`` and ``--extra-descriptors-path`` are passed together
(see ``resolve_combined_vectors`` there — same naming scheme:
``<latents_stem>_plus_<descriptors_stem>.npy`` next to the latents file).
train.py builds the file on demand if it is missing, but pre-building it
here — together with its z-score stats cache — means a batch of concurrent
SLURM jobs only ever reads it.

Both inputs must be row-aligned to the same station list (the global
``tessera_global/station_list_filtered.csv``). NaN semantics compose
correctly for free: the loader drops any row containing NaN, so a station
survives iff it is valid in BOTH sources — the same intersection filtering
a two-slot implementation would need to enforce by hand.

Per-dim z-score stats for the combined file are pre-warmed here with the
standard global-stats convention (computed over the file's own valid rows).
Because the extra descriptors have zero NaN rows, the combined file's valid
set equals the VAE latents' valid set, so the VAE dims get identical stats
to a pure-VAE run — the comparison stays exact.

Usage (from the repo root):
    uv run python scripts/data/concat_station_vectors.py \
        --inputs  <latents>.npy <data root>/processed/station_vectors/extra_descriptors.npy \
        --output  <latents>_plus_extra_descriptors.npy
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

LOG_FORMAT = "%(asctime)s | %(levelname)s | %(message)s"
logger = logging.getLogger("concat_station_vectors")


def main() -> int:
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--inputs",
        type=Path,
        nargs="+",
        required=True,
        help="Two or more (n_stations, d_i) .npy files, all "
        "row-aligned to the same station list.",
    )
    p.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output .npy; a _global_stats.npz sibling is pre-warmed next to it.",
    )
    args = p.parse_args()

    arrays = []
    for path in args.inputs:
        arr = np.load(path).astype(np.float32)
        if arr.ndim != 2:
            raise ValueError(f"{path}: expected 2-D, got shape {arr.shape}")
        n_valid = int((~np.isnan(arr).any(axis=1)).sum())
        logger.info(
            "%s: shape=%s, %d/%d valid rows",
            path.name,
            arr.shape,
            n_valid,
            arr.shape[0],
        )
        arrays.append(arr)
    n_rows = {a.shape[0] for a in arrays}
    if len(n_rows) != 1:
        raise ValueError(
            f"Row-count mismatch across inputs: { {p.name: a.shape for p, a in zip(args.inputs, arrays, strict=False)} } "
            "— all inputs must be aligned to the same station list."
        )

    combined = np.hstack(arrays)
    valid = ~np.isnan(combined).any(axis=1)
    logger.info(
        "Combined: shape=%s, %d/%d valid rows (intersection of all sources)",
        combined.shape,
        int(valid.sum()),
        combined.shape[0],
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, combined)
    logger.info("Wrote %s", args.output)

    from tessera_downscaling.data.vae_latents import (
        compute_or_load_global_vae_stats,
    )

    mean, std = compute_or_load_global_vae_stats(args.output)
    logger.info("z-score stats pre-warmed (%d dims)", len(mean))
    return 0


if __name__ == "__main__":
    sys.exit(main())
