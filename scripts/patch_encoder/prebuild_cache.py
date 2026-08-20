#!/usr/bin/env python3
"""Build the dataset cache of a patch file before submitting training jobs.

The cache (valid patch indices + per-channel normalisation statistics) comes
from a single pass over a 326 GB ``.npy``, which takes minutes. Every training
job needs it, so build it once here rather than letting a sweep's jobs scan the
same file concurrently. Crop sizes share a cache, so one call per patch file
covers the whole sweep over that file.

Usage:

    uv run python scripts/patch_encoder/prebuild_cache.py                # 2017
    uv run python scripts/patch_encoder/prebuild_cache.py \\
        --patches processed/tessera_station_patches/patch_embeddings_2024_p128.npy

    # Before the foundation-model sweep (slurm/submit_fm_sweep.sh):
    uv run python scripts/patch_encoder/prebuild_cache.py --patches \\
        processed/alphaearth_station_patches/patch_embeddings_alphaearth_{2017,2024}_p128.npy \\
        processed/olmoearth_station_patches/patch_embeddings_olmoearth_{2017,2024}_g16.npy

Relative paths are interpreted under the data root.
"""

from __future__ import annotations

import argparse
import logging
import time

from tessera_downscaling.patch_encoder.dataset import default_patches_path, prepare_data
from tessera_downscaling.paths import resolve

logger = logging.getLogger("prebuild_cache")


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--patches",
        nargs="+",
        default=[str(default_patches_path())],
        help="Patch files to scan (default: the paper's 2017 file).",
    )
    parser.add_argument("--cache-dir", default=None, help="Root of the cache tree.")
    parser.add_argument(
        "--rebuild", action="store_true", help="Rescan even if a cache exists."
    )
    args = parser.parse_args()

    for patches in args.patches:
        path = resolve(patches)
        if not path.exists():
            logger.warning(f"missing, skipping: {path}")
            continue
        started = time.time()
        cache = prepare_data(path, cache_dir=args.cache_dir, rebuild=args.rebuild)
        logger.info(
            f"{path.name}: {len(cache['valid_indices'])} valid patches "
            f"in {time.time() - started:.0f}s"
        )


if __name__ == "__main__":
    main()
