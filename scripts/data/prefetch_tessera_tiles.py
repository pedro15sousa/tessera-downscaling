"""Download the TESSERA tiles needed for station patches via GeoTessera.

Populates the local tile tree that ``extract_tessera_patches_local.py`` reads
(``--tiles-dir`` here == its ``--mount-dir``) with every tile any station's
patch touches, using the GeoTessera library's native per-tile download.

Why per-tile downloads driven by the patch geometry, rather than the
library's ``download_tiles_for_points``: that helper fetches only the tile
*containing* each point, but a patch centred near a tile edge straddles up to
four tiles. This script instead reuses the extractor's own patch-bbox
geometry (via ``shortlist_tessera_tiles``) to compute the exact overlap set,
so the downloaded tree contains precisely the tiles extraction will read --
no misses at tile edges and no over-fetching. The library's
``download_tile`` re-downloads unconditionally, so already-present tiles
(both the embedding ``.npy`` and its ``_scales.npy``, the two files the
extractor needs) are skipped here unless ``--force`` is given, which makes
re-runs resume where they stopped.

Usage (from the repo root; geotessera is a core dependency):
    uv run python scripts/data/prefetch_tessera_tiles.py \\
        --station-csv ingest/raw/ghcnh/station_list.csv \\
        --tiles-dir /path/to/tessera_tiles --years 2017

Then:
    TESSERA_V2_MOUNT=/path/to/tessera_tiles \\
    uv run python scripts/data/extract_tessera_patches_local.py \\
        --out-dir processed/tessera_station_patches

``--shortlist-csv`` accepts a previously written
``tessera_tiles_shortlist.csv`` instead of recomputing the tile set;
``--dry-run`` reports the tile set and what would be downloaded without
touching the network.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd
from shortlist_tessera_tiles import compute_patch_bbox, load_stations, tiles_for_bbox
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("prefetch_tessera_tiles")


def tile_set_for_stations(
    station_csv: Path, patch_size: int, pixel_res_m: float
) -> pd.DataFrame:
    """Unique tiles overlapping any station's patch bbox, as a dataframe."""
    stations = load_stations(station_csv)
    records: dict[str, tuple[float, float]] = {}
    for lat, lon in zip(
        stations["latitude"].to_numpy(), stations["longitude"].to_numpy(), strict=False
    ):
        bbox = compute_patch_bbox(float(lon), float(lat), patch_size, pixel_res_m)
        for name, c_lon, c_lat, *_ in tiles_for_bbox(*bbox):
            records.setdefault(name, (c_lon, c_lat))
    return pd.DataFrame(
        [(name, c_lon, c_lat) for name, (c_lon, c_lat) in records.items()],
        columns=["tile_name", "center_lon", "center_lat"],
    ).sort_values(["center_lat", "center_lon"])


def tile_present(tiles_dir: Path, name: str, year: int) -> bool:
    """True iff both files the extractor reads exist for this tile."""
    d = tiles_dir / str(year) / name
    return (d / f"{name}.npy").exists() and (d / f"{name}_scales.npy").exists()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--station-csv",
        type=Path,
        help="CSV with station locations (GHCNh or canonical lat/lon columns); "
        "the tile set is computed from their patch bboxes.",
    )
    src.add_argument(
        "--shortlist-csv",
        type=Path,
        help="A tessera_tiles_shortlist.csv written by shortlist_tessera_tiles.py, "
        "used as the tile set instead of recomputing it.",
    )
    parser.add_argument(
        "--tiles-dir",
        type=Path,
        required=True,
        help="Local tile tree to populate (= extract_tessera_patches_local.py's "
        "--mount-dir / $TESSERA_V2_MOUNT).",
    )
    parser.add_argument(
        "--years",
        type=int,
        nargs="+",
        default=[2017],
        help="Embedding years to download (default: 2017, the paper's).",
    )
    parser.add_argument(
        "--dataset-version",
        default="v2",
        help="GeoTessera dataset version (default: v2).",
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        default=128,
        help="Patch side length in pixels at 10 m; sizes the per-station bbox "
        "(default 128, matching the extraction).",
    )
    parser.add_argument("--pixel-res-m", type=float, default=10.0)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download tiles that are already present.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report the tile set and what would be downloaded; no network access.",
    )
    args = parser.parse_args()

    if args.station_csv is not None:
        tiles = tile_set_for_stations(
            args.station_csv, args.patch_size, args.pixel_res_m
        )
    else:
        tiles = pd.read_csv(args.shortlist_csv)[
            ["tile_name", "center_lon", "center_lat"]
        ]
    logger.info(f"Tile set: {len(tiles)} unique tiles, years {args.years}")

    n_present = n_downloaded = 0
    failed: list[str] = []
    for year in args.years:
        todo = [
            row
            for row in tiles.itertuples(index=False)
            if args.force or not tile_present(args.tiles_dir, row.tile_name, year)
        ]
        n_present += len(tiles) - len(todo)
        logger.info(
            f"[{year}] {len(tiles) - len(todo)} tiles already present, "
            f"{len(todo)} to download"
        )
        if args.dry_run or not todo:
            continue

        from geotessera import GeoTessera

        gt = GeoTessera(
            dataset_version=args.dataset_version, embeddings_dir=str(args.tiles_dir)
        )
        for row in tqdm(todo, desc=f"tiles {year}", unit="tile"):
            ok = False
            try:
                ok = gt.download_tile(row.center_lon, row.center_lat, year=year)
            except Exception as e:  # noqa: BLE001 -- keep going, report at the end
                logger.warning(f"{row.tile_name} ({year}): {type(e).__name__}: {e}")
            if ok:
                n_downloaded += 1
            else:
                failed.append(f"{row.tile_name} ({year})")

    logger.info(
        f"Done: {n_present} already present, {n_downloaded} downloaded, "
        f"{len(failed)} failed."
    )
    if failed:
        logger.error(
            "Failed tiles (not released for this dataset version/year, or a "
            "network error) — for unreleased embedding versions, hand the "
            "shortlist to the TESSERA team instead:\n  " + "\n  ".join(failed)
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
