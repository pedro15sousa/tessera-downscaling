#!/usr/bin/env python3
"""Stage 1 of the OlmoEarth arm: fetch the Sentinel-2 L2A monthly composites.

OlmoEarth publishes no precomputed global embedding product, so the benchmark
generates the embeddings locally in two stages:

1. (this script) build, per station, monthly Sentinel-2 L2A composites for the
   target year from the Microsoft Planetary Computer -- 64x64 px at 10 m, 12
   bands, up to 12 monthly timesteps -- stored as one uint16 ``.npy`` per year;
2. ``extract_olmoearth_embed.py`` runs the OlmoEarth encoder over that imagery
   on a GPU and writes the token-grid embeddings the VAE trains on.

Conventions match the AlphaEarth and TESSERA extractions: rows are aligned with
the station CSV, the station's 10 m pixel sits at patch index ``(S//2, S//2)``,
patches are north-up in the station's own UTM zone, and a progress file makes
the run resumable. A station with no imagery all year keeps a zero row and a
zero month mask (counted, not fatal).

Imagery details:

* Source: Planetary Computer STAC, collection ``sentinel-2-l2a``, which carries
  back-processed L2A from 2015 -- that is why 2017 is available here at all,
  unlike in most S2 archives. Early 2017 is sparser because Sentinel-2B only
  became operational mid-year.
* One STAC search per station-year; within each calendar month the least-cloudy
  scenes (by ``eo:cloud_cover``) are composited first-valid, up to
  ``--scenes-per-month`` of them.
* Bands are stored in OlmoEarth's ``sentinel2_l2a`` order:
  B02,B03,B04,B08 (10 m) | B05,B06,B07,B8A,B11,B12 (20 m) | B01,B09 (60 m),
  all warped onto the 64x64 @ 10 m patch grid (bilinear) through a WarpedVRT,
  which also handles scenes served in a neighbouring zone's CRS.
* Values are raw L2A digital numbers (uint16), which is what OlmoEarth's
  ``COMPUTED`` normalisation statistics expect; 0 is S2 nodata.

Optional dependencies (not in the default environment): ``pystac-client``,
``planetary-computer``, ``rasterio``, ``pyproj``. They are imported inside the
functions that need them, so ``--help`` works without them.

Usage (relative paths are interpreted under the data root):

    uv run python scripts/patch_encoder/extract/extract_olmoearth_imagery.py \\
        --years 2024 --limit 10
    uv run python scripts/patch_encoder/extract/extract_olmoearth_imagery.py \\
        --years 2017 --shard 0 4 --workers 32

Output files, per year, in ``--output-dir``:

    s2_<year>_p<S>.npy             (N, S, S, 12 months, 12 bands) uint16
    s2_<year>_p<S>_months.npy      (N, 12) uint8, 1 = the month has imagery
    s2_<year>_p<S>.progress*.json  resume bookkeeping
    imagery_metadata.json          parameters and per-year counts
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

from tessera_downscaling.paths import processed_dir, resolve

logger = logging.getLogger("extract_olmoearth_imagery")

STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
COLLECTION = "sentinel-2-l2a"
# OlmoEarth's sentinel2_l2a band order (its three band sets, concatenated).
BANDS = [
    "B02",
    "B03",
    "B04",
    "B08",
    "B05",
    "B06",
    "B07",
    "B8A",
    "B11",
    "B12",
    "B01",
    "B09",
]
N_MONTHS = 12
PIXEL_RES_M = 10.0
# The Planetary Computer throttles with 429/503 under load, so back off hard
# (5, 15, 45, 135, 300, 300 s) before giving up on a station. A failed station
# is skipped rather than fatal -- it stays un-completed, so a re-run retries it
# -- but many failures mean a real outage and abort the run.
RETRIES = 6
RETRY_SLEEP_S = 5.0
MAX_FAILED = 50


def utm_epsg(lat: float, lon: float) -> str:
    """The station's canonical UTM zone EPSG code.

    No Norway/Svalbard exceptions: the patch grid is ours, and scenes that
    arrive in another CRS are warped onto it.
    """
    zone = int(math.floor((lon + 180.0) / 6.0)) % 60 + 1
    return f"EPSG:{(32600 if lat >= 0 else 32700) + zone}"


def patch_transform(lat: float, lon: float, patch_size: int):
    """North-up affine transform of the station-centred patch grid.

    The station's pixel is at ``(S//2, S//2)`` and the grid is pixel-snapped
    exactly as in the AlphaEarth extractor, so the two products' patches line
    up station by station.
    """
    import rasterio.transform
    from pyproj import Transformer

    crs = utm_epsg(lat, lon)
    transformer = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    x, y = transformer.transform(lon, lat)
    half = patch_size // 2
    west = PIXEL_RES_M * (math.floor(x / PIXEL_RES_M) - half)
    north = PIXEL_RES_M * (math.floor(y / PIXEL_RES_M) + half)
    return crs, rasterio.transform.Affine(
        PIXEL_RES_M, 0.0, west, 0.0, -PIXEL_RES_M, north
    )


def read_scene(item, crs: str, transform, patch_size: int) -> np.ndarray:
    """Read all 12 bands of one STAC item onto the patch grid.

    Scenes with processing baseline >= 04.00 (after 2022-01-25, plus any
    reprocessed older Collection-1 scene) carry a +1000 BOA offset. OlmoEarth's
    pretraining data was harmonised back to the pre-offset convention by rslearn
    (``harmonize: true``, i.e. ``clip(DN, 1000) - 1000``) and its normalisation
    statistics expect harmonised values, so the same formula is applied here per
    scene -- including its quirk that valid dark pixels with DN <= 1000 collapse
    to 0. The Planetary Computer serves 2017 as baseline 02.x (no offset) and
    2024 uniformly as 05.1x.
    """
    import planetary_computer as pc
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.vrt import WarpedVRT

    try:
        baseline = float(item.properties.get("s2:processing_baseline", "0"))
    except ValueError:
        baseline = 0.0
    harmonize = baseline >= 4.0

    signed = pc.sign(item)
    out = np.zeros((patch_size, patch_size, len(BANDS)), dtype=np.uint16)
    for band_index, band in enumerate(BANDS):
        href = signed.assets[band].href
        with rasterio.open(href) as src:
            with WarpedVRT(
                src,
                crs=crs,
                transform=transform,
                width=patch_size,
                height=patch_size,
                resampling=Resampling.bilinear,
            ) as vrt:
                data = vrt.read(1)
        if harmonize:
            data = np.clip(data, 1000, None) - 1000
        out[:, :, band_index] = data
    return out


def fetch_station_year(
    catalog, lat: float, lon: float, year: int, patch_size: int, scenes_per_month: int
) -> tuple[np.ndarray, np.ndarray]:
    """Build one station's ``(S, S, 12, 12)`` cube and its ``(12,)`` month mask."""
    crs, transform = patch_transform(lat, lon, patch_size)
    search = catalog.search(
        collections=[COLLECTION],
        intersects={"type": "Point", "coordinates": [lon, lat]},
        datetime=f"{year}-01-01/{year}-12-31",
    )
    items = list(search.items())

    by_month: dict[int, list] = {}
    for item in items:
        by_month.setdefault(item.datetime.month - 1, []).append(item)

    cube = np.zeros((patch_size, patch_size, N_MONTHS, len(BANDS)), dtype=np.uint16)
    valid = np.zeros(N_MONTHS, dtype=np.uint8)
    for month, month_items in by_month.items():
        month_items.sort(key=lambda it: it.properties.get("eo:cloud_cover", 101))
        composite = None
        for item in month_items[:scenes_per_month]:
            scene = read_scene(item, crs, transform, patch_size)
            if composite is None:
                composite = scene
            else:
                hole = ~np.any(composite != 0, axis=-1)  # all-band nodata pixels
                composite[hole] = scene[hole]
            if np.all(np.any(composite != 0, axis=-1)):
                break  # fully covered, no need for more scenes
        if composite is not None and composite.any():
            cube[:, :, month, :] = composite
            valid[month] = 1
    return cube, valid


def extract_year(
    stations: pd.DataFrame,
    year: int,
    patch_size: int,
    output_dir: Path,
    workers: int,
    scenes_per_month: int,
    shard: tuple[int, int] | None,
    limit: int | None,
    force: bool,
) -> dict:
    """Fetch every station's imagery for one year into memory-mapped ``.npy``s."""
    from pystac_client import Client

    n_stations = len(stations)
    shape = (n_stations, patch_size, patch_size, N_MONTHS, len(BANDS))
    out_path = output_dir / f"s2_{year}_p{patch_size}.npy"
    months_path = output_dir / f"s2_{year}_p{patch_size}_months.npy"
    shard_tag = f".shard{shard[0]}of{shard[1]}" if shard else ""
    progress_path = out_path.with_suffix(f".progress{shard_tag}.json")

    if out_path.exists() and not force:
        cube = np.load(str(out_path), mmap_mode="r")
        if cube.shape != shape:
            raise ValueError(f"{out_path} shape {cube.shape} != {shape}")
        months = np.load(str(months_path), mmap_mode="r")
    else:
        logger.info(f"Pre-allocating {np.prod(shape) * 2 / 1e9:.1f} GB at {out_path}")
        cube = np.lib.format.open_memmap(
            str(out_path), mode="w+", dtype=np.uint16, shape=shape
        )
        months = np.lib.format.open_memmap(
            str(months_path), mode="w+", dtype=np.uint8, shape=(n_stations, N_MONTHS)
        )
    # Positioned pwrite() instead of mmap stores: Lustre mmap writes are not
    # coherent across nodes and .npy rows are not page-aligned, so concurrent
    # shard jobs could clobber each other's row edges (extract_alphaearth.py
    # carries the full rationale).
    cube_off, cube_row_bytes = int(cube.offset), int(np.prod(shape[1:])) * 2
    months_off, months_row_bytes = int(months.offset), N_MONTHS
    del cube, months
    cube_fd = os.open(str(out_path), os.O_RDWR)
    months_fd = os.open(str(months_path), os.O_RDWR)

    # Union of every progress file of this output (see extract_alphaearth.py).
    completed: set[int] = set()
    if not force:
        for progress_file in out_path.parent.glob(out_path.stem + ".progress*.json"):
            try:
                completed |= set(json.loads(progress_file.read_text())["completed"])
            except (json.JSONDecodeError, KeyError):
                logger.warning(f"Ignoring unreadable progress file {progress_file}")

    todo = np.arange(n_stations)
    if limit is not None:
        todo = todo[:limit]
    if shard is not None:
        todo = todo[todo % shard[1] == shard[0]]
    todo = [int(i) for i in todo if int(i) not in completed]
    logger.info(
        f"Year {year}: {len(todo)} stations to fetch "
        f"({len(completed)} already done in this shard)"
    )

    lock = threading.Lock()
    counters = {"done": 0, "no_imagery": 0, "skipped_external": 0, "t0": time.time()}
    failed: dict[int, str] = {}
    failed_path = out_path.with_suffix(f".failed{shard_tag}.json")
    tls = threading.local()

    def flush() -> None:
        progress_path.write_text(json.dumps({"completed": sorted(completed)}))
        if failed:
            failed_path.write_text(json.dumps(failed, indent=1))

    # Live cross-run dedup (see extract_alphaearth.py).
    external: dict = {"done": set(), "t": 0.0}

    def refresh_external_locked() -> None:
        if time.time() - external["t"] < 60:
            return
        done: set[int] = set()
        for progress_file in out_path.parent.glob(out_path.stem + ".progress*.json"):
            if progress_file == progress_path:
                continue
            try:
                done |= set(json.loads(progress_file.read_text())["completed"])
            except (json.JSONDecodeError, KeyError, OSError):
                pass
        external["done"] = done
        external["t"] = time.time()

    def process(i: int) -> None:
        if not hasattr(tls, "catalog"):
            tls.catalog = Client.open(STAC_URL)
        with lock:
            refresh_external_locked()
            if i in external["done"]:
                if i not in completed:
                    completed.add(i)  # another run fetched it
                    counters["skipped_external"] += 1
                return
        lat = float(stations["latitude"].iloc[i])
        lon = float(stations["longitude"].iloc[i])
        last_err: Exception | None = None
        for attempt in range(RETRIES):
            try:
                station_cube, month_mask = fetch_station_year(
                    tls.catalog, lat, lon, year, patch_size, scenes_per_month
                )
                last_err = None
                break
            except Exception as err:
                last_err = err
                time.sleep(min(RETRY_SLEEP_S * 3**attempt, 300))
        if last_err is not None:
            logger.warning(
                f"Station {i} ({lat:.4f}, {lon:.4f}) failed after {RETRIES} "
                f"retries, skipping: {last_err}"
            )
            with lock:
                failed[i] = f"{type(last_err).__name__}: {last_err}"
                n_failed = len(failed)
            if n_failed > MAX_FAILED:
                raise RuntimeError(
                    f"{n_failed} stations failed -- aborting; this looks like a "
                    f"service outage. Failed stations are not marked complete, "
                    f"so a re-run retries them."
                ) from last_err
            return
        os.pwrite(cube_fd, station_cube.tobytes(), cube_off + i * cube_row_bytes)
        os.pwrite(months_fd, month_mask.tobytes(), months_off + i * months_row_bytes)
        with lock:
            completed.add(i)
            counters["done"] += 1
            if not month_mask.any():
                counters["no_imagery"] += 1
            if counters["done"] % 50 == 0:
                flush()
                rate = counters["done"] / max(time.time() - counters["t0"], 1)
                eta_h = (len(todo) - counters["done"]) / max(rate, 1e-9) / 3600
                logger.info(
                    f"  {counters['done']}/{len(todo)} ({rate:.2f}/s, "
                    f"ETA {eta_h:.1f} h, {counters['no_imagery']} without imagery)"
                )

    errors = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(process, i) for i in todo]
        for future in as_completed(futures):
            exc = future.exception()
            if exc is not None:
                errors.append(exc)
                for pending in futures:
                    pending.cancel()
                break
    flush()
    os.close(cube_fd)
    os.close(months_fd)
    if errors:
        raise errors[0]

    logger.info(
        f"Year {year}: shard complete ({counters['done']} fetched, "
        f"{counters['no_imagery']} without imagery, {len(failed)} failed, "
        f"{counters['skipped_external']} done by other runs)"
    )
    return {
        "file": out_path.name,
        "shape": list(shape),
        "shard": list(shard) if shard else None,
        "n_no_imagery_this_shard": counters["no_imagery"],
        "n_failed_this_shard": len(failed),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fetch Sentinel-2 monthly composites for the OlmoEarth encoder."
    )
    parser.add_argument(
        "--station-csv",
        default=str(
            processed_dir("tessera_station_patches", "station_list_filtered.csv")
        ),
        help="Station CSV the output rows are aligned with.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(processed_dir("olmoearth_imagery")),
        help="Where the imagery cubes, month masks and metadata are written.",
    )
    parser.add_argument("--years", type=int, nargs="+", default=[2017, 2024])
    parser.add_argument(
        "--patch-size",
        type=int,
        default=64,
        help="Patch side in 10 m pixels (64 = OlmoEarth's canonical crop).",
    )
    parser.add_argument("--scenes-per-month", type=int, default=3)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only the first N stations (smoke test).",
    )
    parser.add_argument(
        "--shard",
        type=int,
        nargs=2,
        metavar=("K", "N"),
        default=None,
        help="Process the stations with index %% N == K.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recreate the output files and ignore the progress files.",
    )
    parser.add_argument(
        "--init-only",
        action="store_true",
        help="Pre-allocate the output mmaps, then exit; run this before sharded "
        "jobs so they never race on file creation.",
    )
    return parser


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("pystac_client").setLevel(logging.WARNING)

    args = build_parser().parse_args()

    output_dir = resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    station_csv = resolve(args.station_csv)
    stations = pd.read_csv(station_csv)
    logger.info(f"Loaded {len(stations)} stations from {station_csv}")

    meta_path = output_dir / "imagery_metadata.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    meta.setdefault("files", {})
    meta.update(
        {
            "source": f"Planetary Computer STAC, collection {COLLECTION}",
            "station_csv": str(station_csv),
            "n_stations": len(stations),
            "patch_size": args.patch_size,
            "bands": BANDS,
            "layout": "(N, H, W, T=12 months, C=12 bands) uint16 raw L2A DNs, "
            "north-up, station pixel at (S//2, S//2), native UTM zone "
            "per station",
            "compositing": f"per calendar month, least-cloudy first-valid, up to "
            f"{args.scenes_per_month} scenes",
        }
    )

    if args.init_only:
        for year in args.years:
            cube_path = output_dir / f"s2_{year}_p{args.patch_size}.npy"
            months_path = output_dir / f"s2_{year}_p{args.patch_size}_months.npy"
            if not cube_path.exists():
                logger.info(f"Pre-allocating {cube_path}")
                np.lib.format.open_memmap(
                    str(cube_path),
                    mode="w+",
                    dtype=np.uint16,
                    shape=(
                        len(stations),
                        args.patch_size,
                        args.patch_size,
                        N_MONTHS,
                        len(BANDS),
                    ),
                )
                np.lib.format.open_memmap(
                    str(months_path),
                    mode="w+",
                    dtype=np.uint8,
                    shape=(len(stations), N_MONTHS),
                )
        logger.info("Init complete.")
        return

    shard = tuple(args.shard) if args.shard else None
    for year in args.years:
        logger.info(f"=== Year {year} ===")
        info = extract_year(
            stations,
            year,
            args.patch_size,
            output_dir,
            args.workers,
            args.scenes_per_month,
            shard,
            args.limit,
            args.force,
        )
        meta["files"].setdefault(info["file"], {}).update(
            {k: v for k, v in info.items() if k != "file"}
        )
    meta_path.write_text(json.dumps(meta, indent=2))
    logger.info(f"Metadata saved to {meta_path}")


if __name__ == "__main__":
    main()
