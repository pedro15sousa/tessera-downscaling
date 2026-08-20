#!/usr/bin/env python3
"""Extract AlphaEarth (Satellite Embedding v1) patches at station locations.

The foundation-model benchmark of the paper swaps the TESSERA surface
embedding for another one and retrains the patch encoder on it. This script
produces the AlphaEarth arm's input: 64-channel annual embedding patches
centred on the GHCNh stations, read straight from the public GCS bucket
``gs://alphaearth_foundations`` over HTTP range requests -- no Earth Engine
account and no quota.

The output follows the same contract as the TESSERA extraction
(``scripts/data/extract_tessera_patches_local.py``): one float32 ``.npy`` per
year of shape ``(N, S, S, 64)``, row-aligned with the station CSV, all-zero
rows for stations without coverage. The station sits at pixel ``(S//2, S//2)``,
so the dataset's station-centred crop behaves identically to TESSERA's.

Source-data facts that the code below depends on (from the bucket's README,
each verified against the index during the extraction):

* COGs are 8192x8192 px, 64 channels, int8, 10 m, one directory per UTM zone;
  ``-128`` is nodata (a masked pixel is masked in every channel).
* Values de-quantise to the native ``[-1, 1]`` range as
  ``((v / 127.5) ** 2) * sign(v)``.
* The COGs are stored **south-up** (y pixel size ``+10``, transform origin at
  ``utm_south``). Everything here is converted to conventional north-up patches
  (row 0 = north). The flip was validated against the TESSERA 2024 patches by
  edge-map correlation -- the identity orientation won 27 of 28 test stations.
* Tiles inside a UTM zone share one 10 m grid, so a patch straddling a COG
  boundary is stitched from up to four neighbours by integer arithmetic alone.

Failure semantics: a station with no AlphaEarth coverage (e.g. the high Arctic;
109 of 38,870) keeps its zero row and is counted in the metadata. Read errors
are retried and then raised, so a network-restricted node aborts loudly instead
of quietly writing zeros.

Optional dependencies (not in the default environment): ``rasterio``,
``pyproj``, ``pyarrow``, ``shapely``. They are imported inside the functions
that need them, so ``--help`` works without them.

Usage (relative paths are interpreted under the data root):

    # Smoke test: first 30 stations of one year.
    uv run python scripts/patch_encoder/extract/extract_alphaearth.py \\
        --years 2024 --limit 30

    # One shard of a sharded run (see slurm/submit_extract.sh).
    uv run python scripts/patch_encoder/extract/extract_alphaearth.py \\
        --years 2017 --shard 0 4 --workers 24

    # Recount nonzero rows once every shard has finished.
    uv run python scripts/patch_encoder/extract/extract_alphaearth.py \\
        --years 2017 2024 --report

Output files, per year, in ``--output-dir``:

    patch_embeddings_alphaearth_<year>_p<S>.npy            (N, S, S, 64) float32
    patch_embeddings_alphaearth_<year>_p<S>.progress*.json resume bookkeeping
    extraction_metadata.json                               parameters and counts
    index/aef_index.parquet                                tile index (downloaded)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

from tessera_downscaling.paths import processed_dir, resolve

logger = logging.getLogger("extract_alphaearth")

GCS_HTTP = "https://storage.googleapis.com/alphaearth_foundations"
INDEX_URL = f"{GCS_HTTP}/satellite_embedding/v1/annual/aef_index.parquet"
TILE_SIZE_PX = 8192
PIXEL_RES_M = 10.0
TILE_SIZE_M = TILE_SIZE_PX * PIXEL_RES_M
EMBED_DIM = 64
NODATA = -128

# GDAL/curl tuning. The COGs are band-interleaved with 1024^2 zstd blocks, so a
# 128^2 window is ~64 scattered ~1 MB range reads; a large chunk size plus
# HTTP/2 multiplexing roughly halves the cold-read latency (51 s -> 23 s).
GDAL_ENV = dict(
    GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
    CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tiff",
    CPL_VSIL_CURL_CHUNK_SIZE=2 * 1024 * 1024,
    CPL_VSIL_CURL_CACHE_SIZE=200 * 1024 * 1024,
    VSI_CACHE=True,
    VSI_CACHE_SIZE=100 * 1024 * 1024,
    GDAL_HTTP_MULTIPLEX="YES",
    GDAL_HTTP_VERSION="2",
    GDAL_HTTP_MAX_RETRY=5,
    GDAL_HTTP_RETRY_DELAY=2,
    GDAL_NUM_THREADS="2",
    # GDAL sets no curl timeout by default, and a tarpitted connection then
    # hangs its worker forever. With these, a stalled read fails into the retry
    # path below, which closes the reader and reconnects.
    GDAL_HTTP_TIMEOUT=120,
    GDAL_HTTP_CONNECTTIMEOUT=30,
)

READ_RETRIES = 3
RETRY_SLEEP_S = 5.0
# Stations per worker task: long enough to share a COG's block cache, short
# enough that no single task holds a worker at the tail of a year.
TILE_CHUNK = 25


def dequantize(raw: np.ndarray) -> np.ndarray:
    """int8 quantised values -> float32 in ``[-1, 1]``; nodata (-128) -> 0."""
    v = raw.astype(np.float32)
    out = ((v / 127.5) ** 2) * np.sign(v)
    out[raw == NODATA] = 0.0
    return out


def ensure_index(index_path: Path) -> Path:
    """Download the AlphaEarth GeoParquet tile index unless it is already here."""
    if index_path.exists():
        return index_path
    index_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Downloading AEF tile index to {index_path} (~70 MB)")
    tmp = index_path.with_suffix(".parquet.part")
    urllib.request.urlretrieve(INDEX_URL, tmp)
    tmp.rename(index_path)
    return index_path


def load_index(index_path: Path, years: list[int]) -> pd.DataFrame:
    """Load the tile index, restricted to ``years``, with an HTTP URL per tile."""
    import pyarrow.parquet as pq

    cols = [
        "path",
        "crs",
        "year",
        "utm_zone",
        "utm_west",
        "utm_south",
        "utm_east",
        "utm_north",
        "wgs84_west",
        "wgs84_south",
        "wgs84_east",
        "wgs84_north",
    ]
    df = pq.read_table(index_path, columns=cols).to_pandas()
    df = df[df["year"].isin(years)].reset_index(drop=True)
    df["url"] = df["path"].str.replace(
        "gs://alphaearth_foundations", GCS_HTTP, regex=False
    )
    logger.info(f"Tile index: {len(df)} tiles for years {years}")
    return df


def assign_stations_to_tiles(
    stations: pd.DataFrame, tiles: pd.DataFrame, year: int
) -> pd.DataFrame:
    """Find the COG containing each station for ``year``.

    Candidates come from an STRtree over the tiles' WGS84 bounding boxes; each
    candidate is then tested precisely in its own UTM CRS. A point inside more
    than one tile (zone-boundary corner cases) goes to the tile where it is
    furthest from any edge, which keeps the surrounding context on one grid.

    Returns:
        A frame row-aligned with ``stations`` holding ``tile_idx`` (``-1`` when
        there is no coverage) and the station's ``utm_x`` / ``utm_y`` in the
        winning tile's CRS.
    """
    import shapely
    from pyproj import Transformer

    yt = tiles[tiles["year"] == year].reset_index(drop=True)
    boxes = shapely.box(
        yt["wgs84_west"].values,
        yt["wgs84_south"].values,
        yt["wgs84_east"].values,
        yt["wgs84_north"].values,
    )
    tree = shapely.STRtree(boxes)
    pts = shapely.points(stations["longitude"].values, stations["latitude"].values)
    pt_idx, box_idx = tree.query(pts, predicate="intersects")

    n = len(stations)
    tile_idx = np.full(n, -1, dtype=np.int64)
    utm_x = np.full(n, np.nan)
    utm_y = np.full(n, np.nan)
    best_margin = np.full(n, -np.inf)

    # Group the candidate pairs by CRS so each coordinate transform is one
    # vectorised call rather than one per station.
    cand = pd.DataFrame({"si": pt_idx, "ti": box_idx})
    cand["crs"] = yt["crs"].values[cand["ti"].values]
    for crs, grp in cand.groupby("crs"):
        transformer = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
        xs, ys = transformer.transform(
            stations["longitude"].values[grp["si"].values],
            stations["latitude"].values[grp["si"].values],
        )
        xs, ys = np.asarray(xs), np.asarray(ys)
        tw = yt["utm_west"].values[grp["ti"].values]
        te = yt["utm_east"].values[grp["ti"].values]
        ts = yt["utm_south"].values[grp["ti"].values]
        tn = yt["utm_north"].values[grp["ti"].values]
        margin = np.minimum(np.minimum(xs - tw, te - xs), np.minimum(ys - ts, tn - ys))
        inside = margin >= 0
        for si, ti, x, y, m in zip(
            grp["si"].values[inside],
            grp["ti"].values[inside],
            xs[inside],
            ys[inside],
            margin[inside],
            strict=True,
        ):
            if m > best_margin[si]:
                best_margin[si] = m
                tile_idx[si] = ti
                utm_x[si] = x
                utm_y[si] = y

    n_missing = int((tile_idx < 0).sum())
    logger.info(
        f"Year {year}: assigned {n - n_missing}/{n} stations to tiles "
        f"({n_missing} without coverage)"
    )
    return pd.DataFrame({"tile_idx": tile_idx, "utm_x": utm_x, "utm_y": utm_y})


class TileReader:
    """Reads north-up patch windows out of the south-up COGs of one UTM zone.

    COG handles are opened lazily and cached per instance. rasterio datasets
    are not thread-safe, so each worker task builds its own reader and closes
    it on the way out; a longer-lived, thread-persistent version accumulated
    curl handles until TIME_WAIT socket exhaustion stalled the run.
    """

    def __init__(self, zone_lookup: dict) -> None:
        # zone_lookup: (year, utm_zone) -> {(utm_west, utm_south): tile row}
        self.zone_lookup = zone_lookup
        self._handles: dict[str, object] = {}

    def _open(self, url: str):
        import rasterio

        if url not in self._handles:
            self._handles[url] = rasterio.open(f"/vsicurl/{url}")
        return self._handles[url]

    def close(self) -> None:
        for handle in self._handles.values():
            try:
                handle.close()
            except Exception:
                pass
        self._handles.clear()

    def read_patch(
        self, year: int, zone: str, utm_x: float, utm_y: float, patch_size: int
    ) -> np.ndarray:
        """Read a north-up ``(S, S, 64)`` patch centred on ``(utm_x, utm_y)``.

        The station's pixel lands at patch index ``(S//2, S//2)`` -- ``S//2``
        rows of context to the north, ``S//2 - 1`` to the south, the same
        convention as the TESSERA extraction. Parts of the window outside the
        available tiles, and nodata pixels, come back as zero.

        All arithmetic is in integer pixel indices on the zone-wide 10 m grid,
        anchored at any tile's origin modulo the tile stride (tiles within a
        zone share that grid; verified for all 240 zone-years of the index).
        """
        half = patch_size // 2
        patch = np.zeros((patch_size, patch_size, EMBED_DIM), dtype=np.float32)

        lookup = self.zone_lookup.get((year, zone), {})
        if not lookup:
            return patch

        import rasterio.windows as windows

        any_tile = next(iter(lookup.values()))
        ox = any_tile["utm_west"] % TILE_SIZE_M
        oy = any_tile["utm_south"] % TILE_SIZE_M

        # Station pixel in absolute grid indices: the column index increases
        # eastwards, the s-row index increases NORTHwards (south-up, as stored).
        c_s = int(np.floor((utm_x - ox) / PIXEL_RES_M))
        s_r = int(np.floor((utm_y - oy) / PIXEL_RES_M))

        c_lo, c_hi = c_s - half, c_s - half + patch_size  # [lo, hi)
        s_lo, s_hi = s_r - half + 1, s_r + half + 1  # [lo, hi)

        for wid in range(c_lo // TILE_SIZE_PX, (c_hi - 1) // TILE_SIZE_PX + 1):
            for sid in range(s_lo // TILE_SIZE_PX, (s_hi - 1) // TILE_SIZE_PX + 1):
                key = (wid * TILE_SIZE_M + ox, sid * TILE_SIZE_M + oy)
                tile = lookup.get(key)
                if tile is None:
                    continue
                tcol0 = wid * TILE_SIZE_PX  # absolute column of the tile's column 0
                trow0 = sid * TILE_SIZE_PX  # absolute s-row of the tile's row 0
                c0, c1 = max(c_lo, tcol0), min(c_hi, tcol0 + TILE_SIZE_PX)
                s0, s1 = max(s_lo, trow0), min(s_hi, trow0 + TILE_SIZE_PX)
                if c1 <= c0 or s1 <= s0:
                    continue
                dataset = self._open(tile["url"])
                window = windows.Window(c0 - tcol0, s0 - trow0, c1 - c0, s1 - s0)
                raw = dataset.read(window=window)
                # raw is (64, h, w) with row 0 southernmost; the patch is
                # north-up, so patch_row = (s_hi - 1) - s.
                pr0, pr1 = s_hi - s1, s_hi - s0
                pc0, pc1 = c0 - c_lo, c1 - c_lo
                patch[pr0:pr1, pc0:pc1, :] = np.transpose(
                    dequantize(raw[:, ::-1, :]), (1, 2, 0)
                )
        return patch


def extract_year(
    stations: pd.DataFrame,
    tiles: pd.DataFrame,
    year: int,
    patch_size: int,
    output_dir: Path,
    workers: int,
    shard: tuple[int, int] | None,
    limit: int | None,
    force: bool,
) -> dict:
    """Extract every patch of one year into a memory-mapped ``.npy``."""
    n_stations = len(stations)
    shape = (n_stations, patch_size, patch_size, EMBED_DIM)
    out_path = output_dir / f"patch_embeddings_alphaearth_{year}_p{patch_size}.npy"

    shard_tag = f".shard{shard[0]}of{shard[1]}" if shard else ""
    progress_path = out_path.with_suffix(f".progress{shard_tag}.json")

    if out_path.exists() and not force:
        logger.info(f"Resuming into existing {out_path}")
        arr = np.load(str(out_path), mmap_mode="r")
        if arr.shape != shape:
            raise ValueError(
                f"{out_path} has shape {arr.shape}, expected {shape}. "
                f"Delete it or change --patch-size."
            )
    else:
        logger.info(f"Pre-allocating {np.prod(shape) * 4 / 1e9:.1f} GB at {out_path}")
        arr = np.lib.format.open_memmap(
            str(out_path), mode="w+", dtype=np.float32, shape=shape
        )
    # Rows are written with positioned pwrite() syscalls rather than through
    # the mmap: Lustre guarantees POSIX coherency for write() from several
    # nodes but not for mmap stores, and .npy rows are not page-aligned, so
    # concurrent shard jobs would share boundary pages and could clobber each
    # other's row edges on writeback.
    data_offset = int(arr.offset)
    row_bytes = int(np.prod(shape[1:])) * 4
    del arr
    out_fd = os.open(str(out_path), os.O_RDWR)

    # Resume from the union of every progress file of this output, so an
    # unsharded run and any shard jobs compose (each still writes only its own
    # file). A station another run finishes after this snapshot may be fetched
    # twice -- wasteful, but both runs write identical bytes.
    completed: set[int] = set()
    if not force:
        for progress_file in out_path.parent.glob(out_path.stem + ".progress*.json"):
            try:
                completed |= set(json.loads(progress_file.read_text())["completed"])
            except (json.JSONDecodeError, KeyError):
                logger.warning(f"Ignoring unreadable progress file {progress_file}")
        if completed:
            logger.info(
                f"Resuming: {len(completed)} stations already done across all "
                f"progress files"
            )

    assign = assign_stations_to_tiles(stations, tiles, year)
    yt = tiles[tiles["year"] == year].reset_index(drop=True)

    # Zone lookup for cross-tile stitching: (year, zone) -> {(west, south): row}
    zone_lookup: dict = {}
    for _, row in yt.iterrows():
        zone_lookup.setdefault((year, row["utm_zone"]), {})[
            (row["utm_west"], row["utm_south"])
        ] = row.to_dict()

    todo = np.arange(n_stations)
    if limit is not None:
        todo = todo[:limit]
    if shard is not None:
        k, n = shard
        todo = todo[todo % n == k]
    todo = [int(i) for i in todo if int(i) not in completed]

    n_missing = 0
    for i in list(todo):
        if assign["tile_idx"].iloc[i] < 0:
            completed.add(i)  # no coverage: leave the zero row in place
            todo.remove(i)
            n_missing += 1

    # Group by primary tile so stations sharing a COG share its block cache.
    by_tile: dict[int, list[int]] = {}
    for i in todo:
        by_tile.setdefault(int(assign["tile_idx"].iloc[i]), []).append(i)
    logger.info(
        f"Year {year}: {len(todo)} stations to fetch across {len(by_tile)} tiles "
        f"({n_missing} no-coverage, {len(completed) - n_missing} already done)"
    )

    lock = threading.Lock()
    counters = {
        "done": 0,
        "centre_nonzero": 0,
        "skipped_external": 0,
        "t0": time.time(),
    }
    tls = threading.local()

    # Live cross-run dedup: re-union the other runs' progress files about once a
    # minute so concurrent runs stop duplicating each other's work quickly.
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

    def flush_progress() -> None:
        # The progress json only -- deliberately no msync of the patch file: an
        # msync over a ~163 GB Lustre mmap serialises every worker (observed 10x
        # slowdown). The kernel writes dirty pages back on its own, so rows
        # survive a crashed process; only a node crash can lose recent pages,
        # which the --report pass then shows as zero-centre rows.
        progress_path.write_text(json.dumps({"completed": sorted(completed)}))

    def process_tile(tile_idx: int, station_ids: list[int]) -> None:
        import rasterio

        if not hasattr(tls, "env"):
            tls.env = rasterio.Env(**GDAL_ENV)
            tls.env.__enter__()
        zone = yt["utm_zone"].iloc[tile_idx]
        reader = TileReader(zone_lookup)
        try:
            for i in station_ids:
                with lock:
                    refresh_external_locked()
                    done_elsewhere = i in external["done"]
                    if done_elsewhere and i not in completed:
                        completed.add(i)  # another run fetched it
                        counters["skipped_external"] += 1
                if done_elsewhere:
                    continue
                last_err: Exception | None = None
                for attempt in range(READ_RETRIES):
                    try:
                        patch = reader.read_patch(
                            year,
                            zone,
                            float(assign["utm_x"].iloc[i]),
                            float(assign["utm_y"].iloc[i]),
                            patch_size,
                        )
                        last_err = None
                        break
                    except Exception as err:
                        last_err = err
                        reader.close()  # reconnect on the retry
                        time.sleep(RETRY_SLEEP_S * (attempt + 1))
                if last_err is not None:
                    raise RuntimeError(
                        f"Station {i} (tile {tile_idx}) failed after "
                        f"{READ_RETRIES} retries: {last_err}"
                    ) from last_err
                # Commit per station, so resume state stays live even while a
                # long tile group is in flight. The row write stays outside the
                # lock (rows are disjoint; serialising 4 MB writes under the
                # global lock cost a factor of ten).
                os.pwrite(out_fd, patch.tobytes(), data_offset + i * row_bytes)
                with lock:
                    completed.add(i)
                    counters["done"] += 1
                    centre = patch_size // 2
                    if np.any(patch[centre, centre, :] != 0):
                        counters["centre_nonzero"] += 1
                    if counters["done"] % 100 == 0:
                        flush_progress()
                        rate = counters["done"] / max(time.time() - counters["t0"], 1)
                        eta_h = (len(todo) - counters["done"]) / max(rate, 1e-9) / 3600
                        logger.info(
                            f"  {counters['done']}/{len(todo)} stations "
                            f"({rate:.2f}/s, ETA {eta_h:.1f} h, "
                            f"{counters['centre_nonzero']} centre-nonzero)"
                        )
        finally:
            reader.close()

    # Largest tile groups first: better thread utilisation at the tail.
    tile_items = [
        (tile_idx, ids[j : j + TILE_CHUNK])
        for tile_idx, ids in sorted(by_tile.items(), key=lambda kv: -len(kv[1]))
        for j in range(0, len(ids), TILE_CHUNK)
    ]
    errors = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(process_tile, tile_idx, ids): tile_idx
            for tile_idx, ids in tile_items
        }
        for future in as_completed(futures):
            exc = future.exception()
            if exc is not None:
                errors.append(exc)
                for pending in futures:
                    pending.cancel()
                break

    flush_progress()
    os.close(out_fd)
    if errors:
        raise errors[0]

    logger.info(
        f"Year {year}: shard complete ({counters['done']} fetched, "
        f"{n_missing} no-coverage, "
        f"{counters['skipped_external']} done by other runs)"
    )
    return {
        "file": out_path.name,
        "shape": list(shape),
        "n_stations": n_stations,
        "shard": list(shard) if shard else None,
        "n_no_coverage_this_shard": n_missing,
    }


def report(years: list[int], patch_size: int, output_dir: Path, meta: dict) -> None:
    """Scan the finished ``.npy`` files and record their nonzero counts."""
    centre = patch_size // 2
    for year in years:
        out_path = output_dir / f"patch_embeddings_alphaearth_{year}_p{patch_size}.npy"
        if not out_path.exists():
            continue
        mmap = np.load(str(out_path), mmap_mode="r")
        n_patches = mmap.shape[0]
        n_nonzero = 0
        n_centre = 0
        for i in range(n_patches):
            if np.any(mmap[i, centre, centre, :] != 0):
                n_centre += 1
                n_nonzero += 1
            elif np.any(mmap[i] != 0):
                n_nonzero += 1
        meta["files"][out_path.name] = {
            "shape": list(mmap.shape),
            "n_stations": n_patches,
            "n_nonzero": n_nonzero,
            "n_centre_nonzero": n_centre,
        }
        logger.info(
            f"{out_path.name}: {n_nonzero} nonzero, {n_centre} centre-nonzero "
            f"of {n_patches}"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract AlphaEarth embedding patches at station locations."
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
        default=str(processed_dir("alphaearth_station_patches")),
        help="Where the patch file, progress files and metadata are written.",
    )
    parser.add_argument("--years", type=int, nargs="+", default=[2017, 2024])
    parser.add_argument("--patch-size", type=int, default=128)
    parser.add_argument(
        "--workers",
        type=int,
        default=24,
        help="Download threads. Stay near the default: 64 or more exhausted the "
        "node's TIME_WAIT sockets, which corrupted in-flight zstd streams.",
    )
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
        "--report",
        action="store_true",
        help="Only (re)compute the metadata of existing outputs.",
    )
    parser.add_argument(
        "--init-only",
        action="store_true",
        help="Pre-allocate the output mmaps and download the tile index, then "
        "exit; run this before sharded jobs so they never race on file creation.",
    )
    return parser


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    args = build_parser().parse_args()

    output_dir = resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    station_csv = resolve(args.station_csv)
    stations = pd.read_csv(station_csv)
    if "latitude" not in stations or "longitude" not in stations:
        raise SystemExit(f"{station_csv} must have latitude/longitude columns")
    logger.info(f"Loaded {len(stations)} stations from {station_csv}")

    meta_path = output_dir / "extraction_metadata.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    meta.setdefault("files", {})
    meta.update(
        {
            "source": "gs://alphaearth_foundations (Satellite Embedding v1 annual)",
            "license": "CC-BY 4.0 -- dataset produced by Google and Google DeepMind",
            "station_csv": str(station_csv),
            "n_stations": len(stations),
            "years": sorted(set(meta.get("years", []) + args.years)),
            "patch_size": args.patch_size,
            "embed_dim": EMBED_DIM,
            "grid": "native UTM per zone, 10 m, north-up (COGs are stored "
            "south-up and are flipped on read)",
            "dequantization": "((v/127.5)**2)*sign(v), nodata -128 -> 0.0",
        }
    )

    if args.report:
        report(args.years, args.patch_size, output_dir, meta)
        meta_path.write_text(json.dumps(meta, indent=2))
        logger.info(f"Metadata saved to {meta_path}")
        return

    index_path = ensure_index(output_dir / "index" / "aef_index.parquet")

    if args.init_only:
        for year in args.years:
            shape = (len(stations), args.patch_size, args.patch_size, EMBED_DIM)
            path = (
                output_dir
                / f"patch_embeddings_alphaearth_{year}_p{args.patch_size}.npy"
            )
            if not path.exists():
                logger.info(f"Pre-allocating {path}")
                np.lib.format.open_memmap(
                    str(path), mode="w+", dtype=np.float32, shape=shape
                )
        logger.info("Init complete.")
        return

    tiles = load_index(index_path, args.years)

    shard = tuple(args.shard) if args.shard else None
    for year in args.years:
        logger.info(f"=== Year {year} ===")
        info = extract_year(
            stations,
            tiles,
            year,
            args.patch_size,
            output_dir,
            args.workers,
            shard,
            args.limit,
            args.force,
        )
        meta["files"].setdefault(info["file"], {}).update(
            {k: v for k, v in info.items() if k != "file"}
        )

    meta_path.write_text(json.dumps(meta, indent=2))
    logger.info(f"Metadata saved to {meta_path}")
    logger.info("Run again with --report once every shard has finished.")


if __name__ == "__main__":
    main()
