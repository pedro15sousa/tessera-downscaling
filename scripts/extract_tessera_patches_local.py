"""Extract TESSERA patch embeddings from the LOCAL mount (no downloads).

Context
-------
The v2 embeddings are mounted read-only at

    /tessera/v2/global_0.1_degree_representation/<year>/grid_<lon>_<lat>/
        grid_<lon>_<lat>.npy         (H, W, 128)  int8   (quantized)
        grid_<lon>_<lat>_scales.npy  (H, W)       float32 (per-pixel scale)

Dequantized embedding = quantized.astype(f32) * scales[..., None].

Each tile is stored in its own local UTM zone at 10 m. To place a patch on a
station we need the tile's CRS + affine transform, which geotessera stores in a
*landmask* GeoTIFF (`global_0.1_degree_tiff_all/grid_<lon>_<lat>.tiff`). The
mount does NOT ship landmasks, so we source them from local caches and, for any
gaps, download the tiny (~13 KB) tiff from the version-independent v1 endpoint
    https://dl2.geotessera.org/v1/global_0.1_degree_tiff_all/<grid>.tiff
(v1 landmask dims were verified to equal the v2 embedding dims).

What this produces is byte-for-byte identical to geotessera's
`fetch_mosaic_for_region(bbox, year, target_crs="EPSG:4326")` followed by the
centre-crop that `data/tessera.py::extract_patch_embeddings` performs — verified
against geotessera directly for single-tile, 2-tile and 4-tile (corner) patches
— but reads embeddings straight from the mount, does no tile downloads, and
never deletes or copies mount data.

Output (per year Y and patch size P), row-aligned with the station CSV:

    <out-dir>/patch_embeddings_<Y>_p<P>.npy   (N, P, P, 128) float32, channels-last
    <out-dir>/station_list_filtered.csv       renamed input list (row i <-> patch row i)
    <out-dir>/extraction_metadata.json        params + per-file stats

Note: like the original extract_tessera.py, `station_list_filtered.csv` is just
the (renamed) input station list. It is NOT filtered by patch validity — that
health check happens downstream in data/helpers.py::filter_stations_by_tessera_patches.
A `<npy>.progress.json` sidecar records completed station indices so a run can
resume after interruption.

Usage
-----
    uv run --group core python \
      projects/tessera_downscaling/scripts/extract_tessera_patches_local.py \
        --station-csv station_list.csv \
        --out-dir     /data/weather-downscaling/processed/tessera_station_patches \
        --years 2024 2017 --patch-sizes 128 --workers 8

    # quick trial on the first 8 stations:
    ... --limit 8 --out-dir /tmp/tessera_trial
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import time
import urllib.request
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.io import MemoryFile
from rasterio.merge import merge
from rasterio.transform import array_bounds, rowcol
from rasterio.warp import Resampling, calculate_default_transform, reproject

from geotessera.core import dequantize_embedding
from geotessera.registry import tile_from_world, tile_to_grid_name

LOGGER = logging.getLogger("extract_tessera_local")

# --- defaults -------------------------------------------------------------
MOUNT_DIR = Path("/tessera/v2/global_0.1_degree_representation")
LANDMASK_SEARCH_DIRS = [
    Path("/home/pmms2/end-to-end-forecasting/global_0.1_degree_tiff_all"),
    Path("/data/weather-downscaling/_cache/geotessera/global_0.1_degree_tiff_all"),
]
LANDMASK_V1_URL = "https://dl2.geotessera.org/v1/global_0.1_degree_tiff_all/{name}.tiff"
TARGET_CRS = "EPSG:4326"
PIXEL_RES_M = 10.0
BBOX_MARGIN = 1.2  # matches data/tessera.py::_compute_patch_bbox
# Valid tile-centre range: tiles are centred on 0.05-deg offsets, so the
# outermost centres are +-89.95 / +-179.95. Clamp enumeration to this so that
# polar / antimeridian stations never blow up tile_from_world.
_LAT_LIM, _LON_LIM = 89.95, 179.95
# A 0.1-deg tile reprojects to ~1200 px per side. A tile straddling the
# antimeridian instead reprojects to a globe-spanning raster (millions of px,
# TB-scale allocation); anything past this bound is that pathology -> skip it.
_MAX_TILE_DIM = 5000

# Module-level worker state (set by _worker_init).
_W: dict = {}


# --- station loading ------------------------------------------------------
def load_stations(csv_path: Path) -> pd.DataFrame:
    """Load a station CSV, mapping GHCNh columns to canonical names.

    Mirrors the renaming in the original extract_tessera.py::load_stations
    (GHCN_ID->station_id, LATITUDE->latitude, LONGITUDE->longitude,
    ELEVATION->elevation). All other columns are kept as-is. No region or
    patch filtering is applied here.
    """
    df = pd.read_csv(csv_path)
    df = df.rename(columns={
        "GHCN_ID": "station_id",
        "LATITUDE": "latitude",
        "LONGITUDE": "longitude",
        "ELEVATION": "elevation",
    })
    for col in ("latitude", "longitude"):
        if col not in df.columns:
            raise SystemExit(
                f"station CSV missing '{col}' (after GHCNh rename). "
                f"Columns: {list(df.columns)}"
            )
    return df


# --- geometry / tile helpers ---------------------------------------------
def compute_bbox(lon: float, lat: float, patch_size: int) -> tuple[float, float, float, float]:
    """WGS84 bbox around (lon, lat) for a patch_size-pixel patch at 10 m.

    Identical to data/tessera.py::_compute_patch_bbox so the set of tiles and
    the resulting mosaic match the reference extraction.
    """
    extent_m = (patch_size / 2) * PIXEL_RES_M * BBOX_MARGIN
    m_lat = 111_320.0
    m_lon = max(111_320.0 * math.cos(math.radians(lat)), 1_000.0)
    dlat = extent_m / m_lat
    dlon = extent_m / m_lon
    return (lon - dlon, lat - dlat, lon + dlon, lat + dlat)


def covering_tiles(bbox: tuple[float, float, float, float]) -> set[tuple[float, float]]:
    """Tile centres whose 0.1x0.1-deg cell intersects bbox.

    The bbox is clamped to the valid world range so that near-polar and
    near-antimeridian stations (whose padded bbox would otherwise run past
    +-90 / +-180) don't raise. Such patches may be partially zero-padded at
    the world edge, which the downstream health check handles.
    """
    min_lon, min_lat, max_lon, max_lat = bbox
    # Clamp BOTH bounds into the valid tile-centre range, so a station hard
    # against +-90 / +-180 still resolves to its edge tile (partial patch)
    # rather than an empty set. Clamping preserves min <= max and is a no-op
    # for the vast majority of stations well inside the world.
    def _clamp(v, lim):
        return min(max(v, -lim), lim)
    min_lat, max_lat = _clamp(min_lat, _LAT_LIM), _clamp(max_lat, _LAT_LIM)
    min_lon, max_lon = _clamp(min_lon, _LON_LIM), _clamp(max_lon, _LON_LIM)
    tiles: set[tuple[float, float]] = set()
    lat = math.floor(min_lat * 10) / 10
    while lat <= max_lat + 1e-9:
        lon = math.floor(min_lon * 10) / 10
        while lon <= max_lon + 1e-9:
            plon = min(max(lon + 1e-6, -_LON_LIM), _LON_LIM)
            plat = min(max(lat + 1e-6, -_LAT_LIM), _LAT_LIM)
            tiles.add(tile_from_world(plon, plat))
            lon += 0.1
        lat += 0.1
    return tiles


def home_tile(lon: float, lat: float) -> tuple[float, float]:
    """Home tile centre for a station, clamped to the valid world range."""
    return tile_from_world(
        min(max(lon, -_LON_LIM), _LON_LIM),
        min(max(lat, -_LAT_LIM), _LAT_LIM),
    )


def tile_embedding_path(mount: Path, name: str, year: int) -> Path:
    return mount / str(year) / name / f"{name}.npy"


# --- landmask provisioning ------------------------------------------------
def find_landmask(name: str, search_dirs: list[Path]) -> Path | None:
    for d in search_dirs:
        p = d / f"{name}.tiff"
        if p.exists():
            return p
    return None


def provision_landmasks(
    tile_names: set[str], search_dirs: list[Path], cache_dir: Path
) -> dict[str, Path]:
    """Return {tile_name: landmask_path}, downloading gaps from the v1 endpoint.

    Downloads go to cache_dir (which is also prepended to the search path).
    Tiles whose landmask cannot be found or fetched (e.g. ocean tiles with no
    TESSERA coverage) are simply omitted; build_mosaic skips them.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    dirs = [cache_dir, *search_dirs]
    resolved: dict[str, Path] = {}
    missing: list[str] = []
    for name in sorted(tile_names):
        p = find_landmask(name, dirs)
        if p is not None:
            resolved[name] = p
        else:
            missing.append(name)

    if missing:
        from concurrent.futures import ThreadPoolExecutor

        LOGGER.info("Downloading up to %d missing landmasks to %s", len(missing), cache_dir)

        def _fetch(name):
            dst = cache_dir / f"{name}.tiff"
            try:
                urllib.request.urlretrieve(LANDMASK_V1_URL.format(name=name), dst)  # noqa: S310
                return name, dst
            except Exception:  # noqa: BLE001  (ocean/no-coverage tiles 404 here)
                if dst.exists():
                    dst.unlink()
                return name, None

        done = failed = 0
        with ThreadPoolExecutor(max_workers=32) as ex:
            for name, dst in ex.map(_fetch, missing):
                done += 1
                if dst is not None:
                    resolved[name] = dst
                else:
                    failed += 1
                if done % 1000 == 0 or done == len(missing):
                    LOGGER.info("  landmasks fetched %d/%d (%d unavailable)", done, len(missing), failed)
    LOGGER.info(
        "Landmasks resolved: %d/%d (%d unavailable -> those tiles skipped)",
        len(resolved), len(tile_names), len(tile_names) - len(resolved),
    )
    return resolved


# --- mosaic construction (bit-identical to geotessera) --------------------
def _load_tile(name: str, year: int) -> tuple[np.ndarray, object, object]:
    d = MOUNT_DIR / str(year) / name
    quant = np.load(d / f"{name}.npy")
    scales = np.load(d / f"{name}_scales.npy")
    emb = dequantize_embedding(quant, scales)
    with rasterio.open(_W["landmasks"][name]) as ds:
        return emb, ds.crs, ds.transform


def _reprojected_dataset(name: str, year: int, target_crs: str = TARGET_CRS):
    """Return an open in-memory dataset of the tile reprojected to target_crs.

    Cached per worker in an LRU keyed by (name, year). Reprojection depends
    only on the full tile (not the station/bbox), so a tile always reprojects
    to the identical array -- caching therefore leaves the merged output
    bit-identical, it only avoids re-reading and re-reprojecting shared tiles.
    """
    cache: OrderedDict = _W["tile_cache"]
    key = (name, year)
    hit = cache.get(key)
    if hit is not None:
        cache.move_to_end(key)
        return hit[1]

    emb, src_crs, src_transform = _load_tile(name, year)
    h, w = emb.shape[:2]
    src_bounds = array_bounds(h, w, src_transform)
    dst_t, dst_w, dst_h = calculate_default_transform(src_crs, target_crs, w, h, *src_bounds)
    dst_w, dst_h = int(dst_w), int(dst_h)
    if dst_w > _MAX_TILE_DIM or dst_h > _MAX_TILE_DIM:
        # Antimeridian globe-wrap: skip (the station gets a partial/zero patch).
        return None
    rep = np.empty((emb.shape[2], dst_h, dst_w), dtype=emb.dtype)
    for b in range(emb.shape[2]):
        reproject(
            source=emb[:, :, b], destination=rep[b],
            src_transform=src_transform, src_crs=src_crs,
            dst_transform=dst_t, dst_crs=target_crs,
            resampling=Resampling.bilinear,
        )
    mf = MemoryFile()
    ds = mf.open(
        driver="GTiff", height=dst_h, width=dst_w, count=emb.shape[2],
        dtype=emb.dtype, crs=target_crs, transform=dst_t,
    )
    ds.write(rep)
    cache[key] = (mf, ds)
    cache.move_to_end(key)
    # Never evict below the tiles the current station still needs (min_keep),
    # otherwise a dataset in `datasets` would be closed before merge() reads it.
    floor = max(_W["cache_cap"], _W.get("min_keep", 0))
    while len(cache) > floor:
        _, (old_mf, old_ds) = cache.popitem(last=False)
        old_ds.close()
        old_mf.close()
    return ds


def _trim_cache():
    cache = _W["tile_cache"]
    while len(cache) > _W["cache_cap"]:
        _, (mf, ds) = cache.popitem(last=False)
        ds.close()
        mf.close()


def build_mosaic(bbox, year, target_crs=TARGET_CRS):
    """Reproject each covering tile to target_crs and merge (bit-identical to
    geotessera.fetch_mosaic_for_region). Returns (HxWx128, transform) or
    (None, None) if no covering tile is available.

    Iterates covering_tiles(bbox) in the same order as the reference, so the
    merge order (which sets the mosaic grid) matches.
    """
    names = [
        tile_to_grid_name(tl, tt) for tl, tt in covering_tiles(bbox)
    ]
    names = [
        n for n in names
        if n in _W["landmasks"] and tile_embedding_path(MOUNT_DIR, n, year).exists()
    ]
    if not names:
        return None, None
    # Pin every tile this station needs for the whole merge (patches at high
    # latitude can span up to ~18 narrow tiles, exceeding cache_cap).
    _W["min_keep"] = len(names)
    datasets = []
    try:
        for name in names:
            ds = _reprojected_dataset(name, year, target_crs)
            if ds is None:  # pathological (antimeridian) tile skipped
                continue
            datasets.append(ds)
        if not datasets:
            return None, None
        merged, mt = merge(datasets)
    finally:
        _W["min_keep"] = 0
        _trim_cache()
    return np.transpose(merged, (1, 2, 0)), mt


def slice_patch(mosaic, mt, lon, lat, patch_size) -> np.ndarray:
    """Centre-crop patch_size x patch_size around (lon, lat); zero-pad edges.

    Mirrors the slicing in data/tessera.py::extract_patch_embeddings.
    """
    out = np.zeros((patch_size, patch_size, mosaic.shape[2]), dtype=np.float32)
    row, col = rowcol(mt, lon, lat)
    row, col = int(row), int(col)
    half = patch_size // 2
    h, w = mosaic.shape[:2]
    rs, re, cs, ce = row - half, row + half, col - half, col + half
    rss, ree, css, cee = max(0, rs), min(h, re), max(0, cs), min(w, ce)
    if ree > rss and cee > css:
        out[rss - rs: rss - rs + (ree - rss), css - cs: css - cs + (cee - css), :] = \
            mosaic[rss:ree, css:cee, :]
    return out


# --- worker ---------------------------------------------------------------
def _worker_init(out_path, shape, landmasks, year, patch_size, cache_cap):
    _W["patches"] = np.lib.format.open_memmap(str(out_path), mode="r+", dtype=np.float32, shape=shape)
    _W["landmasks"] = landmasks
    _W["year"] = year
    _W["patch_size"] = patch_size
    _W["cache_cap"] = cache_cap
    _W["tile_cache"] = OrderedDict()
    _W["min_keep"] = 0


def _worker(task) -> tuple[int, bool, float]:
    i, lat, lon = task
    ps = _W["patch_size"]
    year = _W["year"]
    try:
        mosaic, mt = build_mosaic(compute_bbox(lon, lat, ps), year)
        if mosaic is None:
            return i, False, 0.0
        patch = slice_patch(mosaic, mt, lon, lat, ps)
    except Exception as e:  # noqa: BLE001
        LOGGER.warning("station %d (%.4f,%.4f) year=%d p=%d: %s: %s",
                       i, lat, lon, year, ps, type(e).__name__, e)
        return i, False, 0.0
    _W["patches"][i] = patch
    c = ps // 2
    centre_nonzero = bool(np.any(patch[c, c, :] != 0))
    coverage = float(np.mean(np.any(patch != 0, axis=-1)))
    return i, centre_nonzero, coverage


# --- driver ---------------------------------------------------------------
def extract_one_file(
    stations: pd.DataFrame, year: int, patch_size: int, out_path: Path,
    landmasks: dict[str, Path], workers: int, cache_cap: int, resume: bool,
) -> dict:
    from multiprocessing import Pool

    n = len(stations)
    shape = (n, patch_size, patch_size, 128)
    progress_path = out_path.with_suffix(".progress.json")

    if out_path.exists() and resume:
        np.lib.format.open_memmap(str(out_path), mode="r+", dtype=np.float32, shape=shape)
        completed = set(json.loads(progress_path.read_text()).get("completed", [])) \
            if progress_path.exists() else set()
        LOGGER.info("Resuming %s: %d/%d already done", out_path.name, len(completed), n)
    else:
        gb = float(np.prod(shape)) * 4 / 1e9
        LOGGER.info("Allocating %.1f GB mmap at %s", gb, out_path)
        np.lib.format.open_memmap(str(out_path), mode="w+", dtype=np.float32, shape=shape)
        completed = set()
        progress_path.write_text(json.dumps({"completed": []}))

    lats = stations.latitude.to_numpy(dtype=float)
    lons = stations.longitude.to_numpy(dtype=float)
    tasks = [(i, lats[i], lons[i]) for i in range(n) if i not in completed]
    # Sort by home tile so a worker's contiguous chunk shares tiles -> the
    # per-worker LRU reprojects each tile ~once instead of per station.
    tasks.sort(key=lambda t: home_tile(t[2], t[1]))
    if not tasks:
        LOGGER.info("%s already complete.", out_path.name)
        return _file_stats(out_path, shape)

    LOGGER.info("Extracting %s: %d stations, %d workers, cache_cap=%d",
                out_path.name, len(tasks), workers, cache_cap)
    t0 = time.time()
    done = 0
    with Pool(
        processes=workers, initializer=_worker_init,
        initargs=(out_path, shape, landmasks, year, patch_size, cache_cap),
        maxtasksperchild=2000,
    ) as pool:
        for i, _centre_nz, _cov in pool.imap_unordered(_worker, tasks, chunksize=32):
            completed.add(i)
            done += 1
            if done % 200 == 0 or done == len(tasks):
                progress_path.write_text(json.dumps({"completed": sorted(completed)}))
                rate = done / max(time.time() - t0, 1e-6)
                eta = (len(tasks) - done) / max(rate, 1e-6) / 3600
                LOGGER.info("  %d/%d (%.1f station/s, ETA %.1f h)", done, len(tasks), rate, eta)
    progress_path.write_text(json.dumps({"completed": sorted(completed)}))
    LOGGER.info("Finished %s in %.1f min", out_path.name, (time.time() - t0) / 60)
    return _file_stats(out_path, shape)


def _file_stats(out_path: Path, shape) -> dict:
    """Cheap stats from the centre-pixel slice (avoids re-reading the whole
    hundreds-of-GB array). n_centre_nonzero mirrors the centre component of
    the downstream patch health check.
    """
    patches = np.load(str(out_path), mmap_mode="r")
    centre = shape[1] // 2
    centre_nz = int(np.sum(np.any(patches[:, centre, centre, :] != 0, axis=1)))
    return {"shape": list(shape), "n_stations": int(shape[0]), "n_centre_nonzero": centre_nz}


def _parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--station-csv", type=Path, required=True,
                   help="CSV with station lat/lon (GHCNh columns accepted); defines row order.")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--years", type=int, nargs="+", default=[2024, 2017])
    p.add_argument("--patch-sizes", type=int, nargs="+", default=[128],
                   help="Patch side length(s) in pixels (default: 128; crop 64 downstream).")
    p.add_argument("--mount-dir", type=Path, default=MOUNT_DIR)
    p.add_argument("--landmask-cache", type=Path, default=None,
                   help="Where to store downloaded landmasks (default: <out-dir>/landmasks).")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--cache-cap", type=int, default=6,
                   help="Max reprojected tiles cached per worker (memory vs re-read tradeoff).")
    p.add_argument("--limit", type=int, default=None, help="Only process the first N stations (trial).")
    p.add_argument("--force", action="store_true", help="Re-extract even if outputs exist.")
    p.add_argument("--log-level", default="INFO")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    global MOUNT_DIR
    MOUNT_DIR = args.mount_dir

    stations = load_stations(args.station_csv)
    if args.limit:
        stations = stations.iloc[: args.limit].reset_index(drop=True)
    LOGGER.info("Loaded %d stations from %s", len(stations), args.station_csv)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    stations.to_csv(args.out_dir / "station_list_filtered.csv", index=False)

    # Provision landmasks for every tile any station could need (largest patch).
    max_ps = max(args.patch_sizes)
    needed_tiles: set[str] = set()
    for lon, lat in zip(stations.longitude.to_numpy(float), stations.latitude.to_numpy(float)):
        for tl, tt in covering_tiles(compute_bbox(lon, lat, max_ps)):
            needed_tiles.add(tile_to_grid_name(tl, tt))
    cache_dir = args.landmask_cache or (args.out_dir / "landmasks")
    landmasks = provision_landmasks(needed_tiles, LANDMASK_SEARCH_DIRS, cache_dir)

    metadata = {
        "station_csv": str(args.station_csv),
        "mount_dir": str(args.mount_dir),
        "target_crs": TARGET_CRS,
        "n_stations": len(stations),
        "years": args.years,
        "patch_sizes": args.patch_sizes,
        "n_tiles_needed": len(needed_tiles),
        "n_tiles_resolved": len(landmasks),
        "files": {},
    }
    for year in args.years:
        for ps in args.patch_sizes:
            out_path = args.out_dir / f"patch_embeddings_{year}_p{ps}.npy"
            stats = extract_one_file(
                stations, year, ps, out_path, landmasks, args.workers,
                args.cache_cap, resume=not args.force,
            )
            metadata["files"][out_path.name] = stats
            LOGGER.info("%s: centre_nonzero=%d/%d",
                        out_path.name, stats["n_centre_nonzero"], stats["n_stations"])
            (args.out_dir / "extraction_metadata.json").write_text(json.dumps(metadata, indent=2))

    LOGGER.info("Wrote metadata to %s", args.out_dir / "extraction_metadata.json")
    LOGGER.info("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
