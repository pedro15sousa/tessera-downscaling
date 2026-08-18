"""TESSERA embedding extraction at station locations.

Uses the ``geotessera`` library (https://github.com/ucam-eo/geotessera) to
download and extract pre-computed TESSERA embeddings. TESSERA produces annual,
global, 10m-resolution, 128-dimensional land surface embeddings from fused
Sentinel-1 SAR and Sentinel-2 optical time series.

Two extraction modes:

  **Point extraction**: Uses ``geotessera``'s ``sample_embeddings_at_points``
  to extract the single 128-d embedding at each station's pixel location.

  **Patch extraction**: Uses ``geotessera``'s ``fetch_mosaic_for_region`` to
  extract an H×W patch of embeddings centred on each station. The mosaic API
  handles tile stitching and CRS reprojection, so patches that span tile
  boundaries are seamlessly merged. This supports variable patch sizes for
  ablation studies.

  Patches are written station-by-station into a memory-mapped .npy file so
  that the full array (e.g. 76 GiB for 38k stations at 64×64×128) never has
  to live in RAM. The file is pre-allocated on disk and filled incrementally.
  A companion .progress.json file tracks completed stations so extraction can
  be resumed after interruption.

The ``geotessera`` library handles tile download, caching, dequantisation, and
CRS management. Tiles are downloaded on-demand and cached in the specified
embeddings directory.

Requires: ``pip install geotessera>=0.7.0``
"""

import glob
import json
import logging
import math
import os
from pathlib import Path

import numpy as np
from rasterio.transform import rowcol

logger = logging.getLogger(__name__)


def _init_geotessera(embeddings_dir: str | Path | None = None):
    """Initialise a GeoTessera client."""
    try:
        from geotessera import GeoTessera
    except ImportError:
        logger.error(
            "geotessera is not installed. Install with: pip install geotessera>=0.7.0"
        )
        return None

    gt_kwargs = {}
    if embeddings_dir is not None:
        gt_kwargs["embeddings_dir"] = str(embeddings_dir)
    return GeoTessera(**gt_kwargs)


def extract_point_embeddings(
    station_lats: np.ndarray,
    station_lons: np.ndarray,
    year: int = 2024,
    embeddings_dir: str | Path | None = None,
    embed_dim: int = 128,
) -> np.ndarray:
    """Extract TESSERA point embeddings at station locations.

    Args:
        station_lats: ``(N,)`` array of station latitudes (WGS84).
        station_lons: ``(N,)`` array of station longitudes (WGS84).
        year: Which annual embedding to use (2017–2024).
        embeddings_dir: Local directory for caching downloaded tiles.
        embed_dim: Embedding dimensionality (128 for TESSERA).

    Returns:
        ``(N, embed_dim)`` float32 array. Stations outside TESSERA coverage
        get zero vectors.
    """
    n_stations = len(station_lats)
    gt = _init_geotessera(embeddings_dir)
    if gt is None:
        return np.zeros((n_stations, embed_dim), dtype=np.float32)

    points = list(zip(station_lons.tolist(), station_lats.tolist()))

    logger.info(
        f"Extracting TESSERA point embeddings for {n_stations} stations, year={year}"
    )
    embeddings = gt.sample_embeddings_at_points(points, year=year)
    embeddings = np.nan_to_num(embeddings, nan=0.0).astype(np.float32)

    n_found = int(np.sum(np.any(embeddings != 0, axis=1)))
    logger.info(f"Extracted point embeddings for {n_found}/{n_stations} stations")
    return embeddings


def _compute_patch_bbox(
    lon: float, lat: float, patch_size: int, pixel_res_m: float = 10.0
) -> tuple[float, float, float, float]:
    """Compute a WGS84 bounding box around a point for a given patch size."""
    extent_m = (patch_size / 2) * pixel_res_m * 1.2
    metres_per_deg_lat = 111_320.0
    metres_per_deg_lon = 111_320.0 * math.cos(math.radians(lat))
    metres_per_deg_lon = max(metres_per_deg_lon, 1_000.0)
    dlat = extent_m / metres_per_deg_lat
    dlon = extent_m / metres_per_deg_lon
    return (lon - dlon, lat - dlat, lon + dlon, lat + dlat)


def extract_patch_embeddings(
    station_lats: np.ndarray,
    station_lons: np.ndarray,
    year: int = 2024,
    patch_size: int = 64,
    embeddings_dir: str | Path | None = None,
    embed_dim: int = 128,
    output_path: Path | None = None,
    resume: bool = True,
) -> np.ndarray:
    """Extract spatial patches of TESSERA embeddings around station locations.

    Writes patches station-by-station into a memory-mapped .npy file so the
    full array never has to live in RAM. For 38k stations at 64×64×128 fp32
    this is ~76 GiB on disk but only a few MB of RSS.

    A companion ``<output_path>.progress.json`` file records which stations
    have been completed, enabling resumption after interruption. If
    ``output_path`` is None, a temporary in-memory array is used (only
    suitable for small numbers of stations).

    Args:
        station_lats: ``(N,)`` array of station latitudes (WGS84).
        station_lons: ``(N,)`` array of station longitudes (WGS84).
        year: Which annual embedding to use (2017–2024).
        patch_size: Side length of the square patch in pixels (at 10m).
        embeddings_dir: Local directory for tile downloads. Tiles are deleted
            after each station to avoid filling disk.
        embed_dim: Embedding dimensionality (128 for TESSERA).
        output_path: Path to write the output .npy file. If provided, the
            array is memory-mapped to disk so RAM usage stays constant.
            Strongly recommended for > ~1000 stations.
        resume: If True and output_path already exists with a progress file,
            skip already-completed stations.

    Returns:
        ``(N, patch_size, patch_size, embed_dim)`` float32 array (or mmap).
        Stations where extraction failed have zero patches.
    """
    gt = _init_geotessera(embeddings_dir)
    if gt is None:
        if output_path is None:
            return np.zeros(
                (len(station_lats), patch_size, patch_size, embed_dim),
                dtype=np.float32,
            )
        # Still need to return something — create a zero mmap.
        patches = np.lib.format.open_memmap(
            str(output_path),
            mode="w+",
            dtype=np.float32,
            shape=(len(station_lats), patch_size, patch_size, embed_dim),
        )
        # patches[:] = 0
        return patches

    tile_dir = Path(gt.embeddings_dir) / "global_0.1_degree_representation"
    n_stations = len(station_lats)
    shape = (n_stations, patch_size, patch_size, embed_dim)

    # --- Set up output array (disk-backed mmap or in-memory) ---
    if output_path is not None:
        output_path = Path(output_path)
        progress_path = output_path.with_suffix(".progress.json")

        if output_path.exists() and resume:
            logger.info(f"Opening existing mmap at {output_path} (resuming)")
            patches = np.lib.format.open_memmap(
                str(output_path), mode="r+", dtype=np.float32, shape=shape
            )
        else:
            logger.info(
                f"Pre-allocating {shape[0]*patch_size*patch_size*embed_dim*4/1e9:.1f} GB "
                f"mmap at {output_path}"
            )
            patches = np.lib.format.open_memmap(
                str(output_path), mode="w+", dtype=np.float32, shape=shape
            )
            # patches[:] = 0

        # Load progress.
        if resume and progress_path.exists():
            with open(progress_path) as f:
                progress = json.load(f)
            completed = set(progress.get("completed", []))
            logger.info(
                f"Resuming: {len(completed)}/{n_stations} stations already done"
            )
        else:
            completed = set()
            progress_path.write_text(json.dumps({"completed": []}))
    else:
        # Small run: allocate in RAM.
        patches = np.zeros(shape, dtype=np.float32)
        completed = set()
        progress_path = None

    half = patch_size // 2

    logger.info(
        f"Extracting TESSERA {patch_size}×{patch_size} patches for "
        f"{n_stations} stations, year={year}"
    )

    for i in range(n_stations):
        if i in completed:
            continue

        lat, lon = float(station_lats[i]), float(station_lons[i])
        try:
            bbox = _compute_patch_bbox(lon, lat, patch_size)

            mosaic, transform, crs = gt.fetch_mosaic_for_region(
                bbox, year=year, target_crs="EPSG:4326"
            )

            row, col = rowcol(transform, lon, lat)
            row, col = int(row), int(col)
            h, w = mosaic.shape[0], mosaic.shape[1]

            r_start, r_end = row - half, row + half
            c_start, c_end = col - half, col + half

            r_src_start = max(0, r_start)
            r_src_end = min(h, r_end)
            c_src_start = max(0, c_start)
            c_src_end = min(w, c_end)

            if r_src_end <= r_src_start or c_src_end <= c_src_start:
                logger.warning(
                    f"Station {i} ({lat:.4f}, {lon:.4f}): mosaic too small"
                )
            else:
                dr = r_src_start - r_start
                dc = c_src_start - c_start
                pr = r_src_end - r_src_start
                pc = c_src_end - c_src_start
                patches[i, dr:dr + pr, dc:dc + pc, :] = mosaic[
                    r_src_start:r_src_end, c_src_start:c_src_end, :
                ]

        except Exception as e:
            logger.warning(
                f"Station {i} ({lat:.4f}, {lon:.4f}): "
                f"{type(e).__name__}: {e}"
            )
        finally:
            # Delete large tile .npy files after each station (~100 MB each).
            for npy_file in glob.glob(
                str(tile_dir / "**" / "*.npy"), recursive=True
            ):
                try:
                    os.remove(npy_file)
                except OSError:
                    pass

        completed.add(i)

        # Flush progress every 100 stations.
        if progress_path is not None and len(completed) % 100 == 0:
            with open(progress_path, "w") as f:
                json.dump({"completed": sorted(completed)}, f)
            if hasattr(patches, "flush"):
                patches.flush()
            n_found = int(
                np.sum(np.any(patches[:i + 1].reshape(i + 1, -1) != 0, axis=1))
            )
            logger.info(
                f"  {i + 1}/{n_stations} stations processed "
                f"({n_found} non-zero so far)"
            )

    # Final flush.
    if progress_path is not None:
        with open(progress_path, "w") as f:
            json.dump({"completed": sorted(completed)}, f)
        if hasattr(patches, "flush"):
            patches.flush()

    n_found = int(
        np.sum(np.any(patches.reshape(n_stations, -1) != 0, axis=1))
    )
    logger.info(f"Extracted patches for {n_found}/{n_stations} stations")
    return patches


def load_cached_embeddings(cache_path: Path) -> np.ndarray | None:
    """Load previously extracted embeddings from a numpy file.

    Returns the array (or mmap if large), or ``None`` if file doesn't exist.
    """
    if not Path(cache_path).exists():
        return None
    logger.info(f"Loading cached TESSERA embeddings from {cache_path}")
    # Use mmap for large files to avoid loading into RAM.
    arr = np.load(str(cache_path), mmap_mode="r")
    return arr


def save_cached_embeddings(embeddings: np.ndarray, cache_path: Path) -> None:
    """Save extracted embeddings to a numpy file for reuse.

    Note: if embeddings is already a mmap at cache_path (i.e. extract_patch_embeddings
    wrote directly to output_path), this is a no-op — the data is already on disk.
    """
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if hasattr(embeddings, "filename") and Path(embeddings.filename) == cache_path:
        logger.info(
            f"Embeddings already at {cache_path} (mmap), skipping save"
        )
        return
    np.save(str(cache_path), embeddings)
    logger.info(
        f"Saved TESSERA embeddings to {cache_path} (shape: {embeddings.shape})"
    )