"""TESSERA patch extraction at dense regular grid points (map inference).

Produces the ``(N_grid_points, 64, 64, 128)`` patch array that the frozen VAE
encoder turns into per-grid-point latents for the dense 0.05° downscaled maps
(``scripts/maps/extract_dense_grid_patches.py`` → ``generate_maps.py``). Tiles
are downloaded and mosaicked with ``geotessera``.

The per-station patch file (``patch_embeddings_2024.npy``) is produced by
``scripts/data/extract_tessera_patches_local.py``, which reads the TESSERA
mount directly and is optimised for sparse stations scattered across the
globe. For dense regular grids (e.g. a 0.05° grid over Iberia → ~40k grid
points clustered in one region) the access pattern is fundamentally different:

  * Adjacent grid points share many tiles. A per-point fetch pattern would
    re-download each tile dozens of times.
  * Storage isn't the constraint — we have plenty of disk; throughput is.
  * Ordering grid points by spatial locality lets us fetch each tile
    once, extract all patches within, then discard.

This module processes grid points in **sub-bboxes** (default 0.3° × 0.3°)
that are small enough to hold the dequantised mosaic in RAM but large
enough that each ``fetch_mosaic_for_region`` call amortises across many
grid points. Within a sub-bbox we issue a single mosaic fetch, then slice
64×64 patches from the in-memory mosaic for every grid point that falls
inside the bbox.

CRS consistency: we keep ``target_crs="EPSG:4326"`` to match the
projection used by the station extractor when building the
``patch_embeddings_2024.npy`` file the pre-trained VAE was fit on. Using
native UTM here would produce subtly out-of-distribution patches.

Crash recovery: the output ``.npy`` is memory-mapped, and a sibling
``progress.json`` records which grid-point indices have been completed.
Re-running picks up where it left off.
"""
from __future__ import annotations

import json
import logging
import math
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Grid construction
# ---------------------------------------------------------------------------

def compute_grid_points(
    lat_min: float,
    lat_max: float,
    lon_min: float,
    lon_max: float,
    resolution_deg: float,
) -> pd.DataFrame:
    """Build a regular lat/lon grid covering a bounding box.

    Args:
        lat_min, lat_max: Latitude bounds (inclusive of both).
        lon_min, lon_max: Longitude bounds (inclusive of both).
        resolution_deg: Grid spacing in degrees (e.g. 0.05 for ~5km).

    Returns:
        DataFrame with columns ``grid_idx`` (sequential int), ``lat``, ``lon``.
        Rows are ordered row-major: scanning ``lon`` left-to-right, with
        ``lat`` running top-down (north to south). This ordering helps
        spatial-locality processing downstream.

    """
    n_lat = int(round((lat_max - lat_min) / resolution_deg)) + 1
    n_lon = int(round((lon_max - lon_min) / resolution_deg)) + 1
    lats = np.linspace(lat_max, lat_min, n_lat)  # north-to-south
    lons = np.linspace(lon_min, lon_max, n_lon)
    LON, LAT = np.meshgrid(lons, lats)
    return pd.DataFrame({
        "grid_idx": np.arange(LAT.size),
        "lat": LAT.ravel(),
        "lon": LON.ravel(),
    })


# ---------------------------------------------------------------------------
# Sub-bbox iteration
# ---------------------------------------------------------------------------

def _sub_bboxes(
    lat_min: float,
    lat_max: float,
    lon_min: float,
    lon_max: float,
    sub_size_deg: float,
) -> list[tuple[float, float, float, float]]:
    """Tile an outer bbox into smaller sub-bboxes for batch mosaic fetching.

    The last bbox in each row/column may be smaller than ``sub_size_deg``
    if the outer bbox doesn't divide evenly.
    """
    out = []
    n_lat = math.ceil((lat_max - lat_min) / sub_size_deg)
    n_lon = math.ceil((lon_max - lon_min) / sub_size_deg)
    for i in range(n_lat):
        sub_lat_min = lat_min + i * sub_size_deg
        sub_lat_max = min(sub_lat_min + sub_size_deg, lat_max)
        for j in range(n_lon):
            sub_lon_min = lon_min + j * sub_size_deg
            sub_lon_max = min(sub_lon_min + sub_size_deg, lon_max)
            out.append((sub_lon_min, sub_lat_min, sub_lon_max, sub_lat_max))
    return out


# ---------------------------------------------------------------------------
# Patch extraction from an in-memory mosaic
# ---------------------------------------------------------------------------

def _extract_patches_from_mosaic(
    mosaic: np.ndarray,
    transform,
    grid_points: pd.DataFrame,
    patch_size: int,
    output_array: np.ndarray,
) -> int:
    """Slice 64×64×128 patches from an in-memory mosaic for given grid points.

    Args:
        mosaic: ``(H, W, 128)`` array — the dequantised, reprojected mosaic.
        transform: ``rasterio.Affine`` mapping (col, row) → (lon, lat).
        grid_points: DataFrame with ``grid_idx``, ``lat``, ``lon`` columns.
            Only rows whose patches fit within the mosaic bounds are
            successfully extracted; others are silently skipped (caller's
            sub-bbox iteration ensures coverage).
        patch_size: Patch side length in pixels (typically 64).
        output_array: ``(N, patch_size, patch_size, 128)`` mmap'd output.
            Patches are written at row index ``grid_idx``.

    Returns:
        Number of patches successfully written.

    """
    from rasterio.transform import rowcol

    half = patch_size // 2
    h, w = mosaic.shape[0], mosaic.shape[1]
    n_written = 0

    for row in grid_points.itertuples(index=False):
        idx, lat, lon = int(row.grid_idx), float(row.lat), float(row.lon)
        # Convert lat/lon to mosaic pixel coordinates.
        r, c = rowcol(transform, lon, lat)
        r, c = int(r), int(c)

        r_start, r_end = r - half, r + half
        c_start, c_end = c - half, c + half

        # Clip to mosaic extent. Patches that fall partially outside the
        # mosaic get zero-padded; ones entirely outside are skipped.
        r_src_start = max(0, r_start)
        r_src_end = min(h, r_end)
        c_src_start = max(0, c_start)
        c_src_end = min(w, c_end)

        if r_src_end <= r_src_start or c_src_end <= c_src_start:
            continue  # entirely outside; skip

        dr = r_src_start - r_start
        dc = c_src_start - c_start
        pr = r_src_end - r_src_start
        pc = c_src_end - c_src_start
        # Zero out the patch first (in case of partial coverage), then fill.
        output_array[idx, :, :, :] = 0.0
        output_array[idx, dr:dr + pr, dc:dc + pc, :] = mosaic[
            r_src_start:r_src_end, c_src_start:c_src_end, :
        ]
        n_written += 1

    return n_written


# ---------------------------------------------------------------------------
# Main extraction pipeline
# ---------------------------------------------------------------------------

def extract_dense_grid_patches(
    grid_points: pd.DataFrame,
    bbox: tuple[float, float, float, float],
    output_dir: Path,
    year: int = 2024,
    patch_size: int = 64,
    embed_dim: int = 128,
    sub_bbox_size_deg: float = 0.3,
    embeddings_cache_dir: Path | None = None,
    resume: bool = True,
) -> None:
    """Extract TESSERA patches at every grid point, processing in sub-bboxes.

    The output directory will contain:

      * ``patch_embeddings.npy`` — memory-mapped float32 array of shape
        ``(N_grid_points, patch_size, patch_size, embed_dim)``.
      * ``grid_points.csv`` — copy of ``grid_points`` for reference.
      * ``extraction_metadata.json`` — bbox, resolution, year, etc.
      * ``progress.json`` — sub-bbox completion log for resumability.

    Args:
        grid_points: DataFrame from :func:`compute_grid_points`.
        bbox: ``(lon_min, lat_min, lon_max, lat_max)`` outer bbox.
        output_dir: Destination directory; created if missing.
        year: TESSERA year (2017–2024 typically).
        patch_size: Patch side length in pixels at 10m resolution.
        embed_dim: Embedding channel count (128 for TESSERA).
        sub_bbox_size_deg: Side length of each sub-bbox iteration unit.
            Smaller = lower peak RAM but more network round-trips. Default
            0.3° gives ~5 GB peak mosaic at mid-latitudes — comfortable
            on machines with ≥32 GB RAM.
        embeddings_cache_dir: Where geotessera caches downloaded tiles.
            Defaults to ``output_dir / "_tile_cache"``. Tiles persist
            across runs, so partial progress isn't lost on restart.
        resume: If True and an existing ``progress.json`` is present,
            skip sub-bboxes already marked complete.

    """
    import geotessera

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if embeddings_cache_dir is None:
        embeddings_cache_dir = output_dir / "_tile_cache"
    embeddings_cache_dir = Path(embeddings_cache_dir)
    embeddings_cache_dir.mkdir(parents=True, exist_ok=True)

    n_points = len(grid_points)

    # ---- Output array: memory-mapped so we don't hold 100 GB in RAM ----
    output_path = output_dir / "patch_embeddings.npy"
    shape = (n_points, patch_size, patch_size, embed_dim)
    if output_path.exists() and resume:
        # Verify the existing file matches the expected shape.
        existing = np.load(output_path, mmap_mode="r")
        if existing.shape != shape:
            raise RuntimeError(
                f"Existing {output_path} has shape {existing.shape}, "
                f"expected {shape}. Delete it or change output_dir."
            )
        del existing
    else:
        # Pre-allocate via lib.format header + zero fill.
        from numpy.lib.format import open_memmap
        open_memmap(
            output_path, mode="w+",
            dtype=np.float32, shape=shape,
        ).flush()
        size_gb = n_points * patch_size * patch_size * embed_dim * 4 / 1e9
        logger.info(f"Pre-allocating {size_gb:.1f} GB output at {output_path}")
    patches = np.load(output_path, mmap_mode="r+")

    # ---- Save grid_points + metadata for downstream consumers ----
    grid_points.to_csv(output_dir / "grid_points.csv", index=False)
    metadata = {
        "year": year,
        "patch_size": patch_size,
        "embed_dim": embed_dim,
        "n_grid_points": n_points,
        "bbox": list(bbox),
        "sub_bbox_size_deg": sub_bbox_size_deg,
        "target_crs": "EPSG:4326",  # matches the station patch extractor
    }
    with open(output_dir / "extraction_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    # ---- Resume support ----
    progress_path = output_dir / "progress.json"
    if resume and progress_path.exists():
        with open(progress_path) as f:
            progress = json.load(f)
        completed_bboxes = {tuple(b) for b in progress.get("completed_bboxes", [])}
        completed_count = progress.get("n_patches_written", 0)
        logger.info(
            f"Resuming: {len(completed_bboxes)} sub-bboxes already done, "
            f"{completed_count} patches written so far"
        )
    else:
        completed_bboxes = set()
        completed_count = 0

    # ---- Initialise GeoTessera client ----
    lon_min, lat_min, lon_max, lat_max = bbox
    sub_bboxes = _sub_bboxes(lat_min, lat_max, lon_min, lon_max, sub_bbox_size_deg)
    n_sub = len(sub_bboxes)
    logger.info(
        f"Extracting {n_points} patches over {n_sub} sub-bboxes "
        f"({sub_bbox_size_deg}° each), year={year}, patch_size={patch_size}"
    )

    gt = geotessera.GeoTessera(embeddings_dir=str(embeddings_cache_dir))

    # ---- Main loop: one mosaic per sub-bbox, extract all patches within ----
    for k, sub_bbox in enumerate(sub_bboxes):
        if tuple(sub_bbox) in completed_bboxes:
            continue

        sub_lon_min, sub_lat_min, sub_lon_max, sub_lat_max = sub_bbox
        # Find grid points whose centre lies inside this sub-bbox.
        # Use a half-open interval [min, max) on both axes so that a grid
        # point on a shared boundary is owned by exactly one sub-bbox.
        # The very last sub-bbox along each axis closes its interval at
        # the outer maximum so the corner points aren't dropped.
        eps = 1e-9
        is_last_lat = abs(sub_lat_max - lat_max) < eps
        is_last_lon = abs(sub_lon_max - lon_max) < eps
        in_bbox = (
            (grid_points["lat"] >= sub_lat_min - eps)
            & (
                (grid_points["lat"] < sub_lat_max - eps)
                | (is_last_lat & (grid_points["lat"] <= sub_lat_max + eps))
            )
            & (grid_points["lon"] >= sub_lon_min - eps)
            & (
                (grid_points["lon"] < sub_lon_max - eps)
                | (is_last_lon & (grid_points["lon"] <= sub_lon_max + eps))
            )
        )
        sub_grid = grid_points[in_bbox]

        if len(sub_grid) == 0:
            completed_bboxes.add(tuple(sub_bbox))
            continue

        # Expand the fetch bbox by one patch radius (in degrees) so that
        # patches centred near the sub-bbox edge can be fully sliced
        # without crossing into a neighbouring (uncached) tile region.
        # Mirrors the latitude-aware logic in ``compute_bbox`` of
        # ``scripts/data/extract_tessera_patches_local.py``: a half-patch spans
        # ``(patch_size / 2) * pixel_res_m`` metres, which is fewer
        # degrees of longitude as |lat| grows. Latitude radius is
        # uniform (~111.32 km/deg); longitude radius uses the
        # worst-case (highest-|lat|) edge of the sub-bbox so even
        # patches centred at its polar edge fit. 1.5× safety factor
        # is preserved from the original code; ``max(..., 1000.0)``
        # clamp guards against pathological values very near the poles.
        pixel_res_m = 10.0
        half_patch_m = (patch_size / 2) * pixel_res_m * 1.5
        worst_lat = max(abs(sub_lat_min), abs(sub_lat_max))
        m_per_deg_lon = max(
            111_320.0 * math.cos(math.radians(worst_lat)), 1_000.0
        )
        patch_radius_lat_deg = half_patch_m / 111_320.0
        patch_radius_lon_deg = half_patch_m / m_per_deg_lon
        fetch_bbox = (
            sub_lon_min - patch_radius_lon_deg,
            sub_lat_min - patch_radius_lat_deg,
            sub_lon_max + patch_radius_lon_deg,
            sub_lat_max + patch_radius_lat_deg,
        )

        try:
            mosaic, transform, _crs = gt.fetch_mosaic_for_region(
                fetch_bbox, year=year, target_crs="EPSG:4326",
            )
        except Exception as e:
            logger.warning(
                f"Sub-bbox {k+1}/{n_sub} {sub_bbox}: "
                f"fetch failed ({type(e).__name__}: {e}); skipping"
            )
            continue

        n_written = _extract_patches_from_mosaic(
            mosaic, transform, sub_grid, patch_size, patches,
        )
        # Free mosaic before moving to the next bbox.
        del mosaic

        completed_count += n_written
        completed_bboxes.add(tuple(sub_bbox))
        logger.info(
            f"  sub-bbox {k+1}/{n_sub} {sub_bbox} → "
            f"wrote {n_written}/{len(sub_grid)} patches "
            f"(total: {completed_count}/{n_points})"
        )

        # Flush memory map and progress every sub-bbox so a crash
        # only loses one sub-bbox of work.
        patches.flush()
        with open(progress_path, "w") as f:
            json.dump({
                "completed_bboxes": [list(b) for b in completed_bboxes],
                "n_patches_written": completed_count,
                "n_sub_bboxes_done": len(completed_bboxes),
                "n_sub_bboxes_total": n_sub,
            }, f, indent=2)

    # Final flush and progress write.
    patches.flush()
    with open(progress_path, "w") as f:
        json.dump({
            "completed_bboxes": [list(b) for b in completed_bboxes],
            "n_patches_written": completed_count,
            "n_sub_bboxes_done": len(completed_bboxes),
            "n_sub_bboxes_total": n_sub,
        }, f, indent=2)

    logger.info(
        f"Done: {completed_count}/{n_points} patches written, "
        f"{len(completed_bboxes)}/{n_sub} sub-bboxes processed."
    )
