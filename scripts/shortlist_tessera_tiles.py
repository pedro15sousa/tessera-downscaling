"""Shortlist the TESSERA tiles needed to cover patches centred at stations.

The supervisor's TESSERA v1.1 embedding-generation pipeline produces embeddings
tile-by-tile on the standard global 0.1° grid. To extract an H×W patch centred
at each weather station (see ``extract_patch_embeddings`` in
``tessera_downscaling.data.tessera``), we need every tile that any station's
patch bounding box touches. This script reproduces that extraction-time logic
and emits the deduplicated set of tiles to hand off to the pipeline team.

How it mirrors the extraction script
-------------------------------------
``extract_patch_embeddings`` never names tiles itself — it hands a per-station
WGS84 bounding box to ``geotessera.fetch_mosaic_for_region``, which stitches
whichever 0.1° tiles overlap that box. We replicate the two halves of that:

  1. **Patch bbox** — identical formula to ``_compute_patch_bbox`` in
     ``data/tessera.py``: a half-patch spans ``(patch_size / 2) * pixel_res_m``
     metres, inflated by a 1.2× safety margin, converted to degrees with a
     latitude-aware longitude scaling (and the same ``max(..., 1000)`` clamp
     near the poles).

  2. **Tile grid** — geotessera's ``global_0.1_degree_representation`` grid.
     Each tile is a 0.1°×0.1° cell whose *edges* sit on the 0.1° grid lines and
     whose *centre* — the name geotessera uses — sits at ``(k + 0.5) * 0.1``.
     The point-containment rule in geotessera (``tiles.py``) is
     ``centre - 0.05 <= x < centre + 0.05`` with ``half_size = 0.05``, i.e.
     centres land on …, 0.05, 0.15, 0.25, …. A tile is named
     ``grid_{lon:.2f}_{lat:.2f}`` from its centre.

Output
------
  * ``tessera_tiles_shortlist.csv`` — one row per unique tile:
        tile_name, center_lon, center_lat,
        lon_min, lat_min, lon_max, lat_max, n_stations
  * ``tessera_tiles_shortlist.txt`` — one ``tile_name`` per line, for easy
        copy-paste / hand-off.

Usage (from repo root):
    .venv/bin/python projects/tessera_downscaling/scripts/shortlist_tessera_tiles.py \
        --station-csv projects/tessera_downscaling/.tmp_output/processed/tessera_global/station_list_filtered.csv \
        --out-dir projects/tessera_downscaling/.tmp_output/tile_shortlist \
        --patch-size 128
"""
from __future__ import annotations

import argparse
import logging
import math
from pathlib import Path

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("shortlist_tessera_tiles")

# geotessera global tile grid: 0.1° cells, named by centre to 2 dp.
TILE_DEG = 0.1
# Valid 0.1°-cell index bounds. Latitude is a hard physical bound (poles);
# longitude wraps at the antimeridian. Cells span [k*0.1, (k+1)*0.1):
#   lon k ∈ [-1800, 1799]  → centres -179.95 … 179.95  (3600 cells round the globe)
#   lat k ∈ [-900,  899]   → centres  -89.95 …  89.95
N_LON_CELLS = 3600
LAT_CELL_MIN, LAT_CELL_MAX = -900, 899


def load_stations(csv_path: Path) -> pd.DataFrame:
    """Load station lat/lon, accepting GHCNh or canonical column names."""
    df = pd.read_csv(csv_path)
    df = df.rename(columns={
        "GHCN_ID": "station_id",
        "LATITUDE": "latitude",
        "LONGITUDE": "longitude",
        "ELEVATION": "elevation",
    })
    if "latitude" not in df.columns or "longitude" not in df.columns:
        msg = (
            f"Expected latitude/longitude (or LATITUDE/LONGITUDE) columns. "
            f"Got: {list(df.columns)}"
        )
        raise ValueError(msg)
    n_before = len(df)
    df = df.dropna(subset=["latitude", "longitude"]).reset_index(drop=True)
    if len(df) < n_before:
        logger.info(f"Dropped {n_before - len(df)} rows with missing lat/lon")

    # Drop physically impossible coordinates (e.g. lat/lon swapped or sentinel
    # values). We don't silently "correct" them — that would fabricate tiles —
    # but we must not emit garbage tiles either, so we report and exclude them.
    bad = (df["latitude"].abs() > 90) | (df["longitude"].abs() > 180)
    if bad.any():
        cols = [c for c in ("station_id", "latitude", "longitude") if c in df.columns]
        logger.warning(
            f"Excluding {int(bad.sum())} station(s) with out-of-range "
            f"coordinates (|lat|>90 or |lon|>180):\n"
            f"{df.loc[bad, cols].to_string(index=False)}"
        )
        df = df[~bad].reset_index(drop=True)

    logger.info(f"Loaded {len(df)} usable stations from {csv_path}")
    return df


def compute_patch_bbox(
    lon: float, lat: float, patch_size: int, pixel_res_m: float = 10.0
) -> tuple[float, float, float, float]:
    """WGS84 bbox around a point — mirrors ``_compute_patch_bbox`` in data/tessera.py."""
    extent_m = (patch_size / 2) * pixel_res_m * 1.2
    metres_per_deg_lat = 111_320.0
    metres_per_deg_lon = max(111_320.0 * math.cos(math.radians(lat)), 1_000.0)
    dlat = extent_m / metres_per_deg_lat
    dlon = extent_m / metres_per_deg_lon
    return (lon - dlon, lat - dlat, lon + dlon, lat + dlat)


def _cell_index(coord: float) -> int:
    """0.1° cell index for a coordinate (cell = [k*0.1, (k+1)*0.1))."""
    # +1e-7 (in 0.1°-cell units ≈ 1e-8°) nudges values off exact grid edges
    # consistently; far smaller than any real patch half-width.
    return math.floor(coord / TILE_DEG + 1e-7)


def tiles_for_bbox(
    lon_min: float, lat_min: float, lon_max: float, lat_max: float
) -> list[tuple[str, float, float, float, float, float, float]]:
    """All 0.1° tiles overlapping a bbox, as (name, c_lon, c_lat, bounds...).

    Latitude cells are clamped to the poles; longitude cells wrap across the
    antimeridian so a patch straddling ±180° picks up the tile on the far side
    rather than a non-existent grid_180.05.
    """
    tiles = []
    for ki_raw in range(_cell_index(lon_min), _cell_index(lon_max) + 1):
        # Wrap into [-1800, 1799] so ±180° crossings map to real tiles.
        ki = ((ki_raw + N_LON_CELLS // 2) % N_LON_CELLS) - N_LON_CELLS // 2
        c_lon = (ki + 0.5) * TILE_DEG
        for kj in range(_cell_index(lat_min), _cell_index(lat_max) + 1):
            if kj < LAT_CELL_MIN or kj > LAT_CELL_MAX:
                continue  # patch spilled past a pole; no tile exists there
            c_lat = (kj + 0.5) * TILE_DEG
            name = f"grid_{c_lon:.2f}_{c_lat:.2f}"
            tiles.append((
                name, round(c_lon, 2), round(c_lat, 2),
                round(ki * TILE_DEG, 2), round(kj * TILE_DEG, 2),
                round((ki + 1) * TILE_DEG, 2), round((kj + 1) * TILE_DEG, 2),
            ))
    return tiles


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--station-csv", type=Path, required=True,
        help="CSV with station locations (GHCNh or canonical lat/lon columns).",
    )
    parser.add_argument(
        "--out-dir", type=Path, required=True,
        help="Directory for tessera_tiles_shortlist.csv / .txt.",
    )
    parser.add_argument(
        "--patch-size", type=int, default=128,
        help="Patch side length in pixels at 10m (default 128; covers 64×64 "
             "'or larger' with headroom). Sizes the per-station safety margin.",
    )
    parser.add_argument(
        "--pixel-res-m", type=float, default=10.0,
        help="TESSERA pixel resolution in metres (default 10).",
    )
    args = parser.parse_args()

    stations = load_stations(args.station_csv)

    # tile_name -> [center_lon, center_lat, lon_min, lat_min, lon_max, lat_max, n_stations]
    tile_records: dict[str, list] = {}
    n_multi = 0
    for lat, lon in zip(
        stations["latitude"].to_numpy(), stations["longitude"].to_numpy()
    ):
        bbox = compute_patch_bbox(float(lon), float(lat), args.patch_size, args.pixel_res_m)
        hits = tiles_for_bbox(*bbox)
        if len(hits) > 1:
            n_multi += 1
        for name, c_lon, c_lat, lo_lon, lo_lat, hi_lon, hi_lat in hits:
            rec = tile_records.get(name)
            if rec is None:
                tile_records[name] = [c_lon, c_lat, lo_lon, lo_lat, hi_lon, hi_lat, 1]
            else:
                rec[6] += 1

    tiles_df = pd.DataFrame(
        [[name, *vals] for name, vals in tile_records.items()],
        columns=[
            "tile_name", "center_lon", "center_lat",
            "lon_min", "lat_min", "lon_max", "lat_max", "n_stations",
        ],
    ).sort_values(["center_lat", "center_lon"]).reset_index(drop=True)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.out_dir / "tessera_tiles_shortlist.csv"
    txt_path = args.out_dir / "tessera_tiles_shortlist.txt"
    tiles_df.to_csv(csv_path, index=False)
    txt_path.write_text("\n".join(tiles_df["tile_name"]) + "\n")

    logger.info(
        f"{len(stations)} stations → {len(tiles_df)} unique tiles "
        f"(patch_size={args.patch_size}px ≈ "
        f"{args.patch_size * args.pixel_res_m:.0f}m). "
        f"{n_multi} stations span >1 tile."
    )
    logger.info(
        f"Tile centre extent: lon [{tiles_df['center_lon'].min():.2f}, "
        f"{tiles_df['center_lon'].max():.2f}], lat "
        f"[{tiles_df['center_lat'].min():.2f}, {tiles_df['center_lat'].max():.2f}]"
    )
    logger.info(f"Wrote {csv_path}")
    logger.info(f"Wrote {txt_path}")


if __name__ == "__main__":
    main()
