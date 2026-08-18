"""Extract TESSERA embeddings at station locations.

Given the GHCNh station list CSV (or any CSV with latitude/longitude columns),
this script extracts TESSERA embeddings for multiple years and saves them as
numpy files for use in training. Supports both point extraction (128-d per
station) and patch extraction (H×W×128 per station).

Stations can optionally be filtered to a European bounding box:
latitude 35–75°N, longitude -24–40°E.

Prerequisites:
    pip install geotessera

Usage:
    # Point embeddings for all TESSERA years, European stations only:
    uv run --group core python projects/tessera_downscaling/scripts/extract_tessera.py \\
        --station-csv .tmp_output/raw/ghcnh/station_list.csv \\
        --output-dir .tmp_output/processed/tessera \\
        --years 2017 2024 \\
        --mode point \\
        --region europe

    # Patch embeddings for a single year, all stations:
    uv run --group core python projects/tessera_downscaling/scripts/extract_tessera.py \\
        --station-csv .tmp_output/raw/ghcnh/station_list.csv \\
        --output-dir .tmp_output/processed/tessera \\
        --years 2024 2024 \\
        --mode patch \\
        --patch-size 64

Output files (per year):
    tessera/point_embeddings_2024.npy   — shape (N, 128)
    tessera/patch_embeddings_2024.npy   — shape (N, 64, 64, 128)
    tessera/station_list_filtered.csv   — the filtered station list used
    tessera/extraction_metadata.json    — params, counts per year
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from tessera_downscaling.data.tessera import (
    extract_patch_embeddings,
    extract_point_embeddings,
    load_cached_embeddings,
    save_cached_embeddings,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("extract_tessera")

# European bounding box from Wessel's neuralprocesses codebase
# (temperature.py, data_task="europe"). This covers all VALUE/ECA&D stations.
EUROPE_LON_RANGE = (-24, 40)
EUROPE_LAT_RANGE = (35, 75)


def load_stations(csv_path: Path, region: str | None = None) -> pd.DataFrame:
    """Load station locations from a CSV file.

    Recognises column names from the GHCNh station list (LATITUDE, LONGITUDE,
    ELEVATION, GHCN_ID) as well as common lowercase variants.

    Args:
        csv_path: Path to the CSV file.
        region: Optional region filter. Currently supports "europe" which
            applies the bounding box from Vaughan et al. (2022).

    Returns:
        DataFrame with canonical columns: ``latitude``, ``longitude``,
        and optionally ``elevation`` and ``station_id``.
    """
    df = pd.read_csv(csv_path)

    # Rename GHCNh station list columns to canonical forms.
    df = df.rename(columns={
        "GHCN_ID": "station_id",
        "LATITUDE": "latitude",
        "LONGITUDE": "longitude",
        "ELEVATION": "elevation",
    })

    if "latitude" not in df.columns or "longitude" not in df.columns:
        msg = (
            f"Expected GHCNh station list format with LATITUDE/LONGITUDE columns. "
            f"Got: {list(df.columns)}"
        )
        raise ValueError(msg)

    logger.info(f"Loaded {len(df)} stations from {csv_path}")

    # Apply regional filter if requested.
    if region == "europe":
        lon_min, lon_max = EUROPE_LON_RANGE
        lat_min, lat_max = EUROPE_LAT_RANGE
        mask = (
            (df["latitude"] >= lat_min)
            & (df["latitude"] <= lat_max)
            & (df["longitude"] >= lon_min)
            & (df["longitude"] <= lon_max)
        )
        df = df[mask].reset_index(drop=True)
        logger.info(
            f"Filtered to {len(df)} European stations "
            f"(lat {lat_min}–{lat_max}, lon {lon_min}–{lon_max})"
        )
    elif region is not None:
        msg = f"Unknown region '{region}'. Currently only 'europe' is supported."
        raise ValueError(msg)

    return df


def main():
    parser = argparse.ArgumentParser(
        description="Extract TESSERA embeddings at station locations",
    )
    parser.add_argument(
        "--station-csv",
        type=str,
        required=True,
        help="Path to CSV with station locations (e.g. GHCNh station_list.csv)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=".tmp_output/processed/tessera",
        help="Directory to save extracted embeddings",
    )
    parser.add_argument(
        "--years",
        type=int,
        nargs=2,
        metavar=("START", "END"),
        default=[2017, 2024],
        help="Start and end year inclusive (default: 2017 2024)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="point",
        choices=["point", "patch", "both"],
        help="Extraction mode: point (128-d), patch (H×W×128), or both",
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        default=64,
        help="Patch side length in pixels for patch mode (default: 64)",
    )
    parser.add_argument(
        "--region",
        type=str,
        default=None,
        choices=["europe"],
        help="Filter stations to a predefined region (default: no filter)",
    )
    parser.add_argument(
        "--embeddings-dir",
        type=str,
        default=None,
        help="Local cache directory for geotessera tile downloads",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-extract even if cached files exist",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load and optionally filter station locations.
    stations = load_stations(Path(args.station_csv), region=args.region)
    lats = stations["latitude"].values
    lons = stations["longitude"].values

    # Save the filtered station list so we know exactly which stations
    # correspond to which row indices in the embedding arrays.
    station_list_path = output_dir / "station_list_filtered.csv"
    stations.to_csv(station_list_path, index=False)
    logger.info(f"Saved filtered station list to {station_list_path}")

    year_start, year_end = args.years
    years = list(range(year_start, year_end + 1))
    logger.info(f"Extracting for years: {years}")

    metadata = {
        "station_csv": args.station_csv,
        "region": args.region,
        "n_stations": len(stations),
        "years": years,
        "mode": args.mode,
        "patch_size": args.patch_size,
        "per_year": {},
    }

    for year in years:
        logger.info(f"=== Year {year} ===")
        year_meta = {}

        # Point extraction.
        if args.mode in ("point", "both"):
            cache_path = output_dir / f"point_embeddings_{year}.npy"

            if not args.force:
                cached = load_cached_embeddings(cache_path)
            else:
                cached = None

            if cached is not None:
                logger.info(
                    f"Point embeddings for {year} already cached "
                    f"({cached.shape}), skipping"
                )
                year_meta["point_shape"] = list(cached.shape)
                year_meta["point_nonzero"] = int(
                    np.sum(np.any(cached != 0, axis=1))
                )
            else:
                point_emb = extract_point_embeddings(
                    station_lats=lats,
                    station_lons=lons,
                    year=year,
                    embeddings_dir=args.embeddings_dir,
                )
                save_cached_embeddings(point_emb, cache_path)
                year_meta["point_shape"] = list(point_emb.shape)
                year_meta["point_nonzero"] = int(
                    np.sum(np.any(point_emb != 0, axis=1))
                )

        # Patch extraction.
        if args.mode in ("patch", "both"):
            cache_path = output_dir / f"patch_embeddings_{year}.npy"

            if not args.force:
                cached = load_cached_embeddings(cache_path)
            else:
                cached = None

            if cached is not None:
                logger.info(
                    f"Patch embeddings for {year} already cached "
                    f"({cached.shape}), skipping"
                )
                n_sta = cached.shape[0]
                year_meta["patch_shape"] = list(cached.shape)
                year_meta["patch_nonzero"] = int(
                    np.sum(np.any(cached.reshape(n_sta, -1) != 0, axis=1))
                )
            else:
                patch_emb = extract_patch_embeddings(
                    station_lats=lats,
                    station_lons=lons,
                    year=year,
                    patch_size=args.patch_size,
                    embeddings_dir=args.embeddings_dir,
                    output_path=cache_path,   # <-- writes directly to disk, never OOMs
                    resume=not args.force,    # <-- resumes if interrupted
                )
                # save_cached_embeddings(patch_emb, cache_path)
                n_sta = patch_emb.shape[0]
                year_meta["patch_shape"] = list(patch_emb.shape)
                year_meta["patch_nonzero"] = int(
                    np.sum(np.any(patch_emb.reshape(n_sta, -1) != 0, axis=1))
                )

        metadata["per_year"][str(year)] = year_meta

    # Save metadata.
    meta_path = output_dir / "extraction_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Metadata saved to {meta_path}")

    logger.info("Done!")


if __name__ == "__main__":
    main()