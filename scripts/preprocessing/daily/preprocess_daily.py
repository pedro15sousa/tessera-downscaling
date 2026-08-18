"""Preprocess raw 6-hourly ERA5 and GHCNh data into a daily training-ready dataset.

This script is the bridge between the raw downloaded data and the training
pipeline. It performs:

  1. ERA5 daily aggregation: loads 4 six-hourly files per variable per day,
     computes daily max/mean, crops to the European bounding box, and stacks
     all variables into a single (n_channels, n_lat, n_lon) tensor per day.

  2. GHCNh daily aggregation: loads 4 six-hourly GHCNh files per day,
     extracts hourly temperature observations per station, and computes daily
     Tmax. Records how many hourly observations contributed.

  3. Station filtering: identifies which stations have enough valid data days,
     valid elevation, and fall within the European bounding box.

  4. Delta-elevation: computes station elevation minus ERA5 orography
     interpolated to each station's location.

  5. Dataset assembly: saves everything in a clean directory structure with
     metadata recording channel ordering, grid coordinates, and split
     definitions.

Usage:
    uv run --group core python projects/tessera_downscaling/scripts/preprocessing/daily/preprocess_daily.py \\
        --era5-dir .tmp_output/processed \\
        --ghcnh-dir .tmp_output/processed/ghcnh/data \\
        --static-path .tmp_output/processed/era5_static/era5_static_0p25_all.nc \\
        --station-csv .tmp_output/raw/ghcnh/station_list.csv \\
        --output-dir .tmp_output/dataset_daily \\
        --start-date 2017-01-01 \\
        --end-date 2023-01-10
"""

import argparse
import json
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

# Shared helpers (constants, ERA5/GHCNh aggregation, static fields,
# delta-elevation, split helpers). These used to live inline in this file
# and have been factored out so the multi-region preprocessing script can
# share the exact same logic without duplication.
# Shared helpers live one level up under preprocessing/helpers.py. Adding
# the grandparent directory to sys.path lets us import the module as
# `helpers` without turning `preprocessing/` into a proper package — these
# are still scripts that live next to each other and want the simplest
# possible import path.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from helpers import (  # noqa: E402
    SURFACE_VARS,
    ATMOS_VARS,
    PRESSURE_LEVELS,
    MIN_HOURLY_OBS_PER_DAY,
    MIN_VALID_DAYS,
    TRAIN_STATION_FRACTION,
    TRAIN_END,
    VAL_END,
    SPLIT_SEED,
    GHCNH_TARGET_VARS,
    compute_grid_crop_indices,
    aggregate_era5_daily,
    era5_channel_names,
    filter_valid_elevation,
    load_static_fields,
    aggregate_ghcnh_daily,
    compute_delta_elevation,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("preprocess_daily")

# ---------------------------------------------------------------------------
# Europe-specific constants
# ---------------------------------------------------------------------------

# European bounding box.
EUROPE_LAT_RANGE = (35.0, 75.0)
EUROPE_LON_RANGE = (-24.0, 40.0)



# ---------------------------------------------------------------------------
# Station processing
# ---------------------------------------------------------------------------

def load_and_filter_stations(
    station_csv: Path,
    region: str = "europe",
) -> pd.DataFrame:
    """Load the GHCNh station list, optionally filtered to a region.

    Args:
        station_csv: Path to the GHCNh station list CSV.
        region: ``"europe"`` to apply the bounding box from Vaughan et al.
            (2022), or ``"global"`` to keep all stations.

    Returns:
        DataFrame with columns: station_id, latitude, longitude, elevation.
    """
    df = pd.read_csv(station_csv)
    df = df.rename(columns={
        "GHCN_ID": "station_id",
        "LATITUDE": "latitude",
        "LONGITUDE": "longitude",
        "ELEVATION": "elevation",
    })

    if region == "europe":
        mask = (
            (df["latitude"] >= EUROPE_LAT_RANGE[0])
            & (df["latitude"] <= EUROPE_LAT_RANGE[1])
            & (df["longitude"] >= EUROPE_LON_RANGE[0])
            & (df["longitude"] <= EUROPE_LON_RANGE[1])
        )
        df = df[mask][["station_id", "latitude", "longitude", "elevation"]].reset_index(
            drop=True
        )
        logger.info(f"Loaded {len(df)} European stations with valid elevation")
    elif region == "global":
        df = df[["station_id", "latitude", "longitude", "elevation"]].reset_index(
            drop=True
        )
        logger.info(f"Loaded {len(df)} global stations")
    else:
        raise ValueError(f"Unknown region '{region}'. Use 'europe' or 'global'.")

    # Drops missing-elevation rows AND sentinel/out-of-range values that
    # GHCN encodes as -999.9 or 9999. See helpers.filter_valid_elevation.
    df = filter_valid_elevation(df, logger=logger)

    return df



def assign_splits(
    stations: pd.DataFrame,
    dates: list[str],
) -> tuple[np.ndarray, dict]:
    """Assign spatial and temporal splits.

    Returns:
        (station_split, temporal_split_info) where station_split is an array
        of "train" or "test" per station, and temporal_split_info is a dict
        with date boundaries.
    """
    n_stations = len(stations)
    rng = np.random.RandomState(SPLIT_SEED)
    perm = rng.permutation(n_stations)
    n_train = int(n_stations * TRAIN_STATION_FRACTION)

    station_split = np.array(["test"] * n_stations, dtype="U5")
    station_split[perm[:n_train]] = "train"

    temporal_info = {
        "train_end": TRAIN_END,
        "val_end": VAL_END,
        "train_dates": [d for d in dates if d <= TRAIN_END],
        "val_dates": [d for d in dates if TRAIN_END < d <= VAL_END],
        "test_dates": [d for d in dates if d > VAL_END],
    }

    n_train_dates = len(temporal_info["train_dates"])
    n_val_dates = len(temporal_info["val_dates"])
    n_test_dates = len(temporal_info["test_dates"])
    logger.info(
        f"Temporal split: {n_train_dates} train / {n_val_dates} val / "
        f"{n_test_dates} test days"
    )
    logger.info(
        f"Spatial split: {np.sum(station_split == 'train')} train / "
        f"{np.sum(station_split == 'test')} test stations"
    )

    return station_split, temporal_info


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Preprocess raw ERA5 + GHCNh into a daily training dataset",
    )
    parser.add_argument(
        "--era5-dir", type=Path, required=True,
        help="Root directory containing era5_wb2_quarter_*/data/ subdirectories",
    )
    parser.add_argument(
        "--ghcnh-dir", type=Path, required=True,
        help="Directory containing GHCNh 6-hourly NetCDF files",
    )
    parser.add_argument(
        "--static-path", type=Path, required=True,
        help="Path to the ERA5 static fields NetCDF file",
    )
    parser.add_argument(
        "--station-csv", type=Path, required=True,
        help="Path to the GHCNh station list CSV",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path(".tmp_output/dataset_daily"),
        help="Output directory for the preprocessed dataset",
    )
    parser.add_argument(
        "--start-date", type=str, default="2017-01-01",
        help="Start date (inclusive), format YYYY-MM-DD",
    )
    parser.add_argument(
        "--end-date", type=str, default="2023-01-10",
        help="End date (inclusive), format YYYY-MM-DD",
    )
    parser.add_argument(
        "--target-variables", type=str, nargs="+",
        default=list(GHCNH_TARGET_VARS.keys()),
        choices=list(GHCNH_TARGET_VARS.keys()),
        help=(
            "Station-level target variables to extract and store in the .npz. "
            "Default: all supported variables. "
            "Example: --target-variables tmax wind_mean"
        ),
    )
    parser.add_argument(
        "--region", type=str, default="europe",
        choices=["europe", "global"],
        help=(
            "Spatial region for station filtering and ERA5 grid crop. "
            "'europe' (default): lat 35-75N, lon 24W-40E. "
            "'global': all stations, full ERA5 grid."
        ),
    )
    args = parser.parse_args()

    output_dir = args.output_dir
    era5_daily_dir = output_dir / "era5_daily"
    ghcnh_daily_dir = output_dir / "ghcnh_daily"
    era5_daily_dir.mkdir(parents=True, exist_ok=True)
    ghcnh_daily_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Step 1: Compute grid crop indices from an ERA5 sample file.
    # ------------------------------------------------------------------
    sample_ds = xr.open_dataset(
        args.era5_dir / "era5_wb2_quarter_2m_temperature" / "data" / "2017-01-01-00.nc"
    )
    lats = sample_ds.latitude.values
    lons = sample_ds.longitude.values
    sample_ds.close()

    lat_range = EUROPE_LAT_RANGE if args.region == "europe" else None
    lon_range = EUROPE_LON_RANGE if args.region == "europe" else None

    lat_indices, lon_indices, roll_amount, lats_crop, lons_crop = (
        compute_grid_crop_indices(lats, lons, lat_range=lat_range, lon_range=lon_range)
    )
    logger.info(
        f"Grid crop ({args.region}): {len(lats_crop)} lats x {len(lons_crop)} lons "
        f"(lat {lats_crop.min():.2f}–{lats_crop.max():.2f}, "
        f"lon {lons_crop.min():.2f}–{lons_crop.max():.2f})"
    )

    # ------------------------------------------------------------------
    # Step 2: Load and crop static fields.
    # ------------------------------------------------------------------
    static_array, static_var_names = load_static_fields(
        args.static_path, lat_indices, lon_indices, roll_amount
    )
    logger.info(f"Static fields: {len(static_var_names)} variables, shape {static_array.shape}")
    np.save(output_dir / "static_fields.npy", static_array)

    # ------------------------------------------------------------------
    # Step 3: Load and filter stations.
    # ------------------------------------------------------------------
    stations = load_and_filter_stations(args.station_csv, region=args.region)
    station_ids = stations["station_id"].values

    # ------------------------------------------------------------------
    # Step 4: Compute delta-elevation.
    # ------------------------------------------------------------------
    delta_elev = compute_delta_elevation(stations, args.static_path)
    stations["delta_elevation"] = delta_elev

    # ------------------------------------------------------------------
    # Step 5: Generate date list and process each day.
    # ------------------------------------------------------------------
    start = datetime.strptime(args.start_date, "%Y-%m-%d")
    end = datetime.strptime(args.end_date, "%Y-%m-%d")
    dates = []
    current = start
    while current <= end:
        dates.append(current.strftime("%Y-%m-%d"))
        current += timedelta(days=1)
    logger.info(f"Processing {len(dates)} days from {dates[0]} to {dates[-1]}")

    # Build the channel name list for ERA5 daily data.
    channel_names = []
    for var in SURFACE_VARS:
        if var == "2m_temperature":
            channel_names.extend(["2m_temperature_max", "2m_temperature_mean"])
        elif var == "total_precipitation_6hr":
            channel_names.append("total_precipitation_sum")
        else:
            channel_names.append(f"{var}_mean")
    for var in ATMOS_VARS:
        for level in PRESSURE_LEVELS:
            channel_names.append(f"{var}_{level}hPa_mean")
    logger.info(f"ERA5 dynamic channels ({len(channel_names)}): {channel_names}")

    # Track per-station valid day counts per variable for filtering.
    # A station is kept if it meets MIN_VALID_DAYS for ANY requested variable.
    valid_day_counts: dict[str, np.ndarray] = {
        v: np.zeros(len(station_ids), dtype=np.int32)
        for v in args.target_variables
    }
    days_processed = 0
    days_skipped = 0

    for date_str in dates:
        era5_path = era5_daily_dir / f"{date_str}.npy"
        ghcnh_path = ghcnh_daily_dir / f"{date_str}.npz"

        # Skip if already processed and contains all requested variables.
        if era5_path.exists() and ghcnh_path.exists():
            ghcnh_data = np.load(ghcnh_path)
            missing_vars = [
                v for v in args.target_variables if v not in ghcnh_data.files
            ]
            if not missing_vars:
                for v in args.target_variables:
                    if v in ghcnh_data.files:
                        valid_day_counts[v] += ~np.isnan(ghcnh_data[v])
                days_processed += 1
                continue
            # Fall through to re-process: some requested variables are missing.
            logger.info(
                f"{date_str}: re-processing to add missing variables: {missing_vars}"
            )

        # Aggregate ERA5.
        era5_daily = aggregate_era5_daily(
            args.era5_dir, date_str, lat_indices, lon_indices, roll_amount
        )
        if era5_daily is None:
            days_skipped += 1
            continue

        # Aggregate GHCNh for all requested target variables.
        ghcnh_results = aggregate_ghcnh_daily(
            args.ghcnh_dir, date_str, station_ids,
            target_variables=args.target_variables,
        )

        # Save ERA5 grid and GHCNh targets.
        np.save(era5_path, era5_daily)
        np.savez_compressed(ghcnh_path, **ghcnh_results)

        for v in args.target_variables:
            valid_day_counts[v] += ~np.isnan(ghcnh_results[v])
        days_processed += 1

        if days_processed % 100 == 0:
            logger.info(f"  Processed {days_processed} days, skipped {days_skipped}")

    logger.info(
        f"Finished: {days_processed} days processed, {days_skipped} skipped"
    )

    # ------------------------------------------------------------------
    # Step 6: Filter stations by minimum valid days.
    # A station is kept if it meets MIN_VALID_DAYS for ANY requested variable.
    # ------------------------------------------------------------------
    # Store per-variable counts and compute the union mask.
    for v, counts in valid_day_counts.items():
        stations[f"n_valid_days_{v}"] = counts

    valid_mask = np.zeros(len(station_ids), dtype=bool)
    for v, counts in valid_day_counts.items():
        valid_mask |= (counts >= MIN_VALID_DAYS)

    logger.info(
        f"Station filtering: {valid_mask.sum()}/{len(stations)} stations have "
        f">= {MIN_VALID_DAYS} valid days for at least one of {list(valid_day_counts)}"
    )

    # Keep only valid stations. Re-index the GHCNh files to match.
    stations_filtered = stations[valid_mask].reset_index(drop=True)
    if len(stations_filtered) == 0:
        logger.warning("No stations passed the minimum valid days filter. "
                        "Try processing more days or lowering --min-valid-days.")
        stations_filtered = stations.copy()
        valid_indices = np.arange(len(stations))
    else:
        valid_indices = np.where(valid_mask)[0]

    # Save a mapping so we can re-index GHCNh data during training.
    np.save(output_dir / "valid_station_indices.npy", valid_indices)

    # ------------------------------------------------------------------
    # Step 7: Assign spatial and temporal splits.
    # ------------------------------------------------------------------
    valid_dates = sorted([
        d for d in dates
        if (era5_daily_dir / f"{d}.npy").exists()
    ])
    station_split, temporal_info = assign_splits(stations_filtered, valid_dates)
    stations_filtered["spatial_split"] = station_split

    # ------------------------------------------------------------------
    # Step 8: Save metadata.
    # ------------------------------------------------------------------
    stations_filtered.to_csv(output_dir / "stations.csv", index=False)

    metadata = {
        "era5_dynamic_channels": channel_names,
        "static_channels": static_var_names,
        "n_dynamic_channels": len(channel_names),
        "n_static_channels": len(static_var_names),
        "n_total_channels": len(channel_names) + len(static_var_names),
        "pressure_levels": PRESSURE_LEVELS,
        "grid_shape": [len(lats_crop), len(lons_crop)],
        "lat_range": [float(lats_crop.min()), float(lats_crop.max())],
        "lon_range": [float(lons_crop.min()), float(lons_crop.max())],
        "n_stations_total": len(stations),
        "n_stations_filtered": len(stations_filtered),
        "n_days": days_processed,
        "n_days_skipped": days_skipped,
        "date_range": [dates[0], dates[-1]],
        "valid_dates": valid_dates,
        "temporal_split": {
            "train_end": TRAIN_END,
            "val_end": VAL_END,
            "n_train_days": len(temporal_info["train_dates"]),
            "n_val_days": len(temporal_info["val_dates"]),
            "n_test_days": len(temporal_info["test_dates"]),
        },
        "spatial_split": {
            "n_train_stations": int(np.sum(station_split == "train")),
            "n_test_stations": int(np.sum(station_split == "test")),
            "seed": SPLIT_SEED,
            "train_fraction": TRAIN_STATION_FRACTION,
        },
        "min_valid_days": MIN_VALID_DAYS,
        "min_hourly_obs_per_day": MIN_HOURLY_OBS_PER_DAY,
        "elevation_normalisation": "raw_metres",
    }

    # Save grid coordinates.
    np.save(output_dir / "lats.npy", lats_crop)
    np.save(output_dir / "lons.npy", lons_crop)

    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Dataset saved to {output_dir}")
    logger.info(
        f"  {metadata['n_total_channels']} total channels "
        f"({metadata['n_dynamic_channels']} dynamic + "
        f"{metadata['n_static_channels']} static)"
    )
    logger.info(
        f"  {metadata['n_stations_filtered']} stations, {metadata['n_days']} days"
    )
    logger.info("Done!")


if __name__ == "__main__":
    main()