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
    uv run --group core python projects/tessera_downscaling/scripts/preprocess_daily.py \\
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
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from scipy.interpolate import griddata

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("preprocess_daily")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# European bounding box.
EUROPE_LAT_RANGE = (35.0, 75.0)
EUROPE_LON_RANGE = (-24.0, 40.0)

# ERA5 surface variables (no pressure level dimension).
SURFACE_VARS = [
    "2m_temperature",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "mean_sea_level_pressure",
    "total_precipitation_6hr",
]

# ERA5 atmospheric variables (have a pressure level dimension).
ATMOS_VARS = [
    "temperature",
    "u_component_of_wind",
    "v_component_of_wind",
    "specific_humidity",
    "geopotential",
]

# Pressure levels we downloaded (matching the paper).
PRESSURE_LEVELS = [500, 700, 850]

# Minimum number of valid daily Tmax observations for a station to be included.
MIN_VALID_DAYS = 100

# Minimum number of hourly observations in a day for a valid daily Tmax.
MIN_HOURLY_OBS_PER_DAY = 4

# Spatial train/test split ratio.
TRAIN_STATION_FRACTION = 0.85

# Temporal split boundaries.
TRAIN_END = "2020-12-31"
VAL_END = "2021-12-31"
# Everything after VAL_END until the dataset end is test.

# Random seed for reproducible station split.
SPLIT_SEED = 42


# ---------------------------------------------------------------------------
# ERA5 processing
# ---------------------------------------------------------------------------

def compute_grid_crop_indices(
    lats: np.ndarray,
    lons: np.ndarray,
    lat_range: tuple[float, float] | None = None,
    lon_range: tuple[float, float] | None = None,
) -> tuple[np.ndarray, np.ndarray, int, np.ndarray, np.ndarray]:
    """Compute ERA5 array indices for an optional spatial crop.

    Handles the 0-360 longitude convention used by ERA5 by rolling the
    longitude axis so that the requested range is always contiguous.

    Args:
        lats: ERA5 latitude array (descending, 90 to -90).
        lons: ERA5 longitude array (0 to 360).
        lat_range: (min_lat, max_lat) to keep. None = keep all.
        lon_range: (min_lon, max_lon) in -180/180 convention. None = keep all.

    Returns:
        (lat_indices, lon_indices, roll_amount, lats_crop, lons_crop)
    """
    # --- Latitude ---
    if lat_range is not None:
        lat_mask = (lats >= lat_range[0]) & (lats <= lat_range[1])
        lat_indices = np.where(lat_mask)[0]
    else:
        lat_indices = np.arange(len(lats))

    # --- Longitude ---
    # Always convert to -180/180 after an optional roll so the output
    # coordinates are consistent regardless of region.
    if lon_range is not None:
        lon_min_360 = lon_range[0] % 360
        roll_start_idx = np.argmin(np.abs(lons - lon_min_360))
        roll_amount = -roll_start_idx
        lons_rolled = np.roll(lons, roll_amount)
        lons_rolled_180 = np.where(lons_rolled > 180, lons_rolled - 360, lons_rolled)
        lon_mask = (lons_rolled_180 >= lon_range[0]) & (lons_rolled_180 <= lon_range[1])
        lon_indices = np.where(lon_mask)[0]
    else:
        # No crop: still convert 0-360 → -180/180 for consistency, no roll.
        roll_amount = 0
        lons_rolled_180 = np.where(lons > 180, lons - 360, lons)
        lon_indices = np.arange(len(lons))

    lats_crop = lats[lat_indices]
    lons_crop = lons_rolled_180[lon_indices]

    return lat_indices, lon_indices, roll_amount, lats_crop, lons_crop


def load_era5_6hourly(
    era5_dir: Path, var_name: str, date_str: str, hours: list[int]
) -> list[xr.Dataset]:
    """Load the 4 six-hourly ERA5 files for a given variable and date."""
    var_dir = era5_dir / f"era5_wb2_quarter_{var_name}" / "data"
    datasets = []
    for h in hours:
        path = var_dir / f"{date_str}-{h:02d}.nc"
        if path.exists():
            datasets.append(xr.open_dataset(path))
        else:
            datasets.append(None)
    return datasets


def aggregate_era5_daily(
    era5_dir: Path,
    date_str: str,
    lat_indices: np.ndarray,
    lon_indices: np.ndarray,
    roll_amount: int,
) -> np.ndarray | None:
    """Load and aggregate ERA5 data for one day into a single daily grid.

    Returns:
        Array of shape (n_channels, n_lat_crop, n_lon_crop), or None if
        any required files are missing.
    """
    hours = [0, 6, 12, 18]
    channels = []

    # Process surface variables.
    for var in SURFACE_VARS:
        datasets = load_era5_6hourly(era5_dir, var, date_str, hours)
        if any(ds is None for ds in datasets):
            # Close any that were opened.
            for ds in datasets:
                if ds is not None:
                    ds.close()
            return None

        # Stack the 4 timestamps into (4, lat, lon).
        # The variable name in the NetCDF matches the directory name.
        var_key = list(datasets[0].data_vars)[0]
        arrays = []
        for ds in datasets:
            data = ds[var_key].values
            # Roll longitude axis to make Europe contiguous.
            data_rolled = np.roll(data, roll_amount, axis=-1)
            # Crop to European bounds.
            if data.ndim == 2:
                cropped = data_rolled[lat_indices][:, lon_indices]
            else:
                cropped = data_rolled
            arrays.append(cropped)
            ds.close()

        stacked = np.stack(arrays, axis=0)  # (4, lat, lon)

        if var == "2m_temperature":
            # Daily max and daily mean as separate channels.
            channels.append(np.max(stacked, axis=0))
            channels.append(np.mean(stacked, axis=0))
        elif var == "total_precipitation_6hr":
            # Daily sum for precipitation.
            channels.append(np.sum(stacked, axis=0))
        else:
            # Daily mean for wind and pressure.
            channels.append(np.mean(stacked, axis=0))

    # Process atmospheric variables.
    for var in ATMOS_VARS:
        datasets = load_era5_6hourly(era5_dir, var, date_str, hours)
        if any(ds is None for ds in datasets):
            for ds in datasets:
                if ds is not None:
                    ds.close()
            return None

        var_key = list(datasets[0].data_vars)[0]

        for level in PRESSURE_LEVELS:
            arrays = []
            for ds in datasets:
                data = ds[var_key].values
                # Atmospheric data has shape (level, lat, lon).
                # Find the index for this pressure level.
                level_idx = list(ds.level.values).index(level)
                data_level = data[level_idx]
                data_rolled = np.roll(data_level, roll_amount, axis=-1)
                cropped = data_rolled[lat_indices][:, lon_indices]
                arrays.append(cropped)

            stacked = np.stack(arrays, axis=0)  # (4, lat, lon)
            # Daily mean for all atmospheric variables.
            channels.append(np.mean(stacked, axis=0))

        for ds in datasets:
            ds.close()

    return np.stack(channels, axis=0).astype(np.float32)


# ---------------------------------------------------------------------------
# Static fields processing
# ---------------------------------------------------------------------------

def load_static_fields(
    static_path: Path,
    lat_indices: np.ndarray,
    lon_indices: np.ndarray,
    roll_amount: int,
) -> tuple[np.ndarray, list[str]]:
    """Load and crop the static ERA5 fields.

    Returns:
        (static_array, var_names) where static_array has shape
        (n_static_vars, n_lat_crop, n_lon_crop).
    """
    ds = xr.open_dataset(static_path)
    var_names = list(ds.data_vars)
    channels = []

    for var in var_names:
        data = ds[var].values
        # Remove the time dimension if present (static fields have valid_time=1).
        if data.ndim == 3:
            data = data[0]
        data_rolled = np.roll(data, roll_amount, axis=-1)
        cropped = data_rolled[lat_indices][:, lon_indices]
        channels.append(cropped)

    ds.close()
    return np.stack(channels, axis=0).astype(np.float32), var_names


# ---------------------------------------------------------------------------
# GHCNh processing
# ---------------------------------------------------------------------------

# GHCNh variable name → (aggregation, npz key) for supported target variables.
# "max"  = daily maximum (for temperature)
# "mean" = daily mean    (for wind speed, etc.)
GHCNH_TARGET_VARS: dict[str, tuple[str, str]] = {
    "tmax":      ("temperature", "max"),
    "wind_mean": ("wind_speed",  "mean"),
}


def aggregate_ghcnh_daily(
    ghcnh_dir: Path,
    date_str: str,
    station_ids: np.ndarray,
    target_variables: list[str] | None = None,
) -> dict[str, np.ndarray]:
    """Compute daily station observations from 4 six-hourly GHCNh files.

    Args:
        ghcnh_dir: Directory containing GHCNh 6-hourly NetCDF files.
        date_str: Date string "YYYY-MM-DD".
        station_ids: Array of station ID strings to extract.
        target_variables: List of keys from GHCNH_TARGET_VARS to extract.
            Defaults to ["tmax"]. Example: ["tmax", "wind_mean"].

    Returns:
        Dict mapping each target variable key (e.g. "tmax", "wind_mean")
        to a (n_stations,) float32 array (NaN = missing), plus "obs_count"
        as a (n_stations,) int32 array tracking temperature observations.
    """
    if target_variables is None:
        target_variables = ["tmax"]

    # Validate requested variables.
    for var in target_variables:
        if var not in GHCNH_TARGET_VARS:
            raise ValueError(
                f"Unknown target variable '{var}'. "
                f"Supported: {list(GHCNH_TARGET_VARS)}"
            )

    # Determine which raw GHCNh fields we need to read.
    # e.g. {"temperature": ["tmax"], "wind_speed": ["wind_mean"]}
    raw_field_to_targets: dict[str, list[str]] = {}
    for tv in target_variables:
        raw_field, _ = GHCNH_TARGET_VARS[tv]
        raw_field_to_targets.setdefault(raw_field, []).append(tv)

    n_stations = len(station_ids)
    # Accumulator: raw_field -> {station_id: [observations]}
    accumulators: dict[str, dict[str, list[float]]] = {
        field: {sid: [] for sid in station_ids}
        for field in raw_field_to_targets
    }

    for hour in [0, 6, 12, 18]:
        path = ghcnh_dir / f"{date_str}-{hour:02d}.nc"
        if not path.exists():
            continue
        try:
            ds = xr.open_dataset(path)
        except Exception:
            continue

        file_stations = ds["STATION"].values
        fields_present = [f for f in raw_field_to_targets if f in ds.data_vars]

        if not fields_present:
            ds.close()
            continue

        field_arrays = {f: ds[f].values for f in fields_present}

        for j in range(len(file_stations)):
            sid = file_stations[j]
            if isinstance(sid, (bytes, np.bytes_)):
                sid = sid.decode("utf-8").strip()

            for field, arr in field_arrays.items():
                val = arr[j]
                if sid in accumulators[field] and not np.isnan(val):
                    accumulators[field][sid].append(val)

        ds.close()

    # Aggregate and pack results.
    results: dict[str, np.ndarray] = {}

    # obs_count always tracks temperature (backwards-compatible).
    obs_count = np.zeros(n_stations, dtype=np.int32)

    for tv in target_variables:
        raw_field, agg = GHCNH_TARGET_VARS[tv]
        out = np.full(n_stations, np.nan, dtype=np.float32)
        for i, sid in enumerate(station_ids):
            obs = accumulators[raw_field][sid]
            if raw_field == "temperature":
                obs_count[i] = len(obs)
            if len(obs) >= MIN_HOURLY_OBS_PER_DAY:
                out[i] = np.max(obs) if agg == "max" else np.mean(obs)
        results[tv] = out

    results["obs_count"] = obs_count
    return results


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

    # Remove stations with missing elevation.
    df = df.dropna(subset=["elevation"]).reset_index(drop=True)

    return df


def compute_delta_elevation(
    stations: pd.DataFrame,
    static_path: Path,
) -> np.ndarray:
    """Compute station elevation minus ERA5 orography at each station location.

    The ERA5 geopotential (z) is divided by g=9.81 to get orography in metres,
    then bilinearly interpolated to each station's lat/lon.

    Returns:
        (n_stations,) array of delta-elevation in metres.
    """
    ds = xr.open_dataset(static_path)
    z = ds["z"].values
    if z.ndim == 3:
        z = z[0]
    # Convert geopotential to elevation in metres.
    orog = z / 9.81

    lats = ds.latitude.values
    lons = ds.longitude.values
    ds.close()

    # Convert longitudes to -180/180 for consistency with station coordinates.
    lons_180 = np.where(lons > 180, lons - 360, lons)

    # Create a meshgrid for interpolation.
    lon_grid, lat_grid = np.meshgrid(lons_180, lats)
    points = np.column_stack([lon_grid.ravel(), lat_grid.ravel()])
    values = orog.ravel()

    # Interpolate to station locations.
    station_coords = np.column_stack([
        stations["longitude"].values,
        stations["latitude"].values,
    ])
    orog_at_stations = griddata(points, values, station_coords, method="linear")

    # Delta elevation = station elevation - grid orography.
    delta_elev = stations["elevation"].values - orog_at_stations

    logger.info(
        f"Delta-elevation: mean={np.nanmean(delta_elev):.1f}m, "
        f"std={np.nanstd(delta_elev):.1f}m, "
        f"range=[{np.nanmin(delta_elev):.0f}, {np.nanmax(delta_elev):.0f}]m"
    )
    return delta_elev.astype(np.float32)


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