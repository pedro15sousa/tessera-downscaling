"""Shared helpers for preprocessing scripts.

This module holds the pure data-processing functions and constants used by
both :mod:`preprocess_daily` (European-only) and
:mod:`preprocess_daily_global` (multi-region). Each caller supplies its own
region definitions and wiring; this module stays free of any region-specific
state or argparse logic.

The functions here are identical to the versions that previously lived
inline in ``preprocess_daily.py``; moving them out means the multi-region
script can reuse them verbatim with no divergence risk.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from scipy.interpolate import griddata

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants (shared across all preprocessing pipelines)
# ---------------------------------------------------------------------------

# ERA5 surface variables (no pressure level dimension).
SURFACE_VARS: list[str] = [
    "2m_temperature",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "mean_sea_level_pressure",
    "total_precipitation_6hr",
]

# ERA5 atmospheric variables (have a pressure level dimension).
ATMOS_VARS: list[str] = [
    "temperature",
    "u_component_of_wind",
    "v_component_of_wind",
    "specific_humidity",
    "geopotential",
]

# Pressure levels we downloaded (matching Vaughan et al. 2022).
PRESSURE_LEVELS: list[int] = [500, 700, 850]

# Minimum number of hourly observations in a day for a valid daily aggregate.
MIN_HOURLY_OBS_PER_DAY = 4

# Minimum number of valid daily observations for a station to be included.
MIN_VALID_DAYS = 100

# Minimum number of valid snapshot observations for a station to be included
# in the timestamp-cadence dataset. Default 0 = no filtering (a station with
# even a handful of valid snapshots is kept). The snapshot preprocessing
# script exposes this via --min-valid-episodes so it can be re-enabled for
# specific experiments without code changes.
MIN_VALID_EPISODES = 0

# The 6-hourly timestamps that both ERA5 and GHCNh publish at. Snapshot
# episodes are produced at exactly these hours — the intersection of the
# two sources' native cadences.
SNAPSHOT_HOURS: list[int] = [0, 6, 12, 18]

# Spatial train/test split ratio.
TRAIN_STATION_FRACTION = 0.85

# Temporal split boundaries.
TRAIN_END = "2020-12-31"
VAL_END = "2021-12-31"

# Random seed for reproducible station split.
SPLIT_SEED = 42

# ---------------------------------------------------------------------------
# Elevation sentinel filter
# ---------------------------------------------------------------------------
# GHCNh encodes "missing elevation" as a sentinel value (typically -999.9
# in raw GHCN files), and a few rows carry obvious garbage like 9999. The
# preprocessor's existing dropna() catches NaN but not these sentinels,
# so 591 stations were entering the dataset with physically impossible
# elevations (and equally impossible delta_elev derived from them). This
# silently degraded the no-TESSERA baselines (which feed elevation
# directly into the MLP) and was the root cause of the dataset-vs-VAE
# station-set mismatch — the VAE pre-training filtered these stations
# out, the preprocessor didn't.
#
# Bounds chosen to mirror what the VAE pre-training applies, so the
# dataset and the latents agree on which stations are "elevation-valid":
ELEV_SENTINEL_LOW = -900.0   # rejects -999.9 (raw GHCN missing) and below
ELEV_SENTINEL_HIGH = 8848.0  # rejects 9999 and other high garbage; Mt Everest


def filter_valid_elevation(stations_df, logger=None):
    """Return ``stations_df`` with sentinel/NaN/out-of-range elevations dropped.

    The single source of truth for what counts as a "valid elevation"
    inside the preprocessing pipeline. Used by every preprocess_*.py
    so all four pipelines agree on the same rule.

    Args:
        stations_df: DataFrame with an ``elevation`` column.
        logger: Optional logging.Logger. Receives a one-line summary
            of how many stations were dropped and why.

    Returns:
        Filtered DataFrame, index reset.
    """
    import numpy as np
    elev = stations_df["elevation"].values
    elev_valid = (
        np.isfinite(elev)
        & (elev > ELEV_SENTINEL_LOW)
        & (elev <= ELEV_SENTINEL_HIGH)
    )
    if logger is not None:
        n_before = len(stations_df)
        n_kept = int(elev_valid.sum())
        n_nan = int((~np.isfinite(elev)).sum())
        n_sentinel = int(
            (np.isfinite(elev)
             & ~((elev > ELEV_SENTINEL_LOW) & (elev <= ELEV_SENTINEL_HIGH))).sum()
        )
        logger.info(
            f"Elevation filter: kept {n_kept}/{n_before} stations "
            f"(dropped {n_nan} NaN, {n_sentinel} sentinel/out-of-range; "
            f"bounds = ({ELEV_SENTINEL_LOW}, {ELEV_SENTINEL_HIGH}] m)"
        )
    return stations_df.loc[elev_valid].reset_index(drop=True)

# GHCNh target variable -> (raw field, aggregation). Daily cadence.
GHCNH_TARGET_VARS: dict[str, tuple[str, str]] = {
    "tmax":      ("temperature", "max"),
    "wind_mean": ("wind_speed",  "mean"),
}

# GHCNh target variable -> raw field. Snapshot cadence.
# At the snapshot cadence we take the single hourly observation at the
# episode's timestamp verbatim; there is no aggregation, so the structure
# here is a plain dict rather than the (field, agg) tuples used for the
# daily variables above.
GHCNH_SNAPSHOT_VARS: dict[str, str] = {
    "t2m":    "temperature",
    "wind":   "wind_speed",
    "precip": "precipitation_6_hour",
}

# Snapshot variables whose semantics tie the value to the row's exact
# timestamp. ``precipitation_6_hour`` reports the past-6h accumulation
# *ending at the row's time*: a row at 02:00 UTC inside the (00, 06] UTC
# bin is reporting the window ending at 02:00 — wrong window for the
# snapshot at 06:00 UTC. The aggregator therefore restricts these
# variables to rows whose timestamp is exactly the synoptic hour.
# ``t2m`` and ``wind`` are point-in-time observations and don't need
# this restriction (the latest non-NaN observation in the bin is a
# reasonable proxy for the synoptic-hour state).
_SNAPSHOT_VARS_REQUIRING_SYNOPTIC_TIME: frozenset[str] = frozenset({"precip"})


# ---------------------------------------------------------------------------
# ERA5 grid cropping
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
        lat_range: (min_lat, max_lat) to keep. ``None`` = keep all.
        lon_range: (min_lon, max_lon) in -180/180 convention. ``None`` = keep all.

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
) -> list:
    """Load the 4 six-hourly ERA5 files for a given variable and date.

    Returns a list of :class:`xarray.Dataset` handles (or ``None`` for any
    hour whose file is missing). Callers are responsible for closing the
    datasets.
    """
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

    Returns an array of shape ``(n_channels, n_lat_crop, n_lon_crop)``, or
    ``None`` if any required files are missing. Channel order matches the
    sequence defined by :data:`SURFACE_VARS` + :data:`ATMOS_VARS` ×
    :data:`PRESSURE_LEVELS`; see :func:`era5_channel_names` for the full
    naming convention.
    """
    hours = [0, 6, 12, 18]
    channels: list[np.ndarray] = []

    # Process surface variables.
    for var in SURFACE_VARS:
        datasets = load_era5_6hourly(era5_dir, var, date_str, hours)
        if any(ds is None for ds in datasets):
            for ds in datasets:
                if ds is not None:
                    ds.close()
            return None

        var_key = list(datasets[0].data_vars)[0]
        arrays = []
        for ds in datasets:
            data = ds[var_key].values
            data_rolled = np.roll(data, roll_amount, axis=-1)
            if data.ndim == 2:
                cropped = data_rolled[lat_indices][:, lon_indices]
            else:
                cropped = data_rolled
            arrays.append(cropped)
            ds.close()

        stacked = np.stack(arrays, axis=0)  # (4, lat, lon)

        if var == "2m_temperature":
            channels.append(np.max(stacked, axis=0))
            channels.append(np.mean(stacked, axis=0))
        elif var == "total_precipitation_6hr":
            channels.append(np.sum(stacked, axis=0))
        else:
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
                level_idx = list(ds.level.values).index(level)
                data_level = data[level_idx]
                data_rolled = np.roll(data_level, roll_amount, axis=-1)
                cropped = data_rolled[lat_indices][:, lon_indices]
                arrays.append(cropped)

            stacked = np.stack(arrays, axis=0)
            channels.append(np.mean(stacked, axis=0))

        for ds in datasets:
            ds.close()

    return np.stack(channels, axis=0).astype(np.float32)


def era5_channel_names() -> list[str]:
    """Return the canonical ordered channel names for ERA5 daily tensors.

    Kept in a function (not a module-level constant) so callers can import
    without needing to understand the construction logic.
    """
    names: list[str] = []
    for var in SURFACE_VARS:
        if var == "2m_temperature":
            names.extend(["2m_temperature_max", "2m_temperature_mean"])
        elif var == "total_precipitation_6hr":
            names.append("total_precipitation_sum")
        else:
            names.append(f"{var}_mean")
    for var in ATMOS_VARS:
        for level in PRESSURE_LEVELS:
            names.append(f"{var}_{level}hPa_mean")
    return names


# ---------------------------------------------------------------------------
# ERA5 snapshot aggregation (timestamp cadence)
# ---------------------------------------------------------------------------
#
# The snapshot variants read a single 6-hourly ERA5 file per episode and
# do no temporal aggregation — each episode's context grid is the raw
# state of the atmosphere at that one timestamp. Everything else (spatial
# cropping, roll handling, dtype) matches the daily path exactly.
#
# Channel differences vs daily:
#   * 2m_temperature is a single instantaneous channel (no max, no mean).
#   * total_precipitation_6hr stays as the 6h accumulation ending at this
#     timestamp — that's what the raw file contains; there is no 1-hourly
#     precip in the downloaded data to pass through instead.
#   * Everything else is the same as daily's "mean" channel, just without
#     the averaging across 4 files.
#
# Net effect: snapshot tensors have one fewer channel than daily tensors
# (2m_temperature_max is gone). The snapshot-specific channel name list
# is returned by :func:`era5_snapshot_channel_names` below.


def aggregate_era5_snapshot(
    era5_dir: Path,
    date_str: str,
    hour: int,
    lat_indices: np.ndarray,
    lon_indices: np.ndarray,
    roll_amount: int,
) -> np.ndarray | None:
    """Load ERA5 data for one ``(date, hour)`` episode as a single grid.

    Returns an array of shape ``(n_channels, n_lat_crop, n_lon_crop)``, or
    ``None`` if the required file for *any* variable at this timestamp is
    missing. Channel order matches :func:`era5_snapshot_channel_names`.

    Args:
        era5_dir: Root dir containing ``era5_wb2_quarter_<var>/data/`` subdirs
            (same layout the daily aggregator reads).
        date_str: ``"YYYY-MM-DD"``.
        hour: One of :data:`SNAPSHOT_HOURS` (0, 6, 12, 18).
        lat_indices, lon_indices, roll_amount: Output of
            :func:`compute_grid_crop_indices` for this region.
    """
    channels: list[np.ndarray] = []

    # Process surface variables.
    for var in SURFACE_VARS:
        path = era5_dir / f"era5_wb2_quarter_{var}" / "data" / f"{date_str}-{hour:02d}.nc"
        if not path.exists():
            return None
        ds = xr.open_dataset(path)
        var_key = list(ds.data_vars)[0]
        data = ds[var_key].values
        data_rolled = np.roll(data, roll_amount, axis=-1)
        if data.ndim == 2:
            cropped = data_rolled[lat_indices][:, lon_indices]
        else:
            cropped = data_rolled
        ds.close()

        # No temporal aggregation: every surface variable contributes one
        # channel. 2m_temperature becomes a single instantaneous channel
        # (dropping the daily max/mean pair); precipitation passes through
        # as the 6h accumulation ending at this timestamp.
        channels.append(cropped)

    # Process atmospheric variables (pressure-level data).
    for var in ATMOS_VARS:
        path = era5_dir / f"era5_wb2_quarter_{var}" / "data" / f"{date_str}-{hour:02d}.nc"
        if not path.exists():
            return None
        ds = xr.open_dataset(path)
        var_key = list(ds.data_vars)[0]
        data = ds[var_key].values  # (n_levels, H, W)

        for level in PRESSURE_LEVELS:
            level_idx = list(ds.level.values).index(level)
            data_level = data[level_idx]
            data_rolled = np.roll(data_level, roll_amount, axis=-1)
            cropped = data_rolled[lat_indices][:, lon_indices]
            channels.append(cropped)

        ds.close()

    return np.stack(channels, axis=0).astype(np.float32)


def era5_snapshot_channel_names() -> list[str]:
    """Return the canonical ordered channel names for ERA5 snapshot tensors.

    One channel per surface variable (no max/mean split for 2m temperature)
    plus one channel per atmospheric variable × pressure level. The
    ``total_precipitation_sum`` name from the daily channel list is kept
    as-is for consistency with existing normalisation / column-selection
    code, even though at snapshot cadence we're not summing across files
    — the content is the same 6h accumulation that the daily sum would
    have produced from the single file.
    """
    names: list[str] = []
    for var in SURFACE_VARS:
        if var == "2m_temperature":
            names.append("2m_temperature")
        elif var == "total_precipitation_6hr":
            names.append("total_precipitation_sum")
        else:
            names.append(var)
    for var in ATMOS_VARS:
        for level in PRESSURE_LEVELS:
            names.append(f"{var}_{level}hPa")
    return names


# ---------------------------------------------------------------------------
# Static fields
# ---------------------------------------------------------------------------

def load_static_fields(
    static_path: Path,
    lat_indices: np.ndarray,
    lon_indices: np.ndarray,
    roll_amount: int,
) -> tuple[np.ndarray, list[str]]:
    """Load and crop the static ERA5 fields to a given lat/lon subset."""
    ds = xr.open_dataset(static_path)
    var_names = list(ds.data_vars)
    channels = []

    for var in var_names:
        data = ds[var].values
        if data.ndim == 3:
            data = data[0]  # drop valid_time=1
        data_rolled = np.roll(data, roll_amount, axis=-1)
        cropped = data_rolled[lat_indices][:, lon_indices]
        channels.append(cropped)

    ds.close()
    return np.stack(channels, axis=0).astype(np.float32), var_names


# ---------------------------------------------------------------------------
# GHCNh aggregation
# ---------------------------------------------------------------------------

def aggregate_ghcnh_daily(
    ghcnh_dir: Path,
    date_str: str,
    station_ids: np.ndarray,
    target_variables: list[str] | None = None,
) -> dict[str, np.ndarray]:
    """Compute daily station observations from 4 six-hourly GHCNh files.

    Returns a dict mapping each target variable key (e.g. ``"tmax"``,
    ``"wind_mean"``) to a ``(n_stations,)`` float32 array (NaN = missing),
    plus ``"obs_count"`` as a ``(n_stations,)`` int32 array tracking
    temperature observations.
    """
    if target_variables is None:
        target_variables = ["tmax"]

    for var in target_variables:
        if var not in GHCNH_TARGET_VARS:
            raise ValueError(
                f"Unknown target variable '{var}'. "
                f"Supported: {list(GHCNH_TARGET_VARS)}"
            )

    # e.g. {"temperature": ["tmax"], "wind_speed": ["wind_mean"]}
    raw_field_to_targets: dict[str, list[str]] = {}
    for tv in target_variables:
        raw_field, _ = GHCNH_TARGET_VARS[tv]
        raw_field_to_targets.setdefault(raw_field, []).append(tv)

    n_stations = len(station_ids)
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

    results: dict[str, np.ndarray] = {}
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


def aggregate_ghcnh_snapshot(
    ghcnh_dir: Path,
    date_str: str,
    hour: int,
    station_ids: np.ndarray,
    target_variables: list[str] | None = None,
) -> dict[str, np.ndarray]:
    """Read a single ``(date, hour)`` GHCNh file, no temporal aggregation.

    Returns a dict mapping each requested snapshot variable key (e.g.
    ``"t2m"``, ``"wind"``) to a ``(n_stations,)`` float32 array, with NaN
    where the station has no observation at this exact timestamp. An
    ``"obs_count"`` entry is also returned (int32, ``(n_stations,)``) —
    it's either 0 or 1 per station since we only read one file, but the
    field is kept for symmetry with :func:`aggregate_ghcnh_daily` so
    downstream code can treat both cadences the same way.

    Args:
        ghcnh_dir: Directory containing ``YYYY-MM-DD-HH.nc`` files.
        date_str: ``"YYYY-MM-DD"``.
        hour: One of :data:`SNAPSHOT_HOURS`.
        station_ids: The stations whose observations we want, in the
            output order.
        target_variables: Subset of :data:`GHCNH_SNAPSHOT_VARS` keys to
            read. Default: all of them.
    """
    if target_variables is None:
        target_variables = list(GHCNH_SNAPSHOT_VARS.keys())

    for var in target_variables:
        if var not in GHCNH_SNAPSHOT_VARS:
            raise ValueError(
                f"Unknown snapshot target variable '{var}'. "
                f"Supported: {list(GHCNH_SNAPSHOT_VARS)}"
            )

    n_stations = len(station_ids)
    # Prepare output arrays up front — they all share the same station
    # order, and we'll fill them in a single pass over the file.
    results: dict[str, np.ndarray] = {
        tv: np.full(n_stations, np.nan, dtype=np.float32)
        for tv in target_variables
    }
    obs_count = np.zeros(n_stations, dtype=np.int32)

    path = ghcnh_dir / f"{date_str}-{hour:02d}.nc"
    if not path.exists():
        # Every station stays NaN; the preprocess script treats this the
        # same as "episode skipped" for its own bookkeeping.
        results["obs_count"] = obs_count
        return results

    try:
        ds = xr.open_dataset(path)
    except Exception:
        results["obs_count"] = obs_count
        return results

    file_stations = ds["STATION"].values
    raw_fields_present = {
        GHCNH_SNAPSHOT_VARS[tv]: tv
        for tv in target_variables
        if GHCNH_SNAPSHOT_VARS[tv] in ds.data_vars
    }

    if not raw_fields_present:
        ds.close()
        results["obs_count"] = obs_count
        return results

    field_arrays = {f: ds[f].values for f in raw_fields_present}

    # Build the synoptic timestamp once, for variables whose semantics
    # require strict matching (see _SNAPSHOT_VARS_REQUIRING_SYNOPTIC_TIME
    # — currently just precip). The original ghcnh.py wrote both
    # ``DATE`` (string) and ``time`` (datetime64) columns; we prefer
    # the parsed datetime version where present.
    synoptic_time = pd.Timestamp(f"{date_str} {hour:02d}:00:00")
    if "time" in ds.variables:
        file_times = pd.to_datetime(ds["time"].values)
    else:
        file_times = pd.to_datetime(ds["DATE"].values)

    # Pre-compute per-variable: does this variable require an
    # exactly-synoptic-time row? Avoids the dict lookup inside the
    # inner loop.
    needs_synoptic = {
        raw_field: (tv in _SNAPSHOT_VARS_REQUIRING_SYNOPTIC_TIME)
        for raw_field, tv in raw_fields_present.items()
    }

    # station_id -> target row index in the output arrays.
    station_to_row = {sid: i for i, sid in enumerate(station_ids)}

    for j in range(len(file_stations)):
        sid = file_stations[j]
        if isinstance(sid, (bytes, np.bytes_)):
            sid = sid.decode("utf-8").strip()
        row = station_to_row.get(sid)
        if row is None:
            continue
        is_synoptic_row = (file_times[j] == synoptic_time)
        had_any = False
        for raw_field, tv in raw_fields_present.items():
            val = field_arrays[raw_field][j]
            if np.isnan(val):
                continue
            # Skip off-synoptic-time rows for window-attached variables
            # (e.g. precip-6h: the row's time is the window's RIGHT edge).
            if needs_synoptic[raw_field] and not is_synoptic_row:
                continue
            results[tv][row] = val
            had_any = True
        if had_any:
            obs_count[row] = 1

    ds.close()
    results["obs_count"] = obs_count
    return results


# ---------------------------------------------------------------------------
# Delta-elevation
# ---------------------------------------------------------------------------

def compute_delta_elevation(
    stations: pd.DataFrame,
    static_path: Path,
) -> np.ndarray:
    """Compute station elevation minus ERA5 orography at each station location.

    The ERA5 geopotential (``z``) is divided by g=9.81 to get orography in
    metres, then bilinearly interpolated to each station's lat/lon.
    """
    ds = xr.open_dataset(static_path)
    z = ds["z"].values
    if z.ndim == 3:
        z = z[0]
    orog = z / 9.81

    lats = ds.latitude.values
    lons = ds.longitude.values
    ds.close()

    lons_180 = np.where(lons > 180, lons - 360, lons)
    lon_grid, lat_grid = np.meshgrid(lons_180, lats)
    points = np.column_stack([lon_grid.ravel(), lat_grid.ravel()])
    values = orog.ravel()

    station_coords = np.column_stack([
        stations["longitude"].values,
        stations["latitude"].values,
    ])
    orog_at_stations = griddata(points, values, station_coords, method="linear")

    delta_elev = stations["elevation"].values - orog_at_stations

    logger.info(
        f"Delta-elevation: mean={np.nanmean(delta_elev):.1f}m, "
        f"std={np.nanstd(delta_elev):.1f}m, "
        f"range=[{np.nanmin(delta_elev):.0f}, {np.nanmax(delta_elev):.0f}]m"
    )
    return delta_elev.astype(np.float32)


def lookup_station_mtpi(
    stations: pd.DataFrame,
    mtpi_csv: Path,
    max_missing_fraction: float = 0.5,
) -> np.ndarray:
    """Per-station mTPI aligned to ``stations`` row order, from a lookup CSV.

    Mirrors :func:`compute_delta_elevation`'s contract (returns a float32
    array in ``stations`` order, assigned by the caller as a new column) so
    mTPI slots into the existing per-station feature pipeline.

    ``mtpi_csv`` must have ``station_id`` and ``mtpi`` columns — the output of
    ``projects/dataprocessing/scripts/gee/fetch_station_mtpi.py``, which samples
    GEE's ALOS Global mTPI
    (``CSP/ERGo/1_0/Global/ALOS_mTPI``, Theobald et al. 2015) at each station,
    in metres. Stations absent from the lookup (e.g. ocean/masked pixels) are
    filled with 0.0 — the neutral "neither valley nor ridge" mTPI value — so
    the feature never injects NaN into training. If more than
    ``max_missing_fraction`` of stations are missing, raises instead, since
    that almost always means the ``station_id`` formats don't match.
    """
    mtpi_df = pd.read_csv(mtpi_csv, dtype={"station_id": str})
    missing_cols = {"station_id", "mtpi"} - set(mtpi_df.columns)
    if missing_cols:
        raise ValueError(
            f"mtpi_csv {mtpi_csv} is missing required column(s) "
            f"{sorted(missing_cols)}; found {list(mtpi_df.columns)}."
        )

    lookup = dict(
        zip(mtpi_df["station_id"].astype(str), mtpi_df["mtpi"].astype(float))
    )
    raw = np.array(
        [lookup.get(str(sid), np.nan) for sid in stations["station_id"].values],
        dtype=np.float64,
    )
    n_missing = int(np.isnan(raw).sum())
    frac_missing = n_missing / max(len(raw), 1)
    if frac_missing > max_missing_fraction:
        raise ValueError(
            f"{n_missing}/{len(raw)} stations ({frac_missing:.0%}) have no mTPI "
            f"after the join against {mtpi_csv}. This usually means station_id "
            f"formats don't match between the stations frame and the lookup. "
            f"Refusing to proceed."
        )
    mtpi = np.where(np.isnan(raw), 0.0, raw).astype(np.float32)
    logger.info(
        f"mTPI: matched {len(raw) - n_missing}/{len(raw)} stations "
        f"({n_missing} missing -> 0.0); "
        f"mean={np.mean(mtpi):.1f}m, std={np.std(mtpi):.1f}m, "
        f"range=[{np.min(mtpi):.0f}, {np.max(mtpi):.0f}]m"
    )
    return mtpi


# ---------------------------------------------------------------------------
# Splits
# ---------------------------------------------------------------------------

def random_spatial_split(
    n_stations: int, seed: int = SPLIT_SEED,
    train_fraction: float = TRAIN_STATION_FRACTION,
) -> np.ndarray:
    """Assign each station 'train' or 'test' by random permutation.

    Returns a ``(n_stations,)`` array of strings.
    """
    rng = np.random.RandomState(seed)
    perm = rng.permutation(n_stations)
    n_train = int(n_stations * train_fraction)
    split = np.array(["test"] * n_stations, dtype="U5")
    split[perm[:n_train]] = "train"
    return split


def partition_dates_by_temporal_split(
    dates: list[str],
    train_end: str = TRAIN_END,
    val_end: str = VAL_END,
) -> dict[str, list[str]]:
    """Partition a sorted list of dates into train/val/test by boundary."""
    return {
        "train_dates": [d for d in dates if d <= train_end],
        "val_dates":   [d for d in dates if train_end < d <= val_end],
        "test_dates":  [d for d in dates if d > val_end],
    }


def partition_timestamps_by_temporal_split(
    timestamps: list[str],
    train_end: str = TRAIN_END,
    val_end: str = VAL_END,
) -> dict[str, list[str]]:
    """Partition sorted ``"YYYY-MM-DD-HH"`` strings into train/val/test.

    Same logic as :func:`partition_dates_by_temporal_split` — lexicographic
    comparison with a bare ``"YYYY-MM-DD"`` boundary works because any
    timestamp on the boundary date compares greater than the bare date
    itself (``"2020-12-31-00" > "2020-12-31"``), so *all four* snapshots
    of the boundary date fall into the earlier split. That matches the
    daily semantics exactly: the whole of the train_end day is training,
    the whole of the next day onward is val.
    """
    return {
        "train_timestamps": [t for t in timestamps if t <= train_end],
        "val_timestamps":   [t for t in timestamps if train_end < t <= val_end],
        "test_timestamps":  [t for t in timestamps if t > val_end],
    }