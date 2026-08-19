"""Pure functions and constants shared by the dataset preprocessors.

Used by ``scripts/preprocessing/preprocess_timestamp_global.py`` (the paper's
``dataset_timestamp_global`` builder), ``preprocess_aurora.py`` (the Aurora-lead
datasets), ``scripts/aurora/generate_aurora_forecasts.py`` (the region crop) and
``scripts/data/backfill_station_mtpi.py``. Each caller supplies its own region
definitions and CLI wiring; this module holds no region-specific state.

Contents: the ERA5 variable / pressure-level lists and channel order of the
snapshot tensors, the temporal and spatial split constants, the elevation
sentinel filter, the ERA5 grid crop (with the longitude roll for boxes crossing
0 deg), the per-timestamp ERA5 / GHCNh readers, the ERA5 static-field loader,
the delta-elevation and mTPI per-station features, and the split helpers.
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
# Constants
# ---------------------------------------------------------------------------
# The two variable lists below define the channel order of every dataset tensor
# (see era5_snapshot_channel_names) and therefore of the stored normalisation
# stats and checkpoints. Do not reorder them. (io_utils lists the same
# variables in download order.)

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

# Minimum number of valid snapshot observations for a station to be included
# in the dataset. Default 0 = no filtering (a station with even a handful of
# valid snapshots is kept). The preprocessor exposes this via
# --min-valid-episodes so it can be re-enabled without code changes.
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
    inside the preprocessing pipeline.

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

# GHCNh target variable -> raw field. The single observation at the episode's
# timestamp is taken verbatim; there is no temporal aggregation.
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


# ---------------------------------------------------------------------------
# ERA5 snapshot (one 6-hourly file per episode, no temporal aggregation)
# ---------------------------------------------------------------------------
#
# Each episode's context grid is the raw state of the atmosphere at that one
# timestamp: one channel per surface variable (total_precipitation_6hr is the
# 6 h accumulation ending at the timestamp, as stored in the raw file) plus one
# channel per atmospheric variable x pressure level -- 20 channels in all, in
# the order returned by :func:`era5_snapshot_channel_names`.


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
        era5_dir: Root dir containing ``era5_wb2_quarter_<var>/data/`` subdirs.
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

        # One channel per surface variable; precipitation passes through as
        # the 6 h accumulation ending at this timestamp.
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

    One channel per surface variable plus one channel per atmospheric
    variable × pressure level. The precipitation channel is named
    ``total_precipitation_sum`` (a historical name that the stored
    normalisation stats, ``metadata.json`` files and channel-selection code
    all use); its content is the raw 6 h accumulation.
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
# GHCNh snapshot
# ---------------------------------------------------------------------------

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
    ``"obs_count"`` entry is also returned (int32, ``(n_stations,)``): 0 or
    1 per station, i.e. whether the station has any observation at this
    timestamp.

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
    # — currently just precip). download_ghcnh.py writes a ``time``
    # (datetime64) column; older files carried ``DATE`` (string) instead.
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
    ``scripts/data/fetch_station_mtpi.py``, which samples GEE's ALOS Global mTPI
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
        zip(mtpi_df["station_id"].astype(str), mtpi_df["mtpi"].astype(float), strict=False)
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


def partition_timestamps_by_temporal_split(
    timestamps: list[str],
    train_end: str = TRAIN_END,
    val_end: str = VAL_END,
) -> dict[str, list[str]]:
    """Partition sorted ``"YYYY-MM-DD-HH"`` strings into train/val/test.

    Lexicographic comparison with a bare ``"YYYY-MM-DD"`` boundary works
    because any timestamp on the boundary date compares greater than the
    bare date itself (``"2020-12-31-00" > "2020-12-31"``), so *all four*
    snapshots of the boundary date fall into the earlier split: the whole
    of the train_end day is training, the whole of the next day onward is
    val.
    """
    return {
        "train_timestamps": [t for t in timestamps if t <= train_end],
        "val_timestamps":   [t for t in timestamps if train_end < t <= val_end],
        "test_timestamps":  [t for t in timestamps if t > val_end],
    }
