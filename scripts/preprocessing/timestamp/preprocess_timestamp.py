"""Preprocess raw 6-hourly ERA5 + hourly GHCNh data into a timestamp-cadence dataset.

This is the timestamp-cadence counterpart to :mod:`preprocess_daily`. Whereas
the daily script collapses every (date, 4 × 6h snapshots) into one aggregated
episode per day, this script produces one episode per ``(date, hour)`` pair at
the four timestamps that ERA5 publishes (00, 06, 12, 18 UTC). That gives 4×
more episodes over the same date range.

What's different from daily:

  * Targets are instantaneous observations (``t2m``, ``wind``) rather than
    daily aggregates (``tmax``, ``wind_mean``).
  * ERA5 has one fewer channel: ``2m_temperature_max`` is dropped, since
    there's no max over a single snapshot. Everything else passes through
    unchanged — including ``total_precipitation_sum``, which at snapshot
    cadence is the native 6h precip accumulation ending at the timestamp.
  * No MIN_HOURLY_OBS_PER_DAY threshold on the GHCNh side: at snapshot
    cadence a station either has an observation for this exact hour or
    it doesn't.
  * Station filtering threshold defaults to 0 (no filter). Re-enable via
    ``--min-valid-episodes`` if you need it.

Layout written to disk (single-region, flat):

    dataset_timestamp/
        era5_snapshot/
            2017-01-01-00.npy    # shape (n_dynamic, H, W)
            2017-01-01-06.npy
            2017-01-01-12.npy
            2017-01-01-18.npy
            2017-01-02-00.npy
            ...
        ghcnh_snapshot/
            2017-01-01-00.npz    # fields: t2m, wind, obs_count
            ...
        lats.npy, lons.npy
        static_fields.npy
        stations.csv
        valid_station_indices.npy
        metadata.json            # layout_version="snapshot_v1", cadence="6h"

``--region`` defaults to ``europe`` (matching the paper's setup) but can
be set to ``global`` to process every station without the bounding-box
filter. Multi-region layout (separate US / east_asia / australia / ... grids
sharing one station set) is out of scope for this script; that will be
added later alongside the multi-region refactor.

Usage:
    uv run --group core python projects/tessera_downscaling/scripts/preprocessing/timestamp/preprocess_timestamp.py \\
        --era5-dir .tmp_output/_staging/processed \\
        --ghcnh-dir .tmp_output/_staging/processed/ghcnh/data \\
        --static-path .tmp_output/_staging/processed/era5_static/era5_static_0p25_all.nc \\
        --station-csv .tmp_output/_staging/raw/ghcnh/station_list.csv \\
        --output-dir .tmp_output/dataset_timestamp \\
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

# Shared helpers live two levels up at preprocessing/helpers.py.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from helpers import (  # noqa: E402
    MIN_VALID_EPISODES,
    SNAPSHOT_HOURS,
    SPLIT_SEED,
    TRAIN_END,
    TRAIN_STATION_FRACTION,
    VAL_END,
    GHCNH_SNAPSHOT_VARS,
    aggregate_era5_snapshot,
    aggregate_ghcnh_snapshot,
    compute_delta_elevation,
    compute_grid_crop_indices,
    era5_snapshot_channel_names,
    filter_valid_elevation,
    load_static_fields,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("preprocess_timestamp")


# ---------------------------------------------------------------------------
# Region constants — kept in sync with the daily script
# ---------------------------------------------------------------------------

EUROPE_LAT_RANGE = (35.0, 75.0)
EUROPE_LON_RANGE = (-24.0, 40.0)


# ---------------------------------------------------------------------------
# Station processing (identical logic to preprocess_daily.py)
# ---------------------------------------------------------------------------

def load_and_filter_stations(
    station_csv: Path,
    region: str = "europe",
) -> pd.DataFrame:
    """Load the GHCNh station list, optionally filtered to a region."""
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
    timestamps: list[str],
) -> tuple[np.ndarray, dict]:
    """Assign spatial (per-station) and temporal (per-timestamp) splits."""
    n_stations = len(stations)
    rng = np.random.RandomState(SPLIT_SEED)
    perm = rng.permutation(n_stations)
    n_train = int(n_stations * TRAIN_STATION_FRACTION)

    station_split = np.array(["test"] * n_stations, dtype="U5")
    station_split[perm[:n_train]] = "train"

    temporal_info = {
        "train_end": TRAIN_END,
        "val_end": VAL_END,
        "train_timestamps": [t for t in timestamps if t <= TRAIN_END],
        "val_timestamps":   [t for t in timestamps if TRAIN_END < t <= VAL_END],
        "test_timestamps":  [t for t in timestamps if t > VAL_END],
    }

    n_tr = len(temporal_info["train_timestamps"])
    n_va = len(temporal_info["val_timestamps"])
    n_te = len(temporal_info["test_timestamps"])
    logger.info(
        f"Temporal split: {n_tr} train / {n_va} val / {n_te} test timestamps"
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
        description="Preprocess raw ERA5 + GHCNh into a timestamp-cadence dataset",
    )
    parser.add_argument(
        "--era5-dir", type=Path, required=True,
        help="Root directory containing era5_wb2_quarter_*/data/ subdirectories",
    )
    parser.add_argument(
        "--ghcnh-dir", type=Path, required=True,
        help="Directory containing GHCNh 6-hourly NetCDF files (YYYY-MM-DD-HH.nc)",
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
        "--output-dir", type=Path,
        default=Path(".tmp_output/dataset_timestamp"),
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
        default=list(GHCNH_SNAPSHOT_VARS.keys()),
        choices=list(GHCNH_SNAPSHOT_VARS.keys()),
        help=(
            "Snapshot-cadence target variables to extract. "
            f"Default: all of {list(GHCNH_SNAPSHOT_VARS.keys())}."
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
    parser.add_argument(
        "--min-valid-episodes", type=int, default=MIN_VALID_EPISODES,
        help=(
            "Minimum number of valid snapshot observations for a station to "
            "be kept. Default: 0 (no filter). Use a positive value to drop "
            "stations with very little data, matching the daily script's "
            "--min-valid-days behaviour."
        ),
    )
    parser.add_argument(
        "--hours", type=int, nargs="+", default=list(SNAPSHOT_HOURS),
        choices=list(SNAPSHOT_HOURS),
        help=(
            "Which hours of day to process. Default: all four 6-hourly slots. "
            "Restrict this if you want e.g. only 12:00 UTC snapshots."
        ),
    )
    args = parser.parse_args()

    output_dir = args.output_dir
    era5_snapshot_dir = output_dir / "era5_snapshot"
    ghcnh_snapshot_dir = output_dir / "ghcnh_snapshot"
    era5_snapshot_dir.mkdir(parents=True, exist_ok=True)
    ghcnh_snapshot_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Step 1: Compute grid crop indices from an ERA5 sample file.
    # ------------------------------------------------------------------
    sample_file = (
        args.era5_dir / "era5_wb2_quarter_2m_temperature" / "data"
        / f"{args.start_date}-00.nc"
    )
    if not sample_file.exists():
        # Fall back to the earliest available file — handy when the user
        # starts processing from a date we don't have 00-UTC for.
        candidates = sorted(
            (args.era5_dir / "era5_wb2_quarter_2m_temperature" / "data").glob("*.nc")
        )
        if not candidates:
            raise FileNotFoundError(
                f"No ERA5 2m_temperature files under {args.era5_dir}"
            )
        sample_file = candidates[0]
        logger.info(f"Using {sample_file.name} as grid-definition sample file")

    sample_ds = xr.open_dataset(sample_file)
    lats = sample_ds.latitude.values
    lons = sample_ds.longitude.values
    sample_ds.close()

    lat_range = EUROPE_LAT_RANGE if args.region == "europe" else None
    lon_range = EUROPE_LON_RANGE if args.region == "europe" else None

    lat_indices, lon_indices, roll_amount, lats_crop, lons_crop = (
        compute_grid_crop_indices(lats, lons, lat_range=lat_range, lon_range=lon_range)
    )
    logger.info(
        f"Grid crop ({args.region}): {len(lats_crop)} lats × {len(lons_crop)} lons "
        f"(lat {lats_crop.min():.2f}–{lats_crop.max():.2f}, "
        f"lon {lons_crop.min():.2f}–{lons_crop.max():.2f})"
    )

    # ------------------------------------------------------------------
    # Step 2: Load and crop static fields.
    # ------------------------------------------------------------------
    static_array, static_var_names = load_static_fields(
        args.static_path, lat_indices, lon_indices, roll_amount
    )
    logger.info(
        f"Static fields: {len(static_var_names)} variables, shape {static_array.shape}"
    )
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
    # Step 5: Enumerate all (date, hour) episodes, process each.
    # ------------------------------------------------------------------
    start = datetime.strptime(args.start_date, "%Y-%m-%d")
    end = datetime.strptime(args.end_date, "%Y-%m-%d")

    episodes: list[tuple[str, int]] = []
    current = start
    while current <= end:
        date_str = current.strftime("%Y-%m-%d")
        for h in args.hours:
            episodes.append((date_str, h))
        current += timedelta(days=1)
    logger.info(
        f"Processing {len(episodes)} episodes across "
        f"{(end - start).days + 1} days × {len(args.hours)} hours/day"
    )

    channel_names = era5_snapshot_channel_names()
    logger.info(
        f"ERA5 snapshot dynamic channels ({len(channel_names)}): {channel_names}"
    )

    # Per-station per-variable valid-episode counts. Used for the optional
    # min-valid-episodes filter AND for saving diagnostic metadata columns
    # on stations.csv, whether or not the filter is active.
    valid_episode_counts: dict[str, np.ndarray] = {
        v: np.zeros(len(station_ids), dtype=np.int32)
        for v in args.target_variables
    }
    episodes_processed = 0
    episodes_skipped = 0

    for date_str, hour in episodes:
        era5_path = era5_snapshot_dir / f"{date_str}-{hour:02d}.npy"
        ghcnh_path = ghcnh_snapshot_dir / f"{date_str}-{hour:02d}.npz"

        # Fast path: already fully processed and contains all requested vars.
        if era5_path.exists() and ghcnh_path.exists():
            ghcnh_data = np.load(ghcnh_path)
            missing_vars = [
                v for v in args.target_variables if v not in ghcnh_data.files
            ]
            if not missing_vars:
                for v in args.target_variables:
                    valid_episode_counts[v] += ~np.isnan(ghcnh_data[v])
                episodes_processed += 1
                continue
            logger.info(
                f"{date_str}-{hour:02d}: re-processing to add missing variables: "
                f"{missing_vars}"
            )

        # Aggregate ERA5 for this timestamp.
        era5_snapshot = aggregate_era5_snapshot(
            args.era5_dir, date_str, hour, lat_indices, lon_indices, roll_amount,
        )
        if era5_snapshot is None:
            episodes_skipped += 1
            continue

        # Aggregate GHCNh for this timestamp.
        ghcnh_results = aggregate_ghcnh_snapshot(
            args.ghcnh_dir, date_str, hour, station_ids,
            target_variables=args.target_variables,
        )

        np.save(era5_path, era5_snapshot)
        np.savez_compressed(ghcnh_path, **ghcnh_results)

        for v in args.target_variables:
            valid_episode_counts[v] += ~np.isnan(ghcnh_results[v])
        episodes_processed += 1

        if episodes_processed % 500 == 0:
            logger.info(
                f"  Processed {episodes_processed} episodes, "
                f"skipped {episodes_skipped}"
            )

    logger.info(
        f"Finished: {episodes_processed} episodes processed, "
        f"{episodes_skipped} skipped"
    )

    # ------------------------------------------------------------------
    # Step 6: Station filter (disabled by default).
    # A station is kept if EITHER requested variable has
    # >= min_valid_episodes observations.
    # ------------------------------------------------------------------
    for v, counts in valid_episode_counts.items():
        stations[f"n_valid_episodes_{v}"] = counts

    if args.min_valid_episodes > 0:
        valid_mask = np.zeros(len(station_ids), dtype=bool)
        for counts in valid_episode_counts.values():
            valid_mask |= (counts >= args.min_valid_episodes)
        logger.info(
            f"Station filter (min_valid_episodes={args.min_valid_episodes}): "
            f"{int(valid_mask.sum())}/{len(stations)} stations kept"
        )
    else:
        valid_mask = np.ones(len(station_ids), dtype=bool)
        logger.info(
            f"Station filter disabled (min_valid_episodes=0); "
            f"keeping all {len(stations)} stations"
        )

    stations_filtered = stations[valid_mask].reset_index(drop=True)
    valid_indices = np.where(valid_mask)[0]
    np.save(output_dir / "valid_station_indices.npy", valid_indices)

    # ------------------------------------------------------------------
    # Step 7: Spatial + temporal splits.
    # ------------------------------------------------------------------
    valid_timestamps = sorted([
        f"{d}-{h:02d}" for (d, h) in episodes
        if (era5_snapshot_dir / f"{d}-{h:02d}.npy").exists()
    ])
    station_split, temporal_info = assign_splits(stations_filtered, valid_timestamps)
    stations_filtered["spatial_split"] = station_split

    # ------------------------------------------------------------------
    # Step 8: Write metadata.
    # ------------------------------------------------------------------
    stations_filtered.to_csv(output_dir / "stations.csv", index=False)
    np.save(output_dir / "lats.npy", lats_crop)
    np.save(output_dir / "lons.npy", lons_crop)

    metadata = {
        "layout_version": "snapshot_v1",
        "cadence": "6h",
        "hours_per_day": list(args.hours),
        "era5_dynamic_channels": channel_names,
        "static_channels": static_var_names,
        "n_dynamic_channels": len(channel_names),
        "n_static_channels": len(static_var_names),
        "n_total_channels": len(channel_names) + len(static_var_names),
        "pressure_levels": [500, 700, 850],
        "grid_shape": [len(lats_crop), len(lons_crop)],
        "lat_range": [float(lats_crop.min()), float(lats_crop.max())],
        "lon_range": [float(lons_crop.min()), float(lons_crop.max())],
        "n_stations_total": len(stations),
        "n_stations_filtered": int(len(stations_filtered)),
        "n_episodes": episodes_processed,
        "n_episodes_skipped": episodes_skipped,
        "date_range": [args.start_date, args.end_date],
        # The list of episode identifiers — sorted YYYY-MM-DD-HH strings.
        # Dataset classes key off this for temporal splitting; see the
        # `episodes_for_split` helper. Kept under the name `valid_dates`
        # too (aliased copy) so the daily-cadence DailyDownscalingDataset
        # CAN be pointed at a snapshot dataset root and still find a
        # compatible list to index into — useful for ad-hoc notebook
        # exploration. The SnapshotDownscalingDataset always reads the
        # canonical `valid_timestamps` key.
        "valid_timestamps": valid_timestamps,
        "valid_dates": valid_timestamps,
        "temporal_split": {
            "train_end": TRAIN_END,
            "val_end": VAL_END,
            "n_train_timestamps": len(temporal_info["train_timestamps"]),
            "n_val_timestamps": len(temporal_info["val_timestamps"]),
            "n_test_timestamps": len(temporal_info["test_timestamps"]),
        },
        "spatial_split": {
            "n_train_stations": int(np.sum(station_split == "train")),
            "n_test_stations": int(np.sum(station_split == "test")),
            "seed": SPLIT_SEED,
            "train_fraction": TRAIN_STATION_FRACTION,
        },
        "min_valid_episodes": int(args.min_valid_episodes),
        "elevation_normalisation": "raw_metres",
        "region": args.region,
    }
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Dataset saved to {output_dir}")
    logger.info(
        f"  {metadata['n_total_channels']} total channels "
        f"({metadata['n_dynamic_channels']} dynamic + "
        f"{metadata['n_static_channels']} static)"
    )
    logger.info(
        f"  {metadata['n_stations_filtered']} stations, "
        f"{metadata['n_episodes']} episodes"
    )
    logger.info("Done!")


if __name__ == "__main__":
    main()