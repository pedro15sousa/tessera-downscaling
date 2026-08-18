"""Preprocess raw 6-hourly ERA5 + GHCNh into a multi-region snapshot dataset.

Multi-region counterpart to :mod:`preprocess_timestamp`: produces episodes
at the four 6-hourly timestamps (00, 06, 12, 18 UTC) rather than one
aggregated daily episode, across all requested regions:

    dataset_timestamp_global/
        metadata.json                              <-- layout_version="multi_region_snapshot_v1"
        stations.csv                               <-- global, with `region` column
        valid_station_indices.npy
        ghcnh_snapshot/<date>-<hh>.npz             <-- shared GHCNh files (one per timestamp)
        regions/
            europe/
                lats.npy, lons.npy
                static_fields.npy
                normalisation_stats.npz
                normalisation_stats_no_static.npz
                era5_snapshot/<date>-<hh>.npy      <-- region-cropped ERA5 per timestamp
                region_metadata.json
            us/
                ...
            east_asia/
                ...
            australia/
                ...
            southern_africa/
                ...

Design notes:

    - GHCNh observations are read once per timestamp for the union of
      stations across all regions and saved flat, keyed by station index.
      No per-region duplication.
    - ERA5 is cropped per region (different geographic bounds), so one
      ERA5 file is written per (region, timestamp) pair.
    - Temporal split is global — same TRAIN_END / VAL_END boundaries
      from :mod:`helpers` across all regions.
    - Spatial 85/15 split is per-region with the same seed — each region
      has ~85% of its stations in the train split, ~15% in test.
    - Regions are non-overlapping by construction. Stations outside all
      region boxes are dropped.

Layout designed so that :class:`SnapshotDownscalingDataset` can read a
single region from this tree directly (via a ``region`` kwarg on the
class), avoiding dataset duplication between the single-region and
multi-region pipelines.

Usage (from repo root):
    uv run --group core python projects/tessera_downscaling/scripts/preprocessing/timestamp/preprocess_timestamp_global.py \\
        --era5-dir .tmp_output/_staging/processed \\
        --ghcnh-dir .tmp_output/_staging/processed/ghcnh/data \\
        --static-path .tmp_output/_staging/processed/era5_static/era5_static_0p25_all.nc \\
        --station-csv .tmp_output/_staging/raw/ghcnh/station_list.csv \\
        --output-dir .tmp_output/dataset_timestamp_global \\
        --start-date 2017-01-01 \\
        --end-date 2023-01-10 \\
        --regions europe us east_asia australia southern_africa
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
    lookup_station_mtpi,
    compute_grid_crop_indices,
    era5_snapshot_channel_names,
    filter_valid_elevation,
    load_static_fields,
    partition_timestamps_by_temporal_split,
    random_spatial_split,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("preprocess_timestamp_global")


# ---------------------------------------------------------------------------
# Region definitions — IDENTICAL to preprocess_daily_global.py to avoid any
# drift between the two pipelines. Stations that fall inside `europe` in the
# daily pipeline will also fall inside `europe` here.
# ---------------------------------------------------------------------------

REGIONS: dict[str, tuple[float, float, float, float]] = {
    "europe":          (35.0,  75.0,  -24.0,  40.0),
    "us":              (24.0,  50.0, -125.0, -66.0),
    "east_asia":       (20.0,  46.0,  100.0, 146.0),
    "australia":      (-44.0, -10.0,  112.0, 154.0),
    "southern_africa":(-35.0, -15.0,   15.0,  35.0),
}


# ---------------------------------------------------------------------------
# Station assignment — identical in behaviour to the daily-global version.
# Duplicated here (rather than imported) because preprocess_daily_global.py
# is a script not a module; importing it would run its argparse. Keep both
# in sync if you ever touch either.
# ---------------------------------------------------------------------------

def assign_stations_to_regions(
    stations: pd.DataFrame,
    region_bboxes: dict[str, tuple[float, float, float, float]],
) -> pd.DataFrame:
    """Assign each station to exactly one region by bbox membership."""
    lats = stations["latitude"].values
    lons = stations["longitude"].values

    assignments = np.array(["__none__"] * len(stations), dtype=object)
    multi_hits = 0

    for name, (lat_min, lat_max, lon_min, lon_max) in region_bboxes.items():
        in_box = (
            (lats >= lat_min) & (lats <= lat_max)
            & (lons >= lon_min) & (lons <= lon_max)
        )
        newly_assigned = in_box & (assignments == "__none__")
        collision = in_box & (assignments != "__none__")
        if collision.any():
            multi_hits += int(collision.sum())
        assignments[newly_assigned] = name

    if multi_hits:
        logger.warning(
            f"{multi_hits} stations fell inside multiple region bboxes; "
            f"kept the first-matching region. Check that bboxes are non-overlapping."
        )

    kept_mask = assignments != "__none__"
    out = stations[kept_mask].copy().reset_index(drop=True)
    out["region"] = assignments[kept_mask]

    counts = out["region"].value_counts().to_dict()
    logger.info(
        f"Region assignment: {len(out)}/{len(stations)} stations kept. "
        f"Per-region counts: {counts}"
    )
    return out


# ---------------------------------------------------------------------------
# Per-region ERA5 grid setup
# ---------------------------------------------------------------------------

def build_region_grids(
    region_names: list[str],
    era5_dir: Path,
    static_path: Path,
    region_bboxes: dict[str, tuple[float, float, float, float]],
) -> dict[str, dict]:
    """Compute grid crop indices and static fields for each region."""
    sample_path = None
    for candidate in ["2m_temperature", "10m_u_component_of_wind"]:
        sample_dir = era5_dir / f"era5_wb2_quarter_{candidate}" / "data"
        if sample_dir.exists():
            files = sorted(sample_dir.glob("*.nc"))
            if files:
                sample_path = files[0]
                break
    if sample_path is None:
        raise FileNotFoundError(f"Could not find a sample ERA5 file under {era5_dir}")

    ds = xr.open_dataset(sample_path)
    lats = ds.latitude.values
    lons = ds.longitude.values
    ds.close()

    out: dict[str, dict] = {}
    for name in region_names:
        if name not in region_bboxes:
            raise ValueError(f"Unknown region '{name}'. Defined: {list(region_bboxes)}")
        lat_min, lat_max, lon_min, lon_max = region_bboxes[name]
        lat_idx, lon_idx, roll, lats_c, lons_c = compute_grid_crop_indices(
            lats, lons,
            lat_range=(lat_min, lat_max),
            lon_range=(lon_min, lon_max),
        )
        static_array, static_var_names = load_static_fields(
            static_path, lat_idx, lon_idx, roll,
        )
        logger.info(
            f"Region '{name}': grid {len(lats_c)}×{len(lons_c)} cells "
            f"(lat {lats_c.min():.2f}–{lats_c.max():.2f}, "
            f"lon {lons_c.min():.2f}–{lons_c.max():.2f}); "
            f"{len(static_var_names)} static channels"
        )
        out[name] = {
            "lat_indices": lat_idx,
            "lon_indices": lon_idx,
            "roll_amount": roll,
            "lats_crop": lats_c,
            "lons_crop": lons_c,
            "static_array": static_array,
            "static_var_names": static_var_names,
        }
    return out


def write_region_scaffolding(
    output_dir: Path,
    region_name: str,
    region_grid: dict,
) -> Path:
    """Create <output_dir>/regions/<n>/ and write the static per-region files."""
    region_dir = output_dir / "regions" / region_name
    (region_dir / "era5_snapshot").mkdir(parents=True, exist_ok=True)

    np.save(region_dir / "lats.npy", region_grid["lats_crop"])
    np.save(region_dir / "lons.npy", region_grid["lons_crop"])
    np.save(region_dir / "static_fields.npy", region_grid["static_array"])

    region_meta = {
        "region_name": region_name,
        "cadence": "6h",
        "grid_shape": [
            int(len(region_grid["lats_crop"])),
            int(len(region_grid["lons_crop"])),
        ],
        "lat_range": [
            float(region_grid["lats_crop"].min()),
            float(region_grid["lats_crop"].max()),
        ],
        "lon_range": [
            float(region_grid["lons_crop"].min()),
            float(region_grid["lons_crop"].max()),
        ],
        "static_channels": region_grid["static_var_names"],
        "n_static_channels": len(region_grid["static_var_names"]),
    }
    with open(region_dir / "region_metadata.json", "w") as f:
        json.dump(region_meta, f, indent=2)

    return region_dir


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Preprocess raw ERA5 + GHCNh into a multi-region "
                    "snapshot (6-hourly) training dataset",
    )
    parser.add_argument("--era5-dir", type=Path, required=True,
                        help="Root containing era5_wb2_quarter_*/data/ subdirectories")
    parser.add_argument("--ghcnh-dir", type=Path, required=True,
                        help="Directory containing GHCNh 6-hourly NetCDF files")
    parser.add_argument("--static-path", type=Path, required=True,
                        help="Path to the ERA5 static fields NetCDF file")
    parser.add_argument("--mtpi-csv", type=Path, default=None,
                        help="Optional CSV (station_id, mtpi) of per-station "
                             "ALOS mTPI from "
                             "projects/dataprocessing/scripts/gee/"
                             "fetch_station_mtpi.py. "
                             "When given, an `mtpi` column is added to "
                             "stations.csv and training auto-enables the "
                             "3-feature (elevation, delta_elevation, mTPI) "
                             "per-station vector of Vaughan et al. (2022).")
    parser.add_argument("--station-csv", type=Path, required=True,
                        help="Path to the global GHCNh station list CSV")
    parser.add_argument("--output-dir", type=Path,
                        default=Path(".tmp_output/dataset_timestamp_global"),
                        help="Output directory for the multi-region snapshot dataset")
    parser.add_argument("--start-date", type=str, default="2017-01-01")
    parser.add_argument("--end-date", type=str, default="2023-01-10")
    parser.add_argument(
        "--target-variables", type=str, nargs="+",
        default=list(GHCNH_SNAPSHOT_VARS.keys()),
        choices=list(GHCNH_SNAPSHOT_VARS.keys()),
        help="Snapshot target variables to extract. Default: all supported.",
    )
    parser.add_argument(
        "--regions", type=str, nargs="+",
        default=list(REGIONS.keys()),
        choices=list(REGIONS.keys()),
        help="Which regions to process. Default: all defined regions.",
    )
    parser.add_argument(
        "--hours", type=int, nargs="+", default=list(SNAPSHOT_HOURS),
        choices=list(SNAPSHOT_HOURS),
        help="Which hours of day to process. Default: all four 6-hourly slots.",
    )
    parser.add_argument(
        "--min-valid-episodes", type=int, default=MIN_VALID_EPISODES,
        help="Minimum valid snapshot observations per station for inclusion. "
             "Default 0 (no filter). Matches the single-region snapshot script.",
    )
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "ghcnh_snapshot").mkdir(exist_ok=True)

    # ------------------------------------------------------------------
    # Step 1: Build per-region grid crops and static fields.
    # ------------------------------------------------------------------
    logger.info(f"Processing regions: {args.regions}")
    logger.info(f"Processing hours: {args.hours}")
    region_bboxes = {k: REGIONS[k] for k in args.regions}
    region_grids = build_region_grids(
        args.regions, args.era5_dir, args.static_path, region_bboxes,
    )
    for name, grid in region_grids.items():
        write_region_scaffolding(output_dir, name, grid)

    # ------------------------------------------------------------------
    # Step 2: Load all stations, assign to regions, drop unassigned.
    # ------------------------------------------------------------------
    stations_all = pd.read_csv(args.station_csv)
    stations_all = stations_all.rename(columns={
        "GHCN_ID": "station_id",
        "LATITUDE": "latitude",
        "LONGITUDE": "longitude",
        "ELEVATION": "elevation",
    })
    # Drops missing-elevation rows AND sentinel/out-of-range values that
    # GHCN encodes as -999.9 or 9999. See helpers.filter_valid_elevation.
    stations_all = filter_valid_elevation(stations_all, logger=logger)
    stations_all = stations_all[[
        "station_id", "latitude", "longitude", "elevation",
    ]]
    stations = assign_stations_to_regions(stations_all, region_bboxes)
    station_ids = stations["station_id"].values

    # ------------------------------------------------------------------
    # Step 3: Delta-elevation.
    # ------------------------------------------------------------------
    stations["delta_elevation"] = compute_delta_elevation(stations, args.static_path)
    if args.mtpi_csv is not None:
        stations["mtpi"] = lookup_station_mtpi(stations, args.mtpi_csv)

    # ------------------------------------------------------------------
    # Step 4: Enumerate all (date, hour) episodes.
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
        f"Processing {len(episodes)} episodes = "
        f"{(end - start).days + 1} days × {len(args.hours)} hours/day, "
        f"across {len(args.regions)} regions"
    )

    channel_names = era5_snapshot_channel_names()
    logger.info(
        f"ERA5 snapshot dynamic channels ({len(channel_names)}): {channel_names}"
    )

    # ------------------------------------------------------------------
    # Step 5: Per-episode loop.
    #
    # For each (date, hour):
    #   * Process each region's ERA5 separately (different crops). If any
    #     region's file is missing upstream, skip this episode globally —
    #     we don't want partial episodes where some regions have data and
    #     others don't, since the dataset class can't represent that.
    #   * Aggregate GHCNh once across the union of all regionally-assigned
    #     stations. Shared at the top level, keyed by station index.
    # ------------------------------------------------------------------
    valid_episode_counts: dict[str, np.ndarray] = {
        v: np.zeros(len(station_ids), dtype=np.int32)
        for v in args.target_variables
    }

    episodes_processed = 0
    episodes_skipped = 0

    for date_str, hour in episodes:
        ts = f"{date_str}-{hour:02d}"
        ghcnh_path = output_dir / "ghcnh_snapshot" / f"{ts}.npz"
        missing_vars: list[str] = []

        # Fast path: skip if fully already processed.
        all_era5_exist = all(
            (output_dir / "regions" / name / "era5_snapshot" / f"{ts}.npy").exists()
            for name in args.regions
        )
        if ghcnh_path.exists() and all_era5_exist:
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
                f"{ts}: re-processing GHCNh to add missing variables: {missing_vars}"
            )

        # ERA5 per region.
        any_era5_missing = False
        for name in args.regions:
            era5_path = output_dir / "regions" / name / "era5_snapshot" / f"{ts}.npy"
            if era5_path.exists():
                continue
            grid = region_grids[name]
            era5_snap = aggregate_era5_snapshot(
                args.era5_dir, date_str, hour,
                grid["lat_indices"], grid["lon_indices"], grid["roll_amount"],
            )
            if era5_snap is None:
                any_era5_missing = True
                break
            np.save(era5_path, era5_snap)

        if any_era5_missing:
            episodes_skipped += 1
            continue

        # GHCNh aggregated once for all regional stations.
        if not ghcnh_path.exists() or missing_vars:
            ghcnh_results = aggregate_ghcnh_snapshot(
                args.ghcnh_dir, date_str, hour, station_ids,
                target_variables=args.target_variables,
            )
            np.savez_compressed(ghcnh_path, **ghcnh_results)
        else:
            ghcnh_results = dict(np.load(ghcnh_path))

        for v in args.target_variables:
            valid_episode_counts[v] += ~np.isnan(ghcnh_results[v])
        episodes_processed += 1

        if episodes_processed % 500 == 0:
            logger.info(
                f"  Processed {episodes_processed} episodes, skipped {episodes_skipped}"
            )

    logger.info(
        f"Finished episode loop: {episodes_processed} processed, "
        f"{episodes_skipped} skipped"
    )

    # ------------------------------------------------------------------
    # Step 6: Station filter (disabled by default at snapshot cadence).
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
            f"Station filter disabled (min_valid_episodes=0); keeping all "
            f"{len(stations)} stations"
        )

    stations_filtered = stations[valid_mask].reset_index(drop=True)
    valid_indices = np.where(valid_mask)[0]
    np.save(output_dir / "valid_station_indices.npy", valid_indices)

    # ------------------------------------------------------------------
    # Step 7: Per-region 85/15 spatial split (same seed across regions).
    # ------------------------------------------------------------------
    stations_filtered["spatial_split"] = ""
    for region_name in args.regions:
        region_mask = stations_filtered["region"].values == region_name
        n_in_region = int(region_mask.sum())
        if n_in_region == 0:
            continue
        split = random_spatial_split(n_in_region, seed=SPLIT_SEED)
        region_indices = np.where(region_mask)[0]
        stations_filtered.loc[region_indices, "spatial_split"] = split

    split_counts = (
        stations_filtered.groupby(["region", "spatial_split"]).size().to_dict()
    )
    logger.info(f"Per-region spatial split counts: {split_counts}")

    # ------------------------------------------------------------------
    # Step 8: Per-region normalisation stats from training-split timestamps.
    #
    # Uses the same helper as the single-region snapshot script (via the
    # dataset class); here we pre-compute and cache so datasets don't have
    # to do it lazily. Same normalisation style as daily-global: two
    # stats files per region, one including static fields and one not.
    # ------------------------------------------------------------------
    valid_timestamps = sorted([
        f"{d}-{h:02d}" for (d, h) in episodes
        if all(
            (output_dir / "regions" / name / "era5_snapshot" / f"{d}-{h:02d}.npy").exists()
            for name in args.regions
        )
    ])
    temporal_split = partition_timestamps_by_temporal_split(
        valid_timestamps, train_end=TRAIN_END, val_end=VAL_END,
    )

    for region_name in args.regions:
        _compute_region_normalisation_stats(
            output_dir=output_dir,
            region_name=region_name,
            region_grid=region_grids[region_name],
            train_timestamps=temporal_split["train_timestamps"],
        )

    # ------------------------------------------------------------------
    # Step 8b: Global normalisation stats computed across all regions'
    # training timestamps. Used by transfer / joint-source experiments
    # where a single normalisation has to fit all training regions.
    # Computed from TRAINING timestamps only — no test leakage.
    # ------------------------------------------------------------------
    _compute_global_normalisation_stats(
        output_dir=output_dir,
        region_grids=region_grids,
        region_names=args.regions,
        train_timestamps=temporal_split["train_timestamps"],
    )

    # ------------------------------------------------------------------
    # Step 9: Write top-level manifest.
    # ------------------------------------------------------------------
    stations_filtered.to_csv(output_dir / "stations.csv", index=False)

    metadata = {
        "layout_version": "multi_region_snapshot_v1",
        "cadence": "6h",
        "hours_per_day": list(args.hours),
        "era5_dynamic_channels": channel_names,
        "n_dynamic_channels": len(channel_names),
        "pressure_levels": [500, 700, 850],
        "regions": {
            name: {
                "bbox_lat_lon": list(region_bboxes[name]),
                "grid_shape": [
                    int(len(region_grids[name]["lats_crop"])),
                    int(len(region_grids[name]["lons_crop"])),
                ],
                "n_static_channels": len(region_grids[name]["static_var_names"]),
            }
            for name in args.regions
        },
        "n_stations_total": int(len(stations_all)),
        "n_stations_regionally_assigned": int(len(stations)),
        "n_stations_filtered": int(len(stations_filtered)),
        "n_episodes_processed": episodes_processed,
        "n_episodes_skipped": episodes_skipped,
        "date_range": [args.start_date, args.end_date],
        # Episode identifiers (sorted YYYY-MM-DD-HH). Dataset classes use
        # this to build their split lists. `valid_dates` alias kept for
        # compatibility with the single-region snapshot layout, which
        # lets a multi-region dataset root be ad-hoc probed as if flat.
        "valid_timestamps": valid_timestamps,
        "valid_dates": valid_timestamps,
        "temporal_split": {
            "train_end": TRAIN_END,
            "val_end": VAL_END,
            "n_train_timestamps": len(temporal_split["train_timestamps"]),
            "n_val_timestamps": len(temporal_split["val_timestamps"]),
            "n_test_timestamps": len(temporal_split["test_timestamps"]),
        },
        "spatial_split": {
            "train_fraction": TRAIN_STATION_FRACTION,
            "seed": SPLIT_SEED,
            "per_region_counts": {str(k): int(v) for k, v in split_counts.items()},
        },
        "min_valid_episodes": int(args.min_valid_episodes),
        "elevation_normalisation": "raw_metres",
    }
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Dataset saved to {output_dir}")
    logger.info(
        f"  {metadata['n_dynamic_channels']} dynamic channels × "
        f"{len(args.regions)} regions × {len(valid_timestamps)} timestamps"
    )
    logger.info(f"  {metadata['n_stations_filtered']} stations")
    logger.info("Done!")


def _compute_region_normalisation_stats(
    output_dir: Path,
    region_name: str,
    region_grid: dict,
    train_timestamps: list[str],
) -> None:
    """Compute per-region z-score stats from training-split timestamps.

    Writes two files to the region dir:
        * ``normalisation_stats.npz``           (dynamic + static + lat/lon)
        * ``normalisation_stats_no_static.npz`` (dynamic + lat/lon only)

    Note: these are the *per-region* stats. When mixing regions at
    train time the dataset class may recompute joint stats instead; this
    function just writes the per-region baseline, which is correct for
    single-region training against a snapshot-global dataset.
    """
    region_dir = output_dir / "regions" / region_name
    era5_snapshot_dir = region_dir / "era5_snapshot"
    lats = region_grid["lats_crop"]
    lons = region_grid["lons_crop"]
    static_array = region_grid["static_array"]
    n_static = static_array.shape[0]

    lat_grid = lats[:, None] * np.ones((1, len(lons)), dtype=np.float32)
    lon_grid = np.ones((len(lats), 1), dtype=np.float32) * lons[None, :]

    def _accumulate(include_static: bool) -> tuple[np.ndarray, np.ndarray]:
        first_path = era5_snapshot_dir / f"{train_timestamps[0]}.npy"
        if not first_path.exists():
            raise FileNotFoundError(
                f"No ERA5 snapshot file at {first_path}; cannot compute "
                f"normalisation stats for region '{region_name}'."
            )
        first = np.load(first_path)
        n_dynamic = first.shape[0]
        n_channels = n_dynamic + (n_static if include_static else 0) + 2

        running_sum = np.zeros(n_channels, dtype=np.float64)
        running_sq_sum = np.zeros(n_channels, dtype=np.float64)
        n_pixels = 0

        for ts in train_timestamps:
            p = era5_snapshot_dir / f"{ts}.npy"
            if not p.exists():
                continue
            era5 = np.load(p)
            parts = [era5]
            if include_static:
                parts.append(static_array)
            parts.extend([lat_grid[None, :, :], lon_grid[None, :, :]])
            combined = np.concatenate(parts, axis=0)
            running_sum += combined.reshape(n_channels, -1).sum(axis=1)
            running_sq_sum += (combined.reshape(n_channels, -1) ** 2).sum(axis=1)
            n_pixels += combined.shape[1] * combined.shape[2]

        mean = running_sum / n_pixels
        std = np.sqrt(running_sq_sum / n_pixels - mean ** 2)
        std = np.maximum(std, 1e-8)
        return mean.astype(np.float32), std.astype(np.float32)

    mean_full, std_full = _accumulate(include_static=True)
    np.savez(
        region_dir / "normalisation_stats.npz",
        era5_mean=mean_full, era5_std=std_full,
    )

    mean_nostatic, std_nostatic = _accumulate(include_static=False)
    np.savez(
        region_dir / "normalisation_stats_no_static.npz",
        era5_mean=mean_nostatic, era5_std=std_nostatic,
    )

    logger.info(
        f"Region '{region_name}': wrote normalisation stats "
        f"({len(mean_full)} channels w/ static, "
        f"{len(mean_nostatic)} w/o static)"
    )


def _compute_global_normalisation_stats(
    output_dir: Path,
    region_grids: dict,
    region_names: list[str],
    train_timestamps: list[str],
) -> None:
    """Compute global (cross-region) normalisation stats.

    Accumulates per-channel sum / sum-of-squares across EVERY region's
    training timestamps, then divides by total pixel count. The result
    is stats that describe the distribution of ERA5 + static + lat/lon
    values jointly across all training regions.

    These stats are used by transfer / joint-source experiments where
    a single normalisation has to fit all training regions (e.g.
    train on US, test on EU). Per-region stats would mismatch scale
    between train and test in that case.

    Critically: ``train_timestamps`` excludes val/test windows, so
    there is no test-set leakage — only train-window data drives
    the normalisation.

    Writes two files to the top-level ``output_dir``:
        * ``normalisation_stats_global.npz``           (dynamic + static + lat/lon)
        * ``normalisation_stats_no_static_global.npz`` (dynamic + lat/lon only)

    Static channels: if region grids have different ``n_static``
    across regions (unlikely but possible if static masks differ),
    stats are computed using max-common-n_static; we log a warning
    and truncate if mismatched. In practice every region uses the
    same 13 static channels so this is a no-op.
    """
    logger.info(
        "Computing GLOBAL (cross-region) normalisation stats from "
        f"{len(train_timestamps)} training timestamps × {len(region_names)} regions"
    )

    # Check n_static consistency across regions.
    n_statics = {name: region_grids[name]["static_array"].shape[0] for name in region_names}
    if len(set(n_statics.values())) > 1:
        logger.warning(
            f"Mixed n_static across regions {n_statics}; truncating to min."
        )
    n_static_common = min(n_statics.values())

    def _accumulate_global(include_static: bool) -> tuple[np.ndarray, np.ndarray]:
        """Sweep across all regions × timestamps, accumulate sums."""
        running_sum: np.ndarray | None = None
        running_sq_sum: np.ndarray | None = None
        n_pixels_total = 0
        n_files_read = 0
        n_files_missing = 0

        for region_name in region_names:
            region_dir = output_dir / "regions" / region_name
            era5_snapshot_dir = region_dir / "era5_snapshot"
            grid = region_grids[region_name]
            lats = grid["lats_crop"]
            lons = grid["lons_crop"]
            static_array = grid["static_array"]
            lat_grid = lats[:, None] * np.ones((1, len(lons)), dtype=np.float32)
            lon_grid = np.ones((len(lats), 1), dtype=np.float32) * lons[None, :]

            for ts in train_timestamps:
                p = era5_snapshot_dir / f"{ts}.npy"
                if not p.exists():
                    n_files_missing += 1
                    continue
                era5 = np.load(p)
                parts = [era5]
                if include_static:
                    # Truncate static to the common n across regions.
                    parts.append(static_array[:n_static_common])
                parts.extend([lat_grid[None, :, :], lon_grid[None, :, :]])
                combined = np.concatenate(parts, axis=0)
                n_channels = combined.shape[0]

                if running_sum is None:
                    running_sum = np.zeros(n_channels, dtype=np.float64)
                    running_sq_sum = np.zeros(n_channels, dtype=np.float64)

                running_sum += combined.reshape(n_channels, -1).sum(axis=1)
                running_sq_sum += (combined.reshape(n_channels, -1) ** 2).sum(axis=1)
                n_pixels_total += combined.shape[1] * combined.shape[2]
                n_files_read += 1

        if running_sum is None:
            raise FileNotFoundError(
                "No ERA5 snapshot files found across any region; cannot "
                "compute global normalisation stats."
            )

        logger.info(
            f"  {'With' if include_static else 'Without'} static: "
            f"{n_files_read} snapshot files read, {n_files_missing} missing, "
            f"{n_pixels_total} total pixels across channels"
        )

        mean = running_sum / n_pixels_total
        std = np.sqrt(running_sq_sum / n_pixels_total - mean ** 2)
        std = np.maximum(std, 1e-8)
        return mean.astype(np.float32), std.astype(np.float32)

    mean_full, std_full = _accumulate_global(include_static=True)
    np.savez(
        output_dir / "normalisation_stats_global.npz",
        era5_mean=mean_full, era5_std=std_full,
    )

    mean_nostatic, std_nostatic = _accumulate_global(include_static=False)
    np.savez(
        output_dir / "normalisation_stats_no_static_global.npz",
        era5_mean=mean_nostatic, era5_std=std_nostatic,
    )

    logger.info(
        f"Wrote global normalisation stats: "
        f"{len(mean_full)} channels w/ static, "
        f"{len(mean_nostatic)} w/o static"
    )


if __name__ == "__main__":
    main()