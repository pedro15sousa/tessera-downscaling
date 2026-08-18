"""Preprocess raw 6-hourly ERA5 + GHCNh data into a multi-region daily training dataset.

This is the multi-region counterpart to :mod:`preprocess_daily`. Whereas
the original script produces a single Europe-only dataset, this one
produces a layered layout where each region has its own cropped ERA5
grid + static fields + normalisation stats, sharing a single global
GHCNh observation set keyed by station index:

    dataset_daily_global/
        metadata.json                       <-- top-level manifest
        stations.csv                        <-- global, with `region` column
        valid_station_indices.npy
        ghcnh_daily/<date>.npz              <-- shared, keyed by station idx
        regions/
            us/
                lats.npy
                lons.npy
                static_fields.npy
                normalisation_stats.npz
                normalisation_stats_no_static.npz
                era5_daily/<date>.npy
                region_metadata.json
            europe/
                ...
            ...

Design notes:

    - GHCNh observations are computed **once** per day (for the union of
      stations across all requested regions) and saved flat. This avoids
      duplicating per-region aggregates; region-specific training uses a
      station-index mask into the shared file.
    - ERA5 grid data and static fields are naturally per-region because
      the crops differ. They live under ``regions/<name>/``.
    - Temporal split (train/val/test) is global — same boundary dates
      across all regions, set by the shared constants in
      :mod:`helpers`.
    - Spatial 85/15 split is computed *per region* using the same seed,
      so each region has roughly 85% of its stations for training.
    - Regions are **non-overlapping by construction**. Each station is
      assigned to exactly one region based on bbox membership; stations
      outside all defined regions are dropped.
    - Delta-elevation is computed per station using the global static
      fields file (the same one is referenced by all regions — only the
      *crop* differs per region, not the underlying orography source).

Usage (from repo root):
    uv run --group core python projects/tessera_downscaling/scripts/preprocessing/daily/preprocess_daily_global.py \\
        --era5-dir .tmp_output/_staging/processed \\
        --ghcnh-dir .tmp_output/_staging/processed/ghcnh/data \\
        --static-path .tmp_output/_staging/processed/era5_static/era5_static_0p25_all.nc \\
        --station-csv .tmp_output/_staging/raw/ghcnh/station_list.csv \\
        --output-dir .tmp_output/dataset_daily_global \\
        --start-date 2017-01-01 \\
        --end-date 2023-01-10 \\
        --regions us europe east_asia australia southern_africa
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
    MIN_VALID_DAYS,
    SPLIT_SEED,
    TRAIN_END,
    TRAIN_STATION_FRACTION,
    VAL_END,
    GHCNH_TARGET_VARS,
    aggregate_era5_daily,
    aggregate_ghcnh_daily,
    compute_delta_elevation,
    lookup_station_mtpi,
    compute_grid_crop_indices,
    era5_channel_names,
    filter_valid_elevation,
    load_static_fields,
    partition_dates_by_temporal_split,
    random_spatial_split,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("preprocess_daily_global")


# ---------------------------------------------------------------------------
# Region definitions
# ---------------------------------------------------------------------------
#
# Each entry: region_name -> (lat_min, lat_max, lon_min, lon_max) in
# -180/180 convention. Regions are non-overlapping. Adding a new region is
# a matter of adding an entry here and re-running the script with
# --regions including the new name.

REGIONS: dict[str, tuple[float, float, float, float]] = {
    "europe":          (35.0,  75.0,  -24.0,  40.0),
    "us":              (24.0,  50.0, -125.0, -66.0),
    "east_asia":       (20.0,  46.0,  100.0, 146.0),
    "australia":      (-44.0, -10.0,  112.0, 154.0),
    "southern_africa":(-35.0, -15.0,   15.0,  35.0),
}


# ---------------------------------------------------------------------------
# Station assignment
# ---------------------------------------------------------------------------

def assign_stations_to_regions(
    stations: pd.DataFrame,
    region_bboxes: dict[str, tuple[float, float, float, float]],
) -> pd.DataFrame:
    """Assign each station to exactly one region by bbox membership.

    Stations outside all region boxes are dropped. Since we require
    non-overlapping boxes, each station either matches exactly one or
    none. If a station somehow matches multiple (bad region config),
    the first match in insertion order wins and a warning is logged.

    Args:
        stations: DataFrame with canonical columns ``latitude``,
            ``longitude``, ``elevation``, ``station_id``.
        region_bboxes: Dict of region_name -> (lat_min, lat_max,
            lon_min, lon_max).

    Returns:
        DataFrame filtered to regionally-assigned stations, with an
        added ``region`` column.
    """
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
            f"kept the first-matching region. Check that bboxes are "
            f"non-overlapping."
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
# Per-region ERA5 processing
# ---------------------------------------------------------------------------

def build_region_grids(
    region_names: list[str],
    era5_dir: Path,
    static_path: Path,
    region_bboxes: dict[str, tuple[float, float, float, float]],
) -> dict[str, dict]:
    """Compute grid crop indices and static fields for each region.

    Reads the global ERA5 lats/lons from a sample file and returns a
    dict keyed by region name containing everything the daily ERA5 loop
    needs to crop for that region.

    Returns:
        Dict mapping region name to a dict with keys:
            - ``lat_indices``, ``lon_indices``, ``roll_amount``
            - ``lats_crop``, ``lons_crop``
            - ``static_array``: (n_static, H, W) numpy array
            - ``static_var_names``: list of channel names
    """
    # Read lats/lons from a sample 6-hourly file. Pick whichever surface
    # variable directory is present first.
    sample_date = None
    for candidate in ["2m_temperature", "10m_u_component_of_wind"]:
        sample_dir = era5_dir / f"era5_wb2_quarter_{candidate}" / "data"
        if sample_dir.exists():
            files = sorted(sample_dir.glob("*.nc"))
            if files:
                sample_date = files[0]
                break
    if sample_date is None:
        raise FileNotFoundError(
            f"Could not find a sample ERA5 file under {era5_dir}"
        )

    ds = xr.open_dataset(sample_date)
    lats = ds.latitude.values
    lons = ds.longitude.values
    ds.close()

    out: dict[str, dict] = {}
    for name in region_names:
        if name not in region_bboxes:
            raise ValueError(
                f"Unknown region '{name}'. Defined: {list(region_bboxes)}"
            )
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
    """Create <output_dir>/regions/<name>/ and write the static per-region files.

    Writes:
        - ``lats.npy``, ``lons.npy``
        - ``static_fields.npy``
        - ``region_metadata.json``

    Returns the per-region directory path.
    """
    region_dir = output_dir / "regions" / region_name
    (region_dir / "era5_daily").mkdir(parents=True, exist_ok=True)

    np.save(region_dir / "lats.npy", region_grid["lats_crop"])
    np.save(region_dir / "lons.npy", region_grid["lons_crop"])
    np.save(region_dir / "static_fields.npy", region_grid["static_array"])

    region_meta = {
        "region_name": region_name,
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
        description="Preprocess raw ERA5 + GHCNh into a multi-region daily training dataset",
    )
    parser.add_argument("--era5-dir", type=Path, required=True,
                        help="Root directory containing era5_wb2_quarter_*/data/ subdirectories")
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
                        default=Path(".tmp_output/dataset_daily_global"),
                        help="Output directory for the multi-region dataset")
    parser.add_argument("--start-date", type=str, default="2017-01-01")
    parser.add_argument("--end-date", type=str, default="2023-01-10")
    parser.add_argument(
        "--target-variables", type=str, nargs="+",
        default=list(GHCNH_TARGET_VARS.keys()),
        choices=list(GHCNH_TARGET_VARS.keys()),
        help="Station-level target variables. Default: all supported.",
    )
    parser.add_argument(
        "--regions", type=str, nargs="+",
        default=list(REGIONS.keys()),
        choices=list(REGIONS.keys()),
        help="Which regions to process. Default: all defined regions.",
    )
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "ghcnh_daily").mkdir(exist_ok=True)

    # ------------------------------------------------------------------
    # Step 1: Build per-region grid crops and static fields.
    # ------------------------------------------------------------------
    logger.info(f"Processing regions: {args.regions}")
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
    # Step 3: Delta-elevation (uses the global static file, works for
    # every station since the underlying orography covers the full
    # globe — only the per-region crops differ downstream).
    # ------------------------------------------------------------------
    stations["delta_elevation"] = compute_delta_elevation(stations, args.static_path)
    if args.mtpi_csv is not None:
        stations["mtpi"] = lookup_station_mtpi(stations, args.mtpi_csv)

    # ------------------------------------------------------------------
    # Step 4: Generate date list.
    # ------------------------------------------------------------------
    start = datetime.strptime(args.start_date, "%Y-%m-%d")
    end = datetime.strptime(args.end_date, "%Y-%m-%d")
    dates: list[str] = []
    current = start
    while current <= end:
        dates.append(current.strftime("%Y-%m-%d"))
        current += timedelta(days=1)
    logger.info(f"Processing {len(dates)} days from {dates[0]} to {dates[-1]}")

    channel_names = era5_channel_names()
    logger.info(f"ERA5 dynamic channels ({len(channel_names)}): {channel_names}")

    # ------------------------------------------------------------------
    # Step 5: Per-day loop.
    #
    # For each day we process each region's ERA5 separately (different
    # crops), but aggregate GHCNh observations **once** across the
    # union of all regionally-assigned stations. This avoids duplicating
    # GHCNh work per region.
    # ------------------------------------------------------------------
    valid_day_counts: dict[str, np.ndarray] = {
        v: np.zeros(len(station_ids), dtype=np.int32)
        for v in args.target_variables
    }

    days_processed = 0
    days_skipped = 0

    for date_str in dates:
        ghcnh_path = output_dir / "ghcnh_daily" / f"{date_str}.npz"
        missing_vars: list[str] = []

        # Fast path: skip date if already fully processed.
        all_era5_exist = all(
            (output_dir / "regions" / name / "era5_daily" / f"{date_str}.npy").exists()
            for name in args.regions
        )
        if ghcnh_path.exists() and all_era5_exist:
            ghcnh_data = np.load(ghcnh_path)
            missing_vars = [
                v for v in args.target_variables if v not in ghcnh_data.files
            ]
            if not missing_vars:
                for v in args.target_variables:
                    valid_day_counts[v] += ~np.isnan(ghcnh_data[v])
                days_processed += 1
                continue
            logger.info(
                f"{date_str}: re-processing GHCNh to add missing variables: "
                f"{missing_vars}"
            )

        # ERA5 per region.
        any_era5_missing = False
        for name in args.regions:
            era5_path = output_dir / "regions" / name / "era5_daily" / f"{date_str}.npy"
            if era5_path.exists():
                continue
            grid = region_grids[name]
            era5_daily = aggregate_era5_daily(
                args.era5_dir, date_str,
                grid["lat_indices"], grid["lon_indices"], grid["roll_amount"],
            )
            if era5_daily is None:
                any_era5_missing = True
                break
            np.save(era5_path, era5_daily)

        if any_era5_missing:
            days_skipped += 1
            continue

        # GHCNh aggregated once for all regional stations (cached across regions).
        if not ghcnh_path.exists() or missing_vars:
            ghcnh_results = aggregate_ghcnh_daily(
                args.ghcnh_dir, date_str, station_ids,
                target_variables=args.target_variables,
            )
            np.savez_compressed(ghcnh_path, **ghcnh_results)
        else:
            ghcnh_results = dict(np.load(ghcnh_path))

        for v in args.target_variables:
            valid_day_counts[v] += ~np.isnan(ghcnh_results[v])
        days_processed += 1

        if days_processed % 100 == 0:
            logger.info(f"  Processed {days_processed} days, skipped {days_skipped}")

    logger.info(
        f"Finished daily loop: {days_processed} processed, {days_skipped} skipped"
    )

    # ------------------------------------------------------------------
    # Step 6: Filter stations by MIN_VALID_DAYS (union across target variables).
    # ------------------------------------------------------------------
    for v, counts in valid_day_counts.items():
        stations[f"n_valid_days_{v}"] = counts

    valid_mask = np.zeros(len(station_ids), dtype=bool)
    for counts in valid_day_counts.values():
        valid_mask |= (counts >= MIN_VALID_DAYS)

    logger.info(
        f"Station filtering: {int(valid_mask.sum())}/{len(stations)} stations "
        f"have >= {MIN_VALID_DAYS} valid days for at least one target"
    )

    stations_filtered = stations[valid_mask].reset_index(drop=True)
    valid_indices = np.where(valid_mask)[0]
    np.save(output_dir / "valid_station_indices.npy", valid_indices)

    # ------------------------------------------------------------------
    # Step 7: Per-region 85/15 spatial split.
    # ------------------------------------------------------------------
    stations_filtered["spatial_split"] = ""
    for region_name in args.regions:
        region_mask = stations_filtered["region"].values == region_name
        n_in_region = int(region_mask.sum())
        if n_in_region == 0:
            continue
        split = random_spatial_split(n_in_region, seed=SPLIT_SEED)
        # Assign back preserving DataFrame order.
        region_indices = np.where(region_mask)[0]
        stations_filtered.loc[region_indices, "spatial_split"] = split

    split_counts = (
        stations_filtered.groupby(["region", "spatial_split"]).size().to_dict()
    )
    logger.info(f"Per-region spatial split counts: {split_counts}")

    # ------------------------------------------------------------------
    # Step 8: Compute per-region normalisation stats (train dates only).
    # ------------------------------------------------------------------
    valid_dates = sorted([
        d for d in dates
        if all(
            (output_dir / "regions" / name / "era5_daily" / f"{d}.npy").exists()
            for name in args.regions
        )
    ])
    temporal_split = partition_dates_by_temporal_split(
        valid_dates, train_end=TRAIN_END, val_end=VAL_END,
    )

    for region_name in args.regions:
        _compute_region_normalisation_stats(
            output_dir=output_dir,
            region_name=region_name,
            region_grid=region_grids[region_name],
            train_dates=temporal_split["train_dates"],
        )

    # ------------------------------------------------------------------
    # Step 9: Write top-level manifest.
    # ------------------------------------------------------------------
    stations_filtered.to_csv(output_dir / "stations.csv", index=False)

    metadata = {
        "layout_version": "multi_region_v1",
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
        "n_days_processed": days_processed,
        "n_days_skipped": days_skipped,
        "date_range": [dates[0], dates[-1]],
        "valid_dates": valid_dates,
        "temporal_split": {
            "train_end": TRAIN_END,
            "val_end": VAL_END,
            "n_train_days": len(temporal_split["train_dates"]),
            "n_val_days": len(temporal_split["val_dates"]),
            "n_test_days": len(temporal_split["test_dates"]),
        },
        "spatial_split": {
            "train_fraction": TRAIN_STATION_FRACTION,
            "seed": SPLIT_SEED,
            "per_region_counts": {
                str(k): int(v) for k, v in split_counts.items()
            },
        },
        "min_valid_days": MIN_VALID_DAYS,
    }
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Dataset saved to {output_dir}")
    logger.info("Done!")


def _compute_region_normalisation_stats(
    output_dir: Path,
    region_name: str,
    region_grid: dict,
    train_dates: list[str],
) -> None:
    """Compute per-region z-score stats from training dates.

    Writes two files to the region dir:
        - ``normalisation_stats.npz``           (dynamic + static + lat/lon)
        - ``normalisation_stats_no_static.npz`` (dynamic + lat/lon only)
    """
    region_dir = output_dir / "regions" / region_name
    era5_daily_dir = region_dir / "era5_daily"
    lats = region_grid["lats_crop"]
    lons = region_grid["lons_crop"]
    static_array = region_grid["static_array"]  # (n_static, H, W)
    n_static = static_array.shape[0]

    lat_grid = lats[:, None] * np.ones((1, len(lons)), dtype=np.float32)
    lon_grid = np.ones((len(lats), 1), dtype=np.float32) * lons[None, :]

    def _accumulate(include_static: bool) -> tuple[np.ndarray, np.ndarray]:
        """Compute per-channel mean/std over training dates."""
        # Dynamic channel count comes from the first loaded file.
        first_path = era5_daily_dir / f"{train_dates[0]}.npy"
        if not first_path.exists():
            raise FileNotFoundError(
                f"No ERA5 daily file at {first_path}; cannot compute "
                f"normalisation stats for region '{region_name}'."
            )
        first = np.load(first_path)
        n_dynamic = first.shape[0]

        n_channels = n_dynamic + (n_static if include_static else 0) + 2
        running_sum = np.zeros(n_channels, dtype=np.float64)
        running_sq_sum = np.zeros(n_channels, dtype=np.float64)
        n_pixels = 0

        for d in train_dates:
            p = era5_daily_dir / f"{d}.npy"
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


if __name__ == "__main__":
    main()