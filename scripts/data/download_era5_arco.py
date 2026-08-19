"""Download ERA5 from the ARCO-ERA5 GCS bucket into the WeatherBench2 staging layout.

Fallback for dates the WeatherBench2 zarr does not cover (it ends 2023-01-10):
reads the daily NetCDFs of ``gs://gcp-public-data-arco-era5`` (one file per
variable and day; one per pressure level for atmospheric variables), slices out
the four 6-hourly timestamps and writes them exactly as
``download_era5_wb2.py`` does:

    <root>/era5_wb2_quarter_<variable>/data/YYYY-MM-DD-HH.nc

so the preprocessors read both sources interchangeably. Resume-safe via
:func:`atomic`. Downloads are anonymous.

Usage (from the repo root; needs the ``ingest`` extra for google-cloud-storage):
    uv run python scripts/data/download_era5_arco.py \
        --variables 2m_temperature 10m_u_component_of_wind \
        --atmos-vars temperature --start 2023-01-11 --end 2023-12-31
"""

from __future__ import annotations

import argparse
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import xarray as xr

from tessera_downscaling.io_utils import (
    atomic,
    atomic_completed,
    compute_file_name,
    parallel_foreach,
    write_and_flush_dataset,
)
from tessera_downscaling.paths import staging_dir

if TYPE_CHECKING:
    from google.cloud import storage

ARCO_BUCKET = "gcp-public-data-arco-era5"


def arco_surf_path(date: pd.Timestamp, var: str) -> str:
    y = date.year
    m = date.month
    day = date.day
    return f"raw/date-variable-single_level/{y}/{m:02d}/{day:02d}/{var}/surface.nc"


def arco_atmos_path(date: pd.Timestamp, var: str, level: int) -> str:
    y = date.year
    m = date.month
    d = date.day
    return f"raw/date-variable-pressure_level/{y}/{m:02d}/{d:02d}/{var}/{level!s}.nc"


def download_arco_file(
    variable_date_pair: tuple[str, pd.Timestamp],
    client: storage.Client,
    tmp_dir: Path,
    dest_dir: Path,
) -> None:
    variable, date = variable_date_pair

    # If all files have been successfully created, do not recreate.
    hours = [0, 6, 12, 18]
    dts = [pd.Timestamp(date.to_datetime64() + np.timedelta64(h, "h")) for h in hours]
    var_dir = dest_dir / f"era5_wb2_quarter_{variable}" / "data"
    var_dir.mkdir(parents=True, exist_ok=True)
    file_paths = [var_dir / compute_file_name(dt) for dt in dts]
    if all(atomic_completed(file_path) for file_path in file_paths):
        return

    # Download the data, and write to temporary file.
    blob = client.bucket(ARCO_BUCKET).blob(arco_surf_path(date, variable))
    tmp_path = tmp_dir / f"{variable}-{date.year}-{date.month:02d}-{date.day:02d}.nc"
    blob.download_to_filename(tmp_path)

    try:
        ds = xr.open_dataset(tmp_path, engine="netcdf4")

        # Atomically slice out each of the hours that we need from the data.
        for _dt, file_path in zip(dts, file_paths, strict=False):
            ds_h = ds.sel(time=np.datetime64(_dt).astype("datetime64[s]"))
            atomic(write_and_flush_dataset, file_path, ds_h, engine="h5netcdf")
    finally:
        tmp_path.unlink()


def download_arco_atmos_file(
    variable_levels_date: tuple[str, tuple[int, ...], pd.Timestamp],
    client: storage.Client,
    tmp_dir: Path,
    dest_dir: Path,
) -> None:
    """Atmospheric (pressure-level) version of download_arco_file.

    Downloads N pressure-level files for one (variable, date), stacks them
    on a level dimension, and slices into 4 per-timestamp files matching
    the WB2 output convention (one NetCDF per (var, timestamp), each with
    a level dim).
    """
    variable, levels, date = variable_levels_date

    hours = [0, 6, 12, 18]
    dts = [pd.Timestamp(date.to_datetime64() + np.timedelta64(h, "h")) for h in hours]
    var_dir = dest_dir / f"era5_wb2_quarter_{variable}" / "data"
    var_dir.mkdir(parents=True, exist_ok=True)
    file_paths = [var_dir / compute_file_name(dt) for dt in dts]
    if all(atomic_completed(file_path) for file_path in file_paths):
        return

    tmp_paths = []
    try:
        # Download one file per pressure level
        for level in levels:
            blob = client.bucket(ARCO_BUCKET).blob(
                arco_atmos_path(date, variable, level)
            )
            tmp_path = (
                tmp_dir
                / f"{variable}-{level}-{date.year}-{date.month:02d}-{date.day:02d}.nc"
            )
            blob.download_to_filename(tmp_path)
            tmp_paths.append((tmp_path, level))

        # Open each, add a level coord, concat into one stacked dataset
        datasets = []
        for tmp_path, level in tmp_paths:
            ds_l = xr.open_dataset(tmp_path, engine="netcdf4")
            if "level" not in ds_l.dims:
                ds_l = ds_l.expand_dims(level=[level])
            datasets.append(ds_l)
        ds_stacked = xr.concat(datasets, dim="level")

        # Slice each of the 4 timestamps and write
        for dt, file_path in zip(dts, file_paths, strict=False):
            ds_h = ds_stacked.sel(time=np.datetime64(dt).astype("datetime64[s]"))
            atomic(write_and_flush_dataset, file_path, ds_h, engine="h5netcdf")

        for ds_l in datasets:
            ds_l.close()
    finally:
        for tmp_path, _ in tmp_paths:
            tmp_path.unlink(missing_ok=True)


def main() -> None:
    p = argparse.ArgumentParser(
        description="Fetch ERA5 surface variables from the ARCO bucket"
    )
    p.add_argument(
        "--variables",
        nargs="+",
        default=[],
        help="Surface variables (e.g. 100m_u_component_of_wind).",
    )
    p.add_argument("--start", required=True, help="Start date YYYY-MM-DD (inclusive).")
    p.add_argument("--end", required=True, help="End date YYYY-MM-DD (inclusive).")
    p.add_argument(
        "--root",
        type=Path,
        default=staging_dir("processed"),
        help="Output root. Files are written to "
        "{root}/era5_wb2_quarter_{var}/data/{timestamp}.nc "
        "(default: <data root>/_staging/processed).",
    )
    p.add_argument("--num-processes", type=int, default=4)
    p.add_argument(
        "--tmp-dir",
        type=Path,
        default=staging_dir("tmp", "arco_tmp"),
        help="Temporary directory for daily NetCDFs before splitting "
        "(default: <data root>/_staging/tmp/arco_tmp).",
    )
    p.add_argument(
        "--atmos-vars",
        nargs="+",
        default=[],
        help="Atmospheric variables (e.g. temperature u_component_of_wind).",
    )
    p.add_argument(
        "--pressure-levels",
        type=int,
        nargs="+",
        default=[500, 700, 850],
        help="Pressure levels (hPa) for atmos vars (default 500 700 850).",
    )
    args = p.parse_args()

    if not args.variables and not args.atmos_vars:
        raise ValueError(
            "Specify at least one of --variables (surface) or --atmos-vars."
        )

    from google.cloud import storage  # imported here so --help works without the extra

    dates = pd.date_range(args.start, args.end, freq="1D")
    print(f"Surface vars ({len(args.variables)}): {args.variables}")
    print(f"Atmos vars ({len(args.atmos_vars)}) × levels {args.pressure_levels}")
    print(f"Dates: {dates[0].date()} .. {dates[-1].date()} ({len(dates)} days)")
    print(f"Root: {args.root}")

    args.root.mkdir(parents=True, exist_ok=True)
    args.tmp_dir.mkdir(parents=True, exist_ok=True)

    if args.variables:
        print(f"\n=== Surface phase ({len(args.variables) * len(dates)} items) ===")
        parallel_foreach(
            f=partial(download_arco_file, tmp_dir=args.tmp_dir, dest_dir=args.root),
            init=lambda _: storage.Client.create_anonymous_client(),
            items=[(v, d) for v in args.variables for d in dates],
            num_processes=args.num_processes,
        )

    if args.atmos_vars:
        print(f"\n=== Atmos phase ({len(args.atmos_vars) * len(dates)} items) ===")
        levels = tuple(args.pressure_levels)
        parallel_foreach(
            f=partial(
                download_arco_atmos_file, tmp_dir=args.tmp_dir, dest_dir=args.root
            ),
            init=lambda _: storage.Client.create_anonymous_client(),
            items=[(v, levels, d) for v in args.atmos_vars for d in dates],
            num_processes=args.num_processes,
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
