"""Download 6-hourly ERA5 from the WeatherBench2 GCS bucket into the staging tree.

Reads the public zarr store
``gs://weatherbench2/datasets/era5/1959-2023_01_10-wb13-6h-1440x721_with_derived_variables.zarr``
and writes one NetCDF per (variable, timestamp) at 0.25 deg:

    <root>/era5_wb2_quarter_<variable>/data/YYYY-MM-DD-HH.nc

for the 12 variables the downscaling datasets use (5 surface + 5 atmospheric x
the 3 pressure levels 500/700/850 hPa; see ``tessera_downscaling.io_utils``).
Every file is written through :func:`atomic`, so an interrupted run resumes
where it stopped. The default root is the data root's ``_staging/processed/``.

Usage (from the repo root; needs the ``ingest`` extra for gcsfs):
    uv run python scripts/data/download_era5_wb2.py \\
        --start 2010-01-01 --end 2023-01-10T18:00 --num-processes 8
"""

from __future__ import annotations

import argparse
from functools import partial
from pathlib import Path

import pandas as pd
import xarray as xr

from tessera_downscaling.io_utils import (
    ATMOS_VARIABLES,
    PAPER_LEVELS,
    SURFACE_VARIABLES,
    atomic,
    atomic_completed,
    compute_file_name,
    parallel_foreach,
    write_and_flush_dataset,
)
from tessera_downscaling.paths import staging_dir

REMOTE_ROOT = "gs://weatherbench2/datasets/era5"


# Choose between the `init_<resolution>` functions to pick which resolution to download.
def init_quarter(_: int) -> tuple[xr.Dataset, str]:
    zarr_name = "1959-2023_01_10-wb13-6h-1440x721_with_derived_variables.zarr"
    return xr.open_zarr(f"{REMOTE_ROOT}/{zarr_name}"), "quarter"


def init_one_and_half(_: int) -> tuple[xr.Dataset, str]:
    zarr_name = "1959-2023_01_10-6h-240x121_equiangular_with_poles_conservative.zarr"
    return xr.open_zarr(f"{REMOTE_ROOT}/{zarr_name}"), "one_and_half"


def process_slice(
    var_and_datetime: tuple[str, pd.Timestamp],
    dataset_and_res_name: tuple[xr.Dataset, str],
    root: Path,
) -> None:
    # Extract everything.
    var, datetime = var_and_datetime
    dataset, res_name = dataset_and_res_name

    # Compute paths.
    file_name = compute_file_name(datetime)
    data_dir = root / f"era5_wb2_{res_name}_{var}" / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    output_path = data_dir / file_name

    # Skip if we already have the data.
    if atomic_completed(output_path):
        return

    # Download the data atomically.
    def download_and_process_slice(
        output_path: Path,
        dataset: xr.Dataset,
        datetime: pd.Timestamp,
        var: str,
    ) -> None:
        x = dataset[var].sel(time=datetime).load()
        # Filter to only the pressure levels we need for atmospheric variables.
        if "level" in x.dims:
            x = x.sel(level=PAPER_LEVELS)
        x = x.load()
        write_and_flush_dataset(output_path, x, engine="h5netcdf")

    atomic(download_and_process_slice, output_path, dataset, datetime, var)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--start", default="2010-01-01", help="First timestamp (inclusive).")
    p.add_argument(
        "--end", default="2023-01-10T18:00:00", help="Last timestamp (inclusive)."
    )
    p.add_argument(
        "--root",
        type=Path,
        default=staging_dir("processed"),
        help="Output root; files go to <root>/era5_wb2_quarter_<var>/data/ "
        "(default: <data root>/_staging/processed).",
    )
    p.add_argument("--num-processes", type=int, default=8)
    args = p.parse_args()

    variables = ATMOS_VARIABLES + SURFACE_VARIABLES
    dates = pd.date_range(args.start, args.end, freq="6h")

    parallel_foreach(
        f=partial(process_slice, root=args.root),
        init=init_quarter,
        items=[(v, d) for d in dates for v in variables],
        num_processes=args.num_processes,
    )


if __name__ == "__main__":
    main()
