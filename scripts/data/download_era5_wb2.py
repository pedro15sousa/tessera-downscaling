"""Download 6-hourly ERA5 from the WeatherBench2 GCS bucket into the staging tree.

Reads the public zarr store
``gs://weatherbench2/datasets/era5/1959-2023_01_10-wb13-6h-1440x721_with_derived_variables.zarr``
and writes one NetCDF per (variable, timestamp) at 0.25 deg:

    <root>/era5_wb2_quarter_<variable>/data/YYYY-MM-DD-HH.nc

for the 10 variables the downscaling datasets use (5 surface + 5 atmospheric;
see ``tessera_downscaling.io_utils``). Every file is written through
:func:`atomic`, so an interrupted run resumes where it stopped.

``--levels`` selects which pressure levels the atmospheric variables keep:

* ``paper`` (default): the 3 levels the downscaling datasets use
  (500/700/850 hPa), written under ``<data root>/ingest/processed/``.
* ``aurora``: all 13 WeatherBench2 levels (50..1000 hPa), which the Aurora
  0.25 deg model needs as initial conditions, written under
  ``<data root>/ingest/aurora_inputs/`` so they never overwrite the 3-level
  staging. Pass this only when preparing inputs for
  ``scripts/aurora/generate_aurora_forecasts.py``.

Usage (from the repo root; needs the ``ingest`` extra for gcsfs):
    uv run python scripts/data/download_era5_wb2.py \\
        --start 2010-01-01 --end 2023-01-10T18:00 --num-processes 8
    uv run python scripts/data/download_era5_wb2.py --levels aurora \\
        --start 2009-12-28 --end 2021-12-26T18:00 --num-processes 16
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
    WB_LEVELS,
    atomic,
    atomic_completed,
    compute_file_name,
    parallel_foreach,
    write_and_flush_dataset,
)
from tessera_downscaling.paths import ingest_dir

REMOTE_ROOT = "gs://weatherbench2/datasets/era5"

# Pressure-level set and default output root per --levels choice.
LEVEL_CHOICES = {
    "paper": (PAPER_LEVELS, "processed"),
    "aurora": (WB_LEVELS, "aurora_inputs"),
}


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
    levels: list[int],
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
        # Filter atmospheric variables to the pressure levels this run keeps.
        if "level" in x.dims:
            x = x.sel(level=levels)
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
        "--levels",
        choices=sorted(LEVEL_CHOICES),
        default="paper",
        help="Pressure levels for atmospheric variables: 'paper' (500/700/850 hPa, "
        "the downscaling datasets) or 'aurora' (all 13 WB2 levels, Aurora initial "
        "conditions). Default: paper.",
    )
    p.add_argument(
        "--root",
        type=Path,
        default=None,
        help="Output root; files go to <root>/era5_wb2_quarter_<var>/data/. "
        "Default: <data root>/ingest/processed for --levels paper, "
        "<data root>/ingest/aurora_inputs for --levels aurora.",
    )
    p.add_argument("--num-processes", type=int, default=8)
    args = p.parse_args()

    levels, default_subdir = LEVEL_CHOICES[args.levels]
    root = args.root if args.root is not None else ingest_dir(default_subdir)

    variables = ATMOS_VARIABLES + SURFACE_VARIABLES
    dates = pd.date_range(args.start, args.end, freq="6h")

    parallel_foreach(
        f=partial(process_slice, root=root, levels=levels),
        init=init_quarter,
        items=[(v, d) for d in dates for v in variables],
        num_processes=args.num_processes,
    )


if __name__ == "__main__":
    main()
