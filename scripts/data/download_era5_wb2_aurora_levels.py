"""Download 6-hourly ERA5 at all 13 WeatherBench2 pressure levels (Aurora inputs).

Same source, layout and resume semantics as ``download_era5_wb2.py``, but the
atmospheric variables keep all 13 pressure levels (50..1000 hPa) that the Aurora
0.25 deg model needs as initial conditions. Written to a separate root (default
``_staging/aurora_inputs``) so it never overwrites the 3-level paper staging.

Usage (from the repo root; needs the ``ingest`` extra for gcsfs):
    uv run python scripts/data/download_era5_wb2_aurora_levels.py \\
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
    SURFACE_VARIABLES,
    WB_LEVELS,
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
        # Keep all 13 pressure levels for atmospheric variables.
        if "level" in x.dims:
            x = x.sel(level=WB_LEVELS)
        x = x.load()
        write_and_flush_dataset(output_path, x, engine="h5netcdf")

    atomic(download_and_process_slice, output_path, dataset, datetime, var)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--start", default="2009-12-28", help="First timestamp (inclusive).")
    p.add_argument(
        "--end", default="2021-12-26T18:00:00", help="Last timestamp (inclusive)."
    )
    p.add_argument(
        "--root",
        type=Path,
        default=staging_dir("aurora_inputs"),
        help="Output root; files go to <root>/era5_wb2_quarter_<var>/data/ "
        "(default: <data root>/_staging/aurora_inputs).",
    )
    p.add_argument("--num-processes", type=int, default=16)
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
