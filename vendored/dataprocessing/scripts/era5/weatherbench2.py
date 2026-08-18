from functools import partial
from pathlib import Path

import pandas as pd
import xarray as xr

from dataprocessing.utils import (
    ATMOS_VARIABLES,
    SURFACE_VARIABLES,
    atomic,
    atomic_completed,
    compute_file_name,
    parallel_foreach,
    write_and_flush_dataset,
)

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
            x = x.sel(level=[500, 700, 850])
        x = x.load()
        write_and_flush_dataset(output_path, x, engine="h5netcdf")

    atomic(download_and_process_slice, output_path, dataset, datetime, var)


def main() -> None:
    variables = ATMOS_VARIABLES + SURFACE_VARIABLES
    # dates = pd.date_range("1979-01-01", "2023-01-10T18:00:00", freq="6h")
    # dates = pd.date_range("2017-01-01", "2023-01-10T18:00:00", freq="6h")
    # dates = pd.date_range("2010-01-01", "2023-01-10T18:00:00", freq="6h")
    dates = pd.date_range("2010-01-01", "2023-01-10T18:00:00", freq="6h")

    parallel_foreach(
        # f=partial(process_slice, root=Path(".tmp_output/processed")),
        # f=partial(process_slice, root=Path("projects/tessera_downscaling/.tmp_output/_staging/processed")),
        f=partial(process_slice, root=Path("/data/weather-downscaling/_staging/processed")),
        init=init_quarter,
        items=[(v, d) for d in dates for v in variables],
        # num_processes=2,
        num_processes=8,
    )


if __name__ == "__main__":
    main()
