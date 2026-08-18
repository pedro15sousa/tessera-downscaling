import multiprocessing.synchronize as ms
import shutil
import time
from functools import partial
from multiprocessing import Semaphore
from pathlib import Path
from typing import Any

import pandas as pd
import xarray as xr
from ecmwf.datastores import Client
from urllib3 import request

from dataprocessing.utils import (
    ATMOS_VARIABLES,
    SURFACE_VARIABLES,
    atomic,
    atomic_completed,
    compute_file_name,
    flush_and_sync,
    parallel_foreach,
    write_and_flush_dataset,
)

ROOT_DIR = Path(".tmp_out")


def request_template(
    variables: list[str],
    year: int,
    month: int,
    hours: list[int],
    *,
    is_atmos: bool = False,
) -> dict[str, Any]:
    r = {
        "product_type": ["reanalysis"],
        "variable": variables,
        "year": [str(year)],
        "month": [f"{month:02d}"],
        "day": [f"{n:02d}" for n in range(1, 32)],
        "time": hours,
        "data_format": "grib",
        "download_format": "unarchived",
    }
    if is_atmos:
        p = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]
        r["pressure_level"] = [str(x) for x in p]
    return r


def download_variable_month_raw(
    output_path: Path,
    year_month_var: tuple[int, int, str],
    submission_semaphore: ms.Semaphore,
    download_semaphore: ms.Semaphore,
) -> None:
    year, month, variable = year_month_var

    # Pause inside of a semaphore to ensure that there's no chance of
    # sending a request too quickly after another worker sent a request.
    with submission_semaphore:
        time.sleep(0.2)
    is_atmos = variable in ATMOS_VARIABLES
    r = request_template([variable], year, month, [0, 6, 12, 18], is_atmos=is_atmos)
    if is_atmos:
        cds_dataset_name = "reanalysis-era5-pressure-levels"
    else:
        cds_dataset_name = "reanalysis-era5-single-levels"
    results = Client().submit_and_wait_on_results(cds_dataset_name, r)

    # Acquire download semaphore and download.
    with download_semaphore:
        resp = request("GET", results.location, preload_content=True)
        if resp.status != 200:  # noqa: PLR2004
            message = "Failed to download, status: {result.status}.\n"
            raise RuntimeError(message)

    # Write results out to permanent storage.
    with output_path.open("wb") as f:
        f.write(resp.data)
        flush_and_sync(f)


def process_variable_slice(output_path: Path, subset: xr.Dataset, var: str) -> None:  # noqa: C901
    # In order to line up with the weatherbench2 data, we need to pull out some
    # specific fields, and discard the rest. Furthermore, the name of the
    # main field differs, so we need to align that. Moreover, weatherbench uses
    # hours since 1959-01-01, while the grib data uses seconds since 1979.
    if var == "2m_temperature":
        cds_name = "t2m"
        is_atmos = False
    elif var == "10m_u_component_of_wind":
        cds_name = "u10"
        is_atmos = False
    elif var == "10m_v_component_of_wind":
        cds_name = "v10"
        is_atmos = False
    elif var == "mean_sea_level_pressure":
        cds_name = "msl"
        is_atmos = False
    elif var == "u_component_of_wind":
        cds_name = "u"
        is_atmos = True
    elif var == "v_component_of_wind":
        cds_name = "v"
        is_atmos = True
    elif var == "temperature":
        cds_name = "t"
        is_atmos = True
    elif var == "specific_humidity":
        cds_name = "q"
        is_atmos = True
    elif var == "geopotential":
        cds_name = "z"
        is_atmos = True
    else:
        message = f"Unexpected variable {var}, subset is {subset}."
        raise RuntimeError(message)

    # Pull out only the required fields, and rename main variable.
    if is_atmos:
        subset = subset[["latitude", "longitude", "time", "isobaricInhPa", cds_name]]
        subset = subset.rename({cds_name: var, "isobaricInhPa": "level"})
        subset = subset.assign_coords(level=subset.level.astype(int)).sortby("level")
        subset = subset.drop_vars(["number", "step", "time"])
    else:
        subset = subset[["latitude", "longitude", "time", cds_name]]
        subset = subset.rename({cds_name: var})
        subset = subset.drop_vars(["number", "step", "surface", "time"])
    subset = subset.rename({"valid_time": "time"})

    # Write out to temporary storage as netcdf.
    encoding_dict = {
        "time": {"units": "hours since 1959-01-01", "calendar": "proleptic_gregorian"},
    }
    write_and_flush_dataset(
        output_path, subset, engine="h5netcdf", encoding=encoding_dict
    )


def download_variable_month(
    year_month_var: tuple[int, int, str],
    _: None,
    submission_semaphore: ms.Semaphore,
    download_semaphore: ms.Semaphore,
) -> None:
    # Specify paths.
    year, month, var = year_month_var
    raw_name = f"{year}-{month:02}.grib"
    raw_root = ROOT_DIR / "raw" / f"era5-cds-{var}"
    raw_root.mkdir(parents=True, exist_ok=True)
    raw_path = raw_root / raw_name

    # Get + permanently store a copy of the raw data.
    if not atomic_completed(raw_path):
        print(f"Downloading {raw_path}.")
        atomic(
            download_variable_month_raw,
            raw_path,
            year_month_var,
            submission_semaphore,
            download_semaphore,
        )

    # Specify dates which are expected to be in the resulting file.
    month = pd.Timestamp(year, month, 1)
    end_of_month = month + pd.offsets.MonthEnd() + pd.offsets.Hour(18)
    times = pd.date_range(start=month, end=end_of_month, freq="6h")
    dir_path = ROOT_DIR / "processed" / f"era5_quarter_{var}" / "data"
    dir_path.mkdir(parents=True, exist_ok=True)
    file_paths = [dir_path / compute_file_name(t) for t in times]

    # Return early if the data has already been processed.
    if all(atomic_completed(f) for f in file_paths):
        return

    # Grab raw data from storage, and put in temporary storage. It is a good idea to
    # ensure that the `.tmp` directory is memory-backed using a linux `tmpfs`.
    tmp_raw_path = Path(".tmp") / f"{var}-{raw_name}"
    with tmp_raw_path.open("wb") as tmp, raw_path.open("rb") as raw:
        shutil.copyfileobj(fsrc=raw, fdst=tmp)
        flush_and_sync(tmp)

    try:
        # Process data -- split up into 6 hour chunks.
        d = xr.open_dataset(
            tmp_raw_path,
            engine="cfgrib",
            decode_timedelta=False,
            backend_kwargs={
                "indexpath": ""
            },  # index created in memory rather than disk
        )
        for t, file_path in zip(times, file_paths, strict=False):
            ds_slice = d.sel({"time": t.to_datetime64()})
            atomic(process_variable_slice, file_path, ds_slice, var)

    finally:
        # Tidy up -- remove temporary raw data file.
        tmp_raw_path.unlink()


def main() -> None:
    vs = ATMOS_VARIABLES + SURFACE_VARIABLES
    items = [(y, m, v) for y in [2023, 2024, 2025] for m in range(1, 13) for v in vs]
    parallel_foreach(
        f=partial(
            download_variable_month,
            submission_semaphore=Semaphore(1),
            download_semaphore=Semaphore(2),
        ),
        init=lambda _: None,
        items=items,
        num_processes=4,
    )


if __name__ == "__main__":
    main()
