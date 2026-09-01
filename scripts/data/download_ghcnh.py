"""Download GHCNh station observations and bin them into 6-hourly NetCDFs.

Two phases per year, for every station in NOAA's global GHCNh station list:

1. Fetch the raw per-station-year PSV files from NCEI into
   ``<root>/raw/ghcnh/<station>_<year>.psv`` (404 -> empty file, so a
   station-year with no data is not re-requested on resume).
2. Parse them (quality-code filter: keep rows whose QC flags are all missing
   or 1), keep temperature / dew point / pressure / wind / precipitation plus
   station metadata, and write one NetCDF per 6-hour bin, labelled by the bin's
   right edge, to ``<root>/processed/ghcnh/data/YYYY-MM-DD-HH.nc``.

The station list itself is cached at ``<root>/raw/ghcnh/station_list.csv``
(it is also the station list every downstream step is aligned to). Both phases
are resume-safe via :func:`atomic`. The default root is the data root's
``ingest/``.

Usage (from the repo root):
    uv run python scripts/data/download_ghcnh.py --years 2010 2023 --num-processes 4
"""

from __future__ import annotations

import argparse
import math
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd
from urllib3 import request

from tessera_downscaling.io_utils import (
    atomic,
    atomic_completed,
    compute_file_name,
    flush_and_sync,
    parallel_foreach,
    write_and_flush_dataset,
)
from tessera_downscaling.paths import ingest_dir

# Set from --root in main(); module-level so the worker functions can see them.
RAW_ROOT = ingest_dir("raw", "ghcnh")
PROCESSED_ROOT = ingest_dir("processed", "ghcnh")


def filtering_cols() -> list[str]:
    return [
        "temperature_Quality_Code",
        "dew_point_temperature_Quality_Code",
        "station_level_pressure_Quality_Code",
        "wind_direction_Quality_Code",
        "wind_speed_Quality_Code",
        "precipitation_Quality_Code",
        "precipitation_6_hour_Quality_Code",
    ]


def ghcn_station_year_url(year: int, station_id: str) -> str:
    return f"https://www.ncei.noaa.gov/oa/global-historical-climatology-network/hourly/access/by-year/{year}/psv/GHCNh_{station_id}_{year}.psv"


def ghcnh_file_path(year: int, station_id: str) -> Path:
    return RAW_ROOT / f"{station_id}_{year}.psv"


def download(path: Path, url: str) -> None:
    print(path)
    resp = request("GET", url)
    if resp.status == 404:  # noqa: PLR2004
        data = b""
    elif resp.status != 200:  # noqa: PLR2004
        message = f"Bad response status: {resp.status}"
        raise RuntimeError(message)
    else:
        data = resp.data
    with path.open("wb") as f:
        f.write(data)
        flush_and_sync(f)


def download_ghcnh_station(year_id_pair: tuple[int, str], _: None) -> None:
    year, station_id = year_id_pair
    file_path = ghcnh_file_path(year, station_id)
    if not atomic_completed(file_path):
        atomic(download, file_path, ghcn_station_year_url(year, station_id))


def open_ghcnh_df(year_id_pair: tuple[int, str]) -> pd.DataFrame:
    year, station_id = year_id_pair

    file_path = ghcnh_file_path(year, station_id)
    if not file_path.is_file():
        return None

    try:
        with file_path.open("rb") as data:
            try:
                df = pd.read_csv(data, sep="|", low_memory=False, parse_dates=["DATE"])
            except pd.errors.EmptyDataError:
                return None

            # Find the elevation column regardless of case
            elev_col = None
            for c in df.columns:
                if c.lower() == "elevation":
                    elev_col = c
                    break

            cols = [
                "STATION",
                "DATE",
                "LATITUDE",
                "LONGITUDE",
                "temperature",
                "dew_point_temperature",
                "station_level_pressure",
                "wind_direction",
                "wind_speed",
                "precipitation",
                "precipitation_6_hour",
            ]

            if elev_col is not None:
                cols.insert(4, elev_col)  # Insert after LONGITUDE
                df = df[df[filtering_cols()].isin({math.nan, 1}).all(axis=1)][cols]
                if elev_col != "Elevation":
                    df = df.rename(columns={elev_col: "Elevation"})
            else:
                df = df[df[filtering_cols()].isin({math.nan, 1}).all(axis=1)][cols]

            # Ensure efficient correct data type for all columns.
            try:
                df["DATE"] = df["DATE"].to_numpy().astype("datetime64[s]")
            except ValueError:  # Sometimes happens with bad time stamps.
                return None

            df = df.rename(columns={"DATE": "time"})
            df["STATION"] = df["STATION"].to_numpy().astype("S11")

            # Numeric observation columns — force all to float32. Some GHCNh
            # PSV files store wind_direction etc as mixed int/float across
            # stations in a year; pandas reads the column as object dtype,
            # which xarray can't write to NetCDF. Explicit coercion sidesteps
            # this. pd.to_numeric(errors="coerce") turns unparseable cells
            # into NaN, then we downcast to float32.
            obs_cols = [
                "temperature",
                "dew_point_temperature",
                "station_level_pressure",
                "wind_direction",
                "wind_speed",
                "precipitation",
                "precipitation_6_hour",
            ]
            if elev_col is not None:
                obs_cols.append("Elevation")
            obs_cols += ["LATITUDE", "LONGITUDE"]
            for c in obs_cols:
                if c in df.columns:
                    df[c] = pd.to_numeric(df[c], errors="coerce").astype("float32")

            float64_cols = df.select_dtypes(include=["float64"]).columns
            df[float64_cols] = df[float64_cols].astype("float32")
            return df
    except Exception:
        return None


def main() -> None:
    global RAW_ROOT, PROCESSED_ROOT

    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--root",
        type=Path,
        default=ingest_dir(),
        help="Staging root; raw PSVs go to <root>/raw/ghcnh/, 6-hourly NetCDFs to "
        "<root>/processed/ghcnh/data/ (default: <data root>/ingest).",
    )
    p.add_argument(
        "--years",
        type=int,
        nargs=2,
        default=[2010, 2023],
        metavar=("FIRST", "LAST"),
        help="Inclusive year range.",
    )
    p.add_argument(
        "--num-processes",
        type=int,
        default=4,
        help="Workers for both the download and the parse phase.",
    )
    args = p.parse_args()
    RAW_ROOT = args.root / "raw" / "ghcnh"
    PROCESSED_ROOT = args.root / "processed" / "ghcnh"

    # Ensure that expected directories exist.
    RAW_ROOT.mkdir(parents=True, exist_ok=True)
    (PROCESSED_ROOT / "data").mkdir(parents=True, exist_ok=True)

    # Ensure that we have a copy of the station list.
    station_path = RAW_ROOT / "station_list.csv"
    if not atomic_completed(station_path):
        station_list_url = (
            "https://www.ncei.noaa.gov/oa/"
            "global-historical-climatology-network/hourly/doc/ghcnh-station-list.csv"
        )
        atomic(download, station_path, station_list_url)

    # Download and parse the station list, and extract the station IDs.
    with station_path.open("rb") as f:
        station_ids = pd.read_csv(f)["GHCN_ID"].to_numpy()

    # Define all year-station pairs.
    for year in range(args.years[0], args.years[1] + 1):
        year_station_pairs = [(year, s) for s in station_ids]

        # Iterate over all station IDs for this year, downloading all available data.
        print("Downloading raw data")
        parallel_foreach(
            f=download_ghcnh_station,
            init=lambda _: None,
            items=year_station_pairs,
            num_processes=args.num_processes,
        )

        # Load in all of the data, and turn into a single xarray.
        print("Loading + concat all data")
        with ProcessPoolExecutor(max_workers=args.num_processes) as pool:
            dfs = list(pool.map(open_ghcnh_df, year_station_pairs))
        df = pd.concat([df for df in dfs if df is not None])

        # Split into a single xarray per 6 hours, and write out to temporary storage.
        print("Writing data out to disk")
        grouper = pd.Grouper(key="time", freq="6h", closed="right", label="right")
        for t, gr in df.groupby(grouper):
            output_path = PROCESSED_ROOT / "data" / compute_file_name(t)
            if not atomic_completed(output_path):
                atomic(
                    write_and_flush_dataset,
                    output_path,
                    gr.to_xarray(),
                    engine="h5netcdf",
                )


if __name__ == "__main__":
    main()
