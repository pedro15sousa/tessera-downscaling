"""Sample ALOS Global mTPI at weather-station locations via Google Earth Engine.

Produces the per-station multi-scale Topographic Position Index (mTPI) used as
the third per-station auxiliary feature in the ConvCNP downscaling baseline,
matching the (elevation, elevation-difference, mTPI) vector of Vaughan et al.
(2022, GMD). mTPI quantifies whether a location sits in a valley (negative) or
on a ridge (positive) relative to its surroundings, evaluated over multiple
neighbourhood scales.

Source dataset (the exact product used by the paper):
    CSP/ERGo/1_0/Global/ALOS_mTPI  (Theobald et al., 2015), band ``AVE``,
    ~270 m native resolution, units = metres.

Output:
    A CSV with columns ``station_id,mtpi`` (mTPI in metres), one row per input
    station. Feed it to the preprocessors via ``--mtpi-csv`` (which calls
    :func:`tessera_downscaling ... lookup_station_mtpi`) or to
    ``scripts/preprocessing/backfill_station_mtpi.py`` to add an ``mtpi`` column
    to an already-built ``stations.csv`` without re-preprocessing.

Authentication (one-time). On a headless HPC node ``gcloud`` is usually absent,
so the default ``earthengine authenticate`` fails with "gcloud command not
found". Use the notebook flow instead — it prints a URL you open in a browser
on any machine, then paste the returned token back:
    earthengine authenticate --auth_mode=notebook
    # writes ~/.config/earthengine/credentials
Alternatively, for a fully non-interactive run, set
GOOGLE_APPLICATION_CREDENTIALS to a service-account key JSON.
You must also have a Google Cloud project with the Earth Engine API enabled;
pass it via ``--gee-project`` (or set EARTHENGINE_PROJECT).

Usage (from repo root):
    uv run --project projects/dataprocessing python \
        projects/dataprocessing/scripts/gee/fetch_station_mtpi.py \
        --stations-csv  <path>/stations.csv \
        --output-csv    <path>/station_mtpi.csv \
        --gee-project   my-ee-project

The station list can be any CSV with id/lat/lon columns — point it at the
broadest list you have (e.g. the global GHCNh station list) so a single run
covers every dataset built from a subset of those stations.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

LOG_FORMAT = "%(asctime)s | %(levelname)s | %(message)s"
logger = logging.getLogger("fetch_station_mtpi")

# The ALOS Global mTPI image and its single data band.
MTPI_IMAGE_ID = "CSP/ERGo/1_0/Global/ALOS_mTPI"
MTPI_BAND = "AVE"
# Native resolution of the product, in metres. Sampling at this scale reads the
# product as published rather than resampling it.
MTPI_NATIVE_SCALE_M = 270.0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stations-csv", type=Path, required=True,
                   help="Input CSV of stations (needs id/lat/lon columns).")
    p.add_argument("--output-csv", type=Path, required=True,
                   help="Where to write the station_id,mtpi lookup CSV.")
    p.add_argument("--gee-project", type=str, default=None,
                   help="Google Cloud project with the Earth Engine API "
                        "enabled. Falls back to the EARTHENGINE_PROJECT env "
                        "var / your default EE project if omitted.")
    p.add_argument("--id-col", type=str, default="station_id")
    p.add_argument("--lat-col", type=str, default="latitude")
    p.add_argument("--lon-col", type=str, default="longitude")
    p.add_argument("--batch-size", type=int, default=2000,
                   help="Stations per Earth Engine reduceRegions request. "
                        "Smaller is safer against payload/compute limits.")
    p.add_argument("--scale", type=float, default=MTPI_NATIVE_SCALE_M,
                   help="Sampling scale in metres (default: native 270 m).")
    return p.parse_args()


def _resolve_columns(df: pd.DataFrame, args: argparse.Namespace) -> tuple[str, str, str]:
    """Resolve id/lat/lon column names, tolerating common upper-case variants."""
    def pick(requested: str, *fallbacks: str) -> str:
        for cand in (requested, *fallbacks):
            if cand in df.columns:
                return cand
        raise ValueError(
            f"None of {[requested, *fallbacks]} found in stations CSV columns "
            f"{list(df.columns)}."
        )
    id_col = pick(args.id_col, "station_id", "STATION", "STATION_ID", "id")
    lat_col = pick(args.lat_col, "latitude", "LATITUDE", "lat")
    lon_col = pick(args.lon_col, "longitude", "LONGITUDE", "lon")
    return id_col, lat_col, lon_col


def _chunks(seq: list, n: int):
    for i in range(0, len(seq), n):
        yield seq[i : i + n]


def fetch_mtpi(
    records: list[tuple[str, float, float]],
    *,
    scale: float,
    batch_size: int,
) -> dict[str, float]:
    """Sample mTPI for ``(station_id, lat, lon)`` records. Returns id -> mtpi.

    Stations whose pixel is masked (e.g. open ocean) are simply absent from the
    returned dict; the downstream merge fills those with the neutral value 0.0.
    """
    import ee  # imported here so --help works without GEE installed.

    img = ee.Image(MTPI_IMAGE_ID).select(MTPI_BAND)

    out: dict[str, float] = {}
    n_batches = (len(records) + batch_size - 1) // batch_size
    for bi, batch in enumerate(_chunks(records, batch_size), start=1):
        feats = [
            ee.Feature(
                ee.Geometry.Point([float(lon), float(lat)]),
                {"station_id": str(sid)},
            )
            for sid, lat, lon in batch
        ]
        fc = ee.FeatureCollection(feats)
        sampled = img.reduceRegions(
            collection=fc,
            reducer=ee.Reducer.first(),
            scale=scale,
            tileScale=4,
        )
        info = sampled.getInfo()
        got = 0
        for feat in info.get("features", []):
            props = feat.get("properties", {})
            sid = props.get("station_id")
            # reduceRegions names the output after the band; tolerate the
            # generic 'first' name too, just in case.
            val = props.get(MTPI_BAND, props.get("first"))
            if sid is not None and val is not None:
                out[str(sid)] = float(val)
                got += 1
        logger.info(
            "batch %d/%d: sampled %d/%d stations (cumulative %d)",
            bi, n_batches, got, len(batch), len(out),
        )
    return out


def main() -> int:
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
    args = parse_args()

    df = pd.read_csv(args.stations_csv, dtype={args.id_col: str})
    id_col, lat_col, lon_col = _resolve_columns(df, args)
    logger.info(
        "Loaded %d stations from %s (id=%s, lat=%s, lon=%s)",
        len(df), args.stations_csv, id_col, lat_col, lon_col,
    )

    # Drop rows with missing coordinates — they can't be sampled.
    coords_ok = df[lat_col].notna() & df[lon_col].notna()
    if not coords_ok.all():
        logger.warning("Dropping %d stations with missing lat/lon",
                       int((~coords_ok).sum()))
    df = df[coords_ok]

    records = [
        (str(sid), float(lat), float(lon))
        for sid, lat, lon in zip(df[id_col], df[lat_col], df[lon_col])
    ]

    try:
        import ee
    except ImportError:
        logger.error(
            "earthengine-api is not installed. Run "
            "`uv sync --project projects/dataprocessing` first."
        )
        return 1
    ee.Initialize(project=args.gee_project)
    logger.info("Earth Engine initialised (project=%s)", args.gee_project)

    mtpi_by_id = fetch_mtpi(
        records, scale=args.scale, batch_size=args.batch_size,
    )

    all_ids = [r[0] for r in records]
    mtpi = np.array([mtpi_by_id.get(sid, np.nan) for sid in all_ids],
                    dtype=np.float64)
    n_missing = int(np.isnan(mtpi).sum())
    logger.info(
        "Sampled mTPI for %d/%d stations (%d masked/missing). "
        "mean=%.1fm, range=[%.0f, %.0f]m",
        len(all_ids) - n_missing, len(all_ids), n_missing,
        np.nanmean(mtpi) if n_missing < len(mtpi) else float("nan"),
        np.nanmin(mtpi) if n_missing < len(mtpi) else float("nan"),
        np.nanmax(mtpi) if n_missing < len(mtpi) else float("nan"),
    )

    out_df = pd.DataFrame({"station_id": all_ids, "mtpi": mtpi})
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.output_csv, index=False)
    logger.info("Wrote %d rows to %s", len(out_df), args.output_csv)
    return 0


if __name__ == "__main__":
    sys.exit(main())
