"""Sample hand-crafted surface descriptors at station locations via Earth Engine.

Produces the per-station "extra descriptors" — a hand-crafted alternative to
the learned TESSERA/VAE land-surface representation — mirroring the surface
descriptors of Bakketun et al. (2026, arXiv:2607.02824), who feed SURFEX
physiography fields and topographic neighbourhood indices to the decoder of a
data-driven weather model. Two descriptor groups are sampled, each over the
neighbourhood scale that keeps the comparison it serves fair:

  SFX-equivalents (land cover / vegetation / soil), 320 m radius buffer —
  matching the 640 m footprint of the 64x64 x 10 m TESSERA patches, so the
  hand-crafted vector sees the same ground area as the learned embedding:
    forest_frac    ESA WorldCover tree cover (10) + mangroves (95)
    lowveg_frac    shrubland (20) + grassland (30) + herb. wetland (90)
                   + moss/lichen (100)
    crop_frac      cropland (40)
    built_frac     built-up (50)          [paper: Town — their largest gain]
    bare_frac      bare/sparse (60)
    snowice_frac   snow and ice (70)      [paper: Glacier]
    water_frac     permanent water (80); masked (open-ocean) pixels counted
                   as water
    tree_height    ETH Global Canopy Height 2020 (m), masked -> 0
    clay_frac      SoilGrids clay 0-5cm mean (g/kg)
    sand_frac      SoilGrids sand 0-5cm mean (g/kg)

  TOPO-equivalents (topographic neighbourhood context), 6.25 km radius —
  matching the paper's 12.5 km kernel diameter, from Copernicus GLO-30:
    elev_mean/elev_std/elev_min/elev_max   [paper: Avg/Max/Min ZS, ZS Std]
    slope          mean tan(slope), dimensionless
    dz_dn, dz_de   mean directional derivatives (tan(slope) decomposed by
                   aspect; sign follows EE's downslope-azimuth convention)
                   [paper: ZS SN/WE Derivative]

  Audit-only columns (excluded from the model feature vector downstream):
    wc_masked_frac   fraction of WorldCover-masked pixels in the SFX buffer
    dem_elev_320m    mean GLO-30 elevation over the SFX buffer, for
                     cross-checking against the station elevation column

The paper's Sx horizon angle is deliberately omitted (no cheap EE primitive,
and it was among their weakest descriptors); TPI is already covered by the
existing per-station mTPI feature (see fetch_station_mtpi.py).

Output:
    A CSV keyed by ``station_id`` with one column per descriptor. Rows are
    appended batch-by-batch, so an interrupted run can be resumed with
    ``--resume`` (already-fetched stations are skipped). Feed the CSV to
    ``scripts/data/build_extra_descriptors.py`` to produce the row-aligned
    ``extra_descriptors.npy`` for training. The paper's file is
    ``processed/station_vectors/station_extra_descriptors.csv`` under the data root.

Authentication (one-time) is identical to fetch_station_mtpi.py:
    earthengine authenticate --auth_mode=notebook
Pass a Google Cloud project via ``--gee-project`` / EARTHENGINE_PROJECT, or
rely on the project stored in ~/.config/earthengine/credentials.

Usage (from the repo root; needs the ``ingest`` extra for earthengine-api):
    uv run python scripts/data/fetch_station_extra_descriptors.py \
        --stations-csv <data root>/processed/tessera_global/station_list_filtered.csv \
        --output-csv   <data root>/processed/station_vectors/station_extra_descriptors.csv

Point ``--stations-csv`` at the broadest station list available (the global
TESSERA station list above) so a single run covers every regional dataset.
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
import time
from pathlib import Path

import pandas as pd

LOG_FORMAT = "%(asctime)s | %(levelname)s | %(message)s"
logger = logging.getLogger("fetch_station_extra_descriptors")

# --- Source datasets --------------------------------------------------------
WORLDCOVER_ID = "ESA/WorldCover/v200"  # 10 m, single 2021 image
CANOPY_ID = "users/nlang/ETH_GlobalCanopyHeight_2020_10m_v1"  # 10 m
SOILGRIDS_CLAY_ID = "projects/soilgrids-isric/clay_mean"  # 250 m, g/kg
SOILGRIDS_SAND_ID = "projects/soilgrids-isric/sand_mean"  # 250 m, g/kg
GLO30_ID = "COPERNICUS/DEM/GLO30"  # 30 m ImageCollection

# --- Neighbourhood scales ---------------------------------------------------
# SFX group: 320 m radius = 640 m footprint = the 64x64 x 10 m TESSERA patch,
# so hand-crafted and learned representations see the same ground area.
SFX_BUFFER_M = 320.0
SFX_SCALE_M = 10.0
# TOPO group: 6.25 km radius = the paper's 12.5 km kernel diameter. Reduced at
# 250 m — far finer than the paper's 2.5 km grid, cheap enough for 39k points.
TOPO_BUFFER_M = 6250.0
TOPO_SCALE_M = 250.0

SFX_COLUMNS = [
    "forest_frac",
    "lowveg_frac",
    "crop_frac",
    "built_frac",
    "bare_frac",
    "snowice_frac",
    "water_frac",
    "tree_height",
    "clay_frac",
    "sand_frac",
    "wc_masked_frac",
    "dem_elev_320m",
]
TOPO_COLUMNS = [
    "elev_mean",
    "elev_std",
    "elev_min",
    "elev_max",
    "slope",
    "dz_dn",
    "dz_de",
]
ALL_COLUMNS = ["station_id", *SFX_COLUMNS, *TOPO_COLUMNS]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--stations-csv",
        type=Path,
        required=True,
        help="Input CSV of stations (needs id/lat/lon columns).",
    )
    p.add_argument(
        "--output-csv",
        type=Path,
        required=True,
        help="Where to write the station_id-keyed descriptor CSV.",
    )
    p.add_argument(
        "--gee-project",
        type=str,
        default=None,
        help="Google Cloud project with the Earth Engine API "
        "enabled. Falls back to EARTHENGINE_PROJECT / the "
        "project stored in your EE credentials if omitted.",
    )
    p.add_argument("--id-col", type=str, default="station_id")
    p.add_argument("--lat-col", type=str, default="latitude")
    p.add_argument("--lon-col", type=str, default="longitude")
    p.add_argument(
        "--batch-size",
        type=int,
        default=500,
        help="Stations per reduceRegions request. Heavier per "
        "station than the mTPI fetch, so smaller default.",
    )
    p.add_argument(
        "--resume",
        action="store_true",
        help="Skip stations already present in --output-csv.",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only fetch the first N (remaining) stations. For smoke tests.",
    )
    p.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Retries per batch on transient EE errors.",
    )
    return p.parse_args()


def _resolve_columns(
    df: pd.DataFrame, args: argparse.Namespace
) -> tuple[str, str, str]:
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


def build_sfx_image():
    """Multiband image of the SFX-equivalent descriptors (mean-reducible)."""
    import ee

    wc = ee.ImageCollection(WORLDCOVER_ID).first().select("Map")

    def frac(class_values: list[int], fill: int) -> ee.Image:
        """Boolean image for a set of WorldCover classes; masked px -> fill.

        WorldCover masks open ocean, so filling water with 1 (and every other
        class with 0) counts off-coast pixels as water — the right reading for
        coastal/island stations, which are exactly where the paper found the
        land/water descriptors to matter.
        """
        img = wc.eq(class_values[0])
        for v in class_values[1:]:
            img = img.Or(wc.eq(v))
        return img.unmask(fill)

    canopy = ee.Image(CANOPY_ID).select("b1").unmask(0)  # masked = no trees
    clay = ee.Image(SOILGRIDS_CLAY_ID).select("clay_0-5cm_mean")
    sand = ee.Image(SOILGRIDS_SAND_ID).select("sand_0-5cm_mean")
    dem = ee.ImageCollection(GLO30_ID).select("DEM").mosaic()

    return (
        frac([10, 95], 0)
        .rename("forest_frac")
        .addBands(frac([20, 30, 90, 100], 0).rename("lowveg_frac"))
        .addBands(frac([40], 0).rename("crop_frac"))
        .addBands(frac([50], 0).rename("built_frac"))
        .addBands(frac([60], 0).rename("bare_frac"))
        .addBands(frac([70], 0).rename("snowice_frac"))
        .addBands(frac([80], 1).rename("water_frac"))
        .addBands(canopy.rename("tree_height"))
        .addBands(clay.rename("clay_frac"))
        .addBands(sand.rename("sand_frac"))
        .addBands(wc.mask().Not().unmask(1).rename("wc_masked_frac"))
        .addBands(dem.rename("dem_elev_320m"))
    )


def build_topo_image():
    """Multiband image of per-pixel topography values.

    Buffer means of the per-pixel directional derivatives approximate the
    derivative of the neighbourhood-smoothed elevation field, i.e. the
    paper's ZS SN/WE derivatives; elev stats come from a combined reducer.
    """
    import ee

    # mosaic() carries a default 1-degree projection, and ee.Terrain's
    # neighbourhood ops compute in the INPUT projection (not the request
    # scale) — without an explicit projection, slopes come out ~100x too
    # small and masked near coasts. Pin 90 m before any gradient op.
    dem = (
        ee.ImageCollection(GLO30_ID)
        .select("DEM")
        .mosaic()
        .setDefaultProjection("EPSG:4326", None, 90)
    )
    slope_deg = ee.Terrain.slope(dem)
    aspect_rad = ee.Terrain.aspect(dem).multiply(math.pi / 180.0)
    slope_tan = slope_deg.multiply(math.pi / 180.0).tan()

    return (
        dem.rename("elev")
        .addBands(slope_tan.rename("slope"))
        .addBands(slope_tan.multiply(aspect_rad.cos()).rename("dz_dn"))
        .addBands(slope_tan.multiply(aspect_rad.sin()).rename("dz_de"))
    )


def _reduce_batch(image, feats, reducer, scale: float) -> dict[str, dict]:
    """ReduceRegions one batch; returns station_id -> {output_name: value}."""
    import ee

    fc = ee.FeatureCollection(feats)
    reduced = image.reduceRegions(
        collection=fc,
        reducer=reducer,
        scale=scale,
        tileScale=4,
    )
    out: dict[str, dict] = {}
    for feat in reduced.getInfo().get("features", []):
        props = feat.get("properties", {})
        sid = props.pop("station_id", None)
        if sid is not None:
            out[str(sid)] = props
    return out


def fetch_batch(
    records: list[tuple[str, float, float]], sfx_image, topo_image
) -> pd.DataFrame:
    """Fetch both descriptor groups for one batch of (id, lat, lon)."""
    import ee

    def points(buffer_m: float):
        return [
            ee.Feature(
                ee.Geometry.Point([float(lon), float(lat)]).buffer(buffer_m),
                {"station_id": str(sid)},
            )
            for sid, lat, lon in records
        ]

    sfx = _reduce_batch(
        sfx_image,
        points(SFX_BUFFER_M),
        ee.Reducer.mean(),
        SFX_SCALE_M,
    )
    # Combined reducer outputs '<band>_<stat>' per band; unused combinations
    # (e.g. slope_min) are computed server-side but simply not read out.
    topo_reducer = (
        ee.Reducer.mean()
        .combine(ee.Reducer.stdDev(), sharedInputs=True)
        .combine(ee.Reducer.minMax(), sharedInputs=True)
    )
    topo = _reduce_batch(
        topo_image,
        points(TOPO_BUFFER_M),
        topo_reducer,
        TOPO_SCALE_M,
    )

    rows = []
    for sid, _, _ in records:
        sid = str(sid)
        s, t = sfx.get(sid, {}), topo.get(sid, {})
        rows.append(
            {
                "station_id": sid,
                **{c: s.get(c) for c in SFX_COLUMNS},
                "elev_mean": t.get("elev_mean"),
                "elev_std": t.get("elev_stdDev"),
                "elev_min": t.get("elev_min"),
                "elev_max": t.get("elev_max"),
                "slope": t.get("slope_mean"),
                "dz_dn": t.get("dz_dn_mean"),
                "dz_de": t.get("dz_de_mean"),
            }
        )
    return pd.DataFrame(rows, columns=ALL_COLUMNS)


def fetch_with_retries(
    records: list[tuple[str, float, float]],
    sfx_image,
    topo_image,
    max_retries: int,
) -> tuple[pd.DataFrame, list[str]]:
    """Fetch a batch, retrying and recursively halving on persistent failure.

    Earth Engine occasionally times out server-side on heavy sub-batches
    (observed: Antarctic stations, where polar DEM reprojection is slow).
    Halving isolates the expensive stations instead of killing the run.
    Returns ``(fetched_rows, failed_station_ids)`` — stations that still
    fail at the minimum batch size are skipped and reported, ending up as
    absent -> NaN rows -> dropped at load time, the same semantics as a
    station the VAE couldn't encode.
    """
    for attempt in range(1, max_retries + 1):
        try:
            return fetch_batch(records, sfx_image, topo_image), []
        except Exception as exc:  # ee.EEException and transport errors
            if attempt < max_retries:
                wait = 30 * attempt
                logger.warning(
                    "sub-batch of %d: attempt %d failed (%s); retrying in %ds",
                    len(records),
                    attempt,
                    exc,
                    wait,
                )
                time.sleep(wait)
            elif len(records) >= 8:
                logger.warning(
                    "sub-batch of %d: %d attempts failed (%s); splitting",
                    len(records),
                    max_retries,
                    exc,
                )
                mid = len(records) // 2
                df1, f1 = fetch_with_retries(
                    records[:mid], sfx_image, topo_image, max_retries
                )
                df2, f2 = fetch_with_retries(
                    records[mid:], sfx_image, topo_image, max_retries
                )
                return pd.concat([df1, df2], ignore_index=True), f1 + f2
            else:
                failed = [sid for sid, _, _ in records]
                logger.error(
                    "giving up on %d stations after exhausting retries and "
                    "splits (%s): %s",
                    len(records),
                    exc,
                    failed,
                )
                return pd.DataFrame(columns=ALL_COLUMNS), failed
    raise AssertionError("unreachable")


def main() -> int:
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
    args = parse_args()

    df = pd.read_csv(args.stations_csv, dtype={args.id_col: str})
    id_col, lat_col, lon_col = _resolve_columns(df, args)
    logger.info(
        "Loaded %d stations from %s (id=%s, lat=%s, lon=%s)",
        len(df),
        args.stations_csv,
        id_col,
        lat_col,
        lon_col,
    )

    coords_ok = df[lat_col].notna() & df[lon_col].notna()
    if not coords_ok.all():
        logger.warning(
            "Dropping %d stations with missing lat/lon", int((~coords_ok).sum())
        )
    df = df[coords_ok]

    records = [
        (str(sid), float(lat), float(lon))
        for sid, lat, lon in zip(df[id_col], df[lat_col], df[lon_col], strict=False)
    ]

    done_ids: set[str] = set()
    if args.output_csv.exists():
        if args.resume:
            done_ids = set(
                pd.read_csv(args.output_csv, usecols=["station_id"], dtype=str)[
                    "station_id"
                ]
            )
            logger.info("Resuming: %d stations already fetched", len(done_ids))
        else:
            logger.error(
                "%s already exists. Pass --resume to continue it, or remove "
                "it to start over.",
                args.output_csv,
            )
            return 1
    records = [r for r in records if r[0] not in done_ids]
    if args.limit is not None:
        records = records[: args.limit]
    if not records:
        logger.info("Nothing to fetch.")
        return 0

    try:
        import ee
    except ImportError:
        logger.error("earthengine-api is not installed in this environment.")
        return 1
    ee.Initialize(project=args.gee_project)
    logger.info("Earth Engine initialised (project=%s)", args.gee_project)

    sfx_image = build_sfx_image()
    topo_image = build_topo_image()

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    write_header = not args.output_csv.exists()
    n_batches = (len(records) + args.batch_size - 1) // args.batch_size
    t0 = time.time()
    n_written = 0
    all_failed: list[str] = []
    for bi, batch in enumerate(_chunks(records, args.batch_size), start=1):
        batch_df, failed = fetch_with_retries(
            batch,
            sfx_image,
            topo_image,
            args.max_retries,
        )
        all_failed.extend(failed)
        if len(batch_df):
            batch_df.to_csv(args.output_csv, mode="a", header=write_header, index=False)
            write_header = False
        n_written += len(batch_df)
        n_missing_sfx = int(batch_df["forest_frac"].isna().sum())
        elapsed = time.time() - t0
        logger.info(
            "batch %d/%d: wrote %d stations (%d missing SFX values, "
            "%d failed) — %.0fs elapsed, ~%.0fs remaining",
            bi,
            n_batches,
            len(batch_df),
            n_missing_sfx,
            len(failed),
            elapsed,
            elapsed / bi * (n_batches - bi),
        )

    if all_failed:
        failed_path = args.output_csv.with_suffix(".failed_ids.txt")
        failed_path.write_text("\n".join(all_failed) + "\n")
        logger.warning(
            "%d stations could not be fetched (ids in %s); they will be "
            "NaN rows in the descriptor npy and dropped at load time. "
            "Retry them later with --resume.",
            len(all_failed),
            failed_path,
        )
    logger.info(
        "Done: %d stations appended to %s (total on disk: %d)",
        n_written,
        args.output_csv,
        len(done_ids) + n_written,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
