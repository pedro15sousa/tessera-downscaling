"""Add an ``mtpi`` column to an already-built ``stations.csv`` in place.

Lets you enable the 3-feature (elevation, delta_elevation, mTPI) per-station
vector on existing datasets *without* re-running the full preprocessing
pipeline. It reuses the exact same join logic the preprocessor uses
(:func:`tessera_downscaling.preprocessing.helpers.lookup_station_mtpi`), so a
backfilled dataset is identical to one built fresh with ``--mtpi-csv``.

Once the ``mtpi`` column is present, train.py auto-detects it and trains with
n_elev_features=3 (unless ``--no-mtpi`` is passed); datasets without the column
keep serving the legacy 2-feature layout, so this is safe to run only on the
datasets you actually want to upgrade.

Inputs:
    --mtpi-csv    station_id,mtpi lookup from scripts/data/fetch_station_mtpi.py
                  (the paper's: <data root>/processed/station_mtpi.csv)
    --dataset-dir directory containing stations.csv  (OR --stations-csv <file>)

By default the target stations.csv is rewritten in place after saving a
``stations.csv.bak`` next to it; pass ``--no-backup`` to skip the backup or
``--output`` to write elsewhere instead of overwriting.

Usage (from the repo root):
    uv run python scripts/data/backfill_station_mtpi.py \
        --dataset-dir <data root>/dataset_timestamp_global \
        --mtpi-csv    <data root>/processed/station_mtpi.csv
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
from pathlib import Path

import pandas as pd

from tessera_downscaling.preprocessing.helpers import lookup_station_mtpi

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("backfill_station_mtpi")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--dataset-dir",
        type=Path,
        default=None,
        help="Dataset directory containing stations.csv.",
    )
    src.add_argument(
        "--stations-csv",
        type=Path,
        default=None,
        help="Path to a stations.csv directly.",
    )
    p.add_argument(
        "--mtpi-csv",
        type=Path,
        required=True,
        help="station_id,mtpi lookup CSV (from fetch_station_mtpi.py).",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Write here instead of overwriting the input.",
    )
    p.add_argument(
        "--no-backup",
        action="store_true",
        help="Do not write a .bak copy before overwriting in place.",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    stations_csv = (
        args.stations_csv
        if args.stations_csv is not None
        else args.dataset_dir / "stations.csv"
    )
    if not stations_csv.exists():
        logger.error("stations.csv not found at %s", stations_csv)
        return 1

    stations = pd.read_csv(stations_csv, dtype={"station_id": str})
    if "mtpi" in stations.columns:
        logger.warning(
            "stations.csv already has an `mtpi` column; it will be recomputed "
            "and overwritten from %s.",
            args.mtpi_csv,
        )

    # Same array-aligned join the preprocessors use (fills masked/missing with
    # 0.0, raises if the station_id formats clearly don't match).
    stations["mtpi"] = lookup_station_mtpi(stations, args.mtpi_csv)

    out_path = args.output if args.output is not None else stations_csv
    if out_path == stations_csv and not args.no_backup:
        backup = stations_csv.with_suffix(stations_csv.suffix + ".bak")
        shutil.copy2(stations_csv, backup)
        logger.info("Backed up original to %s", backup)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    stations.to_csv(out_path, index=False)
    logger.info("Wrote %d stations with mtpi column to %s", len(stations), out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
