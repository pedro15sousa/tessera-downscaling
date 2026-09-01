"""Pick a hand-curated, TESSERA-pre-filtered probe set for a temporal
rollout experiment, and write it to ``probe_station_ids.json``.

Driven from the command line so the same script generates probe sets for
Norway (the paper's rollout) or any other bbox-defined region.

What the script does
--------------------

Filters the dataset's ``stations.csv`` down to those stations that all of:

1.  Lie in the requested ``--region``.
2.  Have ``spatial_split == --spatial-split`` (= "train" for the rollout
    experiment).
3.  Fall inside the lat/lon bbox.
4.  Optionally meet an elevation floor (``--elev-min``).
5.  Survive TESSERA + VAE-latent filtering — equivalently, their station_id
    has a *non-NaN* row in the VAE latents file. This is the same pre-filter
    intersection the dataset class applies at training time, so every
    station in the resulting probe set is guaranteed to actually contribute
    training observations (i.e. no "200 in the allowlist, only 11 survive
    the runtime filter" surprises like the station-count experiment had).

Writes a single JSON file:

    {
        "probe_station_ids":  [<sorted station_ids>],
        "n_probe":            <count>,
        "n_pool":             <count of region+spatial_split rows, pre-bbox>,
        "fraction_actual":    n_probe / n_pool,
        "selection_method":   "bbox_only" | "bbox_elev_floor",
        "bbox_lat":           [min, max],
        "bbox_lon":           [min, max],
        "elev_min_m":         <int> | null,
        "description":        <free-text>
    }

Refuses to overwrite an existing file (use ``--force`` to override) — the
same caching pattern that submit.sh uses for its sweep-point sidecars.

Example: build the Norway probe set of the paper
------------------------------------------------

    # Mainland Norway bbox; no elevation floor.
    uv run python scripts/experiments/pick_probe_set.py \\
        --dataset-dir       <data root>/datasets/dataset_timestamp_global \\
        --vae-latents-path  <data root>/processed/station_vectors/station_latents_lat16_grad0.5.npy \\
        --vae-latents-csv   <data root>/processed/tessera_global/station_list_filtered.csv \\
        --region            europe \\
        --spatial-split     train \\
        --bbox-lat 58 71 --bbox-lon 4 31 \\
        --out-dir scripts/experiments/snapshot_14y_eu_temporal_rollout_norway_lat16_mtpi

Pass ``--description "..."`` to override the auto-generated description.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--dataset-dir",
        type=Path,
        required=True,
        help="Path to the dataset directory containing stations.csv "
        "(e.g. <data root>/datasets/dataset_timestamp_global).",
    )
    p.add_argument(
        "--vae-latents-path",
        type=Path,
        required=True,
        help="Path to the VAE-latents .npy file. A station is considered "
        "TESSERA-valid iff its row in this array has no NaN.",
    )
    p.add_argument(
        "--vae-latents-csv",
        type=Path,
        required=True,
        help="CSV row-aligned with --vae-latents-path. Must contain a "
        "station_id column.",
    )
    p.add_argument(
        "--region",
        type=str,
        required=True,
        help="Dataset region name to filter by (e.g. 'europe'). Matched "
        "against the 'region' column of stations.csv.",
    )
    p.add_argument(
        "--spatial-split",
        type=str,
        default="train",
        choices=["train", "test", "all"],
        help="Source split for the probe pool. Default 'train'.",
    )
    p.add_argument(
        "--bbox-lat",
        type=float,
        nargs=2,
        metavar=("LAT_MIN", "LAT_MAX"),
        required=True,
        help="Latitude bounding box (degrees). Stations with latitude in "
        "[LAT_MIN, LAT_MAX] (inclusive) are kept.",
    )
    p.add_argument(
        "--bbox-lon",
        type=float,
        nargs=2,
        metavar=("LON_MIN", "LON_MAX"),
        required=True,
        help="Longitude bounding box (degrees). Stations with longitude in "
        "[LON_MIN, LON_MAX] (inclusive) are kept.",
    )
    p.add_argument(
        "--elev-min",
        type=float,
        default=None,
        help="Optional elevation floor in metres. Stations with elevation "
        "< this value are excluded. Omit for no elevation filter.",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Destination experiment folder. The file probe_station_ids.json "
        "will be written here. Folder must already exist (the script "
        "won't auto-create — that's the experiment author's call).",
    )
    p.add_argument(
        "--description",
        type=str,
        default=None,
        help="Optional free-text description for the JSON's 'description' "
        "field. Auto-generated from bbox/elev/region if omitted.",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Overwrite probe_station_ids.json if it already exists. "
        "Default: refuse and exit non-zero.",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------


def _load_stations(dataset_dir: Path) -> pd.DataFrame:
    csv_path = dataset_dir / "stations.csv"
    if not csv_path.exists():
        sys.exit(f"ERROR: stations.csv not found at {csv_path}.")
    stations = pd.read_csv(csv_path)
    stations["station_id"] = stations["station_id"].astype(str)
    required_cols = {
        "station_id",
        "region",
        "spatial_split",
        "latitude",
        "longitude",
        "elevation",
    }
    missing = required_cols - set(stations.columns)
    if missing:
        sys.exit(
            f"ERROR: stations.csv at {csv_path} is missing required "
            f"column(s): {sorted(missing)}."
        )
    return stations


def _load_vae_valid_ids(
    latents_path: Path,
    csv_path: Path,
) -> set[str]:
    """Return the set of station_ids with non-NaN VAE latent rows.

    Memory-maps the .npy so loading a multi-GB latents file doesn't cost
    RAM. The CSV is small.
    """
    if not latents_path.exists():
        sys.exit(f"ERROR: VAE latents .npy not found at {latents_path}.")
    if not csv_path.exists():
        sys.exit(f"ERROR: VAE latents CSV not found at {csv_path}.")
    csv = pd.read_csv(csv_path)
    csv["station_id"] = csv["station_id"].astype(str)
    if "station_id" not in csv.columns:
        sys.exit(f"ERROR: VAE latents CSV {csv_path} has no station_id column.")
    latents = np.load(str(latents_path), mmap_mode="r")
    if latents.shape[0] != len(csv):
        sys.exit(
            f"ERROR: VAE latents shape {latents.shape} is not row-aligned "
            f"with CSV ({len(csv)} rows). Re-check the --vae-latents-* "
            f"paths point at matched files."
        )
    valid_mask = ~np.isnan(latents).any(axis=1)
    return set(csv.loc[valid_mask, "station_id"].astype(str).tolist())


def _filter_probe_set(
    stations: pd.DataFrame,
    vae_valid_ids: set[str],
    region: str,
    spatial_split: str,
    bbox_lat: tuple[float, float],
    bbox_lon: tuple[float, float],
    elev_min: float | None,
) -> tuple[list[str], int]:
    """Apply the filter chain. Returns (sorted_station_ids, n_pool).

    n_pool is the count of region+spatial_split rows BEFORE the bbox/elev/VAE
    filters — matches the existing JSONs' n_pool semantics so fraction_actual
    is computed against the same reference pool as the legacy 5%-random
    experiment.
    """
    pool_mask = (stations["region"] == region) & (
        stations["spatial_split"] == spatial_split
        if spatial_split != "all"
        else stations["spatial_split"].isin(["train", "test"])
    )
    n_pool = int(pool_mask.sum())
    if n_pool == 0:
        sys.exit(
            f"ERROR: no stations match region={region!r}, "
            f"spatial_split={spatial_split!r}. Check the dataset."
        )

    lat_min, lat_max = bbox_lat
    lon_min, lon_max = bbox_lon
    mask = (
        pool_mask
        & stations["latitude"].between(lat_min, lat_max)
        & stations["longitude"].between(lon_min, lon_max)
    )
    n_post_bbox = int(mask.sum())
    print(f"  region+spatial_split pool:        {n_pool}")
    print(
        f"  after bbox lat=[{lat_min},{lat_max}] lon=[{lon_min},{lon_max}]:  "
        f"{n_post_bbox}"
    )

    if elev_min is not None:
        mask &= stations["elevation"] >= elev_min
        n_post_elev = int(mask.sum())
        print(f"  after elev_min={elev_min}m:                 {n_post_elev}")
    else:
        n_post_elev = n_post_bbox

    # TESSERA+VAE intersection: VAE-valid implies TESSERA-covered.
    candidates = stations.loc[mask, "station_id"].astype(str)
    surviving = candidates[candidates.isin(vae_valid_ids)]
    n_final = len(surviving)
    print(f"  after TESSERA+VAE filter:              {n_final}")

    if n_final == 0:
        sys.exit(
            "ERROR: zero stations remain after all filters. Widen the "
            "bbox, drop the elevation floor, or pick a region with better "
            "TESSERA coverage."
        )

    return sorted(surviving.tolist()), n_pool


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def _build_payload(
    probe_ids: list[str],
    n_pool: int,
    region: str,
    spatial_split: str,
    bbox_lat: tuple[float, float],
    bbox_lon: tuple[float, float],
    elev_min: float | None,
    description: str | None,
) -> dict:
    selection_method = "bbox_elev_floor" if elev_min is not None else "bbox_only"
    if description is None:
        bits = [
            f"All {region}/{spatial_split} stations inside bbox "
            f"lat∈[{bbox_lat[0]}, {bbox_lat[1]}], "
            f"lon∈[{bbox_lon[0]}, {bbox_lon[1]}]"
        ]
        if elev_min is not None:
            bits.append(f"with elevation ≥ {elev_min:g}m")
        bits.append(
            "that survive TESSERA+VAE filtering. Hand-picked probe set; "
            "deterministic in bbox/elev — no random sampling and no "
            "per-config seed."
        )
        description = ", ".join(bits[:1]) + " " + " ".join(bits[1:])
    return {
        "probe_station_ids": probe_ids,
        "n_probe": len(probe_ids),
        "n_pool": n_pool,
        "fraction_actual": len(probe_ids) / n_pool,
        "selection_method": selection_method,
        "bbox_lat": list(bbox_lat),
        "bbox_lon": list(bbox_lon),
        "elev_min_m": elev_min,
        "description": description,
    }


def main() -> None:
    args = _parse_args()
    out_dir: Path = args.out_dir
    if not out_dir.exists():
        sys.exit(
            f"ERROR: --out-dir {out_dir} does not exist. Create the "
            f"experiment folder first."
        )
    if not out_dir.is_dir():
        sys.exit(f"ERROR: --out-dir {out_dir} is not a directory.")
    out_path = out_dir / "probe_station_ids.json"
    if out_path.exists() and not args.force:
        sys.exit(
            f"ERROR: {out_path} already exists. Pass --force to overwrite, "
            f"or delete the file explicitly first."
        )

    print(f"Loading stations from {args.dataset_dir / 'stations.csv'}...")
    stations = _load_stations(args.dataset_dir)
    print(f"  {len(stations)} total rows in stations.csv")

    print(f"Loading VAE-valid station IDs from {args.vae_latents_path.name}...")
    vae_valid_ids = _load_vae_valid_ids(
        args.vae_latents_path,
        args.vae_latents_csv,
    )
    print(f"  {len(vae_valid_ids)} stations have non-NaN VAE latents.")

    print("Filtering...")
    probe_ids, n_pool = _filter_probe_set(
        stations=stations,
        vae_valid_ids=vae_valid_ids,
        region=args.region,
        spatial_split=args.spatial_split,
        bbox_lat=tuple(args.bbox_lat),
        bbox_lon=tuple(args.bbox_lon),
        elev_min=args.elev_min,
    )

    payload = _build_payload(
        probe_ids=probe_ids,
        n_pool=n_pool,
        region=args.region,
        spatial_split=args.spatial_split,
        bbox_lat=tuple(args.bbox_lat),
        bbox_lon=tuple(args.bbox_lon),
        elev_min=args.elev_min,
        description=args.description,
    )
    out_path.write_text(json.dumps(payload, indent=2))
    print()
    print(f"Wrote {out_path}")
    print(
        f"  n_probe={payload['n_probe']}, "
        f"n_pool={payload['n_pool']}, "
        f"fraction_actual={payload['fraction_actual']:.4f}, "
        f"selection_method={payload['selection_method']}"
    )


if __name__ == "__main__":
    main()
