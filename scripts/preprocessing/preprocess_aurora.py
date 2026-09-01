"""Stage 2 of the Aurora-context pipeline: forecasts -> downscaling datasets.

Consumes the *per-region, pre-cropped* Aurora forecasts written by
``scripts/aurora/generate_aurora_forecasts.py`` (Stage 1 now crops each frame to
the regions of interest, writing ``lead{L}h/<region>/processed/...``) and
produces, for each lead, a ``dataset_timestamp_aurora_lead{L}h`` tree that
mirrors the ``multi_region_snapshot_v1`` layout of ``dataset_timestamp_global``
-- so the existing dataset/eval code reads it with no changes.

Key points, all handled here:
  * 19 dynamic channels (Aurora has no precipitation; we drop
    ``total_precipitation_sum`` -> the 19-channel-trained model expects exactly
    this set, in this order). The staging keeps all 13 pressure levels; we select
    the 3 downscaling levels (500/700/850) when stacking.
  * Splits: ``--split`` selects which timestamps to build (train / val / trainval
    / test / all), using the same lexicographic rule as ``episodes_for_split``.
    ``all`` (the default) builds the full per-lead dataset (train+val from the
    generation run, test from the migration) so downstream episode-splitting just
    works; partial splits get a ``_<split>`` suffix so they don't clobber it.
  * Grid: Stage 1 already cropped each frame to the region grid (with the
    longitude roll), so there is NO crop here -- we only ASSERT the staged grid
    is bit-identical to the region's reference grid from
    ``dataset_timestamp_global`` (catches a wrong staging dir or grid change),
    then stack.
  * Store-only regions: a region with no scaffolding in the global dataset is
    skipped for dataset building with a warning -- its staging is left in place.
  * Targets / station list / GHCNh / normalisation stats are taken from the
    global dataset verbatim (stats with precip's entry removed), so train (ERA5,
    precip dropped) and Aurora-context share identical normalisation.

Usage (from the repo root; defaults resolve under the data root):
    uv run python scripts/preprocessing/preprocess_aurora.py \
        --leads 6 24 72 --split all --regions europe east_asia
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

import numpy as np

from tessera_downscaling.paths import data_root, dataset_dir, ingest_dir
from tessera_downscaling.preprocessing.helpers import (
    ATMOS_VARS,
    PRESSURE_LEVELS,
    SURFACE_VARS,
    era5_snapshot_channel_names,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("preprocess_aurora")

PRECIP_CHANNEL_NAME = "total_precipitation_sum"


# --------------------------------------------------------------------------- #
# Split selection (mirror the dataset split exactly).
# --------------------------------------------------------------------------- #


def load_timestamps(global_dataset: Path, split: str) -> list[str]:
    """Valid-times for a split, using the same lexicographic rule as
    ``episodes_for_split`` so the Aurora dataset matches the ERA5 splits exactly:

        train    : s <= train_end            val : train_end < s <= val_end
        test     : s >  val_end              trainval : s <= val_end
        all      : every valid timestamp

    Counts are checked against metadata where available (train/val/test).
    """
    meta = json.loads((global_dataset / "metadata.json").read_text())
    ts = meta["temporal_split"]
    te, ve = ts["train_end"], ts["val_end"]
    vt = meta["valid_timestamps"]
    if split == "train":
        sel = [s for s in vt if s <= te]
    elif split == "val":
        sel = [s for s in vt if te < s <= ve]
    elif split == "test":
        sel = [s for s in vt if s > ve]
    elif split == "trainval":
        sel = [s for s in vt if s <= ve]
    elif split == "all":
        sel = list(vt)
    else:
        raise ValueError(f"Unknown split: {split!r}")
    out = sorted(sel)
    expected = {
        "train": ts.get("n_train_timestamps"),
        "val": ts.get("n_val_timestamps"),
        "test": ts.get("n_test_timestamps"),
    }.get(split)
    if expected is not None and len(out) != expected:
        raise ValueError(
            f"Derived {len(out)} {split} timestamps, metadata says {expected}."
        )
    return out


# --------------------------------------------------------------------------- #
# 19-channel Aurora snapshot aggregator. Stage 1 already cropped each frame to
# the region grid, so this is a straight stack (no roll/crop) of
# surface(4) + atmos(5) x downscaling-levels(3), precip excluded.
# --------------------------------------------------------------------------- #


def aggregate_aurora_snapshot(
    source_root: Path, date_str: str, hour: int
) -> np.ndarray | None:
    """Stack a 19-channel snapshot from PRE-CROPPED per-region Aurora staging.

    Channel order == era5_snapshot_channel_names() with precip removed. The
    staged fields are already on the region grid (Stage 1 cropped them), so there
    is no roll/crop here. Returns None if any required file is missing.
    """
    import xarray as xr

    surface_vars = [v for v in SURFACE_VARS if v != "total_precipitation_6hr"]  # 4 vars
    channels: list[np.ndarray] = []

    for var in surface_vars:
        path = (
            source_root
            / f"era5_wb2_quarter_{var}"
            / "data"
            / f"{date_str}-{hour:02d}.nc"
        )
        if not path.exists():
            return None
        ds = xr.open_dataset(path)
        channels.append(ds[list(ds.data_vars)[0]].values)  # (H, W), already cropped
        ds.close()

    for var in ATMOS_VARS:
        path = (
            source_root
            / f"era5_wb2_quarter_{var}"
            / "data"
            / f"{date_str}-{hour:02d}.nc"
        )
        if not path.exists():
            return None
        ds = xr.open_dataset(path)
        data = ds[list(ds.data_vars)[0]].values  # (n_levels, H, W), already cropped
        levels = list(ds.level.values)
        for (
            level
        ) in PRESSURE_LEVELS:  # select the 3 downscaling levels from the 13 staged
            channels.append(data[levels.index(level)])
        ds.close()

    return np.stack(channels, axis=0).astype(np.float32)


# --------------------------------------------------------------------------- #
# Normalisation-stats: drop the precip entry from the per-region ERA5 stats.
# --------------------------------------------------------------------------- #


def drop_precip_from_stats(src_npz: Path, dst_npz: Path, precip_idx: int) -> None:
    """Copy a normalisation_stats .npz, deleting the precip channel (index
    `precip_idx` in the dynamic block). The array layout is
    [dynamic..., (static...), lat, lon]; precip lives within the dynamic block,
    so a single np.delete at precip_idx is correct.
    """
    d = np.load(src_npz)
    np.savez(
        dst_npz,
        era5_mean=np.delete(d["era5_mean"], precip_idx),
        era5_std=np.delete(d["era5_std"], precip_idx),
    )


# --------------------------------------------------------------------------- #
# Per-region build.
# --------------------------------------------------------------------------- #

_SNAP: dict = {}  # per-worker shared read-only state (set by _snap_winit)


def _snap_winit(source_root: Path, snap_dir: Path) -> None:
    _SNAP["source_root"] = source_root
    _SNAP["snap_dir"] = snap_dir


def _build_one_snapshot(ts: str) -> tuple[int, int]:
    """Build + save one region/ts snapshot. (written, skipped). Resumable via exists()."""
    out_path = _SNAP["snap_dir"] / f"{ts}.npy"
    if out_path.exists():
        return 0, 1
    date_str, hour = ts.rsplit("-", 1)
    snap = aggregate_aurora_snapshot(_SNAP["source_root"], date_str, int(hour))
    if snap is None:
        raise FileNotFoundError(
            f"Missing Aurora forecast for {ts} under {_SNAP['source_root']}"
        )
    np.save(out_path, snap)
    return 1, 0


def build_region(
    region: str,
    aurora_staging_root: Path,
    lead: int,
    global_dataset: Path,
    out_dataset: Path,
    split_times: list[str],
    precip_idx: int,
    workers: int = 1,
) -> tuple[int, int]:
    """Assemble one region's 19-channel snapshots from its pre-cropped staging.
    Returns (written, skipped) snapshot counts. Caller guarantees the region has
    scaffolding in the global dataset.
    """
    import xarray as xr

    global_region = global_dataset / "regions" / region
    out_region = out_dataset / "regions" / region
    (out_region / "era5_snapshot").mkdir(parents=True, exist_ok=True)

    ref_lats = np.load(global_region / "lats.npy")
    ref_lons = np.load(global_region / "lons.npy")

    # Per-region staging written by Stage 1 (already cropped to this region).
    source_root = aurora_staging_root / f"lead{lead}h" / region / "processed"
    probe = (
        source_root
        / "era5_wb2_quarter_2m_temperature"
        / "data"
        / f"{split_times[0]}.nc"
    )
    if not probe.exists():
        raise FileNotFoundError(
            f"No per-region Aurora staging at {probe}; run Stage 1 (cropped) for this lead/region first."
        )
    ads = xr.open_dataset(probe)
    a_lats, a_lons = ads["latitude"].values, ads["longitude"].values
    ads.close()

    # Grid guard: the staged frame must already BE this region's reference grid.
    # (Stage 1 asserted this at write time; re-checking here catches a wrong
    # --aurora-staging-root or a grid change before we write thousands of npys.)
    if not (
        a_lats.shape == ref_lats.shape
        and a_lons.shape == ref_lons.shape
        and np.allclose(a_lats, ref_lats, atol=1e-4)
        and np.allclose(a_lons, ref_lons, atol=1e-4)
    ):
        raise AssertionError(
            f"Region '{region}': staged grid {a_lats.shape}x{a_lons.shape} != dataset reference "
            f"{ref_lats.shape}x{ref_lons.shape}. Wrong staging dir, or the crop changed -- refusing."
        )

    # Copy reference scaffolding verbatim (identical grid + static fields).
    for fname in ("lats.npy", "lons.npy", "static_fields.npy", "region_metadata.json"):
        (out_region / fname).write_bytes((global_region / fname).read_bytes())

    # 19-channel stats from the global per-region stats (precip dropped).
    for stats_name in ("normalisation_stats.npz", "normalisation_stats_no_static.npz"):
        src = global_region / stats_name
        if src.exists():
            drop_precip_from_stats(src, out_region / stats_name, precip_idx)

    snap_dir = out_region / "era5_snapshot"
    n = len(split_times)
    written = skipped = 0
    import time

    t0 = time.time()

    def _tick(i):
        if i % 500 == 0 or i == n:
            rate = i / max(time.time() - t0, 1e-9)
            logger.info(
                f"[lead {lead}h] {region}: {i}/{n} | wrote {written} skipped {skipped} | {rate:.0f} ts/s"
            )

    if workers <= 1:
        _snap_winit(source_root, snap_dir)
        for i, ts in enumerate(split_times, 1):
            w, s = _build_one_snapshot(ts)
            written += w
            skipped += s
            _tick(i)
    else:
        import multiprocessing as mp

        with mp.Pool(
            workers, initializer=_snap_winit, initargs=(source_root, snap_dir)
        ) as pool:
            for i, (w, s) in enumerate(
                pool.imap_unordered(_build_one_snapshot, split_times, chunksize=8), 1
            ):
                written += w
                skipped += s
                _tick(i)
    return written, skipped


# --------------------------------------------------------------------------- #
# Top-level per-lead build.
# --------------------------------------------------------------------------- #


def build_lead(
    lead: int,
    aurora_staging_root: Path,
    global_dataset: Path,
    output_root: Path,
    regions: list[str],
    split: str,
    symlink_ghcnh: bool,
    workers: int = 1,
) -> None:
    full_channels = era5_snapshot_channel_names()
    precip_idx = full_channels.index(PRECIP_CHANNEL_NAME)
    channels_19 = [c for c in full_channels if c != PRECIP_CHANNEL_NAME]
    assert len(channels_19) == 19, f"Expected 19 channels, got {len(channels_19)}"

    global_meta = json.loads((global_dataset / "metadata.json").read_text())

    # Only regions with scaffolding in the global dataset can be built; others
    # (e.g. uk, staged for a later project) are skipped with a warning.
    buildable = []
    for r in regions:
        if (global_dataset / "regions" / r / "lats.npy").exists():
            buildable.append(r)
        else:
            logger.warning(
                f"[lead {lead}h] region '{r}' has no scaffolding in {global_dataset} "
                f"-- skipping dataset build (store-only); its staging is untouched."
            )
    if not buildable:
        logger.warning(
            f"[lead {lead}h] no buildable regions in {regions}; nothing to do."
        )
        return

    # Canonical full dataset for split=all; partial splits get a suffix so they
    # don't clobber it.
    suffix = "" if split == "all" else f"_{split}"
    out_dataset = output_root / f"dataset_timestamp_aurora_lead{lead}h{suffix}"
    out_dataset.mkdir(parents=True, exist_ok=True)

    split_times = load_timestamps(global_dataset, split)
    logger.info(
        f"[lead {lead}h] split='{split}': {len(split_times)} timestamps, regions={buildable}"
    )

    for region in buildable:
        w, s = build_region(
            region,
            aurora_staging_root,
            lead,
            global_dataset,
            out_dataset,
            split_times,
            precip_idx,
            workers,
        )
        logger.info(
            f"[lead {lead}h] region '{region}': wrote {w}, skipped {s} snapshots"
        )

    # Top-level shared artefacts (targets + station list are unchanged).
    (out_dataset / "stations.csv").write_bytes(
        (global_dataset / "stations.csv").read_bytes()
    )
    (out_dataset / "valid_station_indices.npy").write_bytes(
        (global_dataset / "valid_station_indices.npy").read_bytes()
    )
    # GHCNh: identical real observations -> symlink (default) or copy.
    src_ghcnh = global_dataset / "ghcnh_snapshot"
    dst_ghcnh = out_dataset / "ghcnh_snapshot"
    if not dst_ghcnh.exists():
        if symlink_ghcnh:
            # Link relatively (../dataset_timestamp_global/ghcnh_snapshot in
            # the standard layout) so the tree survives being copied or moved
            # to a different root; an absolute target would dangle there.
            target = Path(os.path.relpath(src_ghcnh, dst_ghcnh.parent))
            dst_ghcnh.symlink_to(target, target_is_directory=True)
        else:
            dst_ghcnh.mkdir()
            for ts in split_times:
                f = src_ghcnh / f"{ts}.npz"
                if f.exists():
                    (dst_ghcnh / f.name).write_bytes(f.read_bytes())

    # Metadata: mirror global, but 19 channels, this split's timestamps, tagged Aurora.
    te, ve = (
        global_meta["temporal_split"]["train_end"],
        global_meta["temporal_split"]["val_end"],
    )
    n_train = sum(1 for s in split_times if s <= te)
    n_val = sum(1 for s in split_times if te < s <= ve)
    n_test = sum(1 for s in split_times if s > ve)
    metadata = {
        "layout_version": "multi_region_snapshot_v1",
        "source": "aurora",
        "aurora_model": "aurora-0.25-pretrained",
        "lead_hours": lead,
        "split_coverage": split,
        "cadence": global_meta.get("cadence", "6h"),
        "hours_per_day": global_meta.get("hours_per_day", [0, 6, 12, 18]),
        "era5_dynamic_channels": channels_19,
        "n_dynamic_channels": len(channels_19),
        "pressure_levels": global_meta.get("pressure_levels", [500, 700, 850]),
        "regions": {r: global_meta["regions"][r] for r in buildable},
        "valid_timestamps": split_times,
        "valid_dates": split_times,
        "temporal_split": {
            "train_end": te,
            "val_end": ve,
            "n_train_timestamps": n_train,
            "n_val_timestamps": n_val,
            "n_test_timestamps": n_test,
        },
        "spatial_split": global_meta.get("spatial_split", {}),
        "elevation_normalisation": global_meta.get(
            "elevation_normalisation", "raw_metres"
        ),
        "derived_from": str(global_dataset),
    }
    (out_dataset / "metadata.json").write_text(json.dumps(metadata, indent=2))
    logger.info(f"[lead {lead}h] wrote dataset to {out_dataset}")


def parse_args():
    p = argparse.ArgumentParser(
        description="Stage 2: per-region Aurora forecasts -> downscaling datasets."
    )
    p.add_argument(
        "--global-dataset",
        type=Path,
        default=dataset_dir(),
        help="dataset_timestamp_global directory (default: under the data root)",
    )
    p.add_argument(
        "--aurora-staging-root",
        type=Path,
        default=ingest_dir("aurora"),
        help="Contains lead{L}h/<region>/processed (default: <data root>/ingest/aurora)",
    )
    p.add_argument(
        "--output-root",
        type=Path,
        default=data_root() / "datasets",
        help="Where dataset_timestamp_aurora_lead{L}h dirs are written "
        "(default: <data root>/datasets)",
    )
    p.add_argument("--leads", type=int, nargs="+", default=[6, 24, 72])
    p.add_argument(
        "--split",
        choices=["train", "val", "trainval", "test", "all"],
        default="all",
        help="Which timestamps to build. Default 'all' (the full per-lead dataset); "
        "partial splits get a _<split> suffix so they don't clobber it.",
    )
    p.add_argument(
        "--regions",
        type=str,
        nargs="+",
        default=["europe", "east_asia"],
        help="Dataset regions to build (regions without global-dataset scaffolding are skipped).",
    )
    p.add_argument(
        "--copy-ghcnh",
        action="store_true",
        help="Copy GHCNh files instead of symlinking the dir",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Parallel worker processes over timestamps within each region/lead (default 8). 1 = serial.",
    )
    return p.parse_args()


def main():
    args = parse_args()
    for lead in args.leads:
        build_lead(
            lead=lead,
            aurora_staging_root=args.aurora_staging_root,
            global_dataset=args.global_dataset,
            output_root=args.output_root,
            regions=args.regions,
            split=args.split,
            symlink_ghcnh=not args.copy_ghcnh,
            workers=args.workers,
        )
    logger.info("Done.")


if __name__ == "__main__":
    main()
