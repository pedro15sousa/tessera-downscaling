"""One-off migration: crop existing *global* Aurora staging into the per-region
layout that ``generate_aurora_forecasts.py`` now writes.

The original test-window run wrote global frames at
    <output_root>/lead{L}h/processed/era5_wb2_quarter_<var>/data/<ts>.nc
whereas the cropped pipeline writes
    <output_root>/lead{L}h/<region>/processed/era5_wb2_quarter_<var>/data/<ts>.nc

This script reads each existing global file and writes its per-region crops,
reusing the exact crop logic in generate_aurora_forecasts (same longitude roll,
same europe/east_asia assertion against the dataset reference grid). It is
CPU-only (no GPU, no Aurora), idempotent (skips per-region files that already
exist) and, with --delete-global, removes the global tree once a lead is fully
migrated so you reclaim the space.

Usage (dry run -> migrate -> reclaim space):
    python migrate_global_staging_to_regions.py \
        --output-root /path/_staging/aurora \
        --global-metadata /path/dataset_timestamp_global/metadata.json \
        --dry-run
    python migrate_global_staging_to_regions.py --output-root ... --global-metadata ...
    python migrate_global_staging_to_regions.py --output-root ... --global-metadata ... --delete-global
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parent))
import generate_aurora_forecasts as gen  # noqa: E402


def global_input_path(output_root: Path, lead: int, wb2_var: str, ts: str) -> Path:
    """OLD (pre-crop) global layout: lead{L}h/processed/era5_wb2_quarter_<var>/data/<ts>.nc."""
    return output_root / f"lead{lead}h" / "processed" / f"era5_wb2_quarter_{wb2_var}" / "data" / f"{ts}.nc"


def list_global_timestamps(output_root: Path, lead: int) -> list[str]:
    """Timestamps present in the global 2m_temperature dir for a lead (the done-marker var)."""
    d = output_root / f"lead{lead}h" / "processed" / "era5_wb2_quarter_2m_temperature" / "data"
    if not d.exists():
        return []
    return sorted(p.stem for p in d.glob("*.nc"))


_W = {}  # per-worker shared, read-only state (set by _winit on each process)


def _winit(output_root, lead, region_crops, surf_set):
    _W.update(output_root=output_root, lead=lead, region_crops=region_crops, surf_set=surf_set)


def _migrate_one_ts(ts: str) -> tuple[int, int]:
    """Crop every global var for one timestamp into all regions. (written, skipped)."""
    output_root, lead = _W["output_root"], _W["lead"]
    region_crops, surf_set = _W["region_crops"], _W["surf_set"]
    all_vars = list(gen.SURF_AURORA_TO_WB2.values()) + list(gen.ATMOS_AURORA_TO_WB2.values())
    written = skipped = 0
    for wb2 in all_vars:
        targets = []  # regions that still need this (ts, var)
        for rc in region_crops:
            out = gen.output_path(output_root, lead, rc["name"], wb2, gen.parse_ts(ts))
            if out.exists():
                skipped += 1
            else:
                targets.append((rc, out))
        if not targets:
            continue
        ds = xr.open_dataset(global_input_path(output_root, lead, wb2, ts))
        arr = ds[wb2].values
        levels = list(ds["level"].values) if "level" in ds.dims else None
        ds.close()
        is_surf = wb2 in surf_set
        for rc, out in targets:
            coords = {"latitude": rc["lats"], "longitude": rc["lons"]}
            if is_surf:
                cropped = np.roll(arr, rc["roll"], axis=-1)[rc["lat_idx"]][:, rc["lon_idx"]]
                da = xr.DataArray(cropped, dims=("latitude", "longitude"), coords=coords, name=wb2)
            else:
                cropped = np.roll(arr, rc["roll"], axis=-1)[:, rc["lat_idx"]][:, :, rc["lon_idx"]]
                da = xr.DataArray(cropped, dims=("level", "latitude", "longitude"),
                                  coords={"level": levels, **coords}, name=wb2)
            out.parent.mkdir(parents=True, exist_ok=True)
            tmp = out.with_suffix(".nc.tmp")
            da.to_dataset(name=wb2).to_netcdf(tmp, engine="h5netcdf")
            tmp.replace(out)
            written += 1
    return written, skipped


def migrate_lead(output_root: Path, lead: int, region_crops, dry_run: bool, workers: int) -> tuple[int, int]:
    """Crop every global frame for one lead into all regions, in parallel over timestamps."""
    import time

    surf_set = set(gen.SURF_AURORA_TO_WB2.values())
    timestamps = list_global_timestamps(output_root, lead)
    n = len(timestamps)
    if dry_run:
        # Count only: 6 regions x 9 vars per ts not already present.
        all_vars = list(gen.SURF_AURORA_TO_WB2.values()) + list(gen.ATMOS_AURORA_TO_WB2.values())
        todo = sum(
            1 for ts in timestamps for rc in region_crops for v in all_vars
            if not gen.output_path(output_root, lead, rc["name"], v, gen.parse_ts(ts)).exists()
        )
        return todo, n * len(region_crops) * len(all_vars) - todo

    written = skipped = 0
    t0 = time.time()

    def _tick(i, w, s):
        if i % 100 == 0 or i == n:
            rate = i / max(time.time() - t0, 1e-9)
            eta = (n - i) / max(rate, 1e-9)
            print(f"    {i}/{n} ts | wrote {w:,} skipped {s:,} | {rate:.1f} ts/s | ETA {eta/60:.0f} min", flush=True)

    if workers <= 1:
        _winit(output_root, lead, region_crops, surf_set)
        for i, ts in enumerate(timestamps, 1):
            dw, ds_ = _migrate_one_ts(ts)
            written += dw; skipped += ds_
            _tick(i, written, skipped)
    else:
        import multiprocessing as mp
        with mp.Pool(workers, initializer=_winit,
                     initargs=(output_root, lead, region_crops, surf_set)) as pool:
            for i, (dw, ds_) in enumerate(pool.imap_unordered(_migrate_one_ts, timestamps), 1):
                written += dw; skipped += ds_
                _tick(i, written, skipped)
    return written, skipped


def run(args) -> None:
    output_root = Path(args.output_root)
    global_dataset_dir = Path(args.global_metadata).parent
    region_bboxes = gen.resolve_region_bboxes(args.global_metadata, args.regions, args.uk_bbox)
    region_names = list(region_bboxes)

    print(f"Regions: {region_names}")
    print(f"Leads  : {args.leads}")

    for lead in args.leads:
        ts_list = list_global_timestamps(output_root, lead)
        if not ts_list:
            print(f"lead {lead}h: no global staging found at {output_root}/lead{lead}h/processed -- skipping")
            continue
        # Resolve crops from the global grid (probe any present global file) once per lead.
        probe = global_input_path(output_root, lead, "2m_temperature", ts_list[0])
        ds = xr.open_dataset(probe)
        a_lats, a_lons = ds["latitude"].values, ds["longitude"].values
        ds.close()
        print(f"\nlead {lead}h: {len(ts_list)} global frames on {len(a_lats)}x{len(a_lons)} grid")
        region_crops = gen.resolve_region_crops(region_bboxes, a_lats, a_lons, global_dataset_dir, logger=print)

        written, skipped = migrate_lead(output_root, lead, region_crops, args.dry_run, args.workers)
        verb = "would write" if args.dry_run else "wrote"
        print(f"  {verb} {written:,} per-region files; skipped {skipped:,} already present")

        if args.delete_global and not args.dry_run:
            # Only delete once every region's file is confirmed present for every ts.
            all_vars = list(gen.SURF_AURORA_TO_WB2.values()) + list(gen.ATMOS_AURORA_TO_WB2.values())
            complete = all(
                gen.output_path(output_root, lead, r, v, gen.parse_ts(ts)).exists()
                for ts in ts_list for r in region_names for v in all_vars
            )
            if complete:
                gtree = output_root / f"lead{lead}h" / "processed"
                shutil.rmtree(gtree)
                print(f"  deleted global tree {gtree}")
            else:
                print(f"  NOT deleting global: per-region migration incomplete for lead {lead}h")

    print("\nDone." + ("  (dry run -- nothing written)" if args.dry_run else ""))


def parse_args():
    p = argparse.ArgumentParser(description="Crop existing global Aurora staging into the per-region layout.")
    p.add_argument("--output-root", required=True, help="Aurora staging root (contains lead{L}h/processed/...).")
    p.add_argument("--global-metadata", required=True, help="dataset_timestamp_global/metadata.json (bboxes + ref grids).")
    p.add_argument("--regions", nargs="+", default=gen.DEFAULT_REGIONS)
    p.add_argument("--uk-bbox", type=float, nargs=4, default=list(gen.UK_BBOX),
                   metavar=("LAT_MIN", "LAT_MAX", "LON_MIN", "LON_MAX"))
    p.add_argument("--leads", type=int, nargs="+", default=gen.DEFAULT_LEADS_HOURS)
    p.add_argument("--workers", type=int, default=8,
                   help="Parallel worker processes over timestamps (default 8). 1 = serial.")
    p.add_argument("--dry-run", action="store_true", help="Report counts only; write nothing.")
    p.add_argument("--delete-global", action="store_true",
                   help="After a lead is fully migrated, delete its global tree to reclaim space.")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())