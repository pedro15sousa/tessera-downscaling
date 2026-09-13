#!/usr/bin/env python3
"""Validate a tree of extracted Aurora latents before Phase B builds on it.

Pure JSON + numpy (xarray only for the optional physical drift check), so it
runs on a login node. Per (kind, lead, region) it checks:

  * the directory and its ``latent_meta.json`` exist and the meta is
    self-consistent; the crop's interior is bit-for-bit the region grid of
    ``dataset_timestamp_global`` (``validate_latent_meta``);
  * coverage: one ``.npy`` per expected timestamp (backbone: every valid time
    of the split at that lead; encoder: every init time of the schedule),
    reporting first/last present and the missing ones;
  * spot checks: files reload with the meta's shape and dtype and are finite;
    an abs-max summary;
  * the meta geometry agrees across leads (same crop/shape/storage per
    region and kind), i.e. no shard was written with different settings.

If ``--verify-root`` holds the decoded-field subset written by the extractor,
each of its files is compared with the original forecast staging
(``--physical-root``): RMSE and max-abs of 2t/10u/msl per lead. This is a
tolerance report (different hardware than the original run), not a pass/fail.

Exit code 0 = all checks passed; 1 = at least one failed.

Example:
    uv run python scripts/aurora/verify_aurora_latents.py \
        --leads 6 24 72 --regions europe east_asia --split all
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from _common import build_schedule, load_times

from tessera_downscaling.paths import dataset_dir, ingest_dir
from tessera_downscaling.preprocessing import aurora_latent as al

KIND_SUBDIR = {"backbone": "latent_backbone", "encoder": "latent_encoder"}
CROSS_LEAD_KEYS = (
    "crop",
    "shape",
    "storage_dtype",
    "storage_offset",
    "embed_dim",
    "latent_levels",
    "n_channels",
    "kind",
)


def ts_stem(t) -> str:
    return f"{t.year:04d}-{t.month:02d}-{t.day:02d}-{t.hour:02d}"


def list_stems(directory: Path) -> set[str]:
    with os.scandir(directory) as it:
        return {
            e.name[:-4]
            for e in it
            if e.name.endswith(".npy") and not e.name.startswith("latent_")
        }


def check_dir(
    check,
    directory: Path,
    expected: list[str],
    grid_lats,
    grid_lons,
    spot: int,
    label: str,
) -> dict | None:
    if not directory.is_dir():
        check(False, f"{label}: directory exists ({directory})")
        return None
    meta_path = directory / al.META_NAME
    if not meta_path.exists():
        check(False, f"{label}: {al.META_NAME} present")
        return None
    meta = json.loads(meta_path.read_text())
    try:
        al.validate_latent_meta(meta, grid_lats, grid_lons)
        check(True, f"{label}: meta valid, interior == region grid")
    except (ValueError, KeyError) as e:
        check(False, f"{label}: meta valid ({e})")
    for name in (al.LATS_NAME, al.LONS_NAME):
        check((directory / name).exists(), f"{label}: {name} present")

    present = list_stems(directory)
    missing = [s for s in expected if s not in present]
    extra = sorted(present - set(expected))
    first = min(present) if present else "-"
    last = max(present) if present else "-"
    check(
        not missing,
        f"{label}: {len(present)}/{len(expected)} files present ({first} .. {last})"
        + (f"; missing {len(missing)}, e.g. {missing[:3]}" if missing else ""),
    )
    if extra:
        print(
            f"         note: {len(extra)} files outside the expected set, e.g. {extra[:3]}"
        )

    if spot and present:
        stems = sorted(present)
        picks = [
            stems[int(i)] for i in np.linspace(0, len(stems) - 1, min(spot, len(stems)))
        ]
        bad = []
        absmax = 0.0
        for s in picks:
            arr = np.load(directory / f"{s}.npy")
            if list(arr.shape) != list(meta["shape"]):
                bad.append(f"{s}: shape {arr.shape}")
            if arr.dtype != al.storage_numpy_dtype(meta["storage_dtype"]):
                bad.append(f"{s}: dtype {arr.dtype}")
            if not np.isfinite(arr).all():
                bad.append(f"{s}: non-finite")
            absmax = max(absmax, float(np.abs(arr.astype(np.float32)).max()))
        check(
            not bad,
            f"{label}: {len(picks)} spot-checked files have shape {meta['shape']}, "
            f"dtype {meta['storage_dtype']}, finite (abs-max {absmax:.4g})"
            + (f"; {'; '.join(bad[:3])}" if bad else ""),
        )
    return meta


def physical_drift(
    check, verify_root: Path, physical_root: Path, leads, regions
) -> None:
    import xarray as xr

    variables = ["2m_temperature", "10m_u_component_of_wind", "mean_sea_level_pressure"]
    if not verify_root.is_dir():
        print(
            f"(no physical verification subset under {verify_root}; skipping drift check)"
        )
        return
    print(f"=== physical drift: {verify_root} vs {physical_root} ===")
    for lead in leads:
        for region in regions:
            for var in variables:
                vdir = (
                    verify_root
                    / f"lead{lead}h"
                    / region
                    / "processed"
                    / f"era5_wb2_quarter_{var}"
                    / "data"
                )
                odir = (
                    physical_root
                    / f"lead{lead}h"
                    / region
                    / "processed"
                    / f"era5_wb2_quarter_{var}"
                    / "data"
                )
                if not vdir.is_dir():
                    continue
                files = sorted(vdir.glob("*.nc"))
                rmse, maxabs, n_missing = [], [], 0
                for f in files:
                    o = odir / f.name
                    if not o.exists():
                        n_missing += 1
                        continue
                    a = xr.open_dataset(f)[var].values.astype(np.float64)
                    b = xr.open_dataset(o)[var].values.astype(np.float64)
                    if a.shape != b.shape:
                        n_missing += 1
                        continue
                    d = a - b
                    rmse.append(float(np.sqrt(np.mean(d**2))))
                    maxabs.append(float(np.abs(d).max()))
                if rmse:
                    print(
                        f"  lead {lead:>2}h {region:12s} {var:28s}: n={len(rmse)}, "
                        f"RMSE mean {np.mean(rmse):.4g} max {np.max(rmse):.4g}, "
                        f"max|d| {np.max(maxabs):.4g}"
                        + (f"  ({n_missing} without an original)" if n_missing else "")
                    )
                check(
                    bool(rmse) or not files,
                    f"lead {lead}h {region} {var}: drift computed on {len(rmse)}/{len(files)} files",
                )


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--output-root",
        type=Path,
        default=ingest_dir("aurora"),
        help="Latent root written by extract_aurora_latents.py",
    )
    p.add_argument(
        "--global-dataset",
        type=Path,
        default=dataset_dir(),
        help="dataset_timestamp_global directory (bboxes, timestamps, region grids)",
    )
    p.add_argument("--leads", type=int, nargs="+", default=[6, 24, 72])
    p.add_argument("--regions", nargs="+", default=["europe", "east_asia"])
    p.add_argument(
        "--split", choices=["train", "val", "trainval", "test", "all"], default="all"
    )
    p.add_argument("--start-date", default=None)
    p.add_argument("--end-date", default=None)
    p.add_argument(
        "--no-encoder", action="store_true", help="Skip the encoder directories"
    )
    p.add_argument(
        "--spot-check", type=int, default=5, help="Files per directory to reload"
    )
    p.add_argument("--verify-root", type=Path, default=ingest_dir("aurora_verify"))
    p.add_argument("--physical-root", type=Path, default=ingest_dir("aurora"))
    args = p.parse_args()

    failures: list[str] = []

    def check(cond: bool, msg: str) -> None:
        print(f"  [{'PASS' if cond else 'FAIL'}] {msg}")
        if not cond:
            failures.append(msg)

    meta_path = args.global_dataset / "metadata.json"
    valid_times = load_times(meta_path, args.split, args.start_date, args.end_date)
    inits, per_init = build_schedule(valid_times, args.leads)
    valid_by_lead = {
        lead: sorted(
            {ts_stem(v) for h in per_init.values() for (lh, _s, v) in h if lh == lead}
        )
        for lead in args.leads
    }
    init_stems = [ts_stem(t) for t in inits]
    print(
        f"Expected: {len(valid_times)} valid times ({args.split}), {len(inits)} inits, "
        f"leads {args.leads}, regions {args.regions}"
    )

    for region in args.regions:
        greg = args.global_dataset / "regions" / region
        grid_lats, grid_lons = np.load(greg / "lats.npy"), np.load(greg / "lons.npy")
        metas = []
        print(f"=== {region} ===")
        for lead in args.leads:
            d = args.output_root / f"lead{lead}h" / region / KIND_SUBDIR["backbone"]
            m = check_dir(
                check,
                d,
                valid_by_lead[lead],
                grid_lats,
                grid_lons,
                args.spot_check,
                f"backbone lead {lead}h",
            )
            if m is not None:
                check(
                    m.get("lead_hours") == lead and m.get("kind") == "backbone",
                    f"backbone lead {lead}h: meta kind/lead match the directory",
                )
                metas.append(m)
        if len(metas) > 1:
            same = all(
                all(m.get(k) == metas[0].get(k) for k in CROSS_LEAD_KEYS)
                for m in metas[1:]
            )
            check(same, f"{region}: backbone crop/shape/storage identical across leads")
        if not args.no_encoder:
            d = args.output_root / "lead0h" / region / KIND_SUBDIR["encoder"]
            m = check_dir(
                check,
                d,
                init_stems,
                grid_lats,
                grid_lons,
                args.spot_check,
                "encoder lead 0h",
            )
            if m is not None:
                check(
                    m.get("kind") == "encoder" and m.get("lead_hours") == 0,
                    "encoder: meta kind/lead match the directory",
                )
                if metas:
                    check(
                        m["crop"] == metas[0]["crop"],
                        f"{region}: encoder crop identical to the backbone crop",
                    )
        print()

    physical_drift(
        check, args.verify_root, args.physical_root, args.leads, args.regions
    )

    print("=" * 60)
    if failures:
        print(f"VALIDATION FAILED: {len(failures)} check(s) failed.")
        for m in failures:
            print(f"  - {m}")
        return 1
    print("VALIDATION PASSED: the latent tree is complete and self-consistent.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
