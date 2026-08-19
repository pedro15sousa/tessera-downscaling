#!/usr/bin/env python3
"""Validate the preprocessed Aurora-forecast datasets before committing to a
training+eval run against them.

Pure JSON + numpy (no torch / xarray), so it runs fast on a login node. It
checks exactly the load-bearing assumptions the eval path depends on:

  * metadata: layout multi_region_snapshot_v1, source 'aurora', 19 dynamic
    channels, precip absent from era5_dynamic_channels, the requested regions
    present, test-only split with the same test count as the global dataset;
  * channel names equal the global dataset's list with precip removed (so the
    lenient eval-time drop becomes a no-op rather than a mis-drop);
  * per region: era5_snapshot has one .npy per test timestamp, each with 19
    channels; the region grid (lats/lons) matches the global dataset's grid
    bit-for-bit (the dropped-pole / mis-crop guard); per-region 19-channel
    normalisation stats present.

Exit code 0 = all good; 1 = at least one check failed.

Example (from the repo root; defaults resolve under the data root):
    uv run python scripts/preprocessing/validate_aurora_datasets.py \
        --leads 6 24 72 --regions europe east_asia
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from tessera_downscaling.paths import data_root, dataset_dir

PRECIP = "total_precipitation_sum"


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--global-dataset",
        type=Path,
        default=dataset_dir(),
        help="dataset_timestamp_global directory, the reference "
        "(default: under the data root).",
    )
    p.add_argument(
        "--aurora-output-root",
        type=Path,
        default=data_root(),
        help="Directory containing dataset_timestamp_aurora_lead{L}h "
        "(default: the data root).",
    )
    p.add_argument("--leads", type=int, nargs="+", default=[6, 24, 72])
    p.add_argument("--regions", type=str, nargs="+", default=["europe", "east_asia"])
    p.add_argument(
        "--spot-check",
        type=int,
        default=3,
        help="How many era5_snapshot .npy files per region to open "
        "and shape-check (0 = none, just count files).",
    )
    args = p.parse_args()

    failures: list[str] = []

    def check(cond: bool, msg: str) -> None:
        status = "PASS" if cond else "FAIL"
        print(f"  [{status}] {msg}")
        if not cond:
            failures.append(msg)

    # --- Reference: global dataset ---
    gmeta = json.loads((args.global_dataset / "metadata.json").read_text())
    g_channels = gmeta["era5_dynamic_channels"]
    expected_19 = [c for c in g_channels if c != PRECIP]
    precip_idx = g_channels.index(PRECIP) if PRECIP in g_channels else None
    g_n_test = gmeta["temporal_split"]["n_test_timestamps"]
    print(
        f"Reference global dataset: {len(g_channels)} channels, "
        f"{g_n_test} test timestamps, regions {list(gmeta['regions'])}"
    )
    if PRECIP not in g_channels:
        print(
            f"  NOTE: '{PRECIP}' is not in the global channel list; "
            "expected-19 derived as-is."
        )
    print()

    for lead in args.leads:
        ds = args.aurora_output_root / f"dataset_timestamp_aurora_lead{lead}h"
        print(f"=== lead {lead}h: {ds} ===")
        if not (ds / "metadata.json").exists():
            check(False, f"{ds}/metadata.json exists")
            print()
            continue

        meta = json.loads((ds / "metadata.json").read_text())
        check(
            meta.get("layout_version") == "multi_region_snapshot_v1",
            "layout_version == multi_region_snapshot_v1",
        )
        check(meta.get("source") == "aurora", "source == aurora")
        check(meta.get("lead_hours") == lead, f"lead_hours == {lead}")
        check(meta.get("n_dynamic_channels") == 19, "n_dynamic_channels == 19")

        chans = meta.get("era5_dynamic_channels", [])
        check(len(chans) == 19, "era5_dynamic_channels has 19 entries")
        check(PRECIP not in chans, f"'{PRECIP}' absent from era5_dynamic_channels")
        check(
            chans == expected_19,
            "era5_dynamic_channels == global channels minus precip (order preserved)",
        )

        n_test = meta["temporal_split"]["n_test_timestamps"]
        check(n_test == g_n_test, f"n_test_timestamps == {g_n_test}")
        check(
            len(meta.get("valid_timestamps", [])) == g_n_test,
            f"len(valid_timestamps) == {g_n_test}",
        )
        check(
            meta["temporal_split"].get("n_train_timestamps", -1) == 0
            and meta["temporal_split"].get("n_val_timestamps", -1) == 0,
            "train/val timestamp counts == 0 (test-only)",
        )
        for r in args.regions:
            check(r in meta.get("regions", {}), f"region '{r}' present in metadata")

        # Top-level shared artefacts.
        for fname in ("stations.csv", "valid_station_indices.npy"):
            check((ds / fname).exists(), f"top-level {fname} present")
        check(
            (ds / "ghcnh_snapshot").exists(), "ghcnh_snapshot present (symlink or dir)"
        )

        # Per-region checks.
        for r in args.regions:
            reg = ds / "regions" / r
            greg = args.global_dataset / "regions" / r
            if not reg.is_dir():
                check(False, f"[{r}] region dir exists")
                continue

            for fname in (
                "lats.npy",
                "lons.npy",
                "static_fields.npy",
                "region_metadata.json",
                "normalisation_stats.npz",
            ):
                check((reg / fname).exists(), f"[{r}] {fname} present")

            # Grid must match the global region grid bit-for-bit (dropped-pole guard).
            if (reg / "lats.npy").exists() and (greg / "lats.npy").exists():
                la, lo = np.load(reg / "lats.npy"), np.load(reg / "lons.npy")
                gla, glo = np.load(greg / "lats.npy"), np.load(greg / "lons.npy")
                check(
                    la.shape == gla.shape and lo.shape == glo.shape,
                    f"[{r}] grid shape matches global ({la.shape}x{lo.shape})",
                )
                check(
                    np.allclose(la, gla, atol=1e-4) and np.allclose(lo, glo, atol=1e-4),
                    f"[{r}] grid coords match global (atol 1e-4)",
                )

            # Per-region normalisation stats. era5_mean/std span the WHOLE
            # normalised block [dynamic, static, lat, lon] (not just the 19
            # dynamic channels), so the correct invariant is that Stage 2
            # deleted exactly the precip entry: aurora == global minus precip.
            reg_stats = reg / "normalisation_stats.npz"
            greg_stats = greg / "normalisation_stats.npz"
            if reg_stats.exists() and greg_stats.exists() and precip_idx is not None:
                st, gst = np.load(reg_stats), np.load(greg_stats)
                for key in ("era5_mean", "era5_std"):
                    if key in st.files and key in gst.files:
                        ok = len(st[key]) == len(gst[key]) - 1 and np.allclose(
                            st[key], np.delete(gst[key], precip_idx)
                        )
                        check(
                            ok,
                            f"[{r}] {key} == global minus precip "
                            f"(len {len(gst[key])} -> {len(st[key])})",
                        )

            # One .npy per test timestamp, each 19 channels.
            snap_dir = reg / "era5_snapshot"
            n_npy = len(list(snap_dir.glob("*.npy"))) if snap_dir.is_dir() else 0
            check(
                n_npy == g_n_test,
                f"[{r}] era5_snapshot has {g_n_test} .npy files (found {n_npy})",
            )
            if args.spot_check and snap_dir.is_dir():
                bad = []
                for ts in meta["valid_timestamps"][: args.spot_check]:
                    f = snap_dir / f"{ts}.npy"
                    if not f.exists():
                        bad.append(f"{ts} missing")
                        continue
                    arr = np.load(f)
                    if arr.shape[0] != 19:
                        bad.append(f"{ts} has {arr.shape[0]} channels")
                check(
                    not bad,
                    f"[{r}] spot-checked {args.spot_check} snapshots are 19ch"
                    + (f" ({'; '.join(bad)})" if bad else ""),
                )
        print()

    print("=" * 60)
    if failures:
        print(f"VALIDATION FAILED: {len(failures)} check(s) failed.")
        for m in failures:
            print(f"  - {m}")
        return 1
    print("VALIDATION PASSED: all Aurora datasets look ready for eval.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
