"""Build the row-aligned ``extra_descriptors.npy`` from the GEE descriptor CSV.

Takes the ``station_extra_descriptors.csv`` produced by
``scripts/data/fetch_station_extra_descriptors.py`` and
reindexes it onto the row order of a station-list CSV (normally the global
``tessera_global/station_list_filtered.csv`` that the VAE-latent arrays are
aligned with), producing:

    <output>.npy               (n_stations, 17) float32, raw physical values
    <output>_names.json        column names, units, fill counts, provenance
    <output>_global_stats.npz  z-score stats cache (same convention as VAE
                               latents: computed once over all valid rows,
                               reused across every region/split/experiment)

The npy is consumed by ``train.py --extra-descriptors-path`` which serves it
through the same precomputed-vector pathway as the VAE latents: NaN rows mark
stations without descriptors (dropped by the loader, mirroring NaN latents),
and per-dim z-scoring happens at load time from the cached global stats — so
raw values here stay interpretable and no scale is baked in.

Fill policy (applied before writing, counts recorded in the names json):
  - Station absent from the GEE CSV entirely  -> NaN row (station dropped
    at load, keeping parity with how un-encodable VAE stations behave).
  - clay_frac / sand_frac missing (SoilGrids masks water)  -> global column
    mean, i.e. neutral (~0) after z-scoring.
  - elev_* / slope / dz_* missing (DEM-masked, e.g. offshore)  -> 0.0
    (sea level / flat terrain — the physically neutral value).
  - WorldCover fractions / tree_height are unmasked server-side and should
    never be missing; any stragglers get 0.0 and are logged.

Also prints the two cheap sanity checks: WorldCover fractions summing to ~1,
and the correlation between ``dem_elev_320m`` and the station-list elevation
column (catches lat/lon swaps and unit mistakes immediately).

Usage (from the repo root):
    uv run python scripts/data/build_extra_descriptors.py \
        --descriptors-csv <data root>/processed/station_extra_descriptors.csv \
        --station-csv     <data root>/processed/tessera_global/station_list_filtered.csv \
        --output-npy      <data root>/processed/extra_descriptors.npy
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

LOG_FORMAT = "%(asctime)s | %(levelname)s | %(message)s"
logger = logging.getLogger("build_extra_descriptors")

# Model feature vector, in npy column order. The audit-only CSV columns
# (wc_masked_frac, dem_elev_320m) are deliberately excluded.
FEATURE_COLUMNS = [
    "forest_frac", "lowveg_frac", "crop_frac", "built_frac", "bare_frac",
    "snowice_frac", "water_frac", "tree_height", "clay_frac", "sand_frac",
    "elev_mean", "elev_std", "elev_min", "elev_max",
    "slope", "dz_dn", "dz_de",
]
MEAN_FILL_COLUMNS = {"clay_frac", "sand_frac"}
UNITS = {
    **{c: "fraction [0,1]" for c in FEATURE_COLUMNS if c.endswith("_frac")
       and c not in ("clay_frac", "sand_frac")},
    "tree_height": "m",
    "clay_frac": "g/kg", "sand_frac": "g/kg",
    "elev_mean": "m", "elev_std": "m", "elev_min": "m", "elev_max": "m",
    "slope": "tan(slope), dimensionless",
    "dz_dn": "m/m", "dz_de": "m/m",
}
WC_FRACTION_COLUMNS = [
    "forest_frac", "lowveg_frac", "crop_frac", "built_frac", "bare_frac",
    "snowice_frac", "water_frac",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--descriptors-csv", type=Path, required=True,
                   help="station_extra_descriptors.csv from the GEE fetch.")
    p.add_argument("--station-csv", type=Path, required=True,
                   help="Station list defining the npy row order (use the "
                        "same CSV you pass as --extra-descriptors-station-csv "
                        "at training time).")
    p.add_argument("--output-npy", type=Path, required=True,
                   help="Output .npy path; companion _names.json and "
                        "_global_stats.npz are written next to it.")
    return p.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
    args = parse_args()

    desc = pd.read_csv(args.descriptors_csv, dtype={"station_id": str})
    stations = pd.read_csv(args.station_csv, dtype={"station_id": str})
    if "station_id" not in stations.columns:
        raise ValueError(f"No 'station_id' column in {args.station_csv}")
    missing_cols = [c for c in FEATURE_COLUMNS if c not in desc.columns]
    if missing_cols:
        raise ValueError(
            f"Descriptor CSV is missing columns {missing_cols} — was it "
            "produced by fetch_station_extra_descriptors.py?"
        )
    if desc["station_id"].duplicated().any():
        n_dup = int(desc["station_id"].duplicated().sum())
        logger.warning("Dropping %d duplicate station_ids (keeping first) — "
                       "expected if a --resume rerun overlapped.", n_dup)
        desc = desc.drop_duplicates("station_id", keep="first")

    desc = desc.set_index("station_id")
    aligned = desc.reindex(stations["station_id"].values)
    n_total = len(aligned)
    # Stations never fetched: keep as all-NaN rows so the loader drops them.
    absent = aligned[FEATURE_COLUMNS].isna().all(axis=1)
    n_absent = int(absent.sum())
    if n_absent:
        level = logger.warning if n_absent / n_total > 0.05 else logger.info
        level("%d/%d stations absent from the descriptor CSV -> NaN rows "
              "(dropped at load time)", n_absent, n_total)

    fill_counts: dict[str, int] = {}
    for col in FEATURE_COLUMNS:
        holes = aligned[col].isna() & ~absent
        n_holes = int(holes.sum())
        if not n_holes:
            continue
        fill_value = (
            float(desc[col].mean()) if col in MEAN_FILL_COLUMNS else 0.0
        )
        aligned.loc[holes, col] = fill_value
        fill_counts[col] = n_holes
        logger.info("Filled %d missing '%s' values with %s", n_holes, col,
                    f"{fill_value:.2f}" +
                    (" (global mean)" if col in MEAN_FILL_COLUMNS else ""))

    arr = aligned[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    valid = ~np.isnan(arr).any(axis=1)

    # --- Sanity check 1: WorldCover fractions sum to ~1 -------------------
    frac_sum = arr[valid][:, [FEATURE_COLUMNS.index(c)
                              for c in WC_FRACTION_COLUMNS]].sum(axis=1)
    n_bad = int((np.abs(frac_sum - 1.0) > 0.02).sum())
    logger.info(
        "Sanity: WorldCover fraction sums — mean %.4f, %d/%d rows deviate "
        "from 1 by >0.02%s", frac_sum.mean(), n_bad, int(valid.sum()),
        "  <-- INVESTIGATE" if n_bad > 0.01 * valid.sum() else "",
    )

    # --- Sanity check 2: DEM elevation vs station-list elevation ----------
    if "elevation" in stations.columns and "dem_elev_320m" in aligned.columns:
        st_elev = stations["elevation"].to_numpy(dtype=np.float64)
        dem_elev = aligned["dem_elev_320m"].to_numpy(dtype=np.float64)
        ok = valid & np.isfinite(st_elev) & np.isfinite(dem_elev)
        # Guard against unset-elevation sentinels (GHCN -999.9, 9999.0, and
        # 8191.0 = 2^13-1, seen on lightvessels/polar outposts).
        ok &= (st_elev > -900) & (st_elev < 9000) & (st_elev != 8191.0)
        r = np.corrcoef(st_elev[ok], dem_elev[ok])[0, 1]
        mad = np.median(np.abs(st_elev[ok] - dem_elev[ok]))
        logger.info(
            "Sanity: station elevation vs 320m DEM mean — r=%.3f, median "
            "|diff|=%.0fm over %d stations%s", r, mad, int(ok.sum()),
            "  <-- INVESTIGATE (expect r>0.95)" if r < 0.95 else "",
        )

    # Per-column summary over valid rows.
    for i, col in enumerate(FEATURE_COLUMNS):
        v = arr[valid][:, i]
        logger.info("  %-14s min=%9.2f  mean=%9.2f  max=%9.2f  [%s]",
                    col, v.min(), v.mean(), v.max(), UNITS[col])

    args.output_npy.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output_npy, arr)
    logger.info("Wrote %s  shape=%s  (%d valid / %d NaN rows)",
                args.output_npy, arr.shape, int(valid.sum()),
                n_total - int(valid.sum()))

    names_path = args.output_npy.with_name(args.output_npy.stem + "_names.json")
    with open(names_path, "w") as f:
        json.dump({
            "columns": FEATURE_COLUMNS,
            "units": UNITS,
            "n_stations": n_total,
            "n_valid": int(valid.sum()),
            "fill_counts": fill_counts,
            "n_absent_stations": n_absent,
            "descriptors_csv": str(args.descriptors_csv),
            "station_csv": str(args.station_csv),
            "created": datetime.now(UTC).isoformat(),
            "provenance": "fetch_station_extra_descriptors.py — SFX group "
                          "over 320m radius (WorldCover v200 / ETH canopy "
                          "height 2020 / SoilGrids 0-5cm), TOPO group over "
                          "6.25km radius (Copernicus GLO-30 at 250m), after "
                          "Bakketun et al. 2026 (arXiv:2607.02824).",
        }, f, indent=2)
    logger.info("Wrote %s", names_path)

    # Pre-warm the z-score stats cache (same convention as VAE latents) so
    # training jobs never race to compute it.
    from tessera_downscaling.data.vae_latents import (
        compute_or_load_global_vae_stats,
    )
    mean, std = compute_or_load_global_vae_stats(args.output_npy)
    logger.info("z-score stats cached (%d dims); e.g. %s: mean=%.2f std=%.2f",
                len(mean), FEATURE_COLUMNS[0], mean[0], std[0])
    return 0


if __name__ == "__main__":
    sys.exit(main())
