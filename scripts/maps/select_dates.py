"""Pick the most relevant snapshot per (region, variable) for the dense maps.

For every region in regions.py and each variable, scans the Europe TEST snapshots
and ranks days by the SPATIAL STANDARD DEVIATION of the bilinearly-interpolated
ERA5 field over the region's valid land cells — the day with the strongest spatial
structure (and, for wind, the most active field) to downscale. Each snapshot is
loaded once and scored for every region.

Prints the top candidates + the pick per (region, variable), and writes the picks
to OUTPUTS/selected_dates.json. Use those to set per-region `dates` in regions.py
(this is how the paper's four snapshots -- iberia t2m 2022-07-18-12 / wind
2022-12-12-12, norway t2m 2023-01-02-00 / wind 2022-01-30-00 -- were chosen).

  uv run python scripts/maps/select_dates.py
"""
from __future__ import annotations

import json

import numpy as np
from generate_maps import bilinear_grid_to_points
from regions import OUTPUTS, REGIONS, Region

from tessera_downscaling.data.dataset import MultiRegionSnapshotDownscalingDataset
from tessera_downscaling.paths import dataset_dir, processed_dir

# Snapshots/grid are shared across these europe crops; take them from any region.
EU = Region(next(iter(REGIONS))).region_data
GLATS = np.load(EU / "lats.npy").astype(np.float32)
GLONS = np.load(EU / "lons.npy").astype(np.float32)

# Reference dates currently in regions.py, for context in the printout.
REF = {"t2m": "2022-07-18-12", "wind": "2022-12-12-12"}


def region_cells(name):
    d = np.load(Region(name).dense_npz, allow_pickle=True)
    co, vm = d["coords"], d["valid_mask"]
    return np.stack([co["lat"][vm], co["lon"][vm]], axis=1).astype(np.float32)


def field(era5, var, pts):
    if var == "t2m":
        return bilinear_grid_to_points(era5[0], GLATS, GLONS, pts) - 273.15
    u = bilinear_grid_to_points(era5[1], GLATS, GLONS, pts)
    v = bilinear_grid_to_points(era5[2], GLATS, GLONS, pts)
    return np.sqrt(u * u + v * v)


def main():
    ds = MultiRegionSnapshotDownscalingDataset(
        dataset_dir=dataset_dir("dataset_timestamp_global"),
        region_specs={"europe": "test"}, split="test", target_variables=["t2m"],
        vae_latents_path=processed_dir("station_latents_lat16_grad0.5.npy"),
        vae_latents_station_csv=processed_dir("tessera_global", "station_list_filtered.csv"),
        vae_latents_zscore=True, include_static_fields=False, normalisation_policy="per_region",
    )
    tss = list(ds.timestamps)
    print(f"scanning {len(tss)} europe test snapshots over {list(REGIONS)} ...")

    cells = {name: region_cells(name) for name in REGIONS}
    # records[name][var] = list of (ts, std, mean)
    rec = {name: {"t2m": [], "wind": []} for name in REGIONS}
    for k, ts in enumerate(tss):
        era5 = np.load(EU / "era5_snapshot" / f"{ts}.npy")
        for name, pts in cells.items():
            for var in ("t2m", "wind"):
                f = field(era5, var, pts)
                rec[name][var].append((ts, float(np.std(f)), float(np.mean(f))))
        if k % 250 == 0:
            print(f"  {k}/{len(tss)}")

    picks = {}
    for name in REGIONS:
        print(f"\n===== {name} =====")
        picks[name] = {}
        for var in ("t2m", "wind"):
            rows = sorted(rec[name][var], key=lambda r: r[1], reverse=True)
            ref_rank = next(i for i, r in enumerate(rows) if r[0] == REF[var])
            ref = rows[ref_rank]
            print(f"  -- {var}: top 6 by spatial std (over land cells) --")
            for ts, sd, mn in rows[:6]:
                print(f"     {ts}  std={sd:6.2f}  mean={mn:7.2f}")
            print(f"     [ref {REF[var]}: std={ref[1]:.2f} mean={ref[2]:.2f}  rank {ref_rank+1}/{len(rows)}]")
            picks[name][var] = dict(ts=rows[0][0], std=round(rows[0][1], 3), mean=round(rows[0][2], 3))
            print(f"  PICK {var}: {rows[0][0]}")

    OUTPUTS.mkdir(parents=True, exist_ok=True)
    out = OUTPUTS / "selected_dates.json"
    out.write_text(json.dumps(picks, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
