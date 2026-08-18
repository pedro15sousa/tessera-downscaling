"""Re-render the paper's dense-map figures from saved *_dem.npz (no model re-run).

Overwrites the *_dem.png for the four (region, variable, snapshot) panels used in the
paper with paper-ready styling: no suptitle, panel names ERA5 bilinear interpolation /
ConvCNP (without TESSERA) / ConvCNP with TESSERA. Region, date, resolution, DEM
elevation and the 3-seed ensembling live in the LaTeX caption, not on the figure.

The maps themselves are unchanged -- this only re-labels; the fields come straight from
the arrays generate_maps.py already saved. Run after editing plot_three there:
  .venv/bin/python projects/tessera_downscaling/scripts/maps/replot_paper_maps.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from regions import Region  # noqa: E402
from generate_maps import (  # noqa: E402
    PANEL_TITLES, plot_diff_over_terrain, plot_three, plot_three_plus_diff,
)

# (region, variable, snapshot) -- the paper's fixed selection.
FIGS = [
    ("iberia", "t2m", "2022-07-18-12"),
    ("iberia", "wind", "2022-12-12-12"),
    ("norway", "t2m", "2023-01-02-00"),
    ("norway", "wind", "2022-01-30-00"),
]

for region, var, ts in FIGS:
    R = Region(region)
    npz = R.fig(var, ts, "_dem.npz")
    if not npz.exists():
        print(f"  [skip missing] {npz}")
        continue
    d = np.load(npz, allow_pickle=True)
    arrays = [d["era5_interp"], d["convcnp_baseline"], d["tessera_concat"]]
    style = R.jobs[var]  # cmap/unit are date-independent styling
    out = R.fig(var, ts, "_dem.png")
    plot_three(arrays, PANEL_TITLES, d["lons"], d["lats"], style["cmap"], style["unit"], out)
    print(f"  wrote {out}")
    # Second version: same three panels + Delta = TESSERA - baseline, to show where
    # the texture is placed differently even when the amplitude range matches.
    # "_diff" keeps the raw difference (bulk offset included); "_diff_anom" removes
    # the land-mean offset so only the spatial pattern remains.
    # The mean-removed version also carries a 5th relief-shaded elevation column.
    dem = (np.load(R.dem_path).reshape(len(d["lats"]), len(d["lons"]))
           if R.dem_path.exists() else None)
    for demean, tail in [(False, "_dem_diff.png"), (True, "_dem_diff_anom.png")]:
        out_d = R.fig(var, ts, tail)
        dmax, mean = plot_three_plus_diff(arrays, PANEL_TITLES, d["lons"], d["lats"],
                                          style["cmap"], style["unit"], out_d,
                                          demean=demean,
                                          dem_grid=dem if demean else None)
        print(f"  wrote {out_d.name}  (|delta| p99 = {dmax:.2f}, "
              f"mean = {mean:+.2f} {style['unit_plain']})")

    # Third version: Delta draped over the region's relief (raw + mean-removed).
    if dem is not None:
        out_t = R.fig(var, ts, "_dem_diff_terrain.png")
        plot_diff_over_terrain(arrays, d["lons"], d["lats"], style["unit"], dem, out_t)
        print(f"  wrote {out_t.name}")
    else:
        print(f"  [skip terrain overlay] no DEM at {R.dem_path}")
