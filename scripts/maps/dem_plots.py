"""DEM-version companion plots: TESSERA contribution + field-vs-terrain.

Reads the high-res-DEM map arrays (<region>_<var>_<ts>_dem.npz) and produces:
  (A) *_dem_tessera_contribution.png  — baseline | TESSERA | (TESSERA-baseline)
  (B) *_dem_field_vs_terrain.png       — shaded relief | baseline field | TESSERA
                                         field, draped over terrain (full region)
Mirrors the proxy-version figures so the two can be compared directly.
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.io.img_tiles as cimgt

sys.path.insert(0, str(Path(__file__).parent))
from regions import get_region  # noqa: E402

R = get_region()
PC = ccrs.PlateCarree()
ATTR = "Relief: Esri World Shaded Relief"


class SR(cimgt.GoogleTiles):
    def _image_url(self, tile):
        x, y, z = tile
        return f"https://server.arcgisonline.com/ArcGIS/rest/services/World_Shaded_Relief/MapServer/tile/{z}/{y}/{x}"


ti = SR(cache=True)
for v, job in R.jobs.items():
    ts, u, cm = job["ts"], job["unit_plain"], job["cmap"]
    d = np.load(R.fig(v, ts, "_dem.npz"))
    b, t, vm = d["convcnp_baseline"], d["tessera_concat"], d["valid_mask"]
    lats, lons = d["lats"], d["lons"]
    ext = [float(lons[0]), float(lons[-1]), float(lats[-1]), float(lats[0])]
    lo, hi = (float(np.nanpercentile(np.concatenate([b[vm], t[vm]]), p)) for p in (1, 99))
    diff = t - b
    vmax = float(np.nanpercentile(np.abs(diff), 98))

    # (A) contribution
    fig, ax = plt.subplots(1, 3, figsize=(16.5, 5.2), subplot_kw={"projection": PC})
    panels = [("ConvCNP baseline (DEM)", b, cm, lo, hi),
              ("TESSERA lat16 concat (DEM)", t, cm, lo, hi),
              ("TESSERA − baseline", diff, "RdBu_r", -vmax, vmax)]
    for a, (tl, arr, c, vmn, vmx) in zip(ax, panels):
        im = a.imshow(arr, origin="upper", extent=ext, transform=PC, cmap=c, vmin=vmn, vmax=vmx)
        a.coastlines("50m", linewidth=0.5)
        a.set_extent(ext, crs=PC)
        a.set_title(tl, fontsize=11)
        fig.colorbar(im, ax=a, shrink=0.62, pad=0.02)
    fig.suptitle(f"TESSERA contribution (high-res DEM) — {v} — {ts} UTC", fontsize=13)
    fig.savefig(R.fig(v, ts, "_dem_tessera_contribution.png"), dpi=160, bbox_inches="tight")
    plt.close(fig)

    # (B) field vs terrain (shaded relief, full region, draped fields)
    fig, ax = plt.subplots(1, 3, figsize=(17, 6), subplot_kw={"projection": PC})
    ax[0].set_extent(ext, crs=PC)
    ax[0].add_image(ti, 7)
    ax[0].coastlines("10m", linewidth=0.6, color="white")
    ax[0].set_title("terrain (shaded relief = elevation)", fontsize=10)
    for a, fld, tl in [(ax[1], b, "ConvCNP baseline field (DEM)"), (ax[2], t, "TESSERA field (DEM)")]:
        a.set_extent(ext, crs=PC)
        a.add_image(ti, 7)
        im = a.imshow(fld, origin="upper", extent=ext, transform=PC, cmap=cm, vmin=lo, vmax=hi,
                      alpha=0.40, zorder=2)
        a.coastlines("10m", linewidth=0.6, color="white")
        a.set_title(tl, fontsize=10)
    fig.colorbar(im, ax=ax, shrink=0.7, pad=0.02, label=f"{v} ({u})")
    fig.suptitle(f"Actual {v} field over terrain (high-res DEM) — {ts} UTC", fontsize=12)
    fig.text(0.5, 0.02, ATTR, ha="center", fontsize=7, color="0.4")
    fig.savefig(R.fig(v, ts, "_dem_field_vs_terrain.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(v, "saved dem contribution + field_vs_terrain")
