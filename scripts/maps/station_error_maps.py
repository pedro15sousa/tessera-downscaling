"""Dedicated per-station error figure over the DEM field + terrain.

Three panels (ERA5-interp | ConvCNP baseline | TESSERA). Each drapes its forecast
field faintly over shaded-relief terrain and overlays the per-station absolute
error as markers whose RADIUS and COLOUR INTENSITY both scale with |error|
(shared scale across panels, so the three are directly comparable).

Reads <region>_<var>_<ts>_dem.npz (fields) and <region>_<var>_<ts>_stations.npz (errors).
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
    ts, u, cmf = job["ts"], job["unit_plain"], job["cmap"]
    d = np.load(R.fig(v, ts, "_dem.npz"))
    s = np.load(R.fig(v, ts, "_stations.npz"))
    lats, lons = d["lats"], d["lons"]
    ext = [float(lons[0]), float(lons[-1]), float(lats[-1]), float(lats[0])]
    vm = d["valid_mask"]
    lo, hi = (float(np.nanpercentile(np.concatenate([d["convcnp_baseline"][vm], d["tessera_concat"][vm]]), p))
              for p in (1, 99))
    panels = [("ERA5-interp", d["era5_interp"], s["era5_err"]),
              ("ConvCNP baseline", d["convcnp_baseline"], s["base_err"]),
              ("TESSERA", d["tessera_concat"], s["tess_err"])]
    slat, slon = s["lat"], s["lon"]
    inb = (slon >= ext[0]) & (slon <= ext[1]) & (slat >= ext[2]) & (slat <= ext[3])
    evmax = float(np.nanpercentile(np.concatenate([e[inb] for _, _, e in panels]), 95))

    def msize(e):
        return 16 + 130 * np.clip(e / evmax, 0, 1)

    fig, ax = plt.subplots(1, 3, figsize=(17, 6.2), subplot_kw={"projection": PC})
    sc = None
    for a, (name, fld, err) in zip(ax, panels):
        a.set_extent(ext, crs=PC)
        a.add_image(ti, 7)
        a.imshow(fld, origin="upper", extent=ext, transform=PC, cmap=cmf, vmin=lo, vmax=hi,
                 alpha=0.30, zorder=2)
        e = err[inb]
        sc = a.scatter(slon[inb], slat[inb], transform=PC, s=msize(e), c=e, cmap="plasma",
                       vmin=0, vmax=evmax, edgecolor="k", linewidths=0.4, zorder=5)
        a.coastlines("10m", linewidth=0.6, color="white")
        a.set_title(f"{name}  (station MAE {e.mean():.2f} {u})", fontsize=10)
    cbar = fig.colorbar(sc, ax=ax, shrink=0.7, pad=0.02, extend="max")
    cbar.set_label(f"|station error|  ({u})")
    fig.suptitle(f"Per-station error over forecast field + terrain — {v} — {ts} UTC\n"
                 f"marker radius & colour ∝ |error| (shared scale); field draped faint over shaded relief",
                 fontsize=12)
    fig.text(0.5, 0.02, ATTR, ha="center", fontsize=7, color="0.4")
    fig.savefig(R.fig(v, ts, "_station_errors.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    maes = {n: e[inb].mean() for n, _, e in panels}
    print(f"{v}: in-{R.name} {int(inb.sum())} stns  evmax={evmax:.2f}  "
          f"MAE era5/base/tess = {maes['ERA5-interp']:.2f}/{maes['ConvCNP baseline']:.2f}/{maes['TESSERA']:.2f} {u}")
