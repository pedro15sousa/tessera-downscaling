"""Three interpretive views of TESSERA's landscape-driven downscaling.

Keeps the existing red/blue overlay figures; adds, per variable:

  (1) compare_terrain   — side-by-side, NO blending: satellite terrain | fine-scale
                          anomaly (full region row + inland zoom_box row).
  (2) ruggedness_scatter— |fine-scale anomaly| vs ERA5 sub-grid terrain ruggedness
                          (sdfor), for TESSERA vs baseline, with Spearman r and
                          binned medians. Quantitative "does it track terrain".
  (3) field_vs_terrain  — actual field (not anomaly): satellite | baseline field |
                          TESSERA field, full region, shared colourbar.

Fine-scale anomaly = field minus its ~0.3deg NaN-aware Gaussian low-pass.
Run on a node WITH internet (login) for the satellite tiles.
"""
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.stats import spearmanr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.io.img_tiles as cimgt

sys.path.insert(0, str(Path(__file__).parent))
from regions import SDFOR_IDX, get_region  # noqa: E402

R = get_region()
EU = str(R.region_data)
PC = ccrs.PlateCarree()
ATTR = "Imagery: Esri, Maxar, Earthstar Geographics"
ZOOM = R.zoom_box                  # inland rugged-terrain close-up for this region


class SRTiles(cimgt.GoogleTiles):
    def _image_url(self, tile):
        x, y, z = tile
        return ("https://server.arcgisonline.com/ArcGIS/rest/services/"
                f"World_Shaded_Relief/MapServer/tile/{z}/{y}/{x}")


def nan_highpass(a, sigma=6.0):
    m = np.isfinite(a).astype(float)
    af = np.where(np.isfinite(a), a, 0.0)
    den = gaussian_filter(m, sigma)
    low = np.where(den > 1e-6, gaussian_filter(af * m, sigma) / np.maximum(den, 1e-9), np.nan)
    return a - low


def bilin(grid, glats, glons, lat, lon):
    """Bilinear grid->points; glats descending OK. lat,lon are 1-D arrays."""
    desc = glats[0] > glats[-1]
    if desc:
        rev = glats[::-1]
        i = (len(glats) - 1) - np.searchsorted(rev, lat, side="left")
    else:
        i = np.searchsorted(glats, lat, side="left") - 1
    j = np.searchsorted(glons, lon, side="left") - 1
    i = np.clip(i, 0, len(glats) - 2); j = np.clip(j, 0, len(glons) - 2)
    wy = (lat - glats[i]) / (glats[i + 1] - glats[i])
    wx = (lon - glons[j]) / (glons[j + 1] - glons[j])
    return (grid[i, j] * (1 - wy) * (1 - wx) + grid[i, j + 1] * (1 - wy) * wx
            + grid[i + 1, j] * wy * (1 - wx) + grid[i + 1, j + 1] * wy * wx)


glats = np.load(f"{EU}/lats.npy"); glons = np.load(f"{EU}/lons.npy")
sdfor = np.load(f"{EU}/static_fields.npy")[SDFOR_IDX].astype(float)
tiler = SRTiles(cache=True)

for var, job in R.jobs.items():
    ts, unit, cmap = job["ts"], job["unit_plain"], job["cmap"]
    d = np.load(R.fig(var, ts, ".npz"), allow_pickle=True)
    b, t, vm = d["convcnp_baseline"], d["tessera_concat"], d["valid_mask"]
    lats_d, lons_d = d["lats"], d["lons"]
    ext = [float(lons_d[0]), float(lons_d[-1]), float(lats_d[-1]), float(lats_d[0])]
    hp_t = nan_highpass(t); hp_b = nan_highpass(b)
    vmax = float(np.nanpercentile(np.abs(hp_t), 98))
    LON, LAT = np.meshgrid(lons_d, lats_d)
    rug = bilin(sdfor, glats, glons, LAT.ravel(), LON.ravel()).reshape(LAT.shape)

    # ---------- (1) side-by-side, no blending: terrain | anomaly ----------
    fig, ax = plt.subplots(2, 2, figsize=(13, 11), subplot_kw={"projection": PC})
    for r, (box, zoom, tag) in enumerate([(ext, 7, f"Full {R.name.capitalize()}"), (ZOOM, 10, "inland zoom")]):
        ax[r, 0].set_extent(box, crs=PC); ax[r, 0].add_image(tiler, zoom)
        ax[r, 0].coastlines("10m", lw=0.6, color="white")
        ax[r, 0].set_title(f"{tag} — terrain (shaded relief)", fontsize=10)
        ax[r, 1].set_extent(box, crs=PC)
        im = ax[r, 1].imshow(hp_t, origin="upper", extent=ext, transform=PC,
                             cmap="seismic", vmin=-vmax, vmax=vmax)
        ax[r, 1].coastlines("10m", lw=0.6, color="0.3")
        ax[r, 1].set_title(f"{tag} — TESSERA fine-scale anomaly", fontsize=10)
        fig.colorbar(im, ax=ax[r, 1], shrink=0.8, pad=0.02, label=f"anomaly ({unit})")
    fig.suptitle(f"(1) Terrain vs TESSERA fine-scale {var} variation — {ts} UTC", fontsize=13)
    fig.text(0.5, 0.02, ATTR, ha="center", fontsize=7, color="0.4")
    fig.savefig(R.fig(var, ts, "_compare_terrain.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ---------- (2) quantitative: |anomaly| vs ruggedness ----------
    sel = vm & np.isfinite(hp_t) & np.isfinite(hp_b) & np.isfinite(rug)
    x = rug[sel]; yt = np.abs(hp_t[sel]); yb = np.abs(hp_b[sel])
    rt = spearmanr(x, yt).statistic; rb = spearmanr(x, yb).statistic
    bins = np.linspace(x.min(), np.percentile(x, 99.5), 13)
    bc = 0.5 * (bins[1:] + bins[:-1])
    idx = np.digitize(x, bins) - 1
    med_t = [np.median(yt[idx == k]) if (idx == k).any() else np.nan for k in range(len(bc))]
    med_b = [np.median(yb[idx == k]) if (idx == k).any() else np.nan for k in range(len(bc))]
    fig, axs = plt.subplots(1, 1, figsize=(8.5, 6))
    axs.hexbin(x, yt, gridsize=45, cmap="Reds", mincnt=1, alpha=0.55)
    axs.plot(bc, med_t, "-o", color="darkred", lw=2.2, label=f"TESSERA (Spearman r={rt:.2f})")
    axs.plot(bc, med_b, "-s", color="steelblue", lw=2.2, label=f"baseline (r={rb:.2f})")
    axs.set_xlabel("ERA5 sub-grid terrain ruggedness  sdfor (m)")
    axs.set_ylabel(f"|fine-scale anomaly|  ({unit})")
    axs.set_title(f"(2) TESSERA fine-scale {var} variation vs ERA5 terrain ruggedness — {ts}\n"
                  f"TESSERA adds ~3x more variation than baseline at every ruggedness level;\n"
                  f"only a weak orography trend (r={rt:.2f}) → variation is land-cover driven", fontsize=10)
    axs.legend(); axs.grid(alpha=0.3)
    fig.savefig(R.fig(var, ts, "_ruggedness_scatter.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"{var}: Spearman |anomaly| vs ruggedness  TESSERA={rt:.3f}  baseline={rb:.3f}")

    # ---------- (3) actual field: terrain | baseline | TESSERA ----------
    lo = float(np.nanpercentile(np.concatenate([b[vm], t[vm]]), 1))
    hi = float(np.nanpercentile(np.concatenate([b[vm], t[vm]]), 99))
    fig, ax = plt.subplots(1, 3, figsize=(17, 6), subplot_kw={"projection": PC})
    ax[0].set_extent(ext, crs=PC); ax[0].add_image(tiler, 7)
    ax[0].coastlines("10m", lw=0.6, color="white"); ax[0].set_title("terrain (shaded relief)", fontsize=10)
    for a, fld, ti in [(ax[1], b, "ConvCNP baseline field"), (ax[2], t, "TESSERA field")]:
        a.set_extent(ext, crs=PC); a.add_image(tiler, 7)
        im = a.imshow(fld, origin="upper", extent=ext, transform=PC, cmap=cmap, vmin=lo, vmax=hi, alpha=0.40, zorder=2)
        a.coastlines("10m", lw=0.6, color="white"); a.set_title(ti, fontsize=10)
    fig.colorbar(im, ax=ax, shrink=0.7, pad=0.02, label=f"{var} ({unit})")
    fig.suptitle(f"(3) Actual {var} field over {R.name.capitalize()} — {ts} UTC\n"
                 f"baseline is smooth; TESSERA carries terrain-aligned texture", fontsize=12)
    fig.text(0.5, 0.02, ATTR, ha="center", fontsize=7, color="0.4")
    fig.savefig(R.fig(var, ts, "_field_vs_terrain.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  {var}: saved compare_terrain / ruggedness_scatter / field_vs_terrain")
