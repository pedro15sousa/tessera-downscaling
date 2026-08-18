"""Fine-structure analysis + TESSERA-contribution figures for a region's maps.

Region selected by the REGION env var (default "iberia"); reads
<region>_<var>_<ts>.npz (from generate_maps.py) under outputs/<region>/ and:
  * prints the fine-scale-structure energy (std of a ~0.15deg high-pass) of each
    of the three maps — quantifying how much sub-grid detail each adds,
  * saves a [ConvCNP baseline | TESSERA | TESSERA-baseline] contribution figure.

The high-pass = map - (NaN-aware Gaussian low-pass), so it isolates structure
finer than the smoothing scale; ocean (NaN) cells are excluded throughout.
"""
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from regions import get_region  # noqa: E402

R = get_region()


def nan_highpass(a, sigma=3.0):
    """sigma=3 cells ~= 0.15 deg. NaN-aware (normalised convolution)."""
    m = np.isfinite(a).astype(float)
    af = np.where(np.isfinite(a), a, 0.0)
    den = gaussian_filter(m, sigma)
    low = np.where(den > 1e-6, gaussian_filter(af * m, sigma) / np.maximum(den, 1e-9), np.nan)
    return a - low


for var, job in R.jobs.items():
    ts, unit, cmap = job["ts"], job["unit_plain"], job["cmap"]
    d = np.load(R.fig(var, ts, ".npz"), allow_pickle=True)
    e, b, t = d["era5_interp"], d["convcnp_baseline"], d["tessera_concat"]
    print(f"\n=== {var} @ {ts} ===")
    se, sb, st = (np.nanstd(nan_highpass(x)) for x in (e, b, t))
    print(f"  fine-structure std (0.15deg high-pass):")
    print(f"    ERA5-interp = {se:.3f} {unit}")
    print(f"    ConvCNP base= {sb:.3f} {unit}")
    print(f"    TESSERA     = {st:.3f} {unit}   (x{st/sb:.2f} vs baseline)")
    diff = t - b
    print(f"  TESSERA-baseline: mean={np.nanmean(diff):+.3f}  std={np.nanstd(diff):.3f}  "
          f"p5/p95={np.nanpercentile(diff, 5):+.2f}/{np.nanpercentile(diff, 95):+.2f}  "
          f"maxabs={np.nanmax(np.abs(diff)):.2f} {unit}")

    lats, lons = d["lats"], d["lons"]
    ext = [float(lons[0]), float(lons[-1]), float(lats[-1]), float(lats[0])]
    try:
        import cartopy.crs as ccrs
        proj = ccrs.PlateCarree()
        cart = True
        fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.2), subplot_kw={"projection": proj})
    except Exception:
        cart = False
        fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.2))
    vmax = float(np.nanpercentile(np.abs(diff), 98))
    lo, hi = float(np.nanpercentile(b, 1)), float(np.nanpercentile(b, 99))
    panels = [("ConvCNP baseline", b, cmap, lo, hi),
              ("TESSERA lat16 concat", t, cmap, lo, hi),
              ("TESSERA - baseline", diff, "RdBu_r", -vmax, vmax)]
    for ax, (ti, ar, cm, vm, vx) in zip(axes, panels):
        if cart:
            im = ax.imshow(ar, origin="upper", extent=ext, transform=proj, cmap=cm, vmin=vm, vmax=vx)
            try:
                ax.coastlines("50m", linewidth=0.5)
            except Exception:
                pass
            ax.set_extent(ext, crs=proj)
        else:
            im = ax.imshow(ar, origin="upper", extent=ext, cmap=cm, vmin=vm, vmax=vx, aspect="auto")
        ax.set_title(ti, fontsize=11)
        fig.colorbar(im, ax=ax, shrink=0.62, pad=0.02)
    fig.suptitle(f"TESSERA contribution - {var} - {ts} UTC", fontsize=13)
    p = R.fig(var, ts, "_tessera_contribution.png")
    fig.savefig(p, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {p.name}")
