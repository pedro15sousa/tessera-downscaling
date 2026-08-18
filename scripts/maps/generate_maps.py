"""Dense regional downscaling maps: ERA5-interp vs ConvCNP baseline vs TESSERA lat16-concat.

Produces, for a single test-window snapshot, three directly-comparable 0.05° maps
over the selected region:

  (1) ERA5 bilinear interpolation       — no learned model
  (2) ConvCNP baseline                  — trained, no TESSERA
  (3) ConvCNP + lat16 latents (concat)  — trained, TESSERA "direct concat"

The ConvCNP is an off-grid interpolation model: the dense region grid cells are
fed as an unordered SET of target points (target_coords (1, N, 2)); the model
bilinearly interpolates its CNN-encoded ERA5 features to each point and runs a
pointwise MLP+head. In "concat" mode targets are mutually independent, so all
valid cells are predicted in one forward pass and scattered back to (151, 261).

Contract notes (verified against the training pipeline):
  * Latents fed as (1, N, 16) z-scored with the TRAINING station-latent stats
    (station_latents_lat16_grad0.5_global_stats.npz) — NOT recomputed from the region.
  * NaN/ocean latent cells (valid_mask=False) are dropped, never fed to the model;
    they stay NaN in the output map.
  * Model outputs are already in PHYSICAL units (°C for t2m, m/s for wind) — there
    is no target normalisation / denormalisation step.
  * Elevation (model is include_elevation=True) uses the ERA5-orography proxy:
    elevation = z/g interpolated to each cell, delta_elevation = 0 (offline, no DEM).
    The fine 0.05° structure is therefore attributable to the TESSERA latents.

The region is selected by the REGION env var (default "iberia"); grid dimensions
are derived from the dense npz, so the same script serves iberia, norway, etc.

Run (CPU is sufficient):
  .venv/bin/python projects/tessera_downscaling/scripts/maps/generate_maps.py
  REGION=norway .venv/bin/python projects/tessera_downscaling/scripts/maps/generate_maps.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path("/lus/lfs1aip2/projects/u6do/pmms2/end-to-end-forecasting")
PROJ = REPO / "projects/tessera_downscaling"
sys.path.insert(0, str(PROJ / "scripts/maps"))
sys.path.insert(0, str(PROJ / "src"))

from regions import G, SEEDS, Z_STATIC_IDX, get_region  # noqa: E402
from tessera_downscaling.data.helpers import build_context_grid  # noqa: E402
from tessera_downscaling.model.convcnp import ConvCNPDownscaler  # noqa: E402

# ---------------------------------------------------------------------------
# Region selection / paths (see regions.py). Module-level EU/RUNS/OUT_DIR are
# kept so station_eval.py can import them region-resolved.
# ---------------------------------------------------------------------------
R = get_region()
BASE = PROJ / ".tmp_output"
EU = R.region_data
RUNS = R.runs
DENSE_NPZ = R.dense_npz
VAE_STATS = BASE / "processed/station_latents_lat16_grad0.5_global_stats.npz"
OUT_DIR = R.out_dir
JOBS = R.jobs

# High-res DEM elevation (built by fetch_dem.py). When present, target_elev uses
# the true per-cell DEM elevation and target_delta_elev = DEM - ERA5 orography;
# otherwise we fall back to the smooth ERA5-orography proxy with delta = 0.
# Outputs get a "_dem" suffix so the proxy maps are preserved side by side.
DEM_PATH = R.dem_path
# MAPS_NO_DEM=1 forces the smooth ERA5-orography proxy even when a DEM exists, so
# the proxy ("") and DEM ("_dem") map sets can both be produced for a region.
USE_DEM = DEM_PATH.exists() and os.environ.get("MAPS_NO_DEM") != "1"
SUF = "_dem" if USE_DEM else ""

# ---------------------------------------------------------------------------
# Bilinear grid->points (replicates scripts/baselines/evaluate_simple_baselines.py)
# ---------------------------------------------------------------------------
def bilinear_grid_to_points(grid, glats, glons, pts):
    """grid (H,W); glats (H,) may be descending; glons (W,) ascending; pts (N,2)=lat,lon."""
    lats = pts[:, 0]
    lons = pts[:, 1]
    lat_desc = glats[0] > glats[-1]
    if lat_desc:
        rev = glats[::-1]
        i_rev = np.searchsorted(rev, lats, side="left")
        i = (len(glats) - 1) - i_rev
    else:
        i = np.searchsorted(glats, lats, side="left") - 1
    j = np.searchsorted(glons, lons, side="left") - 1
    i = np.clip(i, 0, len(glats) - 2)
    j = np.clip(j, 0, len(glons) - 2)
    lat0, lat1 = glats[i], glats[i + 1]
    lon0, lon1 = glons[j], glons[j + 1]
    wy = (lats - lat0) / (lat1 - lat0)
    wx = (lons - lon0) / (lon1 - lon0)
    f00 = grid[i, j]; f01 = grid[i, j + 1]
    f10 = grid[i + 1, j]; f11 = grid[i + 1, j + 1]
    out = (f00 * (1 - wy) * (1 - wx) + f01 * (1 - wy) * wx
           + f10 * wy * (1 - wx) + f11 * wy * wx)
    return out.astype(np.float32)


# ---------------------------------------------------------------------------
# Model build + context grid (faithful to evaluate.py)
# ---------------------------------------------------------------------------
def build_model(run_dir, n_ctx, latent_dim):
    cfg = json.load(open(run_dir / "config.json"))
    ck = torch.load(run_dir / "best_model.pt", map_location="cpu", weights_only=False)
    sd = ck["model_state_dict"]
    uses_vae = cfg.get("vae_latents_path") is not None
    tvars = cfg.get("target_variables") or [cfg.get("target_variable")]
    model = ConvCNPDownscaler(
        n_context_channels=n_ctx,
        cnn_hidden=cfg.get("cnn_hidden", 128),
        cnn_layers=cfg.get("cnn_layers", 7),
        cnn_kernel=cfg.get("cnn_kernel", 3),
        setconv_length_scale=cfg.get("setconv_length_scale", 0.5),
        interpolation=cfg.get("interpolation", "setconv"),
        mlp_hidden=cfg.get("mlp_hidden", 128),
        mlp_n_hidden=cfg.get("mlp_n_hidden", 3),
        include_elevation=cfg.get("include_elevation", True),
        target_variables=tvars,
        likelihood_per_variable=cfg.get("likelihood_per_variable"),
        tessera_encoder=None,
        tessera_injection=cfg.get("tessera_injection", "concat"),
        tessera_features_precomputed=uses_vae,
        precomputed_tessera_dim=latent_dim if uses_vae else 0,
        precomputed_drop_prob=0.0,
        precomputed_proj_dim=cfg.get("vae_latents_proj_dim", 0) or 0,
        precomputed_proj_mlp=bool(cfg.get("vae_latents_proj_mlp", False)),
        decoder_kernel=cfg.get("decoder_kernel", "isotropic"),
        use_target_embed_stream=cfg.get("use_target_embed_stream", False),
        target_embed_attention=cfg.get("target_embed_attention", "none"),
    )
    migrated = {(k.replace("setconv.", "interp.", 1) if k.startswith("setconv.") else k): v
                for k, v in sd.items()}
    model.load_state_dict(migrated)
    model.eval()
    return model, cfg, tvars[0]


def build_ctx(cfg, ts, glats, glons):
    include_static = cfg.get("include_static_fields", True)
    if include_static:
        static = np.load(EU / "static_fields.npy")
        stats = np.load(EU / "normalisation_stats.npz")
    else:
        static = None
        stats = np.load(EU / "normalisation_stats_no_static.npz")
    ctx = build_context_grid(
        era5_daily_path=EU / "era5_snapshot" / f"{ts}.npy",
        static_fields=static,
        grid_lats=glats, grid_lons=glons,
        date_str=ts[:10],
        era5_mean=stats["era5_mean"], era5_std=stats["era5_std"],
        hour=int(ts[11:13]),
        drop_dynamic_indices=None, lead_hours=None,
    )
    return ctx.astype(np.float32)


@torch.no_grad()
def run_model(model, var, ctx, glats, glons, pts, elev_pts, delta_pts, latents_z):
    ctx_t = torch.tensor(ctx[None])
    glat_t = torch.tensor(glats.astype(np.float32))
    glon_t = torch.tensor(glons.astype(np.float32))
    tc = torch.tensor(pts[None].astype(np.float32))
    te = torch.tensor(elev_pts[None].astype(np.float32))
    tde = torch.tensor(delta_pts[None].astype(np.float32))   # 0 for proxy, DEM-ERA5 for DEM
    tt = torch.tensor(latents_z[None].astype(np.float32)) if latents_z is not None else None
    out = model(ctx_t, glat_t, glon_t, tc, te, tde, None, tt)
    # MAE-consistent point estimate: the head median. For Gaussian t2m this is
    # exactly mu (byte-identical to before); for the truncated-normal wind head it
    # is the stable truncated median (mae_at_median), matching cross_folder_analysis.
    head = model.heads.heads[var]
    return head.median(out[var])[0].numpy().astype(np.float32)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
# Paper-ready panel names (left -> right). Region, date, resolution, DEM elevation
# and the 3-seed ensembling belong in the LaTeX caption, NOT on the figure. Kept
# module-level so replot_paper_maps.py renders identical labels without a model run.
PANEL_TITLES = [
    "ERA5 bilinear interpolation",
    "ConvCNP (without TESSERA)",
    "ConvCNP with TESSERA",
]
# 4th panel of the *_dem_diff.png variant: where TESSERA places its texture,
# regardless of whether the overall amplitude matches the baseline's.
DIFF_TITLE = r"$\Delta$ = with TESSERA $-$ without TESSERA"
DIFF_CMAP = "RdBu_r"
DIFF_PCT = 99  # symmetric colour limits at this percentile of |delta|
# Hillshade for the terrain-overlay figure (NW sun, standard cartographic default).
HILL_AZ, HILL_ALT, HILL_EXAG = 315.0, 45.0, 2.0
N_CONTOURS = 5   # elevation contours drawn over the draped delta panels


def _make_axes(n, use_cartopy, proj, wspace=None):
    """One row of n map panels, sized so each panel keeps the paper's aspect."""
    import matplotlib.pyplot as plt
    kw = {"subplot_kw": {"projection": proj}} if use_cartopy else {}
    if wspace is not None:
        kw["gridspec_kw"] = {"wspace": wspace}
    return plt.subplots(1, n, figsize=(5.5 * n, 5.2), **kw)


def _inset_cbar(fig, mappable, ax, label):
    """Colorbar hugging `ax`'s right edge WITHOUT stealing subplot space.

    fig.colorbar(..., ax=[...]) shrinks the axes it is attached to, which is what
    made the 4-panel figure's 3rd->4th gap wider than the others. An inset axes is
    positioned in `ax`'s (post-aspect) coordinates instead, so every panel box keeps
    its full width and the inter-panel gaps stay uniform.
    """
    cax = ax.inset_axes([1.035, 0.06, 0.028, 0.88])
    cb = fig.colorbar(mappable, cax=cax)
    cb.set_label(label)
    cb.ax.tick_params(labelsize=9)
    return cb


def _hillshade(elev, lats_d, lons_d):
    """Grey-scale relief from the dense DEM (elev in m, NaN over sea -> 0)."""
    from matplotlib.colors import LightSource
    dlat = abs(float(lats_d[1] - lats_d[0]))
    dlon = abs(float(lons_d[1] - lons_d[0]))
    coslat = float(np.cos(np.deg2rad(np.mean(lats_d))))
    dy, dx = dlat * 111_320.0, dlon * 111_320.0 * coslat      # cell size in metres
    ls = LightSource(azdeg=HILL_AZ, altdeg=HILL_ALT)
    filled = np.where(np.isfinite(elev), elev, 0.0).astype(float)
    return ls, filled, ls.hillshade(filled, vert_exag=HILL_EXAG, dx=dx, dy=dy), dx, dy


def _drape(ax, field, cmap, norm, ls, elev_filled, dx, dy, extent, use_cartopy, proj):
    """Field coloured by (cmap, norm) and relief-shaded by the DEM, NaN transparent."""
    import matplotlib.pyplot as plt
    rgb = plt.get_cmap(cmap)(norm(np.where(np.isfinite(field), field, 0.0)))[..., :3]
    shaded = ls.shade_rgb(rgb, elev_filled, blend_mode="overlay",
                          vert_exag=HILL_EXAG, dx=dx, dy=dy)
    rgba = np.dstack([shaded, np.isfinite(field).astype(float)])
    kw = {"transform": proj} if use_cartopy else {}
    return ax.imshow(rgba, origin="upper", extent=extent, **kw)


def _panel(ax, arr, extent, cmap, vmin, vmax, title, use_cartopy, proj, cfeature):
    """Draw one field panel; returns the image handle (for the colorbar)."""
    if use_cartopy:
        im = ax.imshow(arr, origin="upper", extent=extent, transform=proj,
                       cmap=cmap, vmin=vmin, vmax=vmax)
        ax.coastlines(resolution="10m", linewidth=0.6, color="k")
        try:
            ax.add_feature(cfeature.BORDERS.with_scale("10m"), linewidth=0.4,
                           edgecolor="0.3")
        except Exception:
            pass
        ax.set_extent(extent, crs=proj)
    else:
        im = ax.imshow(arr, origin="upper", extent=extent, cmap=cmap,
                       vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_title(title, fontsize=11)
    return im


def _setup(arrays):
    """Shared plotting prelude: matplotlib backend, cartopy probe, shared vmin/vmax."""
    import matplotlib
    matplotlib.use("Agg")
    stacked = np.concatenate([a[np.isfinite(a)].ravel() for a in arrays])
    vmin, vmax = np.percentile(stacked, [1, 99])
    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
        return ccrs.PlateCarree(), cfeature, True, vmin, vmax
    except Exception:
        return None, None, False, vmin, vmax


def plot_three(arrays, titles, lons_d, lats_d, cmap, unit, out_png):
    import matplotlib.pyplot as plt

    extent = [lons_d[0], lons_d[-1], lats_d[-1], lats_d[0]]  # W,E,S,N
    proj, cfeature, use_cartopy, vmin, vmax = _setup(arrays)
    fig, axes = _make_axes(3, use_cartopy, proj)

    im = None
    for ax, arr, t in zip(axes, arrays, titles):
        im = _panel(ax, arr, extent, cmap, vmin, vmax, t, use_cartopy, proj, cfeature)

    cbar = fig.colorbar(im, ax=list(axes), shrink=0.8, pad=0.02)
    cbar.set_label(unit)
    fig.savefig(out_png, dpi=160, bbox_inches="tight")
    plt.close(fig)


def diff_field(arrays, demean=False):
    """delta = tessera - baseline, its symmetric colour limit, and the removed mean.

    With demean=True the land-mean offset is subtracted, so the panel shows only the
    spatial pattern (where the texture goes) rather than the bulk shift between the
    two models -- the offset is reported separately in the panel title.
    """
    delta = arrays[2] - arrays[1]
    mean = float(np.nanmean(delta))
    if demean:
        delta = delta - mean
    dmax = float(np.nanpercentile(np.abs(delta), DIFF_PCT))
    if not np.isfinite(dmax) or dmax == 0.0:      # degenerate (identical fields)
        dmax = 1e-6
    return delta, dmax, mean


def diff_title(demean, mean, unit):
    if not demean:
        return DIFF_TITLE
    return DIFF_TITLE + "\n" + rf"(mean {mean:+.2f} {unit} removed)"


def _elevation_panel(fig, ax, dem_land, lats_d, lons_d, extent, use_cartopy, proj):
    """Relief-shaded elevation panel + its own metre colorbar (returns e_top)."""
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize

    ls, elev_filled, _, dx, dy = _hillshade(dem_land, lats_d, lons_d)
    e_norm = Normalize(0.0, float(np.nanpercentile(dem_land, 99.5)))
    _drape(ax, dem_land, "terrain", e_norm, ls, elev_filled, dx, dy, extent,
           use_cartopy, proj)
    if use_cartopy:
        ax.coastlines(resolution="10m", linewidth=0.6, color="k")
        ax.set_extent(extent, crs=proj)
    ax.set_title("terrain elevation", fontsize=11)
    _inset_cbar(fig, ScalarMappable(norm=e_norm, cmap="terrain"), ax, "m")


def plot_three_plus_diff(arrays, titles, lons_d, lats_d, cmap, unit, out_png,
                         demean=False, dem_grid=None):
    """plot_three + a 4th panel with the TESSERA-minus-baseline difference map.

    `arrays` is [era5_interp, convcnp_baseline, tessera_concat]; the difference
    is tessera - baseline (arrays[2] - arrays[1]) on a symmetric diverging scale
    with its own colorbar, so texture placement is visible even where the two
    fields have the same overall amplitude/range. demean=True removes the land-mean
    offset first (see diff_field).

    `dem_grid` (n_lat, n_lon metres), when given, adds a 5th relief-shaded elevation
    panel so the delta pattern can be read against the topography in one row.

    All colorbars are inset, so every panel stays equally wide and equally spaced.
    """
    import matplotlib.pyplot as plt

    extent = [lons_d[0], lons_d[-1], lats_d[-1], lats_d[0]]  # W,E,S,N
    proj, cfeature, use_cartopy, vmin, vmax = _setup(arrays)
    delta, dmax, mean = diff_field(arrays, demean)

    n = 5 if dem_grid is not None else 4
    fig, axes = _make_axes(n, use_cartopy, proj, wspace=0.30)
    im = None
    for ax, arr, t in zip(axes, arrays, titles):
        im = _panel(ax, arr, extent, cmap, vmin, vmax, t, use_cartopy, proj, cfeature)
    im_d = _panel(axes[3], delta, extent, DIFF_CMAP, -dmax, dmax,
                  diff_title(demean, mean, unit), use_cartopy, proj, cfeature)

    # Field colorbar sits on the last field panel; the difference gets its own.
    _inset_cbar(fig, im, axes[2], unit)
    _inset_cbar(fig, im_d, axes[3], rf"$\Delta$ {unit}")
    if dem_grid is not None:
        # Sea masked with the model-valid mask so the panel matches the others.
        _elevation_panel(fig, axes[4], np.where(np.isfinite(arrays[2]), dem_grid, np.nan),
                         lats_d, lons_d, extent, use_cartopy, proj)
    fig.savefig(out_png, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return dmax, mean


def plot_diff_over_terrain(arrays, lons_d, lats_d, unit, dem_grid, out_png):
    """Terrain | delta over terrain | demeaned delta over terrain (3 panels).

    `dem_grid` is the region's dense DEM as a (n_lat, n_lon) array in metres (NaN
    over sea). Panel 1 is the elevation itself, relief-shaded; panels 2-3 drape the
    difference over the same relief with `blend_mode="overlay"`, so ridges/valleys
    stay legible through the colour and one can read off whether TESSERA's texture
    follows the topography.
    """
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize

    extent = [lons_d[0], lons_d[-1], lats_d[-1], lats_d[0]]  # W,E,S,N
    proj, cfeature, use_cartopy, _, _ = _setup(arrays)
    ls, elev_filled, _, dx, dy = _hillshade(dem_grid, lats_d, lons_d)
    land = np.isfinite(arrays[2])                       # model-valid cells (no sea)
    dem_land = np.where(land, dem_grid, np.nan)

    raw, dmax, mean = diff_field(arrays)
    anom = raw - mean

    e_top = float(np.nanpercentile(dem_land, 99.5))
    # Contour interval rounded to a readable step (~N_CONTOURS lines over the range).
    step = min([s for s in (100, 200, 250, 500, 1000) if s * N_CONTOURS >= e_top]
               or [1000])
    levels = np.arange(step, e_top, step)

    d_norm = Normalize(-dmax, dmax)
    panels = [
        (DIFF_TITLE, raw),
        (diff_title(True, mean, unit), anom),
    ]
    fig, axes = _make_axes(3, use_cartopy, proj, wspace=0.30)
    _elevation_panel(fig, axes[0], dem_land, lats_d, lons_d, extent, use_cartopy, proj)
    for ax, (title, fld) in zip(axes[1:], panels):
        _drape(ax, fld, DIFF_CMAP, d_norm, ls, elev_filled, dx, dy, extent,
               use_cartopy, proj)
        if len(levels):
            # Relief alone washes out under the pale mid-scale colours; contours give
            # an unambiguous topographic reference to read the delta pattern against.
            ax.contour(lons_d, lats_d, dem_land, levels=levels, colors="k",
                       linewidths=0.25, alpha=0.45,
                       **({"transform": proj} if use_cartopy else {}))
        if use_cartopy:
            ax.coastlines(resolution="10m", linewidth=0.6, color="k")
            ax.set_extent(extent, crs=proj)
        ax.set_title(title, fontsize=11)
        _inset_cbar(fig, ScalarMappable(norm=d_norm, cmap=DIFF_CMAP), ax,
                    rf"$\Delta$ {unit}")
    # Anchored to the middle panel, not the figure: for wide regions (Iberia) the
    # panels occupy only the top of the canvas, and a figure-level caption would sit
    # in dead space that bbox_inches="tight" then has to keep.
    axes[1].text(0.5, -0.06, f"contours every {step:.0f} m", transform=axes[1].transAxes,
                 ha="center", va="top", fontsize=8, color="0.35")
    fig.savefig(out_png, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return dmax, mean


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    glats = np.load(EU / "lats.npy").astype(np.float32)
    glons = np.load(EU / "lons.npy").astype(np.float32)
    elev_grid = (np.load(EU / "static_fields.npy")[Z_STATIC_IDX] / G).astype(np.float32)

    dense = np.load(DENSE_NPZ, allow_pickle=True)
    Z = dense["Z"].astype(np.float32)
    vm = dense["valid_mask"]
    coords = dense["coords"]
    gidx = coords["grid_idx"]
    lat = coords["lat"].astype(np.float32)
    lon = coords["lon"].astype(np.float32)
    assert np.array_equal(gidx, np.arange(len(gidx))), "grid_idx must be 0..N-1 row-major"
    # Grid dims derived from the npz (region-agnostic): rows=lat (desc), cols=lon (asc).
    n_lat, n_lon = len(np.unique(lat)), len(np.unique(lon))
    assert n_lat * n_lon == len(gidx), f"grid {n_lat}x{n_lon} != {len(gidx)} cells"
    lats_d = lat.reshape(n_lat, n_lon)[:, 0]
    lons_d = lon.reshape(n_lat, n_lon)[0, :]

    pts = np.stack([lat[vm], lon[vm]], axis=1).astype(np.float32)   # (Nv, 2)
    era5_orog_pts = bilinear_grid_to_points(elev_grid, glats, glons, pts)  # smooth ERA5 orography
    if USE_DEM:
        dem = np.load(DEM_PATH)[vm].astype(np.float32)
        dem = np.where(np.isfinite(dem), dem, era5_orog_pts)          # fill rare DEM gaps
        elev_pts = dem
        delta_pts = (dem - era5_orog_pts).astype(np.float32)          # true sub-grid anomaly
        print(f"DEM elevation: elev[{elev_pts.min():.0f},{elev_pts.max():.0f}] m  "
              f"delta[{delta_pts.min():.0f},{delta_pts.max():.0f}] m")
    else:
        elev_pts = era5_orog_pts
        delta_pts = np.zeros_like(elev_pts)
    st = np.load(VAE_STATS)
    latents_z = ((Z[vm] - st["mean"]) / st["std"]).astype(np.float32)

    def scatter(vals):
        full = np.full(len(gidx), np.nan, dtype=np.float32)
        full[vm] = vals
        return full.reshape(n_lat, n_lon)

    only_vars = os.environ.get("MAPS_VARS")  # e.g. MAPS_VARS=wind to regenerate wind only
    for var, job in JOBS.items():
        if only_vars and var not in only_vars.split(","):
            continue
        ts = job["ts"]
        print(f"\n=== {var} @ {ts} ===  valid cells: {vm.sum()}/{len(vm)}")

        # (1) ERA5 bilinear interpolation
        era5 = np.load(EU / "era5_snapshot" / f"{ts}.npy")
        if var == "t2m":
            base1 = bilinear_grid_to_points(era5[0], glats, glons, pts) - 273.15
        else:
            u = bilinear_grid_to_points(era5[1], glats, glons, pts)
            v = bilinear_grid_to_points(era5[2], glats, glons, pts)
            base1 = np.sqrt(u * u + v * v)
        m_era5 = scatter(base1)

        # (2)/(3) trained models, ensembled over seeds
        def ensemble(stem, with_latents):
            preds, seed_arrs = [], {}
            for s in SEEDS:
                run = RUNS / f"{stem}_seed{s}"
                if not run.exists():
                    print(f"   [skip missing] {run.name}")
                    continue
                # Build the context grid first (its channel count sets n_ctx),
                # then the model.
                cfg = json.load(open(run / "config.json"))
                ctx = build_ctx(cfg, ts, glats, glons)
                model, _, mvar = build_model(run, n_ctx=ctx.shape[0], latent_dim=Z.shape[1])
                mu = run_model(model, var, ctx, glats, glons, pts, elev_pts, delta_pts,
                               latents_z if with_latents else None)
                preds.append(mu); seed_arrs[s] = scatter(mu)
                print(f"   {run.name}: mu[min/mean/max]={mu.min():.2f}/{mu.mean():.2f}/{mu.max():.2f}")
            return scatter(np.mean(preds, axis=0)), seed_arrs

        m_base, base_seeds = ensemble(job["baseline"], with_latents=False)
        m_tess, tess_seeds = ensemble(job["tessera"], with_latents=True)

        out_png = R.fig(var, ts, f"{SUF}.png")
        plot_three([m_era5, m_base, m_tess], PANEL_TITLES, lons_d, lats_d,
                   job["cmap"], job["unit"], out_png)
        # Same figure + a 4th Delta panel (TESSERA - baseline), raw and mean-removed;
        # the mean-removed one also gets a 5th elevation column when a DEM exists.
        dem_grid = np.load(DEM_PATH).reshape(n_lat, n_lon) if USE_DEM else None
        for dm, tail in [(False, "_diff"), (True, "_diff_anom")]:
            plot_three_plus_diff([m_era5, m_base, m_tess], PANEL_TITLES, lons_d, lats_d,
                                 job["cmap"], job["unit"], R.fig(var, ts, f"{SUF}{tail}.png"),
                                 demean=dm, dem_grid=dem_grid if dm else None)
        if dem_grid is not None:   # Delta draped over the region's relief
            plot_diff_over_terrain([m_era5, m_base, m_tess], lons_d, lats_d, job["unit"],
                                   dem_grid, R.fig(var, ts, f"{SUF}_diff_terrain.png"))

        np.savez(
            R.fig(var, ts, f"{SUF}.npz"),
            era5_interp=m_era5, convcnp_baseline=m_base, tessera_concat=m_tess,
            lats=lats_d, lons=lons_d, valid_mask=vm.reshape(n_lat, n_lon),
            variable=var, timestamp=ts, unit=job["unit"],
            **{f"baseline_seed{s}": a for s, a in base_seeds.items()},
            **{f"tessera_seed{s}": a for s, a in tess_seeds.items()},
        )
        print(f"   saved {out_png.name} and .npz")


if __name__ == "__main__":
    main()
