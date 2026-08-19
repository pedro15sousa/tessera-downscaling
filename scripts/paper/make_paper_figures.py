"""Self-contained generator for the paper's figures.

Produces every figure of "Earth observation embeddings are effective sub-grid
descriptors for probabilistic weather downscaling" into a single folder
(paper/figures/), restyled for journal submission:

  * authored at the exact printed width (\\textwidth = 6.5 in for the 11pt
    article + 1in-margin letter geometry), so \\includegraphics needs no
    scaling and fonts land at their true size;
  * 8 pt base font / 7 pt ticks+legend, constrained layout, NO bbox_inches
    cropping (cropping would change the delivered width and defeat the
    fixed font sizes);
  * vector PDF everywhere, TrueType (fonttype 42) fonts — no Type-3;
  * the dense-map figures (Figs 1, 3, 4, 9) are copied / re-rendered
    losslessly from the cached outputs of scripts/maps/ (see MAPS_OUT below);
    those caches were produced by the v1-generation runs, as documented in
    scripts/maps/generate_maps.py.

This script deliberately REPLICATES the plotting logic of the notebooks and
scripts it mirrors (cross_folder_analysis.ipynb, analyze_cross_lead.ipynb,
data_efficiency_temporal_rollout.ipynb, residual_structure_analysis.ipynb,
scripts/analysis/norway_descriptor_spaces.py) without importing or modifying
them, reading the same run artefacts from the data root
(tessera_downscaling.paths). Every figure that is recomputed carries a
numeric cross-check against the numbers printed in the paper (the *_EXPECTED
tables); a "[warn]" line means the regenerated figure has drifted.

Run:  uv run python scripts/paper/make_paper_figures.py [--only fig02,fig09] [--out DIR]
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import traceback
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from tessera_downscaling import paths  # noqa: E402

# ---------------------------------------------------------------------------
# Locations
# ---------------------------------------------------------------------------
REPO = Path(__file__).resolve().parents[2]  # repository root
DATA = paths.data_root()  # run artefacts, descriptors, latents
# Cached inputs of the map figures, written by scripts/maps/ (override with
# TESSERA_MAPS_OUT, as regions.py does).
MAPS_OUT = Path(os.environ.get("TESSERA_MAPS_OUT") or paths.paper_figure_inputs_dir())
OUT_DEFAULT = REPO / "paper" / "figures"

# ---------------------------------------------------------------------------
# Journal style: author at final printed size, never crop.
# ---------------------------------------------------------------------------
TEXTWIDTH = 6.5  # inches; \textwidth of 11pt article, letter, margin=1in

STYLE = {
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "legend.title_fontsize": 8,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "lines.linewidth": 1.2,
    "lines.markersize": 3.5,
    "grid.linewidth": 0.5,
    "grid.alpha": 0.35,
    "figure.constrained_layout.use": True,
    "savefig.dpi": 400,  # PNG previews + rasterized artists only
    "pdf.fonttype": 42,  # TrueType — many journals reject Type 3
    "ps.fonttype": 42,
}


def save(fig, out_dir: Path, name: str) -> None:
    """Save PDF (canonical, vector) + a PNG preview. No bbox cropping."""
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "preview").mkdir(exist_ok=True)
    fig.savefig(out_dir / f"{name}.pdf")
    fig.savefig(out_dir / "preview" / f"{name}.png")
    w, h = fig.get_size_inches()
    print(f"  [ok] {name}.pdf  ({w:.2f} x {h:.2f} in)")
    plt.close(fig)


# ===========================================================================
# Fig 1 + Figs 3/4 — frozen map figures: copy / wrap losslessly
# ===========================================================================
PAPER_MAPS = [  # (region, var, ts) — the paper's picks (scripts/maps/regions.py `dates`)
    ("iberia", "t2m", "2022-07-18-12"),
    ("iberia", "wind", "2022-12-12-12"),
    ("norway", "t2m", "2023-01-02-00"),
    ("norway", "wind", "2022-01-30-00"),
]


def fig01(out: Path) -> None:
    """Region overview — vector PDF already exists; copy as-is (frozen)."""
    src = MAPS_OUT / "overview" / "region_overview.pdf"
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, out / "fig01_region_overview.pdf")
    print("  [ok] fig01_region_overview.pdf  (copied vector PDF, frozen)")


# --- Figs 3/4 are re-rendered natively from the saved dense-field npz caches
# (generate_maps.py's plot_three / plot_diff_over_terrain, restyled at print
# size; no model re-run — same arrays the paper's PNGs were drawn from).
MAP_PANEL_TITLES = [
    "ERA5 bilinear interpolation",
    "ConvCNP (topography-only)",
    "ConvCNP with TESSERA",
]
MAP_DIFF_TITLE = r"$\Delta$ = TESSERA $-$ topography-only"
MAP_STYLE = {"t2m": ("turbo", "°C"), "wind": ("viridis", "m s$^{-1}$")}
HILL_AZ, HILL_ALT, HILL_EXAG = 315.0, 45.0, 2.0
N_CONTOURS = 5


def _map_inputs(reg, var, ts):
    d = np.load(
        MAPS_OUT / reg / f"{var}_{ts}" / f"{reg}_{var}_{ts}_dem.npz", allow_pickle=True
    )
    arrays = [
        np.asarray(d["era5_interp"]),
        np.asarray(d["convcnp_baseline"]),
        np.asarray(d["tessera_concat"]),
    ]
    lons, lats = np.asarray(d["lons"]), np.asarray(d["lats"])
    dem = np.load(
        DATA / "processed" / "dense" / reg / f"{reg}_0.05deg_dem.npy"
    ).reshape(len(lats), len(lons))
    cmap, unit = MAP_STYLE[var]
    return arrays, lons, lats, dem, cmap, unit


def _fit_height(fig, pad=0.02, iters=4):
    """Shrink the figure height onto its rendered content — sizing the canvas
    instead of bbox-cropping at save, so the authored width survives."""
    for _ in range(iters):
        fig.canvas.draw()
        tb = fig.get_tightbbox(fig.canvas.get_renderer())
        w, h = fig.get_size_inches()
        want = (tb.y1 - tb.y0) + 2 * pad
        if abs(want - h) < 0.03:
            break
        fig.set_size_inches(w, want)


def _hillshade(elev, lats, lons):
    from matplotlib.colors import LightSource

    dlat, dlon = abs(float(lats[1] - lats[0])), abs(float(lons[1] - lons[0]))
    coslat = float(np.cos(np.deg2rad(np.mean(lats))))
    dy, dx = dlat * 111_320.0, dlon * 111_320.0 * coslat
    ls = LightSource(azdeg=HILL_AZ, altdeg=HILL_ALT)
    filled = np.where(np.isfinite(elev), elev, 0.0).astype(float)
    return ls, filled, dx, dy


def _drape(ax, field, cmap, norm, ls, elev_filled, dx, dy, extent, proj):
    rgb = plt.get_cmap(cmap)(norm(np.where(np.isfinite(field), field, 0.0)))[..., :3]
    shaded = ls.shade_rgb(
        rgb, elev_filled, blend_mode="overlay", vert_exag=HILL_EXAG, dx=dx, dy=dy
    )
    rgba = np.dstack([shaded, np.isfinite(field).astype(float)])
    return ax.imshow(rgba, origin="upper", extent=extent, transform=proj)


def _map_cbar(fig, mappable, ax, label):
    """Inset colorbar hugging the (aspect-adjusted) map, as the original."""
    cax = ax.inset_axes([1.04, 0.04, 0.05, 0.92])
    cb = fig.colorbar(mappable, cax=cax)
    cb.set_label(label, fontsize=7)
    cb.ax.tick_params(labelsize=7)
    return cb


def fig03(out: Path) -> None:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    proj = ccrs.PlateCarree()
    for reg, var, ts in PAPER_MAPS:
        arrays, lons, lats, _dem, cmap, unit = _map_inputs(reg, var, ts)
        extent = [lons[0], lons[-1], lats[-1], lats[0]]
        stacked = np.concatenate([a[np.isfinite(a)].ravel() for a in arrays])
        vmin, vmax = np.percentile(stacked, [1, 99])
        fig, axes = plt.subplots(
            1, 3, figsize=(TEXTWIDTH, 3.4), subplot_kw={"projection": proj}
        )
        im = None
        for ax, arr, t in zip(axes, arrays, MAP_PANEL_TITLES, strict=False):
            im = ax.imshow(
                arr,
                origin="upper",
                extent=extent,
                transform=proj,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
            )
            ax.coastlines(resolution="10m", linewidth=0.4, color="k")
            try:
                ax.add_feature(
                    cfeature.BORDERS.with_scale("10m"), linewidth=0.3, edgecolor="0.3"
                )
            except Exception:
                pass
            ax.set_extent(extent, crs=proj)
            ax.set_title(t, fontsize=7.5)
        cb = fig.colorbar(im, ax=list(axes), shrink=0.85, pad=0.015)
        cb.set_label(unit, fontsize=7)
        cb.ax.tick_params(labelsize=7)
        _fit_height(fig)
        save(fig, out, f"fig03_{reg}_{var}_{ts}_dem")


def fig04(out: Path) -> None:
    import cartopy.crs as ccrs
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize

    proj = ccrs.PlateCarree()
    for reg, var, ts in PAPER_MAPS:
        arrays, lons, lats, dem, _cmap, unit = _map_inputs(reg, var, ts)
        extent = [lons[0], lons[-1], lats[-1], lats[0]]
        land = np.isfinite(arrays[2])
        dem_land = np.where(land, dem, np.nan)
        ls, elev_filled, dx, dy = _hillshade(dem_land, lats, lons)

        raw = arrays[2] - arrays[1]
        mean = float(np.nanmean(raw))
        anom = raw - mean
        dmax = float(np.nanpercentile(np.abs(raw), 99)) or 1e-6
        d_norm = Normalize(-dmax, dmax)

        e_top = float(np.nanpercentile(dem_land, 99.5))
        step = min(
            [s for s in (100, 200, 250, 500, 1000) if s * N_CONTOURS >= e_top] or [1000]
        )
        levels = np.arange(step, e_top, step)

        fig, axes = plt.subplots(
            1, 3, figsize=(TEXTWIDTH, 3.4), subplot_kw={"projection": proj}
        )
        # elevation panel
        e_norm = Normalize(0.0, e_top)
        _drape(
            axes[0], dem_land, "terrain", e_norm, ls, elev_filled, dx, dy, extent, proj
        )
        axes[0].coastlines(resolution="10m", linewidth=0.4, color="k")
        axes[0].set_extent(extent, crs=proj)
        axes[0].set_title("terrain elevation", fontsize=7)
        _map_cbar(fig, ScalarMappable(norm=e_norm, cmap="terrain"), axes[0], "m")
        # raw + demeaned increment panels
        panels = [
            (MAP_DIFF_TITLE, raw),
            (MAP_DIFF_TITLE + "\n" + rf"(mean {mean:+.2f} {unit} removed)", anom),
        ]
        for ax, (title, fld) in zip(axes[1:], panels, strict=False):
            _drape(ax, fld, "RdBu_r", d_norm, ls, elev_filled, dx, dy, extent, proj)
            if len(levels):
                ax.contour(
                    lons,
                    lats,
                    dem_land,
                    levels=levels,
                    colors="k",
                    linewidths=0.2,
                    alpha=0.45,
                    transform=proj,
                )
            ax.coastlines(resolution="10m", linewidth=0.4, color="k")
            ax.set_extent(extent, crs=proj)
            ax.set_title(title, fontsize=7)
            _map_cbar(
                fig, ScalarMappable(norm=d_norm, cmap="RdBu_r"), ax, rf"$\Delta$ {unit}"
            )
        axes[1].text(
            0.5,
            -0.06,
            f"contours every {step:.0f} m",
            transform=axes[1].transAxes,
            ha="center",
            va="top",
            fontsize=7,
            color="0.35",
        )
        _fit_height(fig)
        save(fig, out, f"fig04_{reg}_{var}_{ts}_dem_diff_terrain")


# ===========================================================================
# Fig 2 — CRPS uplift per region (replicates cross_folder_analysis cell 17)
# ===========================================================================
SEEDS = [42, 123, 456]

# bbox (deg) -> area km^2, lat-cosine corrected (notebooks/_helpers.py)
_BBOX_DEG = {
    "europe": (35.0, 75.0, -24.0, 40.0),
    "us": (24.0, 50.0, -125.0, -66.0),
    "east_asia": (20.0, 46.0, 100.0, 146.0),
    "australia": (-44.0, -10.0, 112.0, 154.0),
    "southern_africa": (-35.0, -15.0, 15.0, 35.0),
}


def _bbox_area_km2(a, b, c, d):
    import math

    cos_lat = math.cos(math.radians(0.5 * (a + b)))
    return abs((b - a) * 111.0 * (d - c) * 111.0 * cos_lat)


BBOX_AREA = {k: _bbox_area_km2(*v) for k, v in _BBOX_DEG.items()}

_MAE_COL = {"gaussian": "mae", "truncated_normal": "mae_at_median"}
_RMSE_COL = {"gaussian": "rmse", "truncated_normal": "rmse_at_mean"}


def _read(run_dir: Path, var: str):
    """Per-seed metrics from one run dir (cross_folder_analysis cell 14)."""
    res = None
    for fn in ("test_summary.json", "test_results.json"):
        p = run_dir / fn
        if p.exists():
            try:
                res = json.loads(p.read_text())
                break
            except (json.JSONDecodeError, ValueError):
                continue
    if res is None:
        return None
    hs = res.get("head_spec") or {}
    dist = hs.get(var, {}).get("distribution", "gaussian")
    if dist == "truncated_normal":
        for old, new in (("mae", "mae_at_median"), ("rmse", "rmse_at_mean")):
            ok, nk = f"{var}_{old}", f"{var}_{new}"
            if nk not in res and ok in res:
                res[nk] = res[ok]
    g = lambda k: res.get(f"{var}_{k}")  # noqa: E731
    return dict(
        mae=g(_MAE_COL.get(dist, "mae")),
        rmse=g(_RMSE_COL.get(dist, "rmse")),
        nll=g("nll"),
        crps=g("crps"),
        n=g("n_predictions"),
        stations=g("n_test_stations"),
    )


def _agg(region: str, run_name: str, var: str):
    recs = [
        r
        for s in SEEDS
        if (r := _read(DATA / f"training_runs_{region}" / f"{run_name}_seed{s}", var))
        is not None
    ]
    if not recs:
        return None

    def ms(k):
        vals = [r[k] for r in recs if r[k] is not None and r[k] == r[k]]
        return (
            float(np.mean(vals)) if vals else float("nan"),
            float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
        )

    out = {"n_seeds": len(recs)}
    for k in ("mae", "rmse", "nll", "crps", "n", "stations"):
        out[k], out[k + "_std"] = ms(k)
    return out


FIG2_REGIONS = {  # folder -> (display name, bbox key)
    "snapshot_14y_eu": ("Europe", "europe"),
    "snapshot_14y_us": ("USA", "us"),
    "snapshot_14y_east_asia": ("E. Asia", "east_asia"),
    "snapshot_14y_southern_africa": ("S. Africa", "southern_africa"),
    "snapshot_14y_australia": ("Australia", "australia"),
}
TESS_SUFFIX = "_tessera_1B-M_2017"
FIG2_SERIES = [  # (var, label, colour, (baseline run, tessera run))
    (
        "t2m",
        "2 m temperature",
        "#d95f02",
        (
            "t2m_snap_bilinear_baseline_mtpi_wd",
            "t2m_snap_vae_crop64_lat16_auxon_concat_mtpi",
        ),
    ),
    (
        "wind",
        "10 m wind speed",
        "#1b7837",
        (
            "wind_truncnormal_snap_bilinear_baseline_mtpi_wd",
            "wind_truncnormal_snap_vae_crop64_lat16_auxon_concat_mtpi",
        ),
    ),
]
TRAIN_COUNT_RUN = {
    "t2m": "t2m_snap_vae_lat16_concat_with_elev_mtpi_no_static_wd",
    "wind": "wind_truncnormal_snap_vae_lat16_concat_with_elev_mtpi_no_static_wd",
}

# Verified values from the notebook's stored plot_df (fallback + cross-check).
FIG2_EXPECTED = {
    "Europe": dict(
        t2m=6.332653,
        wind=6.240472,
        n_t2m=5825,
        n_wind=3567,
        rho_t2m=321.972445,
        rho_wind=197.163212,
    ),
    "USA": dict(
        t2m=7.934198,
        wind=6.252978,
        n_t2m=2403,
        n_wind=2310,
        rho_t2m=159.196618,
        rho_wind=153.035450,
    ),
    "E. Asia": dict(
        t2m=19.068599,
        wind=5.225167,
        n_t2m=585,
        n_wind=585,
        rho_t2m=47.335539,
        rho_wind=47.335539,
    ),
    "S. Africa": dict(
        t2m=9.231471,
        wind=5.599284,
        n_t2m=223,
        n_wind=223,
        rho_t2m=49.925590,
        rho_wind=49.925590,
    ),
    "Australia": dict(
        t2m=14.980941,
        wind=7.739540,
        n_t2m=140,
        n_wind=134,
        rho_t2m=8.930443,
        rho_wind=8.547709,
    ),
}


def _fig2_rows():
    """Recompute the plotted table from /data; fall back to FIG2_EXPECTED."""
    rows = []
    for region, (disp, bbox_key) in FIG2_REGIONS.items():
        rec = {"label": disp}
        bbox = BBOX_AREA[bbox_key] / 1e6
        for var, _lab, _col, (base_run, tess_run) in FIG2_SERIES:
            b = _agg(region, base_run, var)
            t = _agg(region + TESS_SUFFIX, tess_run, var)
            ok = b and t and b["crps"] == b["crps"] and t["crps"] == t["crps"]
            rec[f"uplift_{var}"] = (
                (b["crps"] - t["crps"]) / b["crps"] * 100 if ok else float("nan")
            )
            n_test = t["stations"] if t else float("nan")
            # Train-station count: prefer the notebook's v1 run, fall back to
            # the baseline run's eval_train_stations (station set is
            # latents-independent; verified to match the stored values).
            tr_vals = []
            for run in (TRAIN_COUNT_RUN[var], base_run):
                for s in SEEDS:
                    r = _read(
                        DATA
                        / f"training_runs_{region}"
                        / f"{run}_seed{s}"
                        / "eval_train_stations",
                        var,
                    )
                    if r is not None and r["stations"] == r["stations"]:
                        tr_vals.append(r["stations"])
                if tr_vals:
                    break
            n_train = float(np.mean(tr_vals)) if tr_vals else float("nan")
            rec[f"n_{var}"] = n_test + n_train
            rec[f"rho_{var}"] = rec[f"n_{var}"] / bbox
        rows.append(rec)

    # Cross-check against the notebook's stored values; fall back on failure.
    for rec in rows:
        exp = FIG2_EXPECTED[rec["label"]]
        for var in ("t2m", "wind"):
            for field, want in (
                (f"uplift_{var}", exp[var]),
                (f"n_{var}", exp[f"n_{var}"]),
                (f"rho_{var}", exp[f"rho_{var}"]),
            ):
                got = rec[field]
                if not np.isfinite(got):
                    print(
                        f"  [warn] fig02 {rec['label']}/{field}: no local "
                        f"data, using stored value {want:.3f}"
                    )
                    rec[field] = want
                elif abs(got - want) > max(0.05, 0.002 * abs(want)):
                    print(
                        f"  [warn] fig02 {rec['label']}/{field}: recomputed "
                        f"{got:.3f} != stored {want:.3f}"
                    )
    return rows


def fig02(out: Path) -> None:
    rows = _fig2_rows()
    xs = np.arange(len(rows))
    width = 0.34

    fig, ax = plt.subplots(figsize=(TEXTWIDTH, 2.9))
    for i, (var, lab, col, _runs) in enumerate(FIG2_SERIES):
        off = (i - 0.5) * (width + 0.02)
        vals = np.array([r[f"uplift_{var}"] for r in rows])
        bars = ax.bar(xs + off, vals, width, label=lab, color=col, zorder=3)
        for b, v in zip(bars, vals, strict=False):
            if v == v:
                ax.annotate(
                    f"{v:.1f}%",
                    (b.get_x() + b.get_width() / 2, v),
                    xytext=(0, 2),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                )

    ax.set_ylabel("CRPS % improvement over\nConvCNP (topography-only)")
    ax.yaxis.set_major_formatter(lambda v, _p: f"{v:.0f}%")
    ax.yaxis.set_major_locator(plt.MultipleLocator(5))
    ymax = max(20.0, max(r["uplift_t2m"] for r in rows) * 1.18)
    ax.set_ylim(0, ymax)
    ax.grid(axis="y", linestyle="--", linewidth=0.5, color="#cccccc", zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_xlim(-0.6, len(rows) - 0.4)
    ax.tick_params(axis="x", length=0)

    # Region name + n / rho annotation rows below the axis (original layout,
    # offsets re-tuned for the 7 pt fonts).
    ax.set_xticks(xs)
    ax.set_xticklabels(
        [f"{r['label']}\n \n " for r in rows], fontsize=7.5, linespacing=1.5
    )
    N_DY, RHO_DY = -20, -32

    def _fmt_rho(v):
        return "—" if v != v else (f"{v:.0f}" if v >= 10 else f"{v:.1f}")

    def _below(x, dy, text, **kw):
        ax.annotate(
            text,
            (x, 0),
            xycoords=("data", "axes fraction"),
            xytext=(0, dy),
            textcoords="offset points",
            ha=kw.pop("ha", "center"),
            va="top",
            **kw,
        )

    for x, r in zip(xs, rows, strict=False):
        for i, (var, *_rest) in enumerate(FIG2_SERIES):
            off = (i - 0.5) * (width + 0.02)
            n = r[f"n_{var}"]
            _below(
                x + off,
                N_DY,
                "—" if n != n else f"{int(n):,}",
                fontsize=7,
                color="#333333",
            )
            _below(
                x + off, RHO_DY, _fmt_rho(r[f"rho_{var}"]), fontsize=7, color="#666666"
            )
        for dy in (N_DY, RHO_DY):
            _below(x, dy, "|", fontsize=7, color="#aaaaaa")
    for dy, txt in ((N_DY, "n"), (RHO_DY, "ρ")):
        ax.annotate(
            txt,
            (0, 0),
            xycoords="axes fraction",
            xytext=(-8, dy),
            textcoords="offset points",
            ha="right",
            va="top",
            fontsize=7,
            color="#666666",
        )

    ax.legend(loc="upper left", frameon=False)
    save(fig, out, "fig02_crps_uplift")


def fig02ext(out: Path) -> None:
    """fig02 plus TESSERA's uplift over the extradesc-augmented baseline
    (ConvCNP + the 17-feature hand-crafted land-surface descriptor).
    Dark shade = vs plain no-TESSERA baseline, light = vs + land surface."""
    from matplotlib.patches import Patch

    EXTRA_RUN = {
        "t2m": "t2m_snap_bilinear_baseline_mtpi_extradesc_wd",
        "wind": "wind_truncnormal_snap_bilinear_baseline_mtpi_extradesc_wd",
    }
    SHADE = {
        ("t2m", "base"): "#d95f02",
        ("t2m", "extra"): "#fdae6b",
        ("wind", "base"): "#1b7837",
        ("wind", "extra"): "#7fbf7b",
    }

    rows = _fig2_rows()
    # add the vs-extradesc uplifts + the paper's appendix cross-check
    extra_gain = {"t2m": [], "wind": []}
    for rec, (region, _d) in zip(rows, FIG2_REGIONS.items(), strict=False):
        for var, _lab, _col, (base_run, tess_run) in FIG2_SERIES:
            b = _agg(region, base_run, var)
            bx = _agg(region, EXTRA_RUN[var], var)
            t = _agg(region + TESS_SUFFIX, tess_run, var)
            rec[f"uplift_extra_{var}"] = (
                (bx["crps"] - t["crps"]) / bx["crps"] * 100
                if bx and t
                else float("nan")
            )
            if b and bx:
                extra_gain[var].append((b["crps"], bx["crps"]))
    # Paper's aggregate (app:extradesc): ratio of region-mean CRPS.
    for var, want in (("t2m", 3.2), ("wind", 2.2)):
        bm = np.mean([g[0] for g in extra_gain[var]])
        xm = np.mean([g[1] for g in extra_gain[var]])
        got = (bm - xm) / bm * 100
        tag = "ok" if abs(got - want) < 0.1 else "warn"
        print(
            f"  [{tag}] extradesc-vs-plain CRPS gain {var}: {got:.2f}% (paper: {want}%)"
        )

    xs = np.arange(len(rows))
    width, gap = 0.19, 0.015
    fig, ax = plt.subplots(figsize=(TEXTWIDTH, 3.3))
    bars_spec = [
        ("t2m", "base", "uplift_t2m"),
        ("t2m", "extra", "uplift_extra_t2m"),
        ("wind", "base", "uplift_wind"),
        ("wind", "extra", "uplift_extra_wind"),
    ]
    ymax = max(20.0, max(r["uplift_t2m"] for r in rows) * 1.18)
    all_bars = []
    for i, (var, ref, field) in enumerate(bars_spec):
        off = (i - 1.5) * (width + gap)
        vals = np.array([r[field] for r in rows])
        bars = ax.bar(xs + off, vals, width, color=SHADE[(var, ref)], zorder=3)
        all_bars.append((bars, vals))
    # value labels, nudged up just enough to clear the left neighbour's line
    PT_PER_UNIT = 2.2 * 72 / ymax  # approx axes points per y-unit
    for gi in range(len(rows)):
        prev_top = -np.inf
        for bars, vals in all_bars:
            v = vals[gi]
            if v != v:
                continue
            dy, y = 2.0, PT_PER_UNIT * v + 2.0
            if abs(y - prev_top) < 7.5:  # would share the line: bump up
                dy += prev_top + 7.5 - y
                y = prev_top + 7.5
            prev_top = y
            b = bars[gi]
            ax.annotate(
                f"{v:.1f}",
                (b.get_x() + b.get_width() / 2, v),
                xytext=(0, dy),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=7,
            )

    ax.set_ylabel("CRPS % improvement over\nConvCNP without TESSERA")
    ax.yaxis.set_major_formatter(lambda v, _p: f"{v:.0f}%")
    ax.yaxis.set_major_locator(plt.MultipleLocator(5))
    ymax = max(20.0, max(r["uplift_t2m"] for r in rows) * 1.25)
    ax.set_ylim(0, ymax)
    ax.grid(axis="y", linestyle="--", linewidth=0.5, color="#cccccc", zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_xlim(-0.6, len(rows) - 0.4)
    ax.tick_params(axis="x", length=0)

    ax.set_xticks(xs)
    ax.set_xticklabels(
        [f"{r['label']}\n \n " for r in rows], fontsize=7.5, linespacing=1.5
    )
    N_DY, RHO_DY = -20, -32

    def _fmt_rho(v):
        return "—" if v != v else (f"{v:.0f}" if v >= 10 else f"{v:.1f}")

    def _below(x, dy, text, **kw):
        ax.annotate(
            text,
            (x, 0),
            xycoords=("data", "axes fraction"),
            xytext=(0, dy),
            textcoords="offset points",
            ha=kw.pop("ha", "center"),
            va="top",
            **kw,
        )

    pair_off = {"t2m": -(width + gap), "wind": (width + gap)}
    for x, r in zip(xs, rows, strict=False):
        for var in ("t2m", "wind"):
            n = r[f"n_{var}"]
            _below(
                x + pair_off[var],
                N_DY,
                "—" if n != n else f"{int(n):,}",
                fontsize=7,
                color="#333333",
            )
            _below(
                x + pair_off[var],
                RHO_DY,
                _fmt_rho(r[f"rho_{var}"]),
                fontsize=7,
                color="#666666",
            )
        for dy in (N_DY, RHO_DY):
            _below(x, dy, "|", fontsize=7, color="#aaaaaa")
    for dy, txt in ((N_DY, "n"), (RHO_DY, "ρ")):
        ax.annotate(
            txt,
            (0, 0),
            xycoords="axes fraction",
            xytext=(-8, dy),
            textcoords="offset points",
            ha="right",
            va="top",
            fontsize=7,
            color="#666666",
        )

    handles = [
        Patch(
            facecolor=SHADE[("t2m", "base")],
            label="2 m temperature — vs ConvCNP (topography-only)",
        ),
        Patch(
            facecolor=SHADE[("wind", "base")],
            label="10 m wind speed — vs ConvCNP (topography-only)",
        ),
        Patch(
            facecolor=SHADE[("t2m", "extra")],
            label="2 m temperature — vs ConvCNP (hand-crafted surface)",
        ),
        Patch(
            facecolor=SHADE[("wind", "extra")],
            label="10 m wind speed — vs ConvCNP (hand-crafted surface)",
        ),
    ]
    fig.legend(
        handles=handles,
        loc="outside lower center",
        ncol=2,
        frameon=False,
        fontsize=7,
        handlelength=1.2,
        columnspacing=1.5,
        labelspacing=0.4,
    )
    save(fig, out, "fig02_crps_uplift_extended")


# ===========================================================================
# Fig 9 — error-alignment scatter (reconstructed; conventions verified to
# reproduce the paper's numbers exactly from the *_stations.npz files)
# ===========================================================================
def fig09(out: Path) -> None:
    from scipy.stats import pearsonr

    panels = {"t2m": {"unit": "°C"}, "wind": {"unit": "m s$^{-1}$"}}
    data = {v: {"iberia": None, "norway": None} for v in panels}
    for reg, var, ts in PAPER_MAPS:
        d = MAPS_OUT / reg / f"{var}_{ts}"
        maps = np.load(d / f"{reg}_{var}_{ts}.npz", allow_pickle=True)
        st = np.load(d / f"{reg}_{var}_{ts}_stations.npz", allow_pickle=True)
        lats, lons = maps["lats"], maps["lons"]
        ext = [float(lons[0]), float(lons[-1]), float(lats[-1]), float(lats[0])]
        inb = (
            (st["lon"] >= ext[0])
            & (st["lon"] <= ext[1])
            & (st["lat"] >= ext[2])
            & (st["lat"] <= ext[3])
        )
        e = (st["obs"] - st["base_pred"])[inb]  # signed baseline error
        dl = (st["tess_pred"] - st["base_pred"])[inb]  # TESSERA increment
        data[var][reg] = (dl, e)

    colours = {"iberia": "#1f77b4", "norway": "#d62728"}
    fig, axes = plt.subplots(1, 2, figsize=(TEXTWIDTH, 3.55))
    fig.get_layout_engine().set(h_pad=0.1)  # breathing room above the legend
    for ax, (var, meta) in zip(axes, panels.items(), strict=False):
        D = np.concatenate([data[var][r][0] for r in ("iberia", "norway")])
        E = np.concatenate([data[var][r][1] for r in ("iberia", "norway")])
        r2 = pearsonr(D, E).statistic ** 2
        hit = 100 * np.mean(np.sign(D) == np.sign(E))
        beta, icpt = np.polyfit(D, E, 1)

        lim = 1.06 * float(np.max(np.abs(np.concatenate([D, E]))))
        # shaded quadrants where sign(increment) == sign(error)
        for x0, y0 in ((0, 0), (-lim, -lim)):
            ax.add_patch(
                plt.Rectangle(
                    (x0, y0), lim, lim, facecolor="0.92", edgecolor="none", zorder=0
                )
            )
        ax.plot([-lim, lim], [-lim, lim], ls="--", lw=0.8, color="0.55", zorder=2)
        xs = np.array([-lim, lim])
        (fit_line,) = ax.plot(
            xs,
            beta * xs + icpt,
            color="k",
            lw=1.1,
            zorder=3,
            label=f"fit ($\\hat\\beta$={beta:.2f}, $R^2$={r2:.2f})",
        )
        for reg in ("iberia", "norway"):
            dl, e = data[var][reg]
            ax.scatter(dl, e, s=8, alpha=0.75, lw=0, color=colours[reg], zorder=4)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect("equal")
        ax.set_title(f"{var}   $R^2$={r2:.2f} · hit={hit:.0f}% · n={len(E)}")
        u = meta["unit"]
        ax.set_xlabel(
            f"TESSERA increment $\\Delta = \\hat{{y}}_T - \\hat{{y}}_b$  ({u})"
        )
        ax.set_ylabel(f"baseline error  $e = y - \\hat{{y}}_b$  ({u})")
        # per-panel legend: only the fit line (its stats differ per panel)
        ax.legend(
            handles=[fit_line],
            loc="upper left",
            frameon=True,
            framealpha=1.0,
            facecolor="white",
            edgecolor="0.8",
            fontsize=7,
            borderpad=0.3,
            handlelength=1.0,
            handletextpad=0.5,
            borderaxespad=0.3,
        )

    # shared entries once, below the panels
    from matplotlib.lines import Line2D

    shared = [
        Line2D(
            [0], [0], ls="--", lw=0.8, color="0.55", label="perfect ($\\Delta = e$)"
        ),
        Line2D(
            [0],
            [0],
            ls="none",
            marker="o",
            ms=4,
            color=colours["iberia"],
            label="iberia",
        ),
        Line2D(
            [0],
            [0],
            ls="none",
            marker="o",
            ms=4,
            color=colours["norway"],
            label="norway",
        ),
    ]
    fig.legend(
        handles=shared, loc="outside lower center", ncol=3, frameon=False, fontsize=7
    )
    save(fig, out, "fig09_error_alignment")


# ===========================================================================
# Figs 6 & 10 — cross-lead uplift and relative skill decay
# (replicates analyze_cross_lead.ipynb cells 1/2/4/8/13/17, reading the same
# training_runs_snapshot_14y_cross_lead* folders under the data root)
# ===========================================================================
XLEAD_ROOT = DATA / "training_runs_snapshot_14y_cross_lead"
XLEAD_TESS = DATA / "training_runs_snapshot_14y_cross_lead_tessera_1B-M_2017"
XLEAD_REGIONS = ["europe", "east_asia"]
EVAL_SOURCES = {
    "eval_lead0h": "ERA5 (lead 0)",
    "eval_lead6h": "Lead +6h",
    "eval_lead24h": "Lead +24h",
    "eval_lead72h": "Lead +72h",
}
EVAL_ORDER = list(EVAL_SOURCES.values())
XLEAD_CFG = {  # (target, variant) -> (root, config dir stem)
    ("t2m", "baseline"): (XLEAD_ROOT, "t2m_xlead_snap_bilinear_baseline_wd"),
    ("t2m", "concat"): (
        XLEAD_TESS,
        "t2m_xlead_snap_vae_lat16_concat_with_elev_no_static_wd",
    ),
    ("wind", "baseline"): (
        XLEAD_ROOT,
        "wind_truncnormal_xlead_snap_bilinear_baseline_wd",
    ),
    ("wind", "concat"): (
        XLEAD_TESS,
        "wind_truncnormal_xlead_snap_vae_lat16_concat_with_elev_no_static_wd",
    ),
}


def _xlead_frame():
    """region/target/variant/seed/eval_label -> rmse/mae/crps (as cell 4)."""
    import pandas as pd

    rows = []
    for region in XLEAD_REGIONS:
        for (target, variant), (root, cfg) in XLEAD_CFG.items():
            for seed in SEEDS:
                for ek, lbl in EVAL_SOURCES.items():
                    p = root / region / f"{cfg}_seed{seed}" / ek / "test_summary.json"
                    if not p.exists():
                        continue
                    d = json.loads(p.read_text())
                    # truncated-normal head remap (wind): MAE@median, RMSE@mean
                    if target == "wind":
                        rmse = d.get("wind_rmse_at_mean", d.get("wind_rmse"))
                        mae = d.get("wind_mae_at_median", d.get("wind_mae"))
                    else:
                        rmse, mae = d.get("t2m_rmse"), d.get("t2m_mae")
                    rows.append(
                        dict(
                            region=region,
                            target=target,
                            variant=variant,
                            seed=seed,
                            eval_label=lbl,
                            rmse=rmse,
                            mae=mae,
                            crps=d.get(f"{target}_crps"),
                        )
                    )
    df = pd.DataFrame(rows)
    df["eval_label"] = pd.Categorical(
        df["eval_label"], categories=EVAL_ORDER, ordered=True
    )
    # Cross-check against the notebook's exported tidy CSV (same run).
    csv = REPO / "notebooks" / "cross_lead_analysis_outputs" / "all_results_tidy.csv"
    if csv.exists():
        ref = pd.read_csv(csv)[
            ["region", "target", "variant", "seed", "eval_label", "rmse", "crps"]
        ]
        m = df.merge(
            ref,
            on=["region", "target", "variant", "seed", "eval_label"],
            suffixes=("", "_ref"),
        )
        for col in ("rmse", "crps"):
            bad = m[(m[col] - m[f"{col}_ref"]).abs() > 1e-6]
            if len(bad):
                print(
                    f"  [warn] fig06/10: {len(bad)} {col} cells differ from "
                    "all_results_tidy.csv"
                )
    return df


def _xlead_interp_rmse():
    """(region, target, eval_label) -> matched-set interp reference RMSE.
    t2m uses the lapse-corrected interpolation, wind the plain one (cell 17)."""
    stems = {"t2m": "era5_interp_lapse_baseline", "wind": "era5_interp_baseline"}
    ref = {}
    for region in XLEAD_REGIONS:
        for target, stem in stems.items():
            cfg = f"{target}_snap_{stem}_matched_seed42"
            for ek, lbl in EVAL_SOURCES.items():
                p = XLEAD_ROOT / region / cfg / ek / "test_summary.json"
                if p.exists():
                    d = json.loads(p.read_text())
                    ref[(region, target, lbl)] = d.get(f"{target}_rmse")
    return ref


# Notebook cell 17's stored RMSE-uplift table (mean %), for cross-checking.
FIG6_EXPECTED_BLUE = {  # vs ConvCNP (no TESSERA), paired per seed
    ("europe", "t2m"): [5.553705, 5.938666, 5.304488, 4.620109],
    ("europe", "wind"): [8.479223, 9.186018, 8.790127, 6.890073],
    ("east_asia", "t2m"): [16.877285, 17.510553, 16.862992, 15.796480],
    ("east_asia", "wind"): [6.082345, 6.225611, 5.884243, 4.910576],
}
FIG6_EXPECTED_BLACK = {  # vs interpolated 0.25deg context field (matched)
    ("europe", "t2m"): [18.165132, 16.497900, 16.869582, 14.277251],
    ("europe", "wind"): [18.957880, 19.248743, 19.001497, 17.385602],
    ("east_asia", "t2m"): [10.211096, 9.000313, 9.744219, 10.376987],
    ("east_asia", "wind"): [18.423413, 16.921463, 17.053155, 17.853257],
}


def fig06(out: Path) -> None:
    metric = "rmse"
    D = _xlead_frame()
    ref = _xlead_interp_rmse()

    # blue: paired per-seed uplift over the no-TESSERA baseline
    piv = D.pivot_table(
        index=["region", "target", "seed", "eval_label"],
        columns="variant",
        values=metric,
        observed=True,
    )
    piv = piv.dropna(subset=["baseline", "concat"])
    up = ((piv["baseline"] - piv["concat"]) / piv["baseline"] * 100).reset_index()
    up = (
        up.groupby(["region", "target", "eval_label"], observed=True)[0]
        .agg(["mean", "std"])
        .reset_index()
    )

    # black: concat seeds vs deterministic interp reference
    mdl = (
        D[D.variant == "concat"]
        .groupby(["region", "target", "eval_label"], observed=True)[metric]
        .agg(m="mean", s="std")
        .reset_index()
    )
    mdl["ref"] = [
        ref.get((r, t, lab))
        for r, t, lab in zip(mdl.region, mdl.target, mdl.eval_label, strict=False)
    ]
    mdl["mean"] = (mdl["ref"] - mdl["m"]) / mdl["ref"] * 100
    mdl["std"] = mdl["s"] / mdl["ref"] * 100

    for (reg, tgt), want in FIG6_EXPECTED_BLUE.items():
        got = (
            up[(up.region == reg) & (up.target == tgt)]
            .set_index("eval_label")["mean"]
            .reindex(EVAL_ORDER)
            .values
        )
        if not np.allclose(got, want, atol=0.01):
            print(f"  [warn] fig06 blue {reg}/{tgt}: {got} != stored {want}")
    for (reg, tgt), want in FIG6_EXPECTED_BLACK.items():
        got = (
            mdl[(mdl.region == reg) & (mdl.target == tgt)]
            .set_index("eval_label")["mean"]
            .reindex(EVAL_ORDER)
            .values
        )
        if not np.allclose(got, want, atol=0.01):
            print(f"  [warn] fig06 black {reg}/{tgt}: {got} != stored {want}")

    from matplotlib.lines import Line2D

    BLUE, BLACK = "#1f77b4", "#111111"
    LBL_BLUE = "vs ConvCNP (topography-only)"
    LBL_BLACK = (
        "vs bilinear interp of the 0.25° context field (+ fitted lapse rate for t2m)"
    )
    fig, axes = plt.subplots(2, 2, figsize=(TEXTWIDTH, 4.3), sharex=True, sharey="row")
    x = np.arange(len(EVAL_ORDER))
    for ri, target in enumerate(["t2m", "wind"]):
        for ci, region in enumerate(XLEAD_REGIONS):
            ax = axes[ri][ci]
            for frame, col, mk in ((up, BLUE, "o"), (mdl, BLACK, "D")):
                cell = frame[(frame.target == target) & (frame.region == region)]
                a = cell.set_index("eval_label").reindex(EVAL_ORDER)
                ax.errorbar(
                    x,
                    a["mean"].values,
                    yerr=a["std"].values,
                    marker=mk,
                    color=col,
                    lw=1.2,
                    ms=3.5,
                    capsize=2,
                )
            ax.axhline(0, color="k", lw=0.6)
            ax.axvline(0.5, color="k", ls=":", lw=0.6, alpha=0.5)
            ax.grid(True, alpha=0.25)
            ax.set_xticks(x)
            ax.set_xticklabels(EVAL_ORDER, rotation=20, ha="right")
            if ri == 0:
                ax.set_title(region.replace("_", " ").title())
            if ci == 0:
                ax.set_ylabel(
                    f"{target}\n{metric.upper()} uplift (% better)", fontweight="bold"
                )
    handles = [
        Line2D([0], [0], color=BLUE, marker="o", lw=1.2, ms=3.5, label=LBL_BLUE),
        Line2D([0], [0], color=BLACK, marker="D", lw=1.2, ms=3.5, label=LBL_BLACK),
    ]
    fig.legend(
        handles=handles, loc="outside lower center", ncol=2, frameon=False, fontsize=7
    )
    save(fig, out, "fig06_tessera_uplift_rmse")


def fig10(out: Path) -> None:
    metric = "crps"
    D = _xlead_frame()
    piv = D.pivot_table(
        index=["region", "target", "variant", "seed"],
        columns="eval_label",
        values=metric,
        observed=True,
    ).reindex(columns=EVAL_ORDER)
    base = piv[EVAL_ORDER[0]]
    rel = piv.sub(base, axis=0).div(base, axis=0) * 100.0
    g = rel.groupby(level=["region", "target", "variant"], observed=True)
    mean, std = g.mean(), g.std()

    from matplotlib.lines import Line2D

    TARGET_COLORS = {"t2m": "#d62728", "wind": "#1f77b4"}
    REL_DASH = {"baseline": "--", "concat": "-"}
    REL_MARKER = {"baseline": "s", "concat": "o"}
    VLABEL = {"baseline": "ConvCNP (topography-only)", "concat": "ConvCNP with TESSERA"}

    fig, axes = plt.subplots(1, 2, figsize=(TEXTWIDTH, 2.7), sharey=True)
    x = np.arange(len(EVAL_ORDER))
    for ci, region in enumerate(XLEAD_REGIONS):
        ax = axes[ci]
        for target in ("t2m", "wind"):
            for variant in ("baseline", "concat"):
                key = (region, target, variant)
                m = mean.loc[key].reindex(EVAL_ORDER).values
                s = std.loc[key].reindex(EVAL_ORDER).fillna(0).values
                ax.plot(
                    x,
                    m,
                    ls=REL_DASH[variant],
                    marker=REL_MARKER[variant],
                    color=TARGET_COLORS[target],
                    lw=1.2,
                    ms=3.5,
                )
                ax.fill_between(
                    x, m - s, m + s, color=TARGET_COLORS[target], alpha=0.12, lw=0
                )
        ax.axhline(0, color="k", lw=0.6)
        ax.axvline(0.5, color="k", ls=":", lw=0.6, alpha=0.5)
        ax.grid(True, alpha=0.25)
        ax.set_xticks(x)
        ax.set_xticklabels(EVAL_ORDER, rotation=20, ha="right")
        ax.set_title(region.replace("_", " ").title())
        if ci == 0:
            ax.set_ylabel(f"Δ {metric.upper()} vs own lead-0  (%)")

    _hdr = lambda t: Line2D([], [], ls="none", marker="", label=t)  # noqa: E731
    handles = [_hdr(r"$\bf{variable}$")]
    handles += [
        Line2D([0], [0], color=TARGET_COLORS[t], lw=1.8, label=t)
        for t in ("t2m", "wind")
    ]
    handles += [_hdr(r"$\bf{variant}$")]
    handles += [
        Line2D(
            [0],
            [0],
            color="0.35",
            lw=1.2,
            ls=REL_DASH[v],
            marker=REL_MARKER[v],
            ms=3.5,
            label=VLABEL[v],
        )
        for v in ("baseline", "concat")
    ]
    axes[0].legend(
        handles=handles,
        loc="upper left",
        fontsize=7,
        framealpha=0.9,
        handlelength=2.0,
        handletextpad=0.5,
        labelspacing=0.35,
        borderaxespad=0.4,
        borderpad=0.35,
    )
    save(fig, out, "fig10_relative_skill_vs_lead_crps")


# ===========================================================================
# Figs 5 & 8 — descriptor-space probes of the persistent ERA5-interp residual
# (replicates residual_structure_analysis.ipynb §3c/3e/3g; the HPC paths stored
# in the runs' config.json are remapped onto the data root by paths.resolve)
# ===========================================================================
SR_DATASET = DATA / "dataset_timestamp_global"
SR_MIN_N = 100  # 'well-sampled' regions: Europe (898), US (357)
SR_WELL_SAMPLED = {  # folder -> (display name, region key)
    "snapshot_14y_eu": ("Europe", "europe"),
    "snapshot_14y_us": ("United States", "us"),
}
SR_MODELS = {  # baseline stems (latents-independent, base folders)
    "t2m": "t2m_snap_bilinear_baseline_mtpi_wd",
    "wind": "wind_truncnormal_snap_bilinear_baseline_mtpi_wd",
}
SR_TESS_STEM = {  # TESSERA arm = v2 1B-M 2017 generation (notebook cell 2)
    "t2m": "t2m_snap_vae_crop64_lat16_auxon_concat_mtpi",
    "wind": "wind_truncnormal_snap_vae_crop64_lat16_auxon_concat_mtpi",
}
SR_SPACES = ["geographic", "elevation+mTPI", "ERA5-static", "TESSERA"]
SRX_SPACES = [
    "geographic",
    "elevation+mTPI",
    "terrain stats (7f)",
    "land cover (10f)",
    "extended surface (17f)",
    "ERA5-static",
    "TESSERA",
]
SRX_NEW = ["terrain stats (7f)", "land cover (10f)", "extended surface (17f)"]
SR_COLOUR = {
    "geographic": "#7f7f7f",
    "elevation+mTPI": "#ff7f0e",
    "ERA5-static": "#9467bd",
    "TESSERA": "#1f77b4",
    "terrain stats (7f)": "#e377c2",
    "land cover (10f)": "#8c564b",
    "extended surface (17f)": "#c49a6c",
}
SRX_TERRAIN = {
    "elev_mean",
    "elev_std",
    "elev_min",
    "elev_max",
    "slope",
    "dz_dn",
    "dz_de",
}

# Stored notebook tables (§3c/§3g outputs) — cross-check targets.
FIG5_EXPECTED = {
    ("Europe", "t2m"): dict(zip(SR_SPACES, (0.014, 0.646, 0.089, 0.186), strict=False)),
    ("Europe", "wind"): dict(
        zip(SR_SPACES, (0.083, 0.117, 0.165, 0.271), strict=False)
    ),
    ("United States", "t2m"): dict(
        zip(SR_SPACES, (-0.115, 0.003, -0.116, 0.012), strict=False)
    ),
    ("United States", "wind"): dict(
        zip(SR_SPACES, (0.254, 0.037, 0.154, 0.312), strict=False)
    ),
}
FIG8_EXPECTED_EXTRA = {  # the three added spaces only
    ("Europe", "t2m"): (0.143, 0.143, 0.303),
    ("Europe", "wind"): (-0.004, 0.259, 0.314),
    ("United States", "t2m"): (-0.127, -0.107, -0.107),
    ("United States", "wind"): (0.118, 0.205, 0.275),
}

_SR_CACHE: dict = {}


def _sr_stations_df():
    import pandas as pd

    if "stations" not in _SR_CACHE:
        s = pd.read_csv(SR_DATASET / "stations.csv")
        s["station_id"] = s["station_id"].astype(str)
        _SR_CACHE["stations"] = s
    return _SR_CACHE["stations"]


def _sr_elev_lut():
    if "elev" not in _SR_CACHE:
        s = _sr_stations_df()
        _SR_CACHE["elev"] = {
            sid: (float(e), float(de), float(m))
            for sid, e, de, m in zip(
                s.station_id, s.elevation, s.delta_elevation, s.mtpi, strict=False
            )
        }
    return _SR_CACHE["elev"]


def _sr_era5_lut(region_key):
    from scipy.interpolate import RegularGridInterpolator

    key = f"era5_{region_key}"
    if key not in _SR_CACHE:
        rdir = SR_DATASET / "regions" / region_key
        sf = np.load(rdir / "static_fields.npy")
        glat, glon = np.load(rdir / "lats.npy"), np.load(rdir / "lons.npy")
        s = _sr_stations_df()
        s = s[s.region == region_key]
        q = np.column_stack(
            [
                np.clip(s.latitude.to_numpy(), glat.min(), glat.max()),
                np.clip(s.longitude.to_numpy(), glon.min(), glon.max()),
            ]
        )
        E = np.column_stack(
            [
                RegularGridInterpolator(
                    (glat, glon),
                    sf[c],
                    method="linear",
                    bounds_error=False,
                    fill_value=None,
                )(q)
                for c in range(sf.shape[0])
            ]
        ).astype(np.float32)
        _SR_CACHE[key] = {sid: E[i] for i, sid in enumerate(s.station_id)}
    return _SR_CACHE[key]


def _sr_latents(folder, var):
    """station_id -> TESSERA latent, via the 1B-M run's config.json."""
    key = f"lat_{folder}_{var}"
    if key not in _SR_CACHE:
        import pandas as pd

        lut = None
        for s in SEEDS:
            cfgp = (
                DATA
                / f"training_runs_{folder}_tessera_1B-M_2017"
                / f"{SR_TESS_STEM[var]}_seed{s}"
                / "config.json"
            )
            if not cfgp.exists():
                continue
            cfg = json.loads(cfgp.read_text())
            lp = paths.resolve(cfg["vae_latents_path"])
            cp = paths.resolve(cfg["vae_latents_station_csv"])
            if lp.exists() and cp.exists():
                arr = np.load(lp)
                ids = pd.read_csv(cp)["station_id"].astype(str).values
                lut = {str(i): arr[k] for k, i in enumerate(ids)}
                break
        _SR_CACHE[key] = lut
    return _SR_CACHE[key]


def _sr_era5_target(folder, var):
    """station_id -> (lat, lon, mean ERA5-interp residual) (§3c)."""
    root = DATA / f"training_runs_{folder}"
    base_stem = SR_MODELS[var]
    for s in SEEDS:
        ep = root / f"{var}_snap_era5_interp_baseline_seed{s}/test_predictions.npz"
        bp = root / f"{base_stem}_seed{s}/test_predictions.npz"
        bse = root / f"{base_stem}_seed{s}/test_station_errors.npz"
        if not (ep.exists() and bp.exists() and bse.exists()):
            continue
        de, db = np.load(ep), np.load(bp)
        te, tb = de[f"{var}_targets"], db[f"{var}_targets"]
        if not (len(te) == len(tb) and np.allclose(te, tb, equal_nan=True)):
            continue
        resid = (de[f"{var}_predictions"] - te).astype(float)
        sidx = db[f"{var}_station_indices"].astype(int)
        m = np.load(bse, allow_pickle=True)
        sids, lats, lons = m["station_ids"], m["station_lats"], m["station_lons"]
        nst = len(sids)
        ok = np.isfinite(resid)
        sums = np.bincount(sidx[ok], weights=resid[ok], minlength=nst)
        cnts = np.bincount(sidx[ok], minlength=nst)
        return {
            str(sids[i]): (float(lats[i]), float(lons[i]), sums[i] / cnts[i])
            for i in range(nst)
            if cnts[i] > 0
        }
    return {}


def _sr_extra_lut():
    """station_id -> 17-feature extended descriptor (+ column names) (§3g)."""
    if "extra" not in _SR_CACHE:
        import pandas as pd

        arr = np.load(DATA / "processed" / "extra_descriptors.npy")
        ids = (
            pd.read_csv(
                DATA / "processed" / "tessera_global" / "station_list_filtered.csv"
            )["station_id"]
            .astype(str)
            .values
        )
        names = json.loads(
            (DATA / "processed" / "extra_descriptors_names.json").read_text()
        )["columns"]
        assert len(ids) == len(arr) and len(names) == arr.shape[1]
        _SR_CACHE["extra"] = ({s: arr[i] for i, s in enumerate(ids)}, list(names))
    return _SR_CACHE["extra"]


def _sr_cv_r2(X, y, seed=0):
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import KFold, cross_val_score

    if len(y) < 25:
        return np.nan
    rf = RandomForestRegressor(
        n_estimators=200, min_samples_leaf=3, random_state=seed, n_jobs=-1
    )
    return float(
        np.mean(
            cross_val_score(
                rf, X, y, cv=KFold(5, shuffle=True, random_state=seed), scoring="r2"
            )
        )
    )


def _sr_probe(spaces):
    """(display region, var) -> {space: CV R^2} over the well-sampled regions."""
    key = ("probe", tuple(spaces))
    if key in _SR_CACHE:
        return _SR_CACHE[key]
    extra_lut, extra_names = (
        _sr_extra_lut() if set(spaces) & set(SRX_NEW) else (None, [])
    )
    ti = [i for i, c in enumerate(extra_names) if c in SRX_TERRAIN]
    ci = [i for i, c in enumerate(extra_names) if c not in SRX_TERRAIN]
    out = {}
    for folder, (disp, rk) in SR_WELL_SAMPLED.items():
        for var in ("t2m", "wind"):
            tgt = _sr_era5_target(folder, var)
            lut = _sr_latents(folder, var)
            elut, slut = _sr_elev_lut(), _sr_era5_lut(rk)
            shared = set(tgt) & set(lut) & set(elut) & set(slut)
            if extra_lut is not None:
                shared &= set(extra_lut)
            shared = sorted(shared)
            lat = np.array([tgt[s][0] for s in shared])
            lon = np.array([tgt[s][1] for s in shared])
            y = np.array([tgt[s][2] for s in shared])
            mlat = np.radians(lat.mean())
            feat = {
                "geographic": np.column_stack([lon * np.cos(mlat), lat]),
                "elevation+mTPI": np.array([elut[s] for s in shared]),
                "ERA5-static": np.array([slut[s] for s in shared]),
                "TESSERA": np.array([lut[s] for s in shared]),
            }
            if extra_lut is not None:
                ext = np.array([extra_lut[s] for s in shared])
                feat["terrain stats (7f)"] = ext[:, ti]
                feat["land cover (10f)"] = ext[:, ci]
                feat["extended surface (17f)"] = ext
            row = {sp: _sr_cv_r2(feat[sp], y) for sp in spaces}
            row["n"] = len(shared)
            out[(disp, var)] = row
            print(
                f"  [probe] {disp}/{var}: n={len(shared)}  "
                + "  ".join(f"{sp}={row[sp]:+.3f}" for sp in spaces)
            )
    _SR_CACHE[key] = out
    return out


def _sr_check(table, expected, figname, atol=0.03):
    for (disp, var), exp in expected.items():
        got = table.get((disp, var))
        if got is None:
            print(f"  [warn] {figname}: no data for {disp}/{var}")
            continue
        vals = exp.items() if isinstance(exp, dict) else zip(SRX_NEW, exp, strict=False)
        for sp, want in vals:
            if abs(got[sp] - want) > atol:
                print(
                    f"  [warn] {figname} {disp}/{var}/{sp}: recomputed "
                    f"{got[sp]:+.3f} != stored {want:+.3f}"
                )


def _sr_bars(ax, spaces, table, var, wrap=False, rotation=30, ha="right"):
    subs = [table[(disp, v)] for (disp, v) in table if v == var]
    means = np.array([np.mean([s[sp] for s in subs]) for sp in spaces])
    sds = np.array([np.std([s[sp] for s in subs], ddof=0) for sp in spaces])
    x = np.arange(len(spaces))
    ax.bar(
        x,
        means,
        yerr=sds,
        color=[SR_COLOUR[s] for s in spaces],
        edgecolor="black",
        linewidth=0.5,
        zorder=2,
        capsize=2.5,
        error_kw=dict(lw=0.8, ecolor="0.2"),
    )
    ax.axhline(0, color="k", lw=0.6)
    labels = [s.replace(" (", "\n(") if wrap else s for s in spaces]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=rotation, ha=ha)
    ax.grid(axis="y", ls=":", alpha=0.5)
    ax.set_axisbelow(True)
    return means, sds


def _sr_tag(ax, tag):
    ax.text(
        0.02,
        0.97,
        tag,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8,
        fontweight="bold",
    )


def fig05(out: Path) -> None:
    table = _sr_probe(SR_SPACES)
    _sr_check(table, FIG5_EXPECTED, "fig05")
    fig, axes = plt.subplots(1, 2, figsize=(TEXTWIDTH, 2.6))
    for ax, tag, var in zip(axes, "ab", ("t2m", "wind"), strict=False):
        _sr_bars(ax, SR_SPACES, table, var)
        if tag == "a":
            ax.set_ylabel("CV $R^2$ predicting\n(ERA5-interp $-$ obs)")
        _sr_tag(ax, f"({tag}) {var}")
    save(fig, out, "fig05_residual_probe_spaces")


def fig08(out: Path) -> None:
    table = _sr_probe(SRX_SPACES)
    _sr_check(table, FIG5_EXPECTED, "fig08")  # shared four spaces
    _sr_check(table, FIG8_EXPECTED_EXTRA, "fig08")  # the three new ones
    newix = [SRX_SPACES.index(s) for s in SRX_NEW]
    fig, axes = plt.subplots(2, 1, figsize=(TEXTWIDTH, 4.3), sharex=True)
    for ax, tag, var in zip(axes, "ab", ("t2m", "wind"), strict=False):
        ax.axvspan(min(newix) - 0.5, max(newix) + 0.5, color="0.92", zorder=0)
        means, sds = _sr_bars(
            ax, SRX_SPACES, table, var, wrap=True, rotation=0, ha="center"
        )
        lo = min(0.0, float((means - sds).min()))
        hi = float((means + sds).max())
        span = max(hi - lo, 1e-6)
        ax.set_ylim(lo - 0.05 * span, hi + 0.24 * span)
        ax.text(
            (min(newix) + max(newix)) / 2,
            0.99,
            "land surface (hand-crafted)",
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=7,
            color="0.35",
        )
        for xi, m, sd in zip(np.arange(len(SRX_SPACES)), means, sds, strict=False):
            ax.text(
                xi,
                m + sd + 0.02 * span,
                f"{m:+.2f}",
                ha="center",
                va="bottom",
                fontsize=7,
                fontweight="bold",
            )
        ax.set_ylabel("CV $R^2$ predicting\n(ERA5-interp $-$ obs)")
        _sr_tag(ax, f"({tag}) {var}")
    save(fig, out, "fig08_residual_probe_extended")


# ===========================================================================
# Fig 7 — simulated Norway deployment (replicates
# data_efficiency_temporal_rollout.ipynb cells 1/5/6/7-8, MAE variant)
# ===========================================================================
ROLL_FOLDER = "snapshot_14y_eu_temporal_rollout_norway_lat16_mtpi"
ROLL_TESS = "snapshot_14y_eu_temporal_rollout_norway_tessera_1B-M_2017"
ROLL_EXP = REPO / "scripts" / "experiments" / ROLL_FOLDER
SWEEP_X_YEARS = {
    "r0": 0.0,
    "r1mo": 1 / 12,
    "r3mo": 0.25,
    "r6mo": 0.50,
    "r1y": 1.0,
    "r2y": 2.0,
    "r3y": 3.0,
    "r4y": 4.0,
    "r5y": 5.0,
    "r6y": 6.0,
}
SWEEP_ORDER = list(SWEEP_X_YEARS)
ROLL_FAMILY = {  # arch_base -> family
    "t2m_snap_vae_lat16_concat_with_elev_mtpi_no_static_wd": "tessera",
    "t2m_snap_bilinear_baseline_mtpi_wd": "baseline",
    "wind_truncnormal_snap_vae_lat16_concat_with_elev_mtpi_no_static_wd": "tessera",
    "wind_truncnormal_snap_bilinear_baseline_mtpi_wd": "baseline",
}
ERA_REF_ARCH = {
    "t2m": "t2m_snap_era5_interp_lapse_baseline",
    "wind": "wind_snap_era5_interp_baseline",
}
FAMILY_COLOUR = {"tessera": "#1f77b4", "baseline": "#7f7f7f"}
FAMILY_MARKER = {"tessera": "o", "baseline": "s"}
FAMILY_LABEL = {
    "tessera": "ConvCNP with TESSERA",
    "baseline": "ConvCNP (topography-only)",
}
VAR_LABEL = {"t2m": "2 m temperature", "wind": "10 m wind speed"}
VAR_UNIT = {"t2m": "°C", "wind": "m/s"}
NB_LAT, NB_LON = (58.0, 71.0), (4.0, 31.0)  # Norway bbox
_AH_FS = {
    "tick": 7.5,
    "label": 8.5,
    "title": 9,
    "toptick": 7,
    "toplabel": 8,
    "legend": 8,
}
_AH_SQRT = (
    lambda x: np.sign(x) * np.sqrt(np.abs(x)),
    lambda x: np.sign(x) * np.square(x),
)


def _roll_sched():
    if "sched" not in _SR_CACHE:
        _SR_CACHE["sched"] = json.loads(
            (ROLL_EXP / "rollout_schedule.json").read_text()
        )
    return _SR_CACHE["sched"]


def _roll_runs():
    """One record per rollout run dir (both families + the interp refs)."""
    import re

    recs = []
    for folder, want_tess in ((ROLL_FOLDER, False), (ROLL_TESS, True)):
        root = DATA / f"training_runs_{folder}"
        for d in sorted(root.iterdir()):
            m = re.match(r"(?P<exp>.+)_seed(?P<seed>\d+)$", d.name)
            if not m:
                continue
            exp = m.group("exp")
            sweep = "all"
            arch = exp
            for lbl in SWEEP_ORDER:
                if exp.endswith(f"_{lbl}"):
                    arch, sweep = exp[: -len(lbl) - 1], lbl
                    break
            family = ROLL_FAMILY.get(arch)
            if family is None and arch not in ERA_REF_ARCH.values():
                continue
            # TESSERA arm exclusively from the 1B-M folder; the rest from base
            if (family == "tessera") != want_tess:
                continue
            var = "t2m" if arch.startswith("t2m") else "wind"
            recs.append(
                dict(
                    run_dir=d,
                    arch=arch,
                    family=family,
                    var=var,
                    sweep=sweep,
                    x=SWEEP_X_YEARS.get(sweep),
                    seed=int(m.group("seed")),
                )
            )
    return recs


def _roll_station_arrays(run_dir, var):
    d = np.load(Path(run_dir) / "test_station_errors.npz", allow_pickle=True)
    if f"{var}_station_mae" not in d.files:
        return None
    return (
        d["station_ids"].astype(str),
        d["station_lats"],
        d["station_lons"],
        d["subset_per_station"],
        d[f"{var}_station_count"].astype(float),
        d[f"{var}_station_mae"].astype(float),
    )


def _roll_col_sel(ids, lat, lon, sub, cnt, sweep, colkey):
    has = cnt > 0
    in_nor = (
        (lat >= NB_LAT[0])
        & (lat <= NB_LAT[1])
        & (lon >= NB_LON[0])
        & (lon <= NB_LON[1])
    )
    dep_ids = {
        sid
        for sid, v in _roll_sched()["sweep_points"][sweep]["probe_active_from"].items()
        if not str(v).startswith("9999")
    }
    is_dep = np.isin(ids, list(dep_ids))
    if colkey == "probe_deployed":
        return (sub == "probe") & is_dep & has
    return (sub == "spatial_test") & in_nor & has


def _roll_micro(val, cnt, sel):
    return float((val[sel] * cnt[sel]).sum() / cnt[sel].sum()) if sel.any() else np.nan


def fig07(out: Path) -> None:
    import pandas as pd

    recs = _roll_runs()
    cols = [
        ("probe_deployed", "Deployed Norwegian stations\n(in training)"),
        (
            "norway_always_heldout",
            "Permanently held-out\nNorwegian stations\n(never in training)",
        ),
    ]

    rows = []
    for r in recs:
        if r["family"] is None:
            continue
        arr = _roll_station_arrays(r["run_dir"], r["var"])
        if arr is None:
            continue
        for colkey, _t in cols:
            sel = _roll_col_sel(*arr[:5], r["sweep"], colkey)
            rows.append(
                dict(
                    var=r["var"],
                    family=r["family"],
                    x=r["x"],
                    seed=r["seed"],
                    col=colkey,
                    mae=_roll_micro(arr[5], arr[4], sel),
                )
            )
    F = pd.DataFrame(rows)

    # ERA5 reference arrays: subset labels + scored population borrowed from a
    # trained baseline run (notebook cell 7's _era_col_arrays).
    era = {}
    for var in ("t2m", "wind"):
        e = next((r for r in recs if r["arch"] == ERA_REF_ARCH[var]), None)
        b = next(
            (r for r in recs if r["var"] == var and r["family"] == "baseline"), None
        )
        if e is None or b is None:
            continue
        d = np.load(Path(e["run_dir"]) / "test_station_errors.npz", allow_pickle=True)
        rr = np.load(Path(b["run_dir"]) / "test_station_errors.npz", allow_pickle=True)
        ids = d["station_ids"].astype(str)
        rids = rr["station_ids"].astype(str)
        sub_of = dict(zip(rids, rr["subset_per_station"].astype(str), strict=False))
        scored = set(rids[rr[f"{var}_station_count"] > 0].tolist())
        sub = np.array([sub_of.get(i, "unmapped") for i in ids])
        cnt = np.where(
            [i in scored for i in ids], d[f"{var}_station_count"].astype(float), 0.0
        )
        era[var] = (
            ids,
            d["station_lats"],
            d["station_lons"],
            sub,
            cnt,
            d[f"{var}_station_mae"].astype(float),
        )

    sched = _roll_sched()
    dep_of = {
        lbl: sum(
            1
            for v in sched["sweep_points"][lbl]["probe_active_from"].values()
            if not str(v).startswith("9999")
        )
        for lbl in SWEEP_ORDER
    }
    dep_sweeps = ["r1mo", "r6mo", "r1y", "r2y", "r3y"]

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(TEXTWIDTH, 4.8),
        sharex="col",
        sharey="row",
        squeeze=False,
        layout="constrained",
    )
    fig.get_layout_engine().set(w_pad=0.01, h_pad=0.01, wspace=0.05, hspace=0.06)
    for i, var in enumerate(("t2m", "wind")):
        for j, (colkey, title) in enumerate(cols):
            ax = axes[i, j]
            xlim = (0.05 if colkey == "probe_deployed" else -0.01, 6.35)
            ax.set_xscale("function", functions=_AH_SQRT)
            ax.set_xticks([0, 0.5, 1, 2, 3, 4, 5, 6])
            ax.set_xticklabels(["0", "0.5", "1", "2", "3", "4", "5", "6"])
            ax.set_xticks([], minor=True)
            ax.set_xlim(*xlim)
            ax.tick_params(labelsize=_AH_FS["tick"], length=2.5, pad=1.5)
            ax.grid(True, alpha=0.3, lw=0.5)
            for family in ("tessera", "baseline"):
                a = (
                    F[(F["var"] == var) & (F.family == family) & (F.col == colkey)]
                    .groupby("x")["mae"]
                    .agg(["mean", "std"])
                    .reset_index()
                    .sort_values("x")
                    .dropna(subset=["mean"])
                )
                ax.errorbar(
                    a["x"],
                    a["mean"],
                    yerr=a["std"].fillna(0),
                    color=FAMILY_COLOUR[family],
                    marker=FAMILY_MARKER[family],
                    linestyle="-",
                    markersize=2.8,
                    lw=1.0,
                    elinewidth=0.8,
                    capsize=1.5,
                    label=FAMILY_LABEL[family],
                )
            if var in era:
                eids, elat, elon, esub, ecnt, eval_ = era[var]
                ex, ey = [], []
                for lbl in SWEEP_ORDER:
                    esel = _roll_col_sel(eids, elat, elon, esub, ecnt, lbl, colkey)
                    m = _roll_micro(eval_, ecnt, esel)
                    if np.isfinite(m):
                        ex.append(SWEEP_X_YEARS[lbl])
                        ey.append(m)
                ax.plot(
                    ex,
                    ey,
                    color="#ff7f0e",
                    ls="--",
                    lw=1.0,
                    label="ERA5 interpolation\n(lapse-corrected for t2m)",
                )
            if i == 0:
                ax.set_title(title, fontsize=_AH_FS["title"], pad=12)
            if j == 0:
                ax.set_ylabel(
                    f"{VAR_LABEL[var]}\nMAE ({VAR_UNIT[var]})", fontsize=_AH_FS["label"]
                )
            if i == 1:
                ax.set_xlabel(
                    "Elapsed years since rollout started",
                    fontsize=_AH_FS["label"],
                    labelpad=2,
                )
            if colkey == "probe_deployed" and i == 0:
                top = ax.twiny()
                top.set_xscale("function", functions=_AH_SQRT)
                xs = [SWEEP_X_YEARS[sw] for sw in dep_sweeps]
                top.set_xticks(xs)
                top.set_xticklabels(
                    [str(dep_of[sw]) for sw in dep_sweeps], fontsize=_AH_FS["toptick"]
                )
                top.set_xlim(*xlim)
                top.tick_params(axis="x", length=2.5, pad=1.5)
                top.set_xlabel(
                    "# probe stations deployed", fontsize=_AH_FS["toplabel"], labelpad=5
                )

    # align the tops of the two column titles (notebook's alignment pass)
    for _ in range(4):
        fig.canvas.draw()
        rend = fig.canvas.get_renderer()
        tops = [ax.title.get_window_extent(rend).y1 for ax in axes[0]]
        target = max(tops)
        if target - min(tops) < 0.5:
            break
        for ax, t in zip(axes[0], tops, strict=False):
            ax.set_title(
                ax.get_title(),
                fontsize=_AH_FS["title"],
                y=ax.title.get_position()[1]
                + (target - t) / ax.get_window_extent().height,
            )

    hl = {}
    for ax in axes.flat:
        for h, lab in zip(*ax.get_legend_handles_labels(), strict=False):
            hl.setdefault(lab, h)
    order = [FAMILY_LABEL["tessera"], FAMILY_LABEL["baseline"]] + [
        lab for lab in hl if lab not in FAMILY_LABEL.values()
    ]
    fig.legend(
        [hl[lab] for lab in order],
        order,
        loc="outside lower center",
        ncol=len(order),
        frameon=False,
        fontsize=_AH_FS["legend"],
        handlelength=1.8,
        columnspacing=1.4,
        handletextpad=0.5,
        borderaxespad=0.0,
    )
    save(fig, out, "fig07_norway_rollout")


# ===========================================================================
# Figs 11 & 12 — descriptor-space views + deployment-resolved reachability
# (replicates scripts/analysis/norway_descriptor_spaces.py, default 4-space
#  paper configuration, reading the same inputs under the data root)
# ===========================================================================
NW_SPACE_ORDER = [
    "geographic\n(lat, lon)",
    "elevation\n+ mTPI",
    "ERA5 static\n(interp)",
    "TESSERA\nlat16 embedding",
]
NW_EXTRA_NAME = "land surface\n(hand-crafted)"
NW_COLOUR = {
    "geographic\n(lat, lon)": "#7f7f7f",
    "elevation\n+ mTPI": "#ff7f0e",
    "ERA5 static\n(interp)": "#9467bd",
    NW_EXTRA_NAME: "#8c564b",
    "TESSERA\nlat16 embedding": "#1f77b4",
}
NW_LABEL = {
    "geographic\n(lat, lon)": "geographic",
    "elevation\n+ mTPI": "elevation + mTPI",
    "ERA5 static\n(interp)": "ERA5 static (interp)",
    NW_EXTRA_NAME: "land surface (hand-crafted)",
    "TESSERA\nlat16 embedding": "TESSERA embedding",
}
NW_LATENTS = (
    DATA / "processed/vae_tessera_1B-M/"
    "station_latents_1B-M_p128_2017_crop64_lat16_grad0.5_auxon.npy"
)
# Docstring headline numbers (probe group) — cross-check targets.
NW_EXPECTED = {
    "geographic\n(lat, lon)": (2, 0.98),
    "elevation\n+ mTPI": (95, 0.61),
    "ERA5 static\n(interp)": (50, 0.96),
    "TESSERA\nlat16 embedding": (96, 0.96),
}


def _nw_setup(extra: bool = False):
    """SPACES arrays, group masks, station table (script lines 100-215).
    extra=True appends the 17-feature hand-crafted land-surface space
    (the script's WITH_EXTRA_DESCRIPTORS=1 opt-in, mean-filled NaN rows)."""
    if ("nw", extra) in _SR_CACHE:
        return _SR_CACHE[("nw", extra)]
    import pandas as pd
    from scipy.interpolate import RegularGridInterpolator

    lat = np.load(NW_LATENTS)
    ll = pd.read_csv(DATA / "processed/tessera_global/station_list_filtered.csv")
    ll["station_id"] = ll["station_id"].astype(str)
    lat_of = {s: i for i, s in enumerate(ll["station_id"])}
    st = pd.read_csv(DATA / "dataset_timestamp_global/stations.csv")
    st["station_id"] = st["station_id"].astype(str)
    st["lrow"] = st["station_id"].map(lat_of)
    st = st[st["lrow"].notna()].copy()
    st["lrow"] = st["lrow"].astype(int)
    st = st[~np.isnan(lat[st["lrow"].to_numpy()]).any(axis=1)].reset_index(drop=True)
    Z16 = lat[st["lrow"].to_numpy()]

    edir = DATA / "dataset_timestamp_global/regions/europe"
    sf = np.load(edir / "static_fields.npy")
    glat, glon = np.load(edir / "lats.npy"), np.load(edir / "lons.npy")
    q = np.column_stack(
        [
            np.clip(st["latitude"].to_numpy(), glat.min(), glat.max()),
            np.clip(st["longitude"].to_numpy(), glon.min(), glon.max()),
        ]
    )
    era5_static = np.column_stack(
        [
            RegularGridInterpolator(
                (glat, glon),
                sf[c],
                method="linear",
                bounds_error=False,
                fill_value=None,
            )(q)
            for c in range(sf.shape[0])
        ]
    ).astype(np.float32)

    in_nb = (
        st.latitude.between(*NB_LAT)
        & st.longitude.between(*NB_LON)
        & (st.region == "europe")
    ).to_numpy()
    eu = (st.region == "europe").to_numpy()
    tr = (st.spatial_split == "train").to_numpy()
    te = (st.spatial_split == "test").to_numpy()
    groups = {
        "norway_test": in_nb & eu & te,
        "norway_probe": in_nb & eu & tr,
        "nonnorway_test": eu & te & ~in_nb,
        "rest_train": eu & tr & ~in_nb,
    }
    spaces = {
        "geographic\n(lat, lon)": st[["latitude", "longitude"]].to_numpy(),
        "elevation\n+ mTPI": st[["elevation", "delta_elevation", "mtpi"]].to_numpy(),
        "ERA5 static\n(interp)": era5_static,
    }
    if extra:
        ext = np.load(DATA / "processed" / "extra_descriptors.npy")
        ext = ext[st["lrow"].to_numpy()]
        n_nan = int(np.isnan(ext).any(axis=1).sum())
        if n_nan:  # mean-fill, keeping the station set identical per space
            col_mean = np.nanmean(ext, axis=0)
            ext = np.where(np.isnan(ext), col_mean, ext)
        spaces[NW_EXTRA_NAME] = ext.astype(np.float32)
    spaces["TESSERA\nlat16 embedding"] = Z16  # rightmost panel by convention
    _SR_CACHE[("nw", extra)] = (spaces, groups, st)
    return _SR_CACHE[("nw", extra)]


def _nw_auc(Xa, ya):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    return float(
        cross_val_score(
            make_pipeline(
                StandardScaler(),
                LogisticRegression(max_iter=2000, class_weight="balanced"),
            ),
            Xa,
            ya,
            cv=StratifiedKFold(4, shuffle=True, random_state=0),
            scoring="roc_auc",
        ).mean()
    )


def _nw_probe_report(extra: bool = False):
    """reachability/AUC of the Norway probe group per space (for fig11 titles)."""
    key = ("nw_report", extra)
    if key in _SR_CACHE:
        return _SR_CACHE[key]
    from sklearn.neighbors import NearestNeighbors
    from sklearn.preprocessing import StandardScaler

    spaces, groups, _st = _nw_setup(extra)
    rep = {}
    for name, X in spaces.items():
        Xs = StandardScaler().fit(X[groups["rest_train"]]).transform(X)
        ref = Xs[groups["rest_train"]]
        qq = Xs[groups["norway_probe"]]
        d_loo = NearestNeighbors(n_neighbors=2).fit(ref).kneighbors(ref)[0][:, 1]
        r95 = float(np.percentile(d_loo, 95))
        dq = NearestNeighbors(n_neighbors=1).fit(ref).kneighbors(qq)[0][:, 0]
        Xa = np.vstack([qq, ref])
        ya = np.r_[np.ones(len(qq)), np.zeros(len(ref))]
        rep[name] = dict(reach=100 * float(np.mean(dq <= r95)), auc=_nw_auc(Xa, ya))
        if name in NW_EXPECTED:
            want_r, want_a = NW_EXPECTED[name]
            if (
                abs(rep[name]["reach"] - want_r) > 4
                or abs(rep[name]["auc"] - want_a) > 0.03
            ):
                print(
                    f"  [warn] fig11/12 {NW_LABEL[name]}: reach "
                    f"{rep[name]['reach']:.0f}% auc {rep[name]['auc']:.2f} "
                    f"vs docstring {want_r}%/{want_a}"
                )
        else:
            print(
                f"  [info] {NW_LABEL[name]}: reach {rep[name]['reach']:.0f}%"
                f"  AUC {rep[name]['auc']:.2f}"
            )
    _SR_CACHE[key] = rep
    return rep


def _nw_cluster_panel(a, name, spaces, groups, norway, sub_mask, rep):
    from matplotlib.ticker import MaxNLocator
    from sklearn.decomposition import PCA
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.preprocessing import StandardScaler

    X = spaces[name]
    Xs = StandardScaler().fit(X[groups["rest_train"]]).transform(X)
    if name.startswith("geographic"):
        P = Xs[:, [1, 0]]
        xl, yl = "longitude (z)", "latitude (z)"
    else:
        ya = np.r_[np.ones(norway.sum()), np.zeros(groups["rest_train"].sum())]
        Xa = np.vstack([Xs[norway], Xs[groups["rest_train"]]])
        w = LinearDiscriminantAnalysis().fit(Xa, ya).scalings_[:, 0]
        w = w / np.linalg.norm(w)
        px = Xs @ w
        resid = Xs - np.outer(px, w)
        py = PCA(n_components=1).fit(resid[groups["rest_train"]]).transform(resid)[:, 0]
        P = np.column_stack([px, py])
        xl, yl = "Norway-vs-rest discriminant", "residual PC1"
    a.scatter(P[sub_mask, 0], P[sub_mask, 1], s=2, c="#cccccc", rasterized=True)
    a.scatter(
        P[groups["nonnorway_test"], 0],
        P[groups["nonnorway_test"], 1],
        s=3,
        c="#2ca02c",
        alpha=0.6,
        lw=0,
        rasterized=True,
    )
    a.scatter(
        P[norway, 0], P[norway, 1], s=3, c="#d62728", alpha=0.6, lw=0, rasterized=True
    )
    r = rep[name]
    a.set_title(
        f"{name.replace(chr(10), ' ')}\nreachable "
        f"{r['reach']:.0f}%,  AUC {r['auc']:.2f}",
        fontsize=8,
    )
    a.set_xlabel(xl, fontsize=8)
    a.set_ylabel(yl, fontsize=8)
    a.tick_params(labelsize=7)
    a.xaxis.set_major_locator(MaxNLocator(5))
    a.yaxis.set_major_locator(MaxNLocator(5))


def _nw_legend_handles():
    from matplotlib.lines import Line2D

    return [
        Line2D([0], [0], ls="none", marker="o", ms=4, color=c, label=lab)
        for c, lab in (
            ("#cccccc", "rest-train"),
            ("#2ca02c", "non-Norway held-out (control)"),
            ("#d62728", "Norway"),
        )
    ]


def _nw_masks(groups, st):
    norway = groups["norway_probe"] | groups["norway_test"]
    rng = np.random.RandomState(0)
    sub = rng.choice(
        np.where(groups["rest_train"])[0],
        size=min(2500, int(groups["rest_train"].sum())),
        replace=False,
    )
    sub_mask = np.zeros(len(st), bool)
    sub_mask[sub] = True
    return norway, sub_mask


def fig11(out: Path) -> None:
    spaces, groups, st = _nw_setup()
    rep = _nw_probe_report()
    norway, sub_mask = _nw_masks(groups, st)
    fig, axes = plt.subplots(2, 2, figsize=(TEXTWIDTH, 5.0))
    for a, name in zip(axes.flat, NW_SPACE_ORDER, strict=False):
        _nw_cluster_panel(a, name, spaces, groups, norway, sub_mask, rep)
    fig.legend(
        handles=_nw_legend_handles(),
        loc="outside lower center",
        ncol=3,
        frameon=False,
        fontsize=7,
    )
    save(fig, out, "fig11_descriptor_clusters")


def fig11ext(out: Path) -> None:
    """fig11 widened with the 17-feature hand-crafted land-surface space:
    two panels per row as in the original, TESSERA centred on its own
    bottom row."""
    spaces, groups, st = _nw_setup(extra=True)
    rep = _nw_probe_report(extra=True)
    norway, sub_mask = _nw_masks(groups, st)
    order = [
        "geographic\n(lat, lon)",
        "elevation\n+ mTPI",
        "ERA5 static\n(interp)",
        NW_EXTRA_NAME,
        "TESSERA\nlat16 embedding",
    ]
    fig = plt.figure(figsize=(TEXTWIDTH, 7.2))
    gs = fig.add_gridspec(3, 4)
    axes = [
        fig.add_subplot(gs[0, :2]),
        fig.add_subplot(gs[0, 2:]),
        fig.add_subplot(gs[1, :2]),
        fig.add_subplot(gs[1, 2:]),
        fig.add_subplot(gs[2, 1:3]),
    ]  # TESSERA, centred
    fig.get_layout_engine().set(hspace=0.06)  # extra air between rows
    for a, name in zip(axes, order, strict=False):
        _nw_cluster_panel(a, name, spaces, groups, norway, sub_mask, rep)
    fig.legend(
        handles=_nw_legend_handles(),
        loc="outside lower center",
        ncol=3,
        frameon=False,
        fontsize=7,
    )
    save(fig, out, "fig11_descriptor_clusters_extended")


def fig12(out: Path, extra: bool = False) -> None:
    import pandas as pd
    from sklearn.neighbors import NearestNeighbors
    from sklearn.preprocessing import StandardScaler

    spaces, groups, st = _nw_setup(extra)
    _nw_probe_report(extra)  # runs the cross-check
    order = list(spaces)
    sched = _roll_sched()
    st_id = st["station_id"].to_numpy()
    norway_probe = groups["norway_probe"]

    prep = {}
    for name, X in spaces.items():
        Xs = StandardScaler().fit(X[groups["rest_train"]]).transform(X)
        d_loo = (
            NearestNeighbors(n_neighbors=2)
            .fit(Xs[groups["rest_train"]])
            .kneighbors(Xs[groups["rest_train"]])[0][:, 1]
        )
        prep[name] = (Xs, float(np.percentile(d_loo, 95)))

    def reach_auc_rows(deployed):
        out_of_training = (norway_probe & ~deployed) | groups["norway_test"]
        ref_mask = groups["rest_train"] | deployed
        n_dep = int(deployed.sum())
        rows = []
        for name, (Xs, r95) in prep.items():
            qq, ref = Xs[out_of_training], Xs[ref_mask]
            dq = NearestNeighbors(n_neighbors=1).fit(ref).kneighbors(qq)[0][:, 0]
            Xa = np.vstack([qq, ref])
            ya = np.r_[np.ones(len(qq)), np.zeros(len(ref))]
            rows.append(
                dict(
                    x=n_dep,
                    space=name,
                    reach=100 * float(np.mean(dq <= r95)),
                    auc=_nw_auc(Xa, ya),
                )
            )
        return rows

    hz = list(reach_auc_rows(np.zeros(len(st), dtype=bool)))
    for lbl in ("r1mo", "r3mo", "r6mo", "r1y", "r2y", "r3y"):
        sp = sched["sweep_points"][lbl]
        dep_ids = {
            sid
            for sid, v in sp["probe_active_from"].items()
            if not str(v).startswith("9999")
        }
        hz.extend(reach_auc_rows(norway_probe & np.isin(st_id, list(dep_ids))))
    hz = pd.DataFrame(hz)

    fig, ax = plt.subplots(1, 2, figsize=(TEXTWIDTH, 2.7))
    for name in order:
        s = hz[hz["space"] == name].sort_values("x")
        for a, ycol in zip(ax, ("reach", "auc"), strict=False):
            a.plot(
                s["x"],
                s[ycol],
                marker="o",
                ms=3,
                color=NW_COLOUR[name],
                lw=1.2,
                label=NW_LABEL[name],
            )
    ax[0].set_ylim(-5, 108)
    ax[0].set_ylabel(
        "% of out-of-training Norwegian\nstations with an in-distribution analogue"
    )
    ax[0].legend(fontsize=7, loc="lower right")
    ax[1].axhline(0.5, color="black", ls=":", lw=0.8)
    ax[1].set_ylim(0.45, 1.0)
    ax[1].set_ylabel("separability AUC\n(out-of-training Norway vs training)")
    for a in ax:
        a.set_xlabel("Norwegian probe stations deployed into training")
        a.grid(alpha=0.3)
        a.axvline(0, color="#bbbbbb", ls=":", lw=0.8, zorder=0)
    ax[0].text(
        15, -3, "cold start", fontsize=7, color="#666666", va="center", ha="left"
    )
    save(fig, out, "fig12_norway_reach_horizon" + ("_extended" if extra else ""))


def fig12ext(out: Path) -> None:
    """fig12 with the 17-feature hand-crafted land-surface curve added."""
    fig12(out, extra=True)


REGISTRY = {}


def register(fn):
    REGISTRY[fn.__name__] = fn
    return fn


for _f in (
    fig01,
    fig02,
    fig02ext,
    fig03,
    fig04,
    fig05,
    fig06,
    fig07,
    fig08,
    fig09,
    fig10,
    fig11,
    fig11ext,
    fig12,
    fig12ext,
):
    register(_f)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=OUT_DEFAULT)
    ap.add_argument(
        "--only",
        type=str,
        default=None,
        help="comma-separated figure names, e.g. fig02,fig09",
    )
    args = ap.parse_args(argv)

    plt.rcParams.update(STYLE)
    names = args.only.split(",") if args.only else sorted(REGISTRY)
    failures = []
    for name in names:
        print(f"== {name} ==")
        try:
            REGISTRY[name](args.out)
        except Exception:
            failures.append(name)
            traceback.print_exc()
    print("\nDone." if not failures else f"\nFAILED: {', '.join(failures)}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
