"""Summary tables for the dense maps: terrain structure added + per-variant
station-performance decomposition, per snapshot.

Scans every OUTPUTS/<region>/<var>_<ts>/ subfolder for the current REGION. For each
snapshot it writes a per-variant table (ERA5-interp / ConvCNP baseline / TESSERA):

  * Station MAE & RMSE at that snapshot (in-region GHCNh test stations),
  * Fine-scale structure = std of the 0.15deg NaN-aware high-pass of the field, and
    its ratio vs the no-TESSERA baseline (= "terrain structure added"),
  * |fine-scale anomaly| vs ERA5 sub-grid terrain ruggedness (Spearman r).

Outputs, per snapshot:  <sub>/<region>_<var>_<ts>_summary.{png,csv}
Region roll-up (all snapshots, original vs auto-selected dates side by side):
                        OUTPUTS/<region>/<region>_summary.{png,csv}

  REGION=norway uv run python scripts/maps/summary_table.py
"""
from __future__ import annotations

import csv

import matplotlib
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.stats import spearmanr

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from generate_maps import bilinear_grid_to_points  # noqa: E402
from regions import SDFOR_IDX, get_region  # noqa: E402

R = get_region()
EU = R.region_data
GLATS = np.load(EU / "lats.npy").astype(np.float32)
GLONS = np.load(EU / "lons.npy").astype(np.float32)
SDFOR = np.load(EU / "static_fields.npy")[SDFOR_IDX].astype(float)

_co = np.load(R.dense_npz, allow_pickle=True)["coords"]
BBOX = (float(_co["lon"].min()), float(_co["lon"].max()),
        float(_co["lat"].min()), float(_co["lat"].max()))


def nan_highpass(a, sigma):
    m = np.isfinite(a).astype(float)
    af = np.where(np.isfinite(a), a, 0.0)
    den = gaussian_filter(m, sigma)
    low = np.where(den > 1e-6, gaussian_filter(af * m, sigma) / np.maximum(den, 1e-9), np.nan)
    return a - low


def ruggedness_grid(lats_d, lons_d):
    LON, LAT = np.meshgrid(lons_d, lats_d)
    pts = np.stack([LAT.ravel(), LON.ravel()], 1).astype(np.float32)
    return bilinear_grid_to_points(SDFOR, GLATS, GLONS, pts).reshape(LAT.shape)


def metrics(var, ts):
    """All metrics for one snapshot, or None if its maps npz is missing."""
    mp, dem = R.fig(var, ts, "_dem.npz"), True
    if not mp.exists():
        mp, dem = R.fig(var, ts, ".npz"), False
    if not mp.exists():
        return None
    d = np.load(mp, allow_pickle=True)
    e, b, t, vm = d["era5_interp"], d["convcnp_baseline"], d["tessera_concat"], d["valid_mask"]
    se, sb, st = (float(np.nanstd(nan_highpass(x, 3.0))) for x in (e, b, t))
    diff = t - b
    rug = ruggedness_grid(d["lats"], d["lons"])
    hb, ht = nan_highpass(b, 6.0), nan_highpass(t, 6.0)
    sel = vm & np.isfinite(hb) & np.isfinite(ht) & np.isfinite(rug)
    rug_b = float(spearmanr(rug[sel], np.abs(hb[sel])).statistic)
    rug_t = float(spearmanr(rug[sel], np.abs(ht[sel])).statistic)
    row = dict(var=var, ts=ts, elevation=("DEM" if dem else "proxy"),
               finestd_era5=se, finestd_base=sb, finestd_tess=st,
               tess_x_base=st / sb if sb else float("nan"),
               diff_mean=float(np.nanmean(diff)), diff_std=float(np.nanstd(diff)),
               diff_p5=float(np.nanpercentile(diff, 5)), diff_p95=float(np.nanpercentile(diff, 95)),
               diff_maxabs=float(np.nanmax(np.abs(diff[np.isfinite(diff)]))),
               rug_r_base=rug_b, rug_r_tess=rug_t,
               n_stn=0, mae_era5=np.nan, mae_base=np.nan, mae_tess=np.nan,
               rmse_era5=np.nan, rmse_base=np.nan, rmse_tess=np.nan, improved_pct=np.nan)
    sp = R.fig(var, ts, "_stations.npz")
    if sp.exists():
        s = np.load(sp)
        m = ((s["lon"] >= BBOX[0]) & (s["lon"] <= BBOX[1]) &
             (s["lat"] >= BBOX[2]) & (s["lat"] <= BBOX[3]))
        if m.any():
            mae = lambda k: float(np.mean(s[k][m]))            # noqa: E731
            rmse = lambda k: float(np.sqrt(np.mean(s[k][m] ** 2)))  # noqa: E731
            # MAE @ median (base_err/tess_err); RMSE @ mean where the mean-based
            # errors are available (truncated-normal wind), else fall back to the
            # median errors (gaussian t2m: mean == median, so identical).
            rk = lambda base: base + "_mean" if (base + "_mean") in s.files else base  # noqa: E731
            row.update(n_stn=int(m.sum()),
                       mae_era5=mae("era5_err"), mae_base=mae("base_err"), mae_tess=mae("tess_err"),
                       rmse_era5=rmse(rk("era5_err")), rmse_base=rmse(rk("base_err")),
                       rmse_tess=rmse(rk("tess_err")),
                       improved_pct=float(np.mean(s["improved"][m]) * 100))
    return row


def per_snapshot_table(row):
    """Render the per-variant decomposition table for one snapshot."""
    var, ts, u = row["var"], row["ts"], R.jobs[row["var"]]["unit_plain"]
    cols = [f"Station MAE ({u})", f"Station RMSE ({u})", f"Fine-scale std ({u})", "x vs base", "Terrain r"]
    cell = [
        ["ERA5-interp (no model)", f"{row['mae_era5']:.2f}", f"{row['rmse_era5']:.2f}",
         f"{row['finestd_era5']:.3f}", f"{row['finestd_era5']/row['finestd_base']:.2f}", "—"],
        ["ConvCNP baseline (no TESSERA)", f"{row['mae_base']:.2f}", f"{row['rmse_base']:.2f}",
         f"{row['finestd_base']:.3f}", "1.00", f"{row['rug_r_base']:+.2f}"],
        ["ConvCNP + lat16 (TESSERA)", f"{row['mae_tess']:.2f}", f"{row['rmse_tess']:.2f}",
         f"{row['finestd_tess']:.3f}", f"{row['tess_x_base']:.2f}", f"{row['rug_r_tess']:+.2f}"],
    ]
    fig, ax = plt.subplots(figsize=(11, 2.4))
    ax.axis("off")
    tbl = ax.table(cellText=[r[1:] for r in cell], rowLabels=[r[0] for r in cell],
                   colLabels=cols, loc="center", cellLoc="center", rowLoc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(9.5); tbl.scale(1, 1.6)
    for j in range(len(cols)):                       # header bold
        tbl[0, j].set_text_props(weight="bold")
    for j in range(-1, len(cols)):                   # highlight TESSERA row
        tbl[3, j].set_facecolor("#e8f0fe")
    ax.set_title(
        f"{R.name.capitalize()} — {var} @ {ts} UTC  ({row['elevation']} elevation)\n"
        f"in-region stations n={row['n_stn']}  ·  TESSERA beats baseline at "
        f"{row['improved_pct']:.0f}% of them  ·  TESSERA-baseline correction: "
        f"mean {row['diff_mean']:+.2f}, std {row['diff_std']:.2f}, "
        f"p5/p95 {row['diff_p5']:+.2f}/{row['diff_p95']:+.2f}, max|.| {row['diff_maxabs']:.2f} {u}",
        fontsize=10)
    p = R.fig(var, ts, "_summary.png")
    fig.savefig(p, dpi=160, bbox_inches="tight"); plt.close(fig)
    # per-snapshot csv (one row per variant)
    with open(R.fig(var, ts, "_summary.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["variant", "mae", "rmse", "fine_std", "fine_x_base", "terrain_r"])
        w.writerow(["era5_interp", row["mae_era5"], row["rmse_era5"], row["finestd_era5"],
                    row["finestd_era5"] / row["finestd_base"], ""])
        w.writerow(["convcnp_baseline", row["mae_base"], row["rmse_base"], row["finestd_base"], 1.0, row["rug_r_base"]])
        w.writerow(["tessera_lat16", row["mae_tess"], row["rmse_tess"], row["finestd_tess"],
                    row["tess_x_base"], row["rug_r_tess"]])
    return p


FIELDS = ["var", "ts", "elevation", "n_stn", "mae_era5", "mae_base", "mae_tess",
          "rmse_era5", "rmse_base", "rmse_tess", "improved_pct",
          "finestd_era5", "finestd_base", "finestd_tess", "tess_x_base",
          "rug_r_base", "rug_r_tess",
          "diff_mean", "diff_std", "diff_p5", "diff_p95", "diff_maxabs"]


def region_rollup(rows):
    rows = sorted(rows, key=lambda r: (r["var"], r["ts"]))
    csv_p = R.out_dir / f"{R.name}_summary.csv"
    with open(csv_p, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in FIELDS})
    # comparison PNG: one row per snapshot, headline columns
    cols = ["var", "date", "n", "MAE base", "MAE tess", "improved%", "fine x base", "r base", "r tess"]
    cell = [[r["var"], r["ts"], r["n_stn"], f"{r['mae_base']:.2f}", f"{r['mae_tess']:.2f}",
             f"{r['improved_pct']:.0f}", f"{r['tess_x_base']:.2f}",
             f"{r['rug_r_base']:+.2f}", f"{r['rug_r_tess']:+.2f}"] for r in rows]
    fig, ax = plt.subplots(figsize=(12, 0.6 + 0.5 * len(cell)))
    ax.axis("off")
    tbl = ax.table(cellText=cell, colLabels=cols, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(9.5); tbl.scale(1, 1.6)
    for j in range(len(cols)):
        tbl[0, j].set_text_props(weight="bold")
    ax.set_title(f"{R.name.capitalize()} — snapshot comparison (MAE in variable units; "
                 f"'fine x base' = TESSERA/baseline fine-scale std)", fontsize=11)
    png_p = R.out_dir / f"{R.name}_summary.png"
    fig.savefig(png_p, dpi=160, bbox_inches="tight"); plt.close(fig)
    return csv_p, png_p


def main():
    subs = sorted(p for p in R.out_dir.glob("*") if p.is_dir())
    rows = []
    for sub in subs:
        var, _, ts = sub.name.partition("_")
        if var not in R.jobs:
            continue
        row = metrics(var, ts)
        if row is None:
            print(f"  [skip] {sub.name}: no maps npz")
            continue
        per_snapshot_table(row)
        rows.append(row)
        print(f"  {sub.name}: MAE base/tess={row['mae_base']:.2f}/{row['mae_tess']:.2f}  "
              f"fine x{row['tess_x_base']:.2f}  terrain r {row['rug_r_base']:+.2f}->{row['rug_r_tess']:+.2f}")
    if rows:
        csv_p, png_p = region_rollup(rows)
        print(f"wrote {csv_p}\nwrote {png_p}")


if __name__ == "__main__":
    main()
