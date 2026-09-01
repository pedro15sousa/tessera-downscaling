"""Study-domain overview figure: the five region bounding boxes + station network.

Draws a single global map carrying

  * the five ERA5 crop boxes of Table 3, labelled with the region name and the
    total station counts n (t2m / wind) reported in Table 1;
  * every station in the evaluated set, coloured by its spatial split
    (training vs permanently held-out).

Both target variables are shown on one map. A station is plotted if it is used
for 2 m temperature, for 10 m wind speed, or for both — the two networks
overlap heavily and separating them would double the figure for very little
information. The per-region labels still carry the counts per variable, so the
figure and Table 1 read against each other.

The plotted station set is reconstructed with exactly the filters the datasets
apply (see data/dataset.py::MultiRegionSnapshotDownscalingDataset and
data/helpers.py), so the counts printed here reproduce Table 1:

    region ∩ spatial_split ∩ TESSERA-patch-valid ∩ VAE-latent-valid
           ∩ (>= 1 valid observation of the variable in the test split)

The last clause is what the evaluator's ``<var>_n_test_stations`` counts, so a
station that exists in the network but never reported the variable in the test
split is excluded — same as in the results tables. The test split is every
snapshot strictly after VAL_END under the lexicographic comparison of
data/helpers.py::episodes_for_split, i.e. 2021-12-31 00Z through 2023-01-10
18Z, and a valid observation is finite AND > -100 (data/helpers.py sentinel
check) — both must match the dataset or the counts drift from the evaluator's.

Usage (from repo root):

    uv run python scripts/maps/plot_region_overview.py

    # restrict the dots to one variable's station set
    uv run python scripts/maps/plot_region_overview.py --variable wind

Outputs go to OUTPUTS/overview/ (see regions.py; the paper's copy is
``make_paper_figures.fig01``). The TESSERA patch-coverage mask is the only
expensive input (it streams the 81 GB patch array once, ~10 s warm) and is
cached under ``processed/overview_cache/``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from regions import OUTPUTS as MAPS_OUTPUTS

from tessera_downscaling.paths import dataset_dir, processed_dir, station_vectors_dir

DATASET = dataset_dir("dataset_timestamp_global")
OUTPUTS = MAPS_OUTPUTS / "overview"

PATCHES = processed_dir("tessera_global", "patch_embeddings_2024.npy")
PATCH_CSV = processed_dir("tessera_global", "station_list_filtered.csv")
LATENTS = station_vectors_dir("station_latents_lat16_grad0.5.npy")
MIN_PATCH_COVERAGE = 0.5

# Region boxes — MUST stay in sync with REGIONS in
# scripts/preprocessing/preprocess_timestamp_global.py (and Table 3).
REGIONS: dict[str, tuple[float, float, float, float]] = {
    "europe": (35.0, 75.0, -24.0, 40.0),
    "us": (24.0, 50.0, -125.0, -66.0),
    "east_asia": (20.0, 46.0, 100.0, 146.0),
    "australia": (-44.0, -10.0, 112.0, 154.0),
    "southern_africa": (-35.0, -15.0, 15.0, 35.0),
}
PRETTY = {
    "europe": "Europe",
    "us": "United States",
    "east_asia": "East Asia",
    "australia": "Australia",
    "southern_africa": "Southern Africa",
}
# Where each region label sits relative to its box. `x`/`y` pick the anchor
# corner, `dx`/`dy` nudge it in degrees, `ha`/`va` align the text to that point.
# Australia is labelled to the side rather than below because its box already
# reaches the southern edge of the plotted extent; Europe is labelled below
# rather than above because its box reaches 75 N and a label above it would
# push past the top of the frame.
LABEL_POS = {
    "europe": dict(x="left", y="bottom", dx=0.0, dy=-2.0, ha="left", va="top"),
    "us": dict(x="left", y="bottom", dx=0.0, dy=-2.0, ha="left", va="top"),
    "east_asia": dict(x="right", y="top", dx=0.0, dy=2.0, ha="right", va="bottom"),
    "australia": dict(x="left", y="top", dx=-2.5, dy=0.0, ha="right", va="top"),
    "southern_africa": dict(x="left", y="bottom", dx=0.0, dy=-2.0, ha="left", va="top"),
}

# Okabe-Ito: colourblind-safe by construction. Training is the recessive
# majority class (85 % of stations); held-out is the figure's subject and is
# drawn on top in the warm hue.
C_TRAIN = "#5B8FC9"
C_TEST = "#D55E00"
C_BOX = "#2E2E2E"
C_LAND = "#EFEDE9"
C_COAST = "#9A9A9A"

# Test split = timestamps strictly after this date string. Full timestamps
# compare lexicographically ("2021-12-31-00" > "2021-12-31"), so every
# snapshot of the boundary day itself falls into the test split — the same
# behaviour as data/helpers.py::episodes_for_split.
VAL_END = "2021-12-31"
# Observation validity, mirroring data/helpers.py: finite AND above the
# sentinel floor.
OBS_SENTINEL_FLOOR = -100.0


# ---------------------------------------------------------------------------
# Station table
# ---------------------------------------------------------------------------


def _patch_valid_mask(cache: Path) -> np.ndarray:
    """Centre-pixel-nonzero AND coverage >= 0.5, per data/helpers.py."""
    if cache.exists():
        return np.load(cache)["valid"]
    print(f"computing TESSERA patch validity from {PATCHES} (one pass) ...")
    m = np.load(str(PATCHES), mmap_mode="r")
    n, h = m.shape[0], m.shape[1]
    c = h // 2
    coverage = np.empty(n, np.float32)
    centre = np.empty(n, bool)
    for i in range(n):
        p = m[i]
        coverage[i] = np.any(p != 0, axis=-1).mean()
        centre[i] = bool(np.any(p[c, c] != 0))
    valid = centre & (coverage >= MIN_PATCH_COVERAGE)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, valid=valid, coverage=coverage, centre=centre)
    print(
        f"  centre {centre.sum()}, coverage {(coverage >= MIN_PATCH_COVERAGE).sum()}, "
        f"both {valid.sum()} / {n}"
    )
    return valid


def _test_split_obs_counts(cache: Path) -> dict[str, np.ndarray]:
    """Per-station count of valid t2m / wind observations in the test split."""
    if cache.exists():
        z = np.load(cache)
        return {"t2m": z["t2m"], "wind": z["wind"]}
    snap_dir = DATASET / "ghcnh_snapshot"
    files = sorted(f for f in snap_dir.glob("*.npz") if f.stem > VAL_END)
    if not files:
        raise SystemExit(f"no test-split snapshots (stem > {VAL_END}) under {snap_dir}")
    print(
        f"scanning {len(files)} test-split snapshots "
        f"({files[0].stem} .. {files[-1].stem}) for observation counts ..."
    )
    out: dict[str, np.ndarray] | None = None
    for f in files:
        z = np.load(f)
        if out is None:
            out = {v: np.zeros(z[v].shape[0], int) for v in ("t2m", "wind")}
        for v in ("t2m", "wind"):
            out[v] += np.isfinite(z[v]) & (z[v] > OBS_SENTINEL_FLOOR)
    assert out is not None
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, **out)
    return out


def build_station_table(cache_dir: Path) -> pd.DataFrame:
    """Stations with their region, split, coordinates and per-variable usability."""
    stations = pd.read_csv(DATASET / "stations.csv")

    patch_ok = _patch_valid_mask(cache_dir / "patch_valid_2024.npz")
    latents = np.load(LATENTS)
    latent_ok = ~np.isnan(latents).any(axis=1)
    descriptor_csv = pd.read_csv(PATCH_CSV, usecols=["station_id"])
    if not (len(descriptor_csv) == len(patch_ok) == len(latents)):
        raise SystemExit(
            "row-count mismatch between station_list_filtered.csv "
            f"({len(descriptor_csv)}), patch mask ({len(patch_ok)}) and "
            f"latents ({len(latents)})"
        )
    usable = set(descriptor_csv.loc[patch_ok & latent_ok, "station_id"])
    stations["descriptor_ok"] = stations["station_id"].isin(usable)

    obs = _test_split_obs_counts(cache_dir / "obs_counts_test_split.npz")
    for v in ("t2m", "wind"):
        if len(obs[v]) != len(stations):
            raise SystemExit(
                f"{v} obs counts ({len(obs[v])}) do not align with stations.csv "
                f"({len(stations)})"
            )
        stations[f"use_{v}"] = stations["descriptor_ok"] & (obs[v] > 0)

    return stations


def summarise(stations: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for region in REGIONS:
        g = stations[stations["region"] == region]
        row = {"region": region}
        for v in ("t2m", "wind"):
            m = g[f"use_{v}"]
            row[f"{v}_train"] = int((m & (g["spatial_split"] == "train")).sum())
            row[f"{v}_test"] = int((m & (g["spatial_split"] == "test")).sum())
            row[f"{v}_n"] = int(m.sum())
        rows.append(row)
    return pd.DataFrame(rows).set_index("region")


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------


def draw(
    stations: pd.DataFrame,
    counts: pd.DataFrame,
    variable: str,
    out_stem: str,
    formats: list[str],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Rectangle

    plt.rcParams.update(
        {
            "font.size": 8,
            "axes.linewidth": 0.6,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
        }
    )

    pc = ccrs.PlateCarree()
    if variable == "both":
        in_use = stations["use_t2m"] | stations["use_wind"]
    else:
        in_use = stations[f"use_{variable}"]
    used = stations[in_use & stations["region"].isin(REGIONS)].copy()
    train = used[used["spatial_split"] == "train"]
    test = used[used["spatial_split"] == "test"]

    # Global extent, cut back to the first and last populated meridian and
    # to just past the northernmost / southernmost box edge, so the figure
    # spends no width or height on empty ocean.
    ext = (-168.0, 179.0, -48.0, 78.0)
    fig_w = 7.2
    map_h = fig_w * (ext[3] - ext[2]) / (ext[1] - ext[0])
    fig = plt.figure(figsize=(fig_w, map_h))
    ax = fig.add_axes([0, 0, 1, 1], projection=pc)
    ax.set_extent(list(ext), crs=pc)

    ax.add_feature(
        cfeature.LAND.with_scale("110m"), facecolor=C_LAND, edgecolor="none", zorder=0
    )
    ax.coastlines(resolution="110m", linewidth=0.3, color=C_COAST, zorder=1)
    ax.spines["geo"].set_edgecolor("#BBBBBB")
    ax.spines["geo"].set_linewidth(0.6)

    gl = ax.gridlines(
        draw_labels=False, linewidth=0.25, color="#D8D8D8", linestyle="-", zorder=1
    )
    gl.xlocator = plt.MultipleLocator(60)
    gl.ylocator = plt.MultipleLocator(30)

    ax.scatter(
        train["longitude"],
        train["latitude"],
        s=0.5,
        c=C_TRAIN,
        linewidths=0,
        alpha=0.85,
        transform=pc,
        zorder=3,
        rasterized=True,
    )
    ax.scatter(
        test["longitude"],
        test["latitude"],
        s=0.7,
        c=C_TEST,
        linewidths=0,
        alpha=0.95,
        transform=pc,
        zorder=4,
        rasterized=True,
    )

    for region, (lat0, lat1, lon0, lon1) in REGIONS.items():
        ax.add_patch(
            Rectangle(
                (lon0, lat0),
                lon1 - lon0,
                lat1 - lat0,
                transform=pc,
                facecolor="none",
                edgecolor=C_BOX,
                linewidth=0.9,
                zorder=5,
            )
        )
        p = LABEL_POS[region]
        tx = lon0 if p["x"] == "left" else lon1
        ty = lat1 if p["y"] == "top" else lat0
        n_t2m = counts.loc[region, "t2m_n"]
        n_wind = counts.loc[region, "wind_n"]
        ax.text(
            tx + p["dx"],
            ty + p["dy"],
            f"{PRETTY[region]}\n$n$ = {n_t2m:,} / {n_wind:,}",
            transform=pc,
            ha=p["ha"],
            va=p["va"],
            fontsize=7.2,
            linespacing=1.25,
            color="#1A1A1A",
            zorder=6,
        )

    # Wrapped so the title stops short of the South American coastline it
    # would otherwise run across.
    var_label = {
        "both": "2 m temperature\nand 10 m wind speed",
        "t2m": "2 m temperature",
        "wind": "10 m wind speed",
    }[variable]
    handles = [
        Line2D(
            [],
            [],
            marker="o",
            linestyle="none",
            markersize=3.2,
            markerfacecolor=C_TRAIN,
            markeredgecolor="none",
            label=f"training stations ({len(train):,})",
        ),
        Line2D(
            [],
            [],
            marker="o",
            linestyle="none",
            markersize=3.8,
            markerfacecolor=C_TEST,
            markeredgecolor="none",
            label=f"held-out stations ({len(test):,})",
        ),
    ]
    ax.legend(
        handles=handles,
        loc="lower left",
        bbox_to_anchor=(0.012, 0.02),
        frameon=False,
        fontsize=7,
        handletextpad=0.35,
        borderaxespad=0.0,
        labelspacing=0.35,
        title=f"Stations shown: {var_label}",
        title_fontsize=7,
        alignment="left",
    )

    OUTPUTS.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        path = OUTPUTS / f"{out_stem}.{fmt}"
        fig.savefig(path, dpi=400 if fmt == "png" else None)
        print(f"wrote {path}")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--variable",
        choices=["both", "t2m", "wind"],
        default="both",
        help="which station set the dots show; 'both' plots the "
        "union of the two variables' sets (default)",
    )
    ap.add_argument("--out-stem", default=None)
    ap.add_argument("--formats", nargs="+", default=["pdf", "png"])
    ap.add_argument("--cache-dir", type=Path, default=processed_dir("overview_cache"))
    args = ap.parse_args()

    stations = build_station_table(args.cache_dir)
    counts = summarise(stations)
    print("\nStation counts (train / held-out / total), by region and variable:")
    print(counts.to_string())
    print("\nTable 1 reports n as t2m/wind totals:")
    for region in REGIONS:
        print(
            f"  {PRETTY[region]:<16} n = {counts.loc[region, 't2m_n']:>5,}"
            f" / {counts.loc[region, 'wind_n']:>5,}"
        )

    OUTPUTS.mkdir(parents=True, exist_ok=True)
    counts.to_csv(OUTPUTS / "region_station_counts.csv")
    (OUTPUTS / "region_boxes.json").write_text(
        json.dumps(
            {
                k: {"lat_min": v[0], "lat_max": v[1], "lon_min": v[2], "lon_max": v[3]}
                for k, v in REGIONS.items()
            },
            indent=2,
        )
    )

    stem = args.out_stem or (
        "region_overview"
        if args.variable == "both"
        else f"region_overview_{args.variable}"
    )
    draw(
        stations=stations,
        counts=counts,
        variable=args.variable,
        out_stem=stem,
        formats=args.formats,
    )


if __name__ == "__main__":
    main()
