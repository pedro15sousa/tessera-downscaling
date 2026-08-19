"""Which descriptor space organises the ERA5-interpolation residual?

The model-free counterpart to the per-region skill table: a random-forest probe
asking, independently of any trained downscaler, which per-station descriptor
space makes the persistent ERA5-interpolation residual predictable at stations
excluded from fitting, re-run over a **wider set of descriptor spaces** so that
the hand-crafted land-surface descriptor is judged on the same footing as the
learned embedding.

**Independent robustness probe -- not the source of the paper's figure.** The
paper's residual-probe figure (preprint Fig 5 + Fig 8, AMS Fig 3) is produced by
``notebooks/residual_structure_analysis.ipynb`` §3c/§3g and re-rendered by
``scripts/paper/make_paper_figures.py`` (``fig05``/``fig08``). This script builds
its station set differently (every station with >= ``--min-snapshots`` valid
test snapshots, no TESSERA-validity filter, v1 latents by default; the paper
uses the stations of the trained runs' ``test_predictions.npz``) and uses its
own random-forest settings (400 trees, 4 folds, ``min_samples_leaf=2``; the
paper: 200 trees, 5 folds, ``min_samples_leaf=3``), so its R² values are *not*
the numbers printed in the paper; they are a check that the ranking of
descriptor spaces is not an artefact of one probe design.

Why this script exists separately from the notebook: the notebook version
compares four spaces (geographic, elevation+mTPI, ERA5-static, TESSERA). The
sharpest objection to the paper is that its hand-crafted baseline is
impoverished — three topographic numbers against a 16-d learned embedding. The
answer has to be a descriptor space with explicit land cover, roughness proxies
and terrain heterogeneity, evaluated identically. That is the
``extra_descriptors.npy`` vector (WorldCover class fractions, tree height, soil
clay/sand, elevation mean/std/min/max, slope, directional gradients), added
here both on its own and stacked with elevation+mTPI — the latter being exactly
the descriptor the ``*_extradesc_*`` ConvCNP arm receives, so the model row and
this model-free row measure the same input.

Target
------
For station s with valid test snapshots T_s, the mean signed interpolation
error

    r_s = (1/|T_s|) Σ_{t∈T_s} ( interp(ERA5)_{s,t} − y_{s,t} )

Averaging over snapshots suppresses transient weather and observation noise,
leaving the persistent, location-specific component that a *static* surface
descriptor could plausibly explain. r_s is constructed without reference to any
descriptor or trained model, so every space is scored against an identical,
neutral target.

Probe
-----
One random forest per descriptor space, scored by pooled K-fold
cross-validation **over stations** (never over snapshots — the question is
generalisation to unseen locations). A positive R² means the space predicts
held-out stations' residuals better than the training-fold mean; near zero
means it exposes no transferable structure.

Outputs (under notebooks/descriptor_analysis_outputs/):
    residual_probe_spaces.json      all regions, variables, spaces
    fig_residual_probe_spaces.png   grouped bars, one panel per variable

Usage:
    uv run python scripts/analysis/residual_probe_spaces.py \\
        --regions europe us --target-variables t2m wind
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from scipy.interpolate import RegularGridInterpolator  # noqa: E402
from sklearn.ensemble import RandomForestRegressor  # noqa: E402
from sklearn.model_selection import KFold  # noqa: E402

from tessera_downscaling.baselines import (  # noqa: E402
    Era5DirResolver,
    build_dataset,
    detect_layout,
    era5_interp_predict,
)
from tessera_downscaling.paths import dataset_dir, processed_dir  # noqa: E402

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger("residual_probe")

REPO_ROOT = Path(__file__).resolve().parents[2]
DATASET = dataset_dir("dataset_timestamp_global")
OUT = REPO_ROOT / "notebooks" / "descriptor_analysis_outputs"

# A station's residual is only a usable target once it averages enough
# snapshots that transient weather has largely cancelled.
DEFAULT_MIN_SNAPSHOTS = 50

# The 17-feature "extra descriptors" vector is NOT purely land-surface: 7 of its
# columns are DEM-derived terrain statistics that the paper's 3-feature
# topography vector lacks (neighbourhood elevation mean/std/min/max, slope, and
# the two directional gradients). Probing the two halves separately is the only
# way to tell whether the vector's advantage comes from land cover or simply
# from richer terrain — a distinction that decides how the result can be
# described.
TERRAIN_FEATURES = {
    "elev_mean",
    "elev_std",
    "elev_min",
    "elev_max",
    "slope",
    "dz_dn",
    "dz_de",
}

SPACE_COLOUR = {
    "geographic": "#7f7f7f",
    "elevation+mTPI": "#ff7f0e",
    "ERA5-static": "#9467bd",
    "terrain stats (7f)": "#e377c2",
    "land cover (10f)": "#8c564b",
    "extended surface (17f)": "#c49a6c",
    "elev+mTPI + extended (20f)": "#bcbd22",
    "TESSERA": "#1f77b4",
}
SPACE_ORDER = list(SPACE_COLOUR)


# ---------------------------------------------------------------------------
# Residual target
# ---------------------------------------------------------------------------


def per_station_residuals(
    args,
    region: str,
    variable: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(station_ids, r_s, n_snapshots)`` over the test split of one region.

    Accumulated as running sums so the pass is O(1) in memory regardless of how
    many snapshots a station reports.
    """
    detect_layout(Path(args.dataset_dir))  # raises unless multi_region_snapshot_v1
    ns = argparse.Namespace(**vars(args))
    ns.train_regions = [region]
    ns.target_variables = [variable]

    dataset = build_dataset(
        ns, [variable], split="test", station_split=args.station_split
    )
    era5_dirs = Era5DirResolver(dataset)

    n_stations = len(dataset.station_ids)
    err_sum = np.zeros(n_stations, dtype=np.float64)
    err_n = np.zeros(n_stations, dtype=np.int64)

    # Episodes report global flat indices under the multi-region layout; map
    # them back to this dataset's dense station ordering.
    offset = dataset.per_region[region].flat_offset

    for idx in range(len(dataset)):
        episode = dataset[idx]
        if episode.get("n_targets", 0) == 0:
            continue
        path = era5_dirs.for_index(idx) / f"{episode['date']}.npy"
        if not path.exists():
            continue
        era5_raw = np.load(path)
        interp = era5_interp_predict(
            variable,
            era5_raw,
            episode["grid_lats"].numpy(),
            episode["grid_lons"].numpy(),
            episode["target_coords"].numpy(),
        )
        obs = episode["target_values"].numpy()
        if obs.ndim > 1:
            obs = obs[:, 0]
        rows = episode["target_station_indices"].numpy() - offset

        good = np.isfinite(interp) & np.isfinite(obs)
        if not good.any():
            continue
        np.add.at(err_sum, rows[good], (interp[good] - obs[good]).astype(np.float64))
        np.add.at(err_n, rows[good], 1)

    keep = err_n >= args.min_snapshots
    logger.info(
        f"[{region}/{variable}] {keep.sum()}/{n_stations} stations with "
        f">= {args.min_snapshots} valid test snapshots "
        f"(median n={int(np.median(err_n[keep])) if keep.any() else 0})"
    )
    return (
        np.asarray(dataset.station_ids)[keep],
        err_sum[keep] / err_n[keep],
        err_n[keep],
    )


# ---------------------------------------------------------------------------
# Descriptor spaces
# ---------------------------------------------------------------------------


def load_descriptor_tables(region: str) -> dict:
    """Per-station descriptor tables, keyed by station_id."""
    stations = pd.read_csv(Path(args_global.dataset_dir) / "stations.csv")
    stations["station_id"] = stations["station_id"].astype(str)

    latents = np.load(args_global.latents_npy)
    listing = pd.read_csv(args_global.latents_csv)
    listing["station_id"] = listing["station_id"].astype(str)
    lat_row = {s: i for i, s in enumerate(listing["station_id"])}

    extra_path = Path(args_global.extra_descriptors_npy)
    extra = np.load(extra_path) if extra_path.exists() else None
    extra_names: list[str] = []
    if extra is None:
        logger.warning(
            f"{extra_path} not found — the hand-crafted descriptor spaces "
            "will be omitted."
        )
    else:
        # Column names come from the sidecar written by
        # build_extra_descriptors.py, so the terrain/cover split is keyed to the
        # actual column order rather than a hard-coded index list.
        names_path = extra_path.with_name(extra_path.stem + "_names.json")
        meta = json.loads(names_path.read_text())
        cols = meta.get("columns") or meta.get("feature_columns") or meta.get("names")
        if cols and isinstance(cols[0], dict):
            cols = [c.get("name") for c in cols]
        if not cols or len(cols) != extra.shape[1]:
            raise ValueError(
                f"{names_path} lists {len(cols) if cols else 0} columns but "
                f"{extra_path.name} has {extra.shape[1]} — cannot split "
                "terrain from land cover safely."
            )
        extra_names = list(cols)
        n_terr = sum(1 for c in extra_names if c in TERRAIN_FEATURES)
        logger.info(
            f"extra descriptors: {extra.shape[1]} columns "
            f"({n_terr} DEM-derived terrain, {extra.shape[1] - n_terr} "
            "land cover / soil / vegetation)"
        )

    # ERA5 static fields interpolated to each station. Deliberately the raw
    # interpolated statics, not the CNN-encoded grid latent, which is dominated
    # by the transient weather being corrected rather than persistent surface
    # character.
    rdir = Path(args_global.dataset_dir) / "regions" / region
    sfield = np.load(rdir / "static_fields.npy")
    glat = np.load(rdir / "lats.npy")
    glon = np.load(rdir / "lons.npy")

    return {
        "stations": stations,
        "latents": latents,
        "lat_row": lat_row,
        "extra": extra,
        "extra_names": extra_names,
        "static": (sfield, glat, glon),
    }


def build_spaces(
    tables: dict,
    station_ids: np.ndarray,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Descriptor matrices aligned to ``station_ids``.

    Returns the spaces plus a boolean mask of stations retained (those present
    in every table). Applying one common mask keeps every space scored on an
    identical station set — otherwise the bars would not be comparable.
    """
    st = tables["stations"].set_index("station_id")
    ids = [str(s) for s in station_ids]

    present = np.array([(s in st.index) and (s in tables["lat_row"]) for s in ids])
    lat_rows = np.array([tables["lat_row"].get(s, -1) for s in ids])
    latent_ok = np.zeros(len(ids), dtype=bool)
    valid_rows = lat_rows >= 0
    latent_ok[valid_rows] = ~np.isnan(tables["latents"][lat_rows[valid_rows]]).any(
        axis=1
    )
    mask = present & latent_ok

    kept_ids = [s for s, m in zip(ids, mask, strict=False) if m]
    sub = st.loc[kept_ids]
    rows = lat_rows[mask]

    sfield, glat, glon = tables["static"]
    qpts = np.column_stack(
        [
            np.clip(sub["latitude"].to_numpy(), glat.min(), glat.max()),
            np.clip(sub["longitude"].to_numpy(), glon.min(), glon.max()),
        ]
    )
    era5_static = np.column_stack(
        [
            RegularGridInterpolator(
                (glat, glon),
                sfield[c],
                method="linear",
                bounds_error=False,
                fill_value=None,
            )(qpts)
            for c in range(sfield.shape[0])
        ]
    )

    topo = sub[["elevation", "delta_elevation", "mtpi"]].to_numpy()
    spaces = {
        "geographic": sub[["latitude", "longitude"]].to_numpy(),
        "elevation+mTPI": topo,
        "ERA5-static": era5_static,
        "TESSERA": tables["latents"][rows],
    }

    if tables["extra"] is not None:
        ex = tables["extra"][rows]
        if np.isnan(ex).any():
            # Mean-fill rather than dropping: dropping would shrink the station
            # set for these spaces alone.
            ex = np.where(np.isnan(ex), np.nanmean(ex, axis=0), ex)
        names = tables["extra_names"]
        terr_idx = [i for i, n in enumerate(names) if n in TERRAIN_FEATURES]
        cover_idx = [i for i, n in enumerate(names) if n not in TERRAIN_FEATURES]
        spaces["terrain stats (7f)"] = ex[:, terr_idx]
        spaces["land cover (10f)"] = ex[:, cover_idx]
        spaces["extended surface (17f)"] = ex
        spaces["elev+mTPI + extended (20f)"] = np.column_stack([topo, ex])

    return spaces, mask


# ---------------------------------------------------------------------------
# Probe
# ---------------------------------------------------------------------------


def cv_r2(x: np.ndarray, y: np.ndarray, args) -> float:
    """Pooled K-fold CV R², folds taken over stations."""
    kf = KFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed)
    pred = np.zeros_like(y, dtype=np.float64)
    for train_idx, test_idx in kf.split(x):
        rf = RandomForestRegressor(
            n_estimators=args.n_trees,
            min_samples_leaf=args.min_samples_leaf,
            random_state=args.seed,
            n_jobs=-1,
        )
        rf.fit(x[train_idx], y[train_idx])
        pred[test_idx] = rf.predict(x[test_idx])
    # Pooled R² against the global mean of the target.
    ss_res = float(((y - pred) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def main():
    global args_global
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--dataset-dir", type=Path, default=DATASET)
    parser.add_argument(
        "--regions",
        nargs="+",
        default=["europe", "us"],
        help="Restricted by default to the well-sampled regions, where the "
        "per-station CV is not dominated by a handful of noisy stations.",
    )
    parser.add_argument("--target-variables", nargs="+", default=["t2m", "wind"])
    parser.add_argument(
        "--station-split", default="test", choices=["train", "test", "all"]
    )
    parser.add_argument("--min-snapshots", type=int, default=DEFAULT_MIN_SNAPSHOTS)
    parser.add_argument("--n-folds", type=int, default=4)
    parser.add_argument("--n-trees", type=int, default=400)
    parser.add_argument("--min-samples-leaf", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--latents-npy",
        type=Path,
        default=processed_dir("station_latents_lat16_grad0.5.npy"),
    )
    parser.add_argument(
        "--latents-csv",
        type=Path,
        default=processed_dir("tessera_global", "station_list_filtered.csv"),
    )
    parser.add_argument(
        "--extra-descriptors-npy",
        type=Path,
        default=processed_dir("extra_descriptors.npy"),
    )
    # Station filters are intentionally NOT applied here: the probe is
    # model-free and should use every station with a usable residual.
    parser.add_argument("--tessera-path", type=Path, default=None)
    parser.add_argument("--tessera-station-csv", type=Path, default=None)
    parser.add_argument("--vae-latents-path", type=Path, default=None)
    parser.add_argument("--vae-latents-station-csv", type=Path, default=None)
    parser.add_argument("--min-tessera-patch-coverage", type=float, default=0.5)
    args = parser.parse_args()
    args_global = args

    OUT.mkdir(parents=True, exist_ok=True)
    report: dict = defaultdict(dict)

    for variable in args.target_variables:
        for region in args.regions:
            ids, resid, n_snap = per_station_residuals(args, region, variable)
            tables = load_descriptor_tables(region)
            spaces, mask = build_spaces(tables, ids)
            y = resid[mask]
            logger.info(
                f"[{region}/{variable}] probing {len(y)} stations, "
                f"residual std={y.std():.3f}"
            )
            if len(y) < args.n_folds * 10:
                logger.warning(
                    f"[{region}/{variable}] only {len(y)} stations — skipping."
                )
                continue
            cell = {}
            for name in SPACE_ORDER:
                if name not in spaces:
                    continue
                r2 = cv_r2(spaces[name], y, args)
                cell[name] = r2
                logger.info(f"    {name:30s} CV R² = {r2:+.4f}")
            cell["_n_stations"] = int(len(y))
            cell["_residual_std"] = float(y.std())
            report[variable][region] = cell

    (OUT / "residual_probe_spaces.json").write_text(json.dumps(report, indent=2))
    logger.info(f"Wrote {OUT / 'residual_probe_spaces.json'}")

    # ---- figure: one panel per variable, grouped bars by region ----
    variables = [v for v in args.target_variables if v in report]
    if not variables:
        logger.warning("Nothing to plot.")
        return
    fig, axes = plt.subplots(
        1,
        len(variables),
        figsize=(7.5 * len(variables), 5.0),
        squeeze=False,
    )
    for ax, variable in zip(axes[0], variables, strict=False):
        regions = list(report[variable])
        names = [n for n in SPACE_ORDER if n in report[variable][regions[0]]]
        xx = np.arange(len(names))
        width = 0.8 / max(len(regions), 1)
        for k, region in enumerate(regions):
            vals = [report[variable][region][n] for n in names]
            off = (k - (len(regions) - 1) / 2) * width
            bars = ax.bar(
                xx + off,
                vals,
                width,
                color=[SPACE_COLOUR[n] for n in names],
                edgecolor="black",
                linewidth=0.6,
                alpha=1.0 if k == 0 else 0.55,
                label=f"{region} (n={report[variable][region]['_n_stations']})",
            )
            for b, v in zip(bars, vals, strict=False):
                ax.text(
                    b.get_x() + b.get_width() / 2,
                    v + (0.012 if v >= 0 else -0.03),
                    f"{v:.2f}",
                    ha="center",
                    va="bottom" if v >= 0 else "top",
                    fontsize=7.5,
                )
        ax.axhline(0.0, color="black", lw=1)
        ax.set_xticks(xx)
        ax.set_xticklabels(
            [n.replace(" (", "\n(").replace(" + ", "\n+ ") for n in names],
            fontsize=8,
        )
        ax.set_ylabel("CV $R^2$ predicting the ERA5-interp residual")
        ax.set_title(variable)
        ax.legend(fontsize=8)

    fig.suptitle(
        "Which descriptor space organises the persistent ERA5-interpolation "
        "residual at unseen stations?\n"
        "Random-forest probe, cross-validated over stations. Identical target "
        "and folds across spaces.",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    path = OUT / "fig_residual_probe_spaces.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Wrote {path}")


if __name__ == "__main__":
    main()
