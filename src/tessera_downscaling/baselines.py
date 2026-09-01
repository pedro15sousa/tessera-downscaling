r"""Non-ML reference baselines (console script ``tessera-baselines``).

Scores three deterministic references on the test split of a
``multi_region_snapshot_v1`` dataset and writes the same output files as
``tessera-evaluate``, so the analysis code treats a baseline run like any
trained run::

    tessera-baselines --baseline era5_interp --dataset-dir datasets/dataset_timestamp_global \
        --target-variables t2m --train-regions europe \
        --tessera-path processed/tessera_global/patch_embeddings_2024.npy \
        --tessera-station-csv processed/tessera_global/station_list_filtered.csv \
        --output-dir training_runs/snapshot_14y_eu/t2m_snap_era5_interp_baseline_seed42

``--baseline``:

* ``era5_interp`` -- bilinearly interpolate the raw ERA5 grid to each station
  and convert units to GHCNh's: ``t2m`` K → °C, ``wind`` = √(u10² + v10²) m/s.

* ``era5_interp_lapse`` (**t2m only**) -- ``era5_interp`` plus a constant
  lapse-rate transfer from the ERA5 orography height to the station height,
  ``T_station = T_interp − Γ · Δelev`` with ``Δelev = station_elevation −
  ERA5_orography`` (the dataset's ``delta_elevation`` column, the same feature
  the ConvCNP receives), so a station above its cell is cooled. ``Γ`` is
  ``--lapse-rate-mode fixed`` (6.5 K/km, ``--lapse-rate``) or ``fitted``: the
  least-squares slope of the signed interpolation error on ``Δelev`` over the
  training split (train stations, train years; ``--lapse-fit-stride`` thins
  the episodes). A fit that is under-sampled falls back to the fixed rate; a
  fit outside the plausible range (e.g. flat terrain) falls back to ``Γ = 0``,
  i.e. plain interpolation. The resolved ``Γ`` and its diagnostics are
  written under ``lapse_rate`` in ``test_summary.json``. Wind has no
  lapse-rate analogue and is refused rather than silently duplicating the
  ``era5_interp`` row.

* ``persistence`` -- the station's most recent valid observation, walking
  back in 6-hour steps up to ``--persistence-max-lookback-hours``; stations
  with no recent observation are excluded.

The baselines have no predictive distribution. To fill the σ / NLL columns
comparably, a single residual std per variable (``std(pred − target)`` over
the test set) is used as a constant predicted σ everywhere; read those
numbers as "if the residuals were Gaussian with this std".

Station filters (``--tessera-path``, ``--vae-latents-path``,
``--min-tessera-patch-coverage``) are passed through to the dataset so a
baseline lands on exactly the station set of the trained runs it is compared
with. Relative paths are interpreted relative to the data root
(:mod:`tessera_downscaling.paths`).

Outputs in ``--output-dir``: ``config.json``, ``test_summary.json`` (=
``test_results.json``), ``test_predictions.npz`` and
``test_station_errors.npz`` (per-station MAE / RMSE / bias / count on the same
station-index basis as ``tessera-evaluate``, without ``subset_per_station`` --
join on ``station_ids`` to re-aggregate over a trained run's subsets).
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from tessera_downscaling.data.dataset import MultiRegionSnapshotDownscalingDataset
from tessera_downscaling.data.helpers import (
    DEFAULT_MIN_TESSERA_PATCH_COVERAGE,
    SUPPORTED_TARGET_VARIABLES,
)
from tessera_downscaling.paths import resolve

logger = logging.getLogger("baselines")

SNAPSHOT_LAYOUT = "multi_region_snapshot_v1"

# ERA5 surface channel order in the snapshot files. Matches SURFACE_VARS in
# tessera_downscaling.preprocessing.helpers -- must stay in sync with it.
ERA5_T2M_CHANNEL = 0  # 2m_temperature (Kelvin)
ERA5_U10_CHANNEL = 1  # 10m_u_component_of_wind (m/s)
ERA5_V10_CHANNEL = 2  # 10m_v_component_of_wind (m/s)

KELVIN_TO_CELSIUS = -273.15

# Persistence: how far back to look for a valid observation, in 6 h steps.
DEFAULT_PERSISTENCE_MAX_LOOKBACK_HOURS = 24
SNAPSHOT_HOUR_STEP = 6

# Standard environmental lapse rate, in K per metre (6.5 K/km).
DEFAULT_LAPSE_RATE_K_PER_M = 0.0065

# Guards on the fitted-lapse-rate estimator. Below MIN_FIT_SAMPLES pairs, or
# below MIN_FIT_DELTA_ELEV_STD metres of Δelev spread, the slope is not
# identifiable and we fall back to the fixed rate. PLAUSIBLE_LAPSE_RATE_RANGE
# brackets physically sensible values -- a fit outside it means the data
# exposes no positive lapse rate (flat terrain) or is broken (unit error,
# lat/lon swap), so we warn loudly and reduce to plain interpolation.
MIN_FIT_SAMPLES = 1000
MIN_FIT_DELTA_ELEV_STD = 10.0
PLAUSIBLE_LAPSE_RATE_RANGE = (0.0, 0.015)

# Targets for which a lapse-rate correction is defined.
LAPSE_SUPPORTED_TARGETS = ("t2m",)

SEASONS = ("DJF", "MAM", "JJA", "SON")


# ---------------------------------------------------------------------------
# Bilinear interpolation
# ---------------------------------------------------------------------------


def bilinear_interp_grid_to_points(
    grid: np.ndarray,
    grid_lats: np.ndarray,
    grid_lons: np.ndarray,
    points: np.ndarray,
) -> np.ndarray:
    """Bilinearly interpolate a 2D field to scattered (lat, lon) points.

    Args:
        grid: ``(H, W)`` array indexed by (lat_idx, lon_idx).
        grid_lats: ``(H,)`` strictly monotonic lat axis (descending or
            ascending -- both are handled).
        grid_lons: ``(W,)`` strictly monotonic lon axis.
        points: ``(N, 2)`` array of (lat, lon) pairs.

    Returns:
        ``(N,)`` array of interpolated values. NaN at any point that falls
        outside the grid envelope.

    """
    lats = points[:, 0]
    lons = points[:, 1]

    # Detect axis direction. ERA5 lat axes are typically descending.
    lat_descending = grid_lats[0] > grid_lats[-1]
    if lat_descending:
        # searchsorted on a descending array: search the reversed array, then
        # unflip to the index of the upper bracket (larger lat).
        rev = grid_lats[::-1]
        i_rev = np.searchsorted(rev, lats, side="left")
        i = (len(grid_lats) - 1) - i_rev
    else:
        i = np.searchsorted(grid_lats, lats, side="left") - 1
    j = np.searchsorted(grid_lons, lons, side="left") - 1

    # Clamp to valid bilinear range.
    i = np.clip(i, 0, len(grid_lats) - 2)
    j = np.clip(j, 0, len(grid_lons) - 2)

    lat0 = grid_lats[i]
    lat1 = grid_lats[i + 1]
    lon0 = grid_lons[j]
    lon1 = grid_lons[j + 1]

    # Fractions along each bracket; when the lat axis descends lat1 < lat0 and
    # the division still yields a fraction in [0, 1].
    wy = (lats - lat0) / (lat1 - lat0)
    wx = (lons - lon0) / (lon1 - lon0)

    # Mark out-of-domain points as NaN, allowing tiny epsilon overshoots so
    # that points exactly on a boundary are interpolated, not dropped.
    eps = 1e-6
    in_domain = (wy >= -eps) & (wy <= 1 + eps) & (wx >= -eps) & (wx <= 1 + eps)

    f00 = grid[i, j]
    f01 = grid[i, j + 1]
    f10 = grid[i + 1, j]
    f11 = grid[i + 1, j + 1]
    interp = (
        f00 * (1 - wy) * (1 - wx)
        + f01 * (1 - wy) * wx
        + f10 * wy * (1 - wx)
        + f11 * wy * wx
    )
    interp = np.where(in_domain, interp, np.nan)
    return interp.astype(np.float32)


# ---------------------------------------------------------------------------
# Per-baseline prediction
# ---------------------------------------------------------------------------


def era5_interp_predict(
    target_variable: str,
    era5_raw: np.ndarray,
    grid_lats: np.ndarray,
    grid_lons: np.ndarray,
    target_coords: np.ndarray,
) -> np.ndarray:
    """Predict via bilinear interpolation of raw ERA5 to station coords.

    Args:
        target_variable: ``"t2m"`` or ``"wind"``.
        era5_raw: ``(C, H, W)`` raw (un-normalised) ERA5 grid for this episode.
        grid_lats: ``(H,)`` grid latitude axis.
        grid_lons: ``(W,)`` grid longitude axis.
        target_coords: ``(N, 2)`` station (lat, lon) array.

    Returns:
        ``(N,)`` predictions in the same units as the GHCNh observations.

    """
    if target_variable == "t2m":
        t2m_kelvin = bilinear_interp_grid_to_points(
            era5_raw[ERA5_T2M_CHANNEL], grid_lats, grid_lons, target_coords
        )
        return t2m_kelvin + KELVIN_TO_CELSIUS
    if target_variable == "wind":
        u = bilinear_interp_grid_to_points(
            era5_raw[ERA5_U10_CHANNEL], grid_lats, grid_lons, target_coords
        )
        v = bilinear_interp_grid_to_points(
            era5_raw[ERA5_V10_CHANNEL], grid_lats, grid_lons, target_coords
        )
        return np.sqrt(u**2 + v**2).astype(np.float32)
    raise ValueError(
        f"era5_interp does not support target {target_variable!r}; "
        f"supported: {sorted(SUPPORTED_TARGET_VARIABLES)}."
    )


def apply_lapse_rate_correction(
    interp_values: np.ndarray,
    delta_elev: np.ndarray,
    lapse_rate: float,
) -> np.ndarray:
    """Transfer an interpolated temperature from ERA5 orography to station height.

    ``T_station = T_interp − Γ · Δelev`` with ``Δelev = station_elev −
    ERA5_orography`` (metres) and ``Γ`` in K/m, so a station above its ERA5
    cell (Δelev > 0) is cooled.

    Stations with a non-finite Δelev receive **no** correction rather than a
    NaN prediction: dropping them would silently change the station set
    relative to the other rows of the same table, which is the one thing this
    comparison must not do. Such stations are counted by the caller.

    Args:
        interp_values: ``(N,)`` interpolated ERA5 temperature, in °C.
        delta_elev: ``(N,)`` station-minus-orography elevation, in metres.
        lapse_rate: Γ in K/m (positive = temperature falls with height).

    Returns:
        ``(N,)`` corrected temperature, in °C.

    """
    correction = np.where(np.isfinite(delta_elev), lapse_rate * delta_elev, 0.0)
    return (interp_values - correction).astype(np.float32)


def fit_lapse_rate_on_train(
    args: argparse.Namespace, target_variable: str
) -> tuple[float, dict]:
    """Least-squares Γ from the training split, plus a diagnostics dict.

    Regresses the signed interpolation error ``e = T_interp − T_obs`` on
    ``Δelev``. Under ``T_obs ≈ T_interp − Γ·Δelev`` we have ``e ≈ Γ·Δelev``,
    so the slope *is* Γ.

    The fit is through the origin unless ``--lapse-fit-intercept`` is passed:
    a free intercept would absorb the domain-mean interpolation bias, turning
    the row into "lapse rate + mean bias correction" -- a strictly stronger
    baseline that is no longer a pure lapse-rate reference.

    Uses ``split="train"`` and ``station_split="train"``: the test year and
    the held-out stations contribute nothing to Γ.

    Returns:
        ``(gamma, diagnostics)``. Falls back to the fixed rate -- recorded in
        ``diagnostics["fallback_reason"]`` -- when the fit is not
        identifiable, and to ``Γ = 0`` when it lands outside the physically
        plausible range.

    """
    fixed = args.lapse_rate
    diagnostics: dict = {
        "mode": "fitted",
        "fit_intercept": bool(args.lapse_fit_intercept),
        "fixed_fallback_k_per_m": fixed,
        "fallback_reason": None,
    }

    logger.info("Fitting lapse rate on the TRAIN split (train stations, train years)…")
    train_dataset = build_dataset(
        args, [target_variable], split="train", station_split="train"
    )
    era5_dirs = Era5DirResolver(train_dataset)

    deltas: list[np.ndarray] = []
    errors: list[np.ndarray] = []
    stride = max(1, args.lapse_fit_stride)
    for idx in range(0, len(train_dataset), stride):
        episode = train_dataset[idx]
        if episode.get("n_targets", 0) == 0:
            continue
        era5_path = era5_dirs.for_index(idx) / f"{episode['date']}.npy"
        if not era5_path.exists():
            continue
        era5_raw = np.load(era5_path)

        interp = era5_interp_predict(
            target_variable,
            era5_raw,
            episode["grid_lats"].numpy(),
            episode["grid_lons"].numpy(),
            episode["target_coords"].numpy(),
        )
        obs = episode["target_values"].numpy()
        if obs.ndim > 1:
            obs = obs[:, 0]
        d_elev = episode["target_delta_elev"].numpy()

        valid = np.isfinite(interp) & np.isfinite(obs) & np.isfinite(d_elev)
        if not valid.any():
            continue
        deltas.append(d_elev[valid].astype(np.float64))
        errors.append((interp[valid] - obs[valid]).astype(np.float64))

    if not deltas:
        diagnostics["fallback_reason"] = "no valid training pairs"
        logger.warning(
            f"Lapse-rate fit found no valid training pairs; falling back to fixed "
            f"Γ={fixed:.5f} K/m."
        )
        diagnostics["gamma_k_per_m"] = fixed
        return fixed, diagnostics

    x = np.concatenate(deltas)
    y = np.concatenate(errors)
    diagnostics["n_fit_samples"] = int(x.size)
    diagnostics["delta_elev_std_m"] = float(x.std())
    diagnostics["fit_stride"] = stride

    if x.size < MIN_FIT_SAMPLES:
        diagnostics["fallback_reason"] = (
            f"only {x.size} fit samples (< {MIN_FIT_SAMPLES})"
        )
    elif x.std() < MIN_FIT_DELTA_ELEV_STD:
        diagnostics["fallback_reason"] = (
            f"Δelev std {x.std():.2f} m < {MIN_FIT_DELTA_ELEV_STD} m — "
            "slope not identifiable"
        )
    if diagnostics["fallback_reason"] is not None:
        logger.warning(
            f"Lapse-rate fit rejected: {diagnostics['fallback_reason']}. "
            f"Falling back to fixed Γ={fixed:.5f} K/m."
        )
        diagnostics["gamma_k_per_m"] = fixed
        return fixed, diagnostics

    design = (
        np.column_stack([x, np.ones_like(x)])
        if args.lapse_fit_intercept
        else x[:, None]
    )
    coeffs, *_ = np.linalg.lstsq(design, y, rcond=None)
    gamma = float(coeffs[0])
    intercept = float(coeffs[1]) if args.lapse_fit_intercept else 0.0

    residual = y - design @ coeffs
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - float((residual**2).sum()) / ss_tot if ss_tot > 0 else float("nan")
    diagnostics.update(
        {
            "gamma_k_per_m": gamma,
            "gamma_k_per_km": gamma * 1000.0,
            "intercept_k": intercept,
            "fit_r2": r2,
        }
    )

    lo, hi = PLAUSIBLE_LAPSE_RATE_RANGE
    if not (lo <= gamma <= hi):
        diagnostics["fallback_reason"] = (
            f"fitted Γ={gamma * 1000:.2f} K/km outside plausible range "
            f"[{lo * 1000:.1f}, {hi * 1000:.1f}] K/km"
        )
        # Fall back to NO correction, not to the fixed rate: a rejected fit
        # means the training data exposes no identifiable positive lapse rate
        # at these stations -- typically flat terrain where Δelev carries
        # little variance (Australia: Γ=-0.89 K/km, R²≈0). Substituting the
        # textbook 6.5 K/km there applies a correction the data has just
        # argued against and measurably degrades the row below plain
        # interpolation (Australia MAE 1.399 -> 1.515); Γ=0 reduces the fitted
        # variant to ERA5 interpolation, the honest reading of "no elevation
        # correction is identifiable here".
        logger.warning(
            f"Lapse-rate fit rejected: {diagnostics['fallback_reason']}. Using Γ=0 "
            "(no correction), so this row reduces to plain ERA5 interpolation."
        )
        diagnostics["gamma_k_per_m"] = 0.0
        diagnostics["reduced_to_plain_interpolation"] = True
        return 0.0, diagnostics

    logger.info(
        f"Fitted Γ = {gamma * 1000:.3f} K/km (intercept {intercept:+.3f} K, "
        f"R²={r2:.4f}, n={x.size:,}, Δelev std={x.std():.1f} m)"
    )
    return gamma, diagnostics


def step_timestamp_back(ts: str, hours: int) -> str:
    """Return ``ts`` shifted back by ``hours`` hours, as ``YYYY-MM-DD-HH``."""
    dt = datetime.strptime(ts, "%Y-%m-%d-%H")
    return (dt - timedelta(hours=hours)).strftime("%Y-%m-%d-%H")


def persistence_predict(
    target_variable: str,
    timestamp: str,
    target_station_indices: np.ndarray,
    ghcnh_snapshot_dir: Path,
    ghcnh_index_for_station: np.ndarray,
    max_lookback_hours: int,
) -> np.ndarray:
    """Predict via the most recent valid GHCNh observation at the same station.

    Walks back in ``SNAPSHOT_HOUR_STEP``-hour steps up to ``max_lookback_hours``,
    returning the first non-NaN observation per target station. Stations with
    no valid observation in the lookback window get NaN, which excludes them
    from downstream metrics.

    Args:
        target_variable: ``"t2m"`` or ``"wind"``.
        timestamp: Episode timestamp ``YYYY-MM-DD-HH`` -- the time at which
            we are *predicting* (so we look at *earlier* timestamps).
        target_station_indices: ``(N,)`` region-local indices into the
            dataset's filtered station list.
        ghcnh_snapshot_dir: Directory containing ``YYYY-MM-DD-HH.npz`` files,
            one per timestamp, shared by all regions.
        ghcnh_index_for_station: Per-station mapping from filtered-station
            index to row index in the GHCNh snapshot files.
        max_lookback_hours: How far back to walk before giving up.

    Returns:
        ``(N,)`` predictions in GHCNh units. NaN for stations whose last
        valid obs is older than the lookback window.

    """
    n = len(target_station_indices)
    preds = np.full(n, np.nan, dtype=np.float32)
    still_missing = np.ones(n, dtype=bool)
    ghcnh_rows = ghcnh_index_for_station[target_station_indices]

    n_steps = max_lookback_hours // SNAPSHOT_HOUR_STEP
    for step in range(1, n_steps + 1):
        if not still_missing.any():
            break
        prev_path = (
            ghcnh_snapshot_dir
            / f"{step_timestamp_back(timestamp, step * SNAPSHOT_HOUR_STEP)}.npz"
        )
        if not prev_path.exists():
            continue
        with np.load(prev_path) as f:
            if target_variable not in f:
                continue
            values_at_prev = f[target_variable]
        candidate_values = values_at_prev[ghcnh_rows]
        fill_mask = still_missing & np.isfinite(candidate_values)
        preds[fill_mask] = candidate_values[fill_mask]
        still_missing &= ~fill_mask
    return preds


# ---------------------------------------------------------------------------
# Shared dataset / layout plumbing
# ---------------------------------------------------------------------------


def detect_layout(dataset_dir: Path) -> str:
    """Return the dataset's ``layout_version``; raise unless it is the snapshot one."""
    md_path = dataset_dir / "metadata.json"
    if not md_path.exists():
        raise FileNotFoundError(f"{md_path} not found")
    with open(md_path) as f:
        layout = json.load(f).get("layout_version")
    if layout != SNAPSHOT_LAYOUT:
        raise ValueError(
            f"{dataset_dir} has layout_version={layout!r}; only {SNAPSHOT_LAYOUT!r} "
            "is supported."
        )
    return layout


def build_dataset(
    args: argparse.Namespace,
    target_variables: list[str],
    split: str,
    station_split: str,
) -> MultiRegionSnapshotDownscalingDataset:
    """Construct a snapshot dataset with this run's station filters applied.

    Single source of truth for the filter set, so the lapse-rate fit on the
    train split and the evaluation on the test split cannot silently disagree
    about which stations exist. Reads ``dataset_dir``, ``train_regions``,
    ``tessera_path`` / ``tessera_station_csv``, ``vae_latents_path`` /
    ``vae_latents_station_csv`` and ``min_tessera_patch_coverage`` from ``args``.
    """
    if not args.train_regions:
        raise ValueError("--train-regions is required.")
    return MultiRegionSnapshotDownscalingDataset(
        dataset_dir=Path(args.dataset_dir),
        regions=list(args.train_regions),
        split=split,
        station_split=station_split,
        target_variables=target_variables,
        tessera_path=args.tessera_path,
        tessera_station_csv=args.tessera_station_csv,
        include_static_fields=False,  # baselines read raw ERA5 from disk
        vae_latents_path=args.vae_latents_path,
        vae_latents_station_csv=args.vae_latents_station_csv,
        min_patch_coverage=args.min_tessera_patch_coverage,
    )


class Era5DirResolver:
    """Map an episode index to the ``era5_snapshot`` directory that serves it.

    Episodes are ``(region, timestamp)`` pairs and each region keeps its raw
    ERA5 fields under ``regions/<region>/era5_snapshot``; wrapping the lookup
    keeps the eval loop and the lapse-rate fit identical.
    """

    def __init__(self, dataset: MultiRegionSnapshotDownscalingDataset) -> None:
        self._dataset = dataset

    def for_index(self, idx: int) -> Path:
        region_name, _local_idx = self._dataset._dispatch(idx)
        return self._dataset.per_region[region_name].region_dir / "era5_snapshot"


# ---------------------------------------------------------------------------
# Main eval loop
# ---------------------------------------------------------------------------


def date_to_season(date_str: str) -> str:
    """Map YYYY-MM-DD or YYYY-MM-DD-HH to meteorological season."""
    month = int(date_str.split("-")[1])
    if month in (12, 1, 2):
        return "DJF"
    if month in (3, 4, 5):
        return "MAM"
    if month in (6, 7, 8):
        return "JJA"
    return "SON"


def evaluate_baseline(args: argparse.Namespace) -> tuple[dict, dict, dict]:
    """Run one baseline over the test set.

    Returns ``(test_summary, npz_data, station_npz)``.
    """
    target_variables = list(args.target_variables)
    layout = detect_layout(Path(args.dataset_dir))

    # Lapse rate: resolve Γ before touching the test set. Fitting reads the
    # train split only, so the test year and held-out stations stay clean.
    lapse_rate = None
    lapse_diagnostics = None
    if args.baseline == "era5_interp_lapse":
        unsupported = [v for v in target_variables if v not in LAPSE_SUPPORTED_TARGETS]
        if unsupported:
            raise ValueError(
                "era5_interp_lapse is defined only for "
                f"{list(LAPSE_SUPPORTED_TARGETS)}, "
                f"got {unsupported}. Wind speed has no lapse-rate analogue; running it "
                "here would duplicate the era5_interp row under a different label."
            )
        if args.lapse_rate_mode == "fitted":
            lapse_rate, lapse_diagnostics = fit_lapse_rate_on_train(
                args, target_variables[0]
            )
        else:
            lapse_rate = args.lapse_rate
            lapse_diagnostics = {
                "mode": "fixed",
                "gamma_k_per_m": lapse_rate,
                "gamma_k_per_km": lapse_rate * 1000.0,
            }
        logger.info(
            f"Applying lapse-rate correction Γ = {lapse_rate * 1000:.3f} K/km "
            f"(mode={args.lapse_rate_mode})"
        )

    # The dataset enforces the temporal-test split and the station filters, so
    # the baseline lands on the same stations as the trained runs.
    test_dataset = build_dataset(
        args, target_variables, split="test", station_split=args.station_split
    )
    logger.info(
        f"Test dataset: {len(test_dataset)} episodes, layout={layout}, "
        f"baseline={args.baseline}"
    )
    era5_dirs = Era5DirResolver(test_dataset)
    ghcnh_snapshot_dir = test_dataset._ghcnh_snapshot_dir

    # Counters for the lapse-rate correction, reported in the summary so a
    # silently-uncorrected run is impossible to mistake for a corrected one.
    n_missing_delta_elev = 0
    corrections_seen: list[np.ndarray] = []

    all_preds = {v: [] for v in target_variables}
    all_targets = {v: [] for v in target_variables}
    all_station_indices = {v: [] for v in target_variables}
    season_errors = {v: defaultdict(list) for v in target_variables}

    n_total = len(test_dataset)
    for idx in range(n_total):
        episode = test_dataset[idx]
        if episode.get("n_targets", 0) == 0:
            continue
        timestamp = episode["date"]
        target_coords = episode["target_coords"].numpy()  # (N, 2)
        target_values = episode["target_values"].numpy()  # (N,) or (N, V)
        target_station_indices = episode["target_station_indices"].numpy()
        grid_lats = episode["grid_lats"].numpy()
        grid_lons = episode["grid_lons"].numpy()

        # The episode's context grid is normalised; the baselines want raw
        # ERA5, re-loaded from disk. Station indices in the episode are flat
        # across regions; GHCNh rows are looked up through the region's own
        # (local) station table.
        region_name, _ = test_dataset._dispatch(idx)
        region_state = test_dataset.per_region[region_name]
        local_target_indices = target_station_indices - region_state.flat_offset
        if args.baseline in ("era5_interp", "era5_interp_lapse"):
            era5_raw = np.load(era5_dirs.for_index(idx) / f"{timestamp}.npy")

        for vi, var in enumerate(target_variables):
            tv = target_values if target_values.ndim == 1 else target_values[:, vi]
            if args.baseline == "era5_interp":
                preds = era5_interp_predict(
                    var, era5_raw, grid_lats, grid_lons, target_coords
                )
            elif args.baseline == "era5_interp_lapse":
                interp = era5_interp_predict(
                    var, era5_raw, grid_lats, grid_lons, target_coords
                )
                # Δelev comes straight from the episode, already sliced to this
                # snapshot's valid target stations -- identical to the e(x⋆)
                # component the ConvCNP receives.
                d_elev = episode["target_delta_elev"].numpy()
                n_missing_delta_elev += int((~np.isfinite(d_elev)).sum())
                preds = apply_lapse_rate_correction(interp, d_elev, lapse_rate)
                corrections_seen.append(
                    (interp - preds)[np.isfinite(interp)].astype(np.float32)
                )
            elif args.baseline == "persistence":
                preds = persistence_predict(
                    var,
                    timestamp,
                    local_target_indices,
                    ghcnh_snapshot_dir,
                    region_state.ghcnh_index_for_station,
                    args.persistence_max_lookback_hours,
                )
            else:
                raise ValueError(f"Unknown baseline: {args.baseline}")

            # Drop pairs where either pred or target is NaN (persistence yields
            # NaN for stations with no recent observation).
            valid = np.isfinite(preds) & np.isfinite(tv)
            preds_v = preds[valid].astype(np.float32)
            targets_v = tv[valid].astype(np.float32)
            all_preds[var].extend(preds_v.tolist())
            all_targets[var].extend(targets_v.tolist())
            all_station_indices[var].extend(target_station_indices[valid].tolist())
            season_errors[var][date_to_season(timestamp)].extend(
                np.abs(preds_v - targets_v).tolist()
            )

        if (idx + 1) % 200 == 0:
            running = np.array(all_preds[target_variables[0]])
            running_t = np.array(all_targets[target_variables[0]])
            mae = np.abs(running - running_t).mean() if len(running) else float("nan")
            logger.info(
                f"  {idx + 1}/{n_total} episodes, {target_variables[0]} MAE so far: "
                f"{mae:.3f}"
            )

    # ------------------------------------------------------------------
    # Metrics -- same keys as tessera-evaluate.
    # ------------------------------------------------------------------
    test_summary = {
        "checkpoint_epoch": 0,  # baselines don't train
        "best_val_loss": None,
        "target_variables": target_variables,
        "baseline_kind": args.baseline,
    }
    npz_data = {}

    # Provenance for the lapse-rate row: Γ, how it was obtained, how large the
    # corrections actually were, and how many stations lacked a usable Δelev.
    if lapse_diagnostics is not None:
        if corrections_seen:
            corr = np.concatenate(corrections_seen)
            lapse_diagnostics.update(
                {
                    "correction_mean_k": float(corr.mean()),
                    "correction_abs_mean_k": float(np.abs(corr).mean()),
                    "correction_std_k": float(corr.std()),
                    "correction_p01_k": float(np.percentile(corr, 1)),
                    "correction_p99_k": float(np.percentile(corr, 99)),
                }
            )
        lapse_diagnostics["n_targets_missing_delta_elev"] = n_missing_delta_elev
        test_summary["lapse_rate"] = lapse_diagnostics
        rounded = {
            k: (round(v, 5) if isinstance(v, float) else v)
            for k, v in lapse_diagnostics.items()
        }
        logger.info(f"Lapse-rate diagnostics: {json.dumps(rounded)}")
        if n_missing_delta_elev:
            logger.warning(
                f"{n_missing_delta_elev} target rows had a non-finite Δelev and were "
                "left uncorrected (kept, to preserve the station set)."
            )

    # Station metadata, shared by every variable's per-station arrays. Same
    # keys and station-index basis as tessera-evaluate, so the two kinds of
    # run can be joined station-by-station downstream.
    station_npz = {
        "station_ids": test_dataset.station_ids,
        "station_lats": test_dataset.station_lats,
        "station_lons": test_dataset.station_lons,
        "station_elevs": test_dataset.station_elevs,
        "station_delta_elevs": test_dataset.station_delta_elevs,
    }
    n_all_stations = len(test_dataset.station_ids)

    for var in target_variables:
        preds = np.array(all_preds[var])
        targets = np.array(all_targets[var])
        if len(preds) == 0:
            logger.warning(f"No valid predictions for {var}; skipping metrics.")
            continue

        errors = np.abs(preds - targets)
        residuals = preds - targets
        bias = float(residuals.mean())
        correlation = (
            float(np.corrcoef(preds, targets)[0, 1]) if len(preds) > 1 else 0.0
        )

        # Calibration under a constant σ = std(pred − target) at every station.
        residual_std = float(residuals.std()) if len(residuals) > 1 else float("nan")
        if residual_std > 0:
            within_1s = float((errors < residual_std).mean() * 100)
            within_2s = float((errors < 2 * residual_std).mean() * 100)
            log_var = 2 * np.log(residual_std)
            nll = float(
                (
                    0.5 * (log_var + residuals**2 / residual_std**2 + np.log(2 * np.pi))
                ).mean()
            )
        else:
            within_1s = within_2s = nll = float("nan")

        logger.info("=" * 60)
        logger.info(f"BASELINE RESULTS — {args.baseline} — {var}")
        logger.info("=" * 60)
        logger.info(f"  Predictions:   {len(errors):,}")
        logger.info(f"  MAE:           {errors.mean():.3f}")
        logger.info(f"  RMSE:          {np.sqrt((errors**2).mean()):.3f}")
        logger.info(f"  Bias:          {bias:+.4f}")
        logger.info(f"  Correlation:   {correlation:.4f}")
        logger.info(f"  Residual σ:    {residual_std:.3f}  (used as constant pred σ)")
        logger.info(f"  Within 1σ:     {within_1s:.1f}%")
        logger.info(f"  Within 2σ:     {within_2s:.1f}%")

        test_summary[f"{var}_n_predictions"] = len(errors)
        test_summary[f"{var}_n_test_stations"] = len(set(all_station_indices[var]))
        test_summary[f"{var}_mae"] = float(errors.mean())
        test_summary[f"{var}_rmse"] = float(np.sqrt((errors**2).mean()))
        test_summary[f"{var}_nll"] = float(nll)
        test_summary[f"{var}_bias"] = bias
        test_summary[f"{var}_correlation"] = correlation
        test_summary[f"{var}_mean_pred_std"] = residual_std
        test_summary[f"{var}_within_1sigma"] = within_1s
        test_summary[f"{var}_within_2sigma"] = within_2s
        for q in (50, 90, 95, 99):
            test_summary[f"{var}_p{q}"] = float(np.percentile(errors, q))
        test_summary[f"{var}_seasonal_mae"] = {
            season: float(np.mean(season_errors[var][season]))
            for season in SEASONS
            if season in season_errors[var]
        }

        # Raw arrays; station_indices is what makes them re-aggregatable.
        sindices = np.array(all_station_indices[var], dtype=np.int64)
        npz_data[f"{var}_predictions"] = preds.astype(np.float32)
        npz_data[f"{var}_targets"] = targets.astype(np.float32)
        npz_data[f"{var}_predicted_stds"] = np.full_like(
            preds, residual_std, dtype=np.float32
        )
        npz_data[f"{var}_station_indices"] = sindices

        # Per-station aggregation over the FULL station list so every array in
        # the npz shares one index basis; stations with no valid pair keep
        # count == 0 (consumers mask on count > 0, as for tessera-evaluate).
        station_mae = np.zeros(n_all_stations)
        station_rmse = np.zeros(n_all_stations)
        station_bias = np.zeros(n_all_stations)
        station_count = np.zeros(n_all_stations, dtype=np.int64)
        for si in np.unique(sindices):
            m = sindices == si
            station_mae[si] = errors[m].mean()
            station_rmse[si] = np.sqrt((errors[m] ** 2).mean())
            station_bias[si] = residuals[m].mean()
            station_count[si] = int(m.sum())
        station_npz[f"{var}_station_mae"] = station_mae
        station_npz[f"{var}_station_rmse"] = station_rmse
        station_npz[f"{var}_station_bias"] = station_bias
        station_npz[f"{var}_station_count"] = station_count

    return test_summary, npz_data, station_npz


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Score the non-ML references (ERA5 interpolation, lapse-rate "
        "corrected interpolation, persistence) on the test split."
    )
    parser.add_argument(
        "--baseline",
        choices=["era5_interp", "era5_interp_lapse", "persistence"],
        required=True,
        help="Which baseline to compute. era5_interp_lapse is t2m-only.",
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        required=True,
        help="multi_region_snapshot_v1 dataset (relative to the data root unless "
        "absolute).",
    )
    parser.add_argument(
        "--target-variables",
        nargs="+",
        required=True,
        choices=sorted(SUPPORTED_TARGET_VARIABLES),
        help="Target variable(s) to score.",
    )
    parser.add_argument(
        "--train-regions",
        nargs="+",
        required=True,
        help="Region(s) whose stations are scored (the trained runs' training "
        "regions).",
    )
    parser.add_argument(
        "--station-split",
        choices=["train", "test", "all"],
        default="test",
        help="Spatial split to score: 'test' (default; held-out stations), 'train', "
        "or 'all' (the station-rollout experiment, for per-subset breakdowns "
        "downstream).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Run directory for the outputs (relative to the data root unless "
        "absolute).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Recorded in config.json so runs sort under <name>_seed<seed>/; the "
        "baselines are deterministic.",
    )
    lapse = parser.add_argument_group("era5_interp_lapse")
    lapse.add_argument(
        "--lapse-rate-mode",
        choices=["fixed", "fitted"],
        default="fixed",
        help="'fixed' uses --lapse-rate; 'fitted' estimates Γ by least squares on "
        "the training split and falls back to fixed / zero when the fit is "
        "degenerate.",
    )
    lapse.add_argument(
        "--lapse-rate",
        type=float,
        default=DEFAULT_LAPSE_RATE_K_PER_M,
        help="Γ in K per METRE (not K/km). Default 0.0065 = 6.5 K/km.",
    )
    lapse.add_argument(
        "--lapse-fit-intercept",
        action="store_true",
        help="Fit a free intercept alongside Γ (a 'lapse rate + mean bias "
        "correction' row rather than a pure lapse-rate reference).",
    )
    lapse.add_argument(
        "--lapse-fit-stride",
        type=int,
        default=1,
        help="Use every Nth training episode when fitting Γ (stride 4-8 changes "
        "the estimate negligibly). Recorded in the summary.",
    )
    parser.add_argument(
        "--persistence-max-lookback-hours",
        type=int,
        default=DEFAULT_PERSISTENCE_MAX_LOOKBACK_HOURS,
        help="persistence: max hours to look back for a valid prior observation.",
    )
    filt = parser.add_argument_group(
        "station filters (pass the same files as the trained runs being compared)"
    )
    filt.add_argument(
        "--tessera-path",
        type=Path,
        default=None,
        help="TESSERA patch .npy: keep only stations with a valid patch.",
    )
    filt.add_argument(
        "--tessera-station-csv",
        type=Path,
        default=None,
        help="CSV row-aligned with --tessera-path. Required with it.",
    )
    filt.add_argument(
        "--vae-latents-path",
        type=Path,
        default=None,
        help="Per-station vector .npy: additionally keep only stations with a "
        "non-NaN row (the TESSERA arm's filter).",
    )
    filt.add_argument(
        "--vae-latents-station-csv",
        type=Path,
        default=None,
        help="CSV row-aligned with --vae-latents-path. Required with it.",
    )
    filt.add_argument(
        "--min-tessera-patch-coverage",
        type=float,
        default=DEFAULT_MIN_TESSERA_PATCH_COVERAGE,
        help="Patch coverage threshold of the TESSERA filter (0 disables). "
        f"Default {DEFAULT_MIN_TESSERA_PATCH_COVERAGE}.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
        level=logging.INFO,
    )
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.baseline != "era5_interp_lapse":
        for flag, default in (
            ("lapse_rate_mode", "fixed"),
            ("lapse_fit_intercept", False),
            ("lapse_fit_stride", 1),
        ):
            if getattr(args, flag) != default:
                parser.error(
                    f"--{flag.replace('_', '-')} only applies to --baseline "
                    f"era5_interp_lapse (got {args.baseline})."
                )
    if args.lapse_rate <= 0:
        parser.error("--lapse-rate must be positive (K per metre, e.g. 0.0065).")
    if args.lapse_fit_stride < 1:
        parser.error("--lapse-fit-stride must be >= 1.")
    if args.tessera_path is not None and args.tessera_station_csv is None:
        parser.error("--tessera-station-csv is required when --tessera-path is set.")
    if args.vae_latents_path is not None and args.vae_latents_station_csv is None:
        parser.error(
            "--vae-latents-station-csv is required when --vae-latents-path is set."
        )
    for name in (
        "dataset_dir",
        "output_dir",
        "tessera_path",
        "tessera_station_csv",
        "vae_latents_path",
        "vae_latents_station_csv",
    ):
        value = getattr(args, name)
        if value is not None:
            setattr(args, name, resolve(value))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with open(args.output_dir / "config.json", "w") as f:
        json.dump(
            {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
            f,
            indent=2,
        )

    test_summary, npz_data, station_npz = evaluate_baseline(args)

    for name in ("test_summary.json", "test_results.json"):
        with open(args.output_dir / name, "w") as f:
            json.dump(test_summary, f, indent=2)
    np.savez(args.output_dir / "test_predictions.npz", **npz_data)
    np.savez(args.output_dir / "test_station_errors.npz", **station_npz)
    logger.info(f"Wrote results to {args.output_dir}")


if __name__ == "__main__":
    main()
