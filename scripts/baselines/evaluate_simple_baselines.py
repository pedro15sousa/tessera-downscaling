"""Evaluate simple non-ML baselines and produce evaluate.py-compatible output.

Three baselines are supported:

  * `era5_interp` — bilinearly interpolate the ERA5 grid to each station's
    (lat, lon) and use that as the prediction. Units are converted to match
    GHCNh observations:
      - t2m: ERA5 K → GHCNh °C   (subtract 273.15)
      - wind: sqrt(u10² + v10²)  (m/s, already matches GHCNh)
      - precip: ERA5 total_precipitation_6hr (m) × 1000 → mm/6h
        (matches GHCNh's `precipitation_6_hour` column, which reports
        the 6h accumulation ending at the synoptic timestamp)

  * `era5_interp_lapse` — `era5_interp` for t2m plus a dry-adiabatic-style
    elevation correction. **t2m only** (see below). The ERA5 2 m temperature
    is representative of the model orography height, which differs from the
    true station elevation; the correction transfers it to the station height
    at a constant lapse rate Γ:

        T_station = T_interp − Γ · Δelev,
        Δelev := station_elevation − ERA5_orography            (metres)

    Δelev is the dataset's `delta_elevation` column, computed in
    `scripts/preprocessing/helpers.py:compute_delta_elevation` as exactly
    `stations["elevation"] − orography_interpolated_to_station`, so a station
    *above* its ERA5 cell has Δelev > 0 and receives a *cooling* correction.
    This is the same feature the ConvCNP receives as part of e(x⋆), so the
    baseline is using no information the trained models lack.

    Γ is set by `--lapse-rate-mode`:
      - `fixed`  — the standard environmental lapse rate, 6.5 K/km
                   (override with `--lapse-rate`).
      - `fitted` — least-squares Γ estimated on the **training** split
                   (train stations, train years) by regressing the signed
                   interpolation error (T_interp − T_obs) on Δelev. Reported
                   alongside its R² and sample count. Fitting on train-only
                   keeps the test year and held-out stations untouched.
                   If the fit is under-sampled or Δelev has too little
                   spread, falls back to `fixed`. If the fitted slope is
                   negative or implausibly large -- i.e. the data exposes no
                   positive lapse rate, as in flat terrain -- falls back to
                   Γ=0, so the row reduces to plain interpolation rather than
                   applying a correction the data argues against.

    Refusing non-t2m targets is deliberate: wind speed and precipitation have
    no lapse-rate analogue, and silently returning the uncorrected field would
    emit a table row numerically identical to `era5_interp` while being
    labelled as something else.

  * `persistence` — predict the station's value at time T from its most
    recent valid GHCNh observation at an earlier timestamp T − Δ. The
    lookback walks backward in 6-hour steps up to a configurable maximum,
    skipping NaN observations. Stations with no valid recent observation
    within the lookback window contribute NaN predictions and are excluded
    from the metrics.

Both baselines are deterministic — they have no learned uncertainty. To
fill the σ-coverage and NLL columns in the analysis tables in a defensible
way, this script computes a single global "residual std" per variable
(standard deviation of pred − target across the test set) and uses that
as a constant predicted σ at every station. The resulting calibration
numbers should be read as "if the residuals were Gaussian with this std".

Output: writes `test_summary.json`, `test_results.json`, `test_predictions.npz`,
`test_station_errors.npz`, and `config.json` into the run directory, mirroring
the format produced by `scripts/evaluate.py` so the analysis notebooks pick
these up without changes. `test_station_errors.npz` carries the per-station
MAE/RMSE/bias/count arrays keyed the same way evaluate.py keys them, which is
what lets a notebook re-aggregate a baseline over an arbitrary subset of
stations — e.g. drawing an ERA5-interpolation reference on exactly the stations
a given panel scores, instead of one global number. It has no
`subset_per_station` (that comes from a trained run's region specs, which the
baselines don't take); join subsets by `station_ids` if you need them.

Usage:
    python scripts/baselines/evaluate_simple_baselines.py \\
        --baseline era5_interp \\
        --dataset-dir .tmp_output/dataset_timestamp_global \\
        --target-variables t2m \\
        --train-regions europe \\
        --normalisation-policy per_region \\
        --output-dir .tmp_output/training_runs_snapshot_14y_eu/t2m_snap_era5_interp_baseline_seed42 \\
        --seed 42

For persistence, optionally pass --persistence-max-lookback-hours (default 24).

For the lapse-rate row (t2m only), either of:
    --baseline era5_interp_lapse --lapse-rate-mode fixed
    --baseline era5_interp_lapse --lapse-rate-mode fitted --lapse-fit-stride 4
The resolved Γ, its fit diagnostics, and the realised correction distribution
are written into `test_summary.json` under the `lapse_rate` key.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

# Add the project root to the path so we can import tessera_downscaling.
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
sys.path.insert(0, str(REPO_ROOT / "projects" / "tessera_downscaling" / "src"))

from tessera_downscaling.data.dataset import (  # noqa: E402
    SnapshotDownscalingDataset,
    MultiRegionSnapshotDownscalingDataset,
)


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger("simple_baselines")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# ERA5 surface channel order in the snapshot files. Matches SURFACE_VARS in
# scripts/preprocessing/helpers.py — must stay in sync with that file.
ERA5_T2M_CHANNEL = 0  # 2m_temperature (Kelvin)
ERA5_U10_CHANNEL = 1  # 10m_u_component_of_wind (m/s)
ERA5_V10_CHANNEL = 2  # 10m_v_component_of_wind (m/s)
ERA5_TP_CHANNEL = 4   # total_precipitation_6hr (m, accumulated over the 6h
                      # ending at the snapshot timestamp). Channel order
                      # follows SURFACE_VARS in scripts/preprocessing/helpers.py.

KELVIN_TO_CELSIUS = -273.15
M_TO_MM = 1000.0

# Bilinear interpolation neighbourhood: 4 nearest grid points.
N_INTERP_NEIGHBOURS = 4

# Persistence: how far back to look for a valid observation. 6h steps.
DEFAULT_PERSISTENCE_MAX_LOOKBACK_HOURS = 24
SNAPSHOT_HOUR_STEP = 6

# Standard environmental lapse rate, in K per metre (6.5 K/km).
DEFAULT_LAPSE_RATE_K_PER_M = 0.0065

# Guards on the fitted-lapse-rate estimator. Below MIN_FIT_SAMPLES pairs, or
# below MIN_FIT_DELTA_ELEV_STD metres of Δelev spread, the slope is not
# identifiable and we fall back to the fixed rate. PLAUSIBLE_LAPSE_RATE_RANGE
# brackets physically sensible values — a fit outside it signals a data
# problem (unit error, lat/lon swap) rather than an unusual climate, so we
# warn loudly and fall back rather than silently publishing it.
MIN_FIT_SAMPLES = 1000
MIN_FIT_DELTA_ELEV_STD = 10.0
PLAUSIBLE_LAPSE_RATE_RANGE = (0.0, 0.015)

# Baselines that require a lapse-rate elevation correction, and the targets
# for which such a correction is defined.
LAPSE_SUPPORTED_TARGETS = ("t2m",)


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
            ascending — both are handled).
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

    # Find lat bracketing index for each point.
    if lat_descending:
        # np.searchsorted on a descending array: invert by searching the
        # reversed array, then unflip.
        rev = grid_lats[::-1]
        i_rev = np.searchsorted(rev, lats, side="left")
        i = (len(grid_lats) - 1) - i_rev  # index of the upper bracket (larger lat)
        # Bracket is [i, i+1] in the descending order, but we want the
        # numerically smaller-then-larger pair. Use (i, i+1) with i clamped.
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

    # Bilinear weights. Note: when lat axis descends, lat1 < lat0, so the
    # division still gives a fraction in [0, 1] as long as we use the
    # correct direction.
    # Fraction along the lat bracket:
    wy = (lats - lat0) / (lat1 - lat0)
    wx = (lons - lon0) / (lon1 - lon0)

    # Mark out-of-domain points as NaN. Allow tiny epsilon overshoots so
    # that points exactly on a boundary are interpolated, not dropped.
    eps = 1e-6
    in_domain = (
        (wy >= -eps) & (wy <= 1 + eps)
        & (wx >= -eps) & (wx <= 1 + eps)
    )

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
        grid_lats, grid_lons: grid axes.
        target_coords: ``(N, 2)`` station (lat, lon) array.

    Returns:
        ``(N,)`` predictions in the same units as the GHCNh observations.
    """
    if target_variable == "t2m":
        t2m_kelvin = bilinear_interp_grid_to_points(
            era5_raw[ERA5_T2M_CHANNEL], grid_lats, grid_lons, target_coords,
        )
        return t2m_kelvin + KELVIN_TO_CELSIUS
    elif target_variable == "wind":
        u = bilinear_interp_grid_to_points(
            era5_raw[ERA5_U10_CHANNEL], grid_lats, grid_lons, target_coords,
        )
        v = bilinear_interp_grid_to_points(
            era5_raw[ERA5_V10_CHANNEL], grid_lats, grid_lons, target_coords,
        )
        return np.sqrt(u ** 2 + v ** 2).astype(np.float32)
    elif target_variable == "precip":
        # total_precipitation_6hr is stored in metres (raw ERA5 unit).
        # GHCNh reports precipitation_6_hour in mm, so multiply by 1000.
        # ERA5's tp accumulator can be slightly negative due to numerical
        # roundoff — clip to >= 0 to match the physical definition.
        tp_m = bilinear_interp_grid_to_points(
            era5_raw[ERA5_TP_CHANNEL], grid_lats, grid_lons, target_coords,
        )
        return np.clip(tp_m * M_TO_MM, 0.0, None).astype(np.float32)
    else:
        raise ValueError(
            f"era5_interp baseline does not yet support target {target_variable!r}. "
            f"Supported: 't2m', 'wind', 'precip'."
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
    correction = np.where(
        np.isfinite(delta_elev), lapse_rate * delta_elev, 0.0,
    )
    return (interp_values - correction).astype(np.float32)


def fit_lapse_rate_on_train(
    args,
    target_variable: str,
    is_multi_region: bool,
) -> tuple[float, dict]:
    """Least-squares Γ from the training split, plus a diagnostics dict.

    Regresses the signed interpolation error ``e = T_interp − T_obs`` on
    ``Δelev``. Under ``T_obs ≈ T_interp − Γ·Δelev`` we have ``e ≈ Γ·Δelev``,
    so the slope *is* Γ.

    The fit is deliberately through the origin unless
    ``--lapse-fit-intercept`` is passed: a free intercept would absorb the
    domain-mean interpolation bias, turning the row into "lapse rate + mean
    bias correction" — a strictly stronger baseline that is no longer a pure
    lapse-rate reference. Both variants are available so the distinction can
    be reported rather than assumed.

    Uses ``split="train"`` and ``station_split="train"``: the test year and
    the held-out stations contribute nothing to Γ.

    Returns:
        ``(gamma, diagnostics)``. Falls back to the fixed rate — recorded in
        ``diagnostics["fallback_reason"]`` — when the fit is not identifiable
        or lands outside the physically plausible range.
    """
    fixed = args.lapse_rate
    diagnostics: dict = {
        "mode": "fitted",
        "fit_intercept": bool(args.lapse_fit_intercept),
        "fixed_fallback_k_per_m": fixed,
        "fallback_reason": None,
    }

    logger.info(
        "Fitting lapse rate on the TRAIN split (train stations, train years)…"
    )
    train_dataset = build_dataset(
        args, [target_variable], is_multi_region,
        split="train", station_split="train",
    )

    era5_dirs = Era5DirResolver(train_dataset, is_multi_region)

    deltas: list[np.ndarray] = []
    errors: list[np.ndarray] = []
    n_episodes = len(train_dataset)
    stride = max(1, args.lapse_fit_stride)
    for idx in range(0, n_episodes, stride):
        episode = train_dataset[idx]
        if episode.get("n_targets", 0) == 0:
            continue
        timestamp = episode["date"]
        era5_dir = era5_dirs.for_index(idx)
        era5_path = era5_dir / f"{timestamp}.npy"
        if not era5_path.exists():
            continue
        era5_raw = np.load(era5_path)

        interp = era5_interp_predict(
            target_variable, era5_raw,
            episode["grid_lats"].numpy(), episode["grid_lons"].numpy(),
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
            "Lapse-rate fit found no valid training pairs; "
            f"falling back to fixed Γ={fixed:.5f} K/m."
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

    if args.lapse_fit_intercept:
        design = np.column_stack([x, np.ones_like(x)])
    else:
        design = x[:, None]
    coeffs, *_ = np.linalg.lstsq(design, y, rcond=None)
    gamma = float(coeffs[0])
    intercept = float(coeffs[1]) if args.lapse_fit_intercept else 0.0

    residual = y - design @ coeffs
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - float((residual ** 2).sum()) / ss_tot if ss_tot > 0 else float("nan")

    diagnostics.update({
        "gamma_k_per_m": gamma,
        "gamma_k_per_km": gamma * 1000.0,
        "intercept_k": intercept,
        "fit_r2": r2,
    })

    lo, hi = PLAUSIBLE_LAPSE_RATE_RANGE
    if not (lo <= gamma <= hi):
        diagnostics["fallback_reason"] = (
            f"fitted Γ={gamma * 1000:.2f} K/km outside plausible range "
            f"[{lo * 1000:.1f}, {hi * 1000:.1f}] K/km"
        )
        # Fall back to NO correction, not to the fixed rate. A rejected fit
        # means the training data exposes no identifiable positive lapse rate at
        # these stations -- typically flat terrain where Delta-elev carries
        # little variance (observed in Australia: Gamma=-0.89 K/km, R2~0).
        # Substituting the textbook 6.5 K/km there applies a correction the data
        # has just argued against, and measurably degrades the row below plain
        # interpolation (Australia MAE 1.399 -> 1.515). Gamma=0 reduces the
        # fitted variant to ERA5 interpolation instead, which is the honest
        # reading of "no elevation correction is identifiable here".
        logger.warning(
            f"Lapse-rate fit rejected: {diagnostics['fallback_reason']}. "
            "Using Γ=0 (no correction), so this row reduces to plain ERA5 "
            "interpolation. A rejected fit means no positive lapse rate is "
            "identifiable in the training data, not that 6.5 K/km should be "
            "assumed."
        )
        diagnostics["gamma_k_per_m"] = 0.0
        diagnostics["reduced_to_plain_interpolation"] = True
        return 0.0, diagnostics

    logger.info(
        f"Fitted Γ = {gamma * 1000:.3f} K/km "
        f"(intercept {intercept:+.3f} K, R²={r2:.4f}, "
        f"n={x.size:,}, Δelev std={x.std():.1f} m)"
    )
    return gamma, diagnostics


def step_timestamp_back(ts: str, hours: int) -> str:
    """Return ts shifted back by `hours` hours, in the same YYYY-MM-DD-HH format."""
    dt = datetime.strptime(ts, "%Y-%m-%d-%H")
    new = dt - timedelta(hours=hours)
    return new.strftime("%Y-%m-%d-%H")


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
        timestamp: Episode timestamp ``YYYY-MM-DD-HH`` — the time at which
            we are *predicting* (so we look at *earlier* timestamps).
        target_station_indices: ``(N,)`` indices into the dataset's filtered
            station list, identifying which stations to predict for.
        ghcnh_snapshot_dir: Directory containing ``YYYY-MM-DD-HH.npz`` files,
            one per timestamp. Single shared dir across regions.
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

    # Look up which row in the GHCNh file each of our target stations lives at.
    ghcnh_rows = ghcnh_index_for_station[target_station_indices]

    n_steps = max_lookback_hours // SNAPSHOT_HOUR_STEP
    for step in range(1, n_steps + 1):
        if not still_missing.any():
            break
        prev_ts = step_timestamp_back(timestamp, step * SNAPSHOT_HOUR_STEP)
        prev_path = ghcnh_snapshot_dir / f"{prev_ts}.npz"
        if not prev_path.exists():
            continue
        with np.load(prev_path) as f:
            if target_variable not in f:
                continue
            values_at_prev = f[target_variable]  # one entry per GHCNh-file station
        # Look up our stations in this file.
        candidate_values = values_at_prev[ghcnh_rows]
        # Where we still need a value AND the candidate is not NaN, fill.
        fill_mask = still_missing & np.isfinite(candidate_values)
        preds[fill_mask] = candidate_values[fill_mask]
        still_missing &= ~fill_mask

    return preds


# ---------------------------------------------------------------------------
# Shared dataset / layout plumbing
# ---------------------------------------------------------------------------

def detect_layout(dataset_dir: Path) -> tuple[str, bool]:
    """Return ``(layout_version, is_multi_region)`` for a snapshot dataset."""
    md_path = dataset_dir / "metadata.json"
    if not md_path.exists():
        raise FileNotFoundError(f"{md_path} not found")
    with open(md_path) as f:
        md = json.load(f)
    layout = md.get("layout_version", "snapshot_v1")
    return layout, layout == "multi_region_snapshot_v1"


def build_dataset(
    args,
    target_variables: list[str],
    is_multi_region: bool,
    split: str,
    station_split: str,
):
    """Construct a snapshot dataset with this run's station filters applied.

    Single source of truth for the filter set, so the lapse-rate fit on the
    train split and the evaluation on the test split cannot silently disagree
    about which stations exist.
    """
    common_kwargs = dict(
        target_variables=target_variables,
        split=split,
        station_split=station_split,
        tessera_path=args.tessera_path,
        tessera_station_csv=args.tessera_station_csv,
        load_tessera_patches=False,
        include_static_fields=False,  # baselines don't need static fields
        vae_latents_path=args.vae_latents_path,
        vae_latents_station_csv=args.vae_latents_station_csv,
        min_patch_coverage=args.min_tessera_patch_coverage,
    )
    if is_multi_region:
        if not args.train_regions:
            raise ValueError(
                "--train-regions is required for the multi-region dataset layout."
            )
        return MultiRegionSnapshotDownscalingDataset(
            dataset_dir=Path(args.dataset_dir),
            regions=list(args.train_regions),
            normalisation_policy=args.normalisation_policy or "per_region",
            **common_kwargs,
        )
    return SnapshotDownscalingDataset(
        dataset_dir=Path(args.dataset_dir), **common_kwargs,
    )


class Era5DirResolver:
    """Map an episode index to the ERA5 snapshot directory that serves it.

    Multi-region datasets nest ``era5_snapshot`` under ``regions/<region>/``;
    single-region datasets keep it at the top level. Wrapping the difference
    here keeps the eval loop and the lapse-rate fit identical.
    """

    def __init__(self, dataset, is_multi_region: bool):
        self._dataset = dataset
        self._is_multi_region = is_multi_region
        if not is_multi_region:
            self._single_dir = dataset._grid_root / "era5_snapshot"

    def for_index(self, idx: int) -> Path:
        if not self._is_multi_region:
            return self._single_dir
        region_name, _local_idx = self._dataset._dispatch(idx)
        return self._dataset.per_region[region_name].region_dir / "era5_snapshot"


# ---------------------------------------------------------------------------
# Main eval loop
# ---------------------------------------------------------------------------

def date_to_season(date_str: str) -> str:
    """Map YYYY-MM-DD or YYYY-MM-DD-HH to meteorological season."""
    month = int(date_str.split("-")[1])
    return {
        12: "DJF", 1: "DJF", 2: "DJF",
        3: "MAM", 4: "MAM", 5: "MAM",
        6: "JJA", 7: "JJA", 8: "JJA",
        9: "SON", 10: "SON", 11: "SON",
    }[month]


def evaluate_baseline(args) -> tuple[dict, dict, dict]:
    """Run one baseline over the test set and return the test_summary dict."""
    target_variables = list(args.target_variables)
    layout, is_multi_region = detect_layout(Path(args.dataset_dir))

    # ------------------------------------------------------------------
    # Lapse rate: resolve Γ before touching the test set. Fitting reads the
    # train split only, so the test year and held-out stations stay clean.
    # ------------------------------------------------------------------
    lapse_rate = None
    lapse_diagnostics = None
    if args.baseline == "era5_interp_lapse":
        unsupported = [
            v for v in target_variables if v not in LAPSE_SUPPORTED_TARGETS
        ]
        if unsupported:
            raise ValueError(
                f"era5_interp_lapse is defined only for "
                f"{list(LAPSE_SUPPORTED_TARGETS)}, got {unsupported}. Wind speed "
                "and precipitation have no lapse-rate analogue; running them "
                "here would duplicate the era5_interp row under a different "
                "label. Use --baseline era5_interp for those targets."
            )
        if args.lapse_rate_mode == "fitted":
            lapse_rate, lapse_diagnostics = fit_lapse_rate_on_train(
                args, target_variables[0], is_multi_region,
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

    # Build the test dataset using the same parameters evaluate.py would.
    # The dataset already enforces the temporal-test + station-test split.
    # Filters (--tessera-path / --vae-latents-path) are passed through so
    # simple baselines land on the same station set as the trained ConvCNPs
    # they are being compared against.
    test_dataset = build_dataset(
        args, target_variables, is_multi_region,
        split="test", station_split=args.station_split,
    )

    logger.info(
        f"Test dataset: {len(test_dataset)} episodes, "
        f"layout={layout}, baseline={args.baseline}"
    )
    era5_dirs = Era5DirResolver(test_dataset, is_multi_region)

    # Counters for the lapse-rate correction, reported in the summary so a
    # silently-uncorrected run is impossible to mistake for a corrected one.
    n_missing_delta_elev = 0
    corrections_seen: list[np.ndarray] = []

    # ------------------------------------------------------------------
    # Iterate the test set, accumulating preds + targets per variable.
    # ------------------------------------------------------------------
    all_preds = {v: [] for v in target_variables}
    all_targets = {v: [] for v in target_variables}
    all_station_indices = {v: [] for v in target_variables}
    season_errors = {v: defaultdict(list) for v in target_variables}

    n_total = len(test_dataset)
    for idx in range(n_total):
        episode = test_dataset[idx]
        if episode.get("n_targets", 0) == 0:
            continue

        timestamp = episode["date"]  # snapshot dataset stores full TS here
        target_coords = episode["target_coords"].numpy()  # (N, 2)
        target_values = episode["target_values"].numpy()  # (N,) or (N, V)
        target_station_indices = episode["target_station_indices"].numpy()
        grid_lats = episode["grid_lats"].numpy()
        grid_lons = episode["grid_lons"].numpy()

        # The dataset returns the normalised context_grid; for the
        # baselines we want raw ERA5. Re-load directly from disk.
        # Multi-region datasets nest era5_snapshot under regions/<region>/;
        # single-region has it at the top level. Both have ghcnh_snapshot
        # at a single shared location.
        era5_snapshot_dir = era5_dirs.for_index(idx)
        if is_multi_region:
            region_name, _local_idx = test_dataset._dispatch(idx)
            region_state = test_dataset.per_region[region_name]
            ghcnh_snapshot_dir = test_dataset._ghcnh_snapshot_dir
            # MR rewrites target_station_indices to global flat indices
            # (region_local + flat_offset). Subtract the offset back to
            # the region-local index before looking up GHCNh rows.
            local_target_indices = (
                target_station_indices - region_state.flat_offset
            )
            ghcnh_index_for_station = region_state.ghcnh_index_for_station
        else:
            ghcnh_snapshot_dir = test_dataset._ghcnh_root / "ghcnh_snapshot"
            local_target_indices = target_station_indices
            ghcnh_index_for_station = test_dataset._ghcnh_index_for_station

        if args.baseline in ("era5_interp", "era5_interp_lapse"):
            era5_raw = np.load(era5_snapshot_dir / f"{timestamp}.npy")

        for vi, var in enumerate(target_variables):
            if target_values.ndim == 1:
                tv = target_values
            else:
                tv = target_values[:, vi]

            if args.baseline == "era5_interp":
                preds = era5_interp_predict(
                    var, era5_raw, grid_lats, grid_lons, target_coords,
                )
            elif args.baseline == "era5_interp_lapse":
                interp = era5_interp_predict(
                    var, era5_raw, grid_lats, grid_lons, target_coords,
                )
                # Δelev comes straight from the episode, already sliced to
                # this snapshot's valid target stations — no index arithmetic,
                # and identical to the e(x⋆) component the ConvCNP receives.
                d_elev = episode["target_delta_elev"].numpy()
                n_missing_delta_elev += int((~np.isfinite(d_elev)).sum())
                preds = apply_lapse_rate_correction(interp, d_elev, lapse_rate)
                corrections_seen.append(
                    (interp - preds)[np.isfinite(interp)].astype(np.float32)
                )
            elif args.baseline == "persistence":
                preds = persistence_predict(
                    var, timestamp, local_target_indices,
                    ghcnh_snapshot_dir, ghcnh_index_for_station,
                    args.persistence_max_lookback_hours,
                )
            else:
                raise ValueError(f"Unknown baseline: {args.baseline}")

            # Drop pairs where either pred or target is NaN. Persistence
            # produces NaN for stations with no recent obs; t2m/wind
            # targets can be NaN if the station happens to have no valid
            # obs at this exact timestamp (though the dataset normally
            # filters those out via valid_indices).
            valid = np.isfinite(preds) & np.isfinite(tv)
            preds_v = preds[valid].astype(np.float32)
            targets_v = tv[valid].astype(np.float32)
            station_indices_v = target_station_indices[valid]

            all_preds[var].extend(preds_v.tolist())
            all_targets[var].extend(targets_v.tolist())
            all_station_indices[var].extend(station_indices_v.tolist())

            errors = np.abs(preds_v - targets_v)
            season = date_to_season(timestamp)
            season_errors[var][season].extend(errors.tolist())

        if (idx + 1) % 200 == 0:
            running = np.array(all_preds[target_variables[0]])
            running_t = np.array(all_targets[target_variables[0]])
            mae = np.abs(running - running_t).mean() if len(running) else float("nan")
            logger.info(
                f"  {idx + 1}/{n_total} episodes, "
                f"{target_variables[0]} MAE so far: {mae:.3f}"
            )

    # ------------------------------------------------------------------
    # Compute metrics — same shape as evaluate.py.
    # ------------------------------------------------------------------
    test_summary = {
        "checkpoint_epoch": 0,           # baselines don't train
        "best_val_loss": None,           # ditto
        "target_variables": target_variables,
        "baseline_kind": args.baseline,
    }
    npz_data = {}

    # Provenance for the lapse-rate row: Γ, how it was obtained, how large the
    # corrections actually were, and how many stations lacked a usable Δelev.
    # Written into the summary so the table row is self-documenting.
    if lapse_diagnostics is not None:
        if corrections_seen:
            corr = np.concatenate(corrections_seen)
            lapse_diagnostics.update({
                "correction_mean_k": float(corr.mean()),
                "correction_abs_mean_k": float(np.abs(corr).mean()),
                "correction_std_k": float(corr.std()),
                "correction_p01_k": float(np.percentile(corr, 1)),
                "correction_p99_k": float(np.percentile(corr, 99)),
            })
        lapse_diagnostics["n_targets_missing_delta_elev"] = n_missing_delta_elev
        test_summary["lapse_rate"] = lapse_diagnostics
        logger.info("Lapse-rate diagnostics: " + json.dumps(
            {k: (round(v, 5) if isinstance(v, float) else v)
             for k, v in lapse_diagnostics.items()},
        ))
        if n_missing_delta_elev:
            logger.warning(
                f"{n_missing_delta_elev} target rows had a non-finite Δelev and "
                "were left uncorrected (kept, to preserve the station set)."
            )

    # Station metadata, shared by every variable's per-station arrays. Same
    # keys and the same station-index basis as evaluate.py, so the two kinds
    # of run can be joined station-by-station downstream.
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
        bias = float(np.mean(preds - targets))
        correlation = (
            float(np.corrcoef(preds, targets)[0, 1]) if len(preds) > 1 else 0.0
        )

        # Calibration via constant-σ assumption: σ = std(pred − target).
        residuals = preds - targets
        residual_std = float(residuals.std()) if len(residuals) > 1 else float("nan")
        # Treat residual_std as the predicted σ at every station.
        if residual_std > 0:
            within_1s = float((errors < residual_std).mean() * 100)
            within_2s = float((errors < 2 * residual_std).mean() * 100)
            # NLL under N(pred, residual_std²).
            log_var = 2 * np.log(residual_std)
            nll = float(0.5 * (
                log_var
                + (residuals ** 2) / (residual_std ** 2)
                + np.log(2 * np.pi)
            ).mean())
        else:
            within_1s = within_2s = nll = float("nan")

        logger.info("=" * 60)
        logger.info(f"BASELINE RESULTS — {args.baseline} — {var}")
        logger.info("=" * 60)
        logger.info(f"  Predictions:   {len(errors):,}")
        logger.info(f"  MAE:           {errors.mean():.3f}")
        logger.info(f"  RMSE:          {np.sqrt((errors ** 2).mean()):.3f}")
        logger.info(f"  Bias:          {bias:+.4f}")
        logger.info(f"  Correlation:   {correlation:.4f}")
        logger.info(f"  Residual σ:    {residual_std:.3f}  (used as constant pred σ)")
        logger.info(f"  Within 1σ:     {within_1s:.1f}%")
        logger.info(f"  Within 2σ:     {within_2s:.1f}%")

        test_summary[f"{var}_n_predictions"] = len(errors)
        # n_test_stations = number of distinct stations contributing at
        # least one valid (pred, target) pair. Useful as the per-row
        # effective-N for any cross-region or cross-experiment comparison.
        test_summary[f"{var}_n_test_stations"] = int(
            len(set(all_station_indices[var]))
        )
        test_summary[f"{var}_mae"] = float(errors.mean())
        test_summary[f"{var}_rmse"] = float(np.sqrt((errors ** 2).mean()))
        test_summary[f"{var}_nll"] = nll
        test_summary[f"{var}_bias"] = bias
        test_summary[f"{var}_correlation"] = correlation
        test_summary[f"{var}_mean_pred_std"] = residual_std
        test_summary[f"{var}_within_1sigma"] = within_1s
        test_summary[f"{var}_within_2sigma"] = within_2s
        test_summary[f"{var}_p50"] = float(np.percentile(errors, 50))
        test_summary[f"{var}_p90"] = float(np.percentile(errors, 90))
        test_summary[f"{var}_p95"] = float(np.percentile(errors, 95))
        test_summary[f"{var}_p99"] = float(np.percentile(errors, 99))

        # Seasonal MAE.
        seasonal_mae = {}
        for season in ("DJF", "MAM", "JJA", "SON"):
            if season in season_errors[var]:
                se = np.array(season_errors[var][season])
                seasonal_mae[season] = float(se.mean())
        test_summary[f"{var}_seasonal_mae"] = seasonal_mae

        # Save raw arrays. station_indices is what makes the raw arrays
        # re-aggregatable; without it the predictions are an anonymous pile.
        sindices = np.array(all_station_indices[var], dtype=np.int64)
        npz_data[f"{var}_predictions"] = preds.astype(np.float32)
        npz_data[f"{var}_targets"] = targets.astype(np.float32)
        npz_data[f"{var}_predicted_stds"] = np.full_like(preds, residual_std, dtype=np.float32)
        npz_data[f"{var}_station_indices"] = sindices

        # Per-station aggregation, indexed over the FULL station list (not
        # just the ones that contributed), so every array in the npz shares
        # one index basis. Stations with no valid pair keep count == 0 and a
        # zero metric — the same convention evaluate.py uses, and the reason
        # consumers must mask on count > 0 rather than on a sentinel.
        station_mae = np.zeros(n_all_stations)
        station_rmse = np.zeros(n_all_stations)
        station_bias = np.zeros(n_all_stations)
        station_count = np.zeros(n_all_stations, dtype=np.int64)
        residuals_all = preds - targets
        for si in np.unique(sindices):
            m = sindices == si
            station_mae[si] = errors[m].mean()
            station_rmse[si] = np.sqrt((errors[m] ** 2).mean())
            station_bias[si] = residuals_all[m].mean()
            station_count[si] = int(m.sum())
        station_npz[f"{var}_station_mae"] = station_mae
        station_npz[f"{var}_station_rmse"] = station_rmse
        station_npz[f"{var}_station_bias"] = station_bias
        station_npz[f"{var}_station_count"] = station_count

    return test_summary, npz_data, station_npz


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate simple non-ML baselines (ERA5 interpolation, persistence).",
    )
    parser.add_argument(
        "--baseline",
        choices=["era5_interp", "era5_interp_lapse", "persistence"],
        required=True,
        help="Which baseline to compute. era5_interp_lapse is t2m-only.",
    )
    parser.add_argument(
        "--lapse-rate-mode", choices=["fixed", "fitted"], default="fixed",
        help="era5_interp_lapse only. 'fixed' uses --lapse-rate (default "
             "6.5 K/km, the standard environmental lapse rate). 'fitted' "
             "estimates Γ by least squares on the TRAIN split (train stations, "
             "train years) and falls back to fixed if the fit is degenerate.",
    )
    parser.add_argument(
        "--lapse-rate", type=float, default=DEFAULT_LAPSE_RATE_K_PER_M,
        help="Γ in K per METRE (not K/km). Default 0.0065 = 6.5 K/km. Used "
             "directly when --lapse-rate-mode=fixed, and as the fallback "
             "when a fitted Γ is rejected.",
    )
    parser.add_argument(
        "--lapse-fit-intercept", action="store_true",
        help="Fit a free intercept alongside Γ. Off by default: an intercept "
             "absorbs the domain-mean interpolation bias, making the row "
             "'lapse rate + mean bias correction' rather than a pure "
             "lapse-rate reference. Enable only to report that variant "
             "explicitly.",
    )
    parser.add_argument(
        "--lapse-fit-stride", type=int, default=1,
        help="Use every Nth training episode when fitting Γ. The slope is "
             "estimated from ~10^6 station-snapshot pairs at stride 1, so a "
             "stride of 4-8 changes the estimate negligibly and cuts fit time "
             "proportionally. Recorded in the summary.",
    )
    parser.add_argument(
        "--dataset-dir", type=Path, required=True,
        help="Path to the snapshot dataset directory.",
    )
    parser.add_argument(
        "--target-variables", nargs="+", required=True,
        choices=["t2m", "wind", "precip"],
        help="Which target variables to evaluate. Multi-task entries pass multiple.",
    )
    parser.add_argument(
        "--train-regions", nargs="+", default=None,
        help="Required for multi-region datasets — passed through so we evaluate "
             "on the correct region's test set.",
    )
    parser.add_argument(
        "--normalisation-policy", choices=["per_region", "global"], default=None,
        help="Required for multi-region datasets.",
    )
    parser.add_argument(
        "--output-dir", type=Path, required=True,
        help="Where to write test_summary.json, test_results.json, "
             "test_predictions.npz, config.json.",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Recorded in config.json so runs sort under {name}_seed{seed}/, "
             "but baselines are deterministic so this has no effect on output.",
    )
    parser.add_argument(
        "--persistence-max-lookback-hours", type=int,
        default=DEFAULT_PERSISTENCE_MAX_LOOKBACK_HOURS,
        help="Max hours to look back for a valid prior observation before "
             "giving up. Stations with no obs in this window get NaN preds "
             "and are excluded from metrics.",
    )
    parser.add_argument(
        "--station-split", choices=["train", "test", "all"], default="test",
        help="Which spatial split to evaluate on. Default 'test' "
             "preserves legacy behaviour (held-out stations only). Pass "
             "'all' for the data-efficiency experiments where we need "
             "the baseline numbers over train + probe + test stations "
             "so per-subset breakdowns can be computed downstream.",
    )
    # --- Station filter flags (passed through to the dataset class) ---
    # Until these existed, simple baselines were evaluated on every
    # spatial-test station, while trained ConvCNPs were filtered to
    # those with valid TESSERA patches (and, for VAE-augmented runs,
    # non-NaN latents). That left the headline-table comparison on
    # different station sets per row. Pass the same paths the matched
    # trained runs used to put every row on identical stations.
    parser.add_argument(
        "--tessera-path", type=Path, default=None,
        help="Path to TESSERA patches .npy. When set, stations are "
             "filtered to those with non-zero centre pixels — same filter "
             "trained ConvCNP baselines apply via their saved config. "
             "Pass this whenever the comparison includes any TESSERA-"
             "filtered ConvCNP row.",
    )
    parser.add_argument(
        "--tessera-station-csv", type=Path, default=None,
        help="CSV row-aligned with --tessera-path. Required when "
             "--tessera-path is set.",
    )
    parser.add_argument(
        "--vae-latents-path", type=Path, default=None,
        help="Path to a VAE latents .npy. When set, stations are also "
             "filtered to those with non-NaN latents — the same filter "
             "trained VAE-augmented runs apply. Use only when the "
             "simple baseline is being compared against VAE rows.",
    )
    parser.add_argument(
        "--vae-latents-station-csv", type=Path, default=None,
        help="CSV row-aligned with --vae-latents-path.",
    )
    parser.add_argument(
        "--min-tessera-patch-coverage", type=float, default=0.5,
        help="Minimum fraction of pixels in the 64x64 patch that must "
             "have any non-zero channel for a station to be kept. Combined "
             "with the centre-pixel-non-zero rule. 0.0 disables the "
             "coverage check (legacy behaviour). Default 0.5. "
             "Pass the same value used at training time for the runs "
             "this baseline is being compared against.",
    )
    args = parser.parse_args()

    # Cross-flag validation.
    if args.baseline != "era5_interp_lapse":
        for flag, default in (
            ("lapse_rate_mode", "fixed"),
            ("lapse_fit_intercept", False),
            ("lapse_fit_stride", 1),
        ):
            if getattr(args, flag) != default:
                parser.error(
                    f"--{flag.replace('_', '-')} only applies to "
                    f"--baseline era5_interp_lapse (got {args.baseline})."
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

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Save config.json so the analysis loader has something to read.
    with open(args.output_dir / "config.json", "w") as f:
        json.dump({k: str(v) if isinstance(v, Path) else v
                   for k, v in vars(args).items()}, f, indent=2)

    test_summary, npz_data, station_npz = evaluate_baseline(args)

    with open(args.output_dir / "test_results.json", "w") as f:
        json.dump(test_summary, f, indent=2)
    with open(args.output_dir / "test_summary.json", "w") as f:
        json.dump(test_summary, f, indent=2)
    np.savez(args.output_dir / "test_predictions.npz", **npz_data)
    np.savez(args.output_dir / "test_station_errors.npz", **station_npz)

    logger.info(f"Wrote results to {args.output_dir}")


if __name__ == "__main__":
    main()