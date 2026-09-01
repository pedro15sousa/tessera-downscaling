"""Evaluate a trained ConvCNP downscaler (console script ``tessera-evaluate``).

Rebuilds the model from the ``config`` dict embedded in a ``tessera-train``
checkpoint, loads the weights with ``strict=True``, runs it over the test
split (held-out years) of a ``multi_region_snapshot_v1`` dataset and writes
the metrics the paper's tables and figures are built from::

    tessera-evaluate --checkpoint <run>/best_model.pt
        [--region-specs-test-file specs.json | --station-split test|train|all]
        [--dataset-dir <dataset> --lead-hours <h> --output-dir <run>/eval_lead<h>h]
        [--filter-vae-latents-path ... --filter-vae-latents-station-csv ...]

Everything the model needs -- dataset, regions, station filters, per-station
vector, head types, topographic features -- comes from the checkpoint config;
stored paths from the machine a run was trained on are remapped onto the
current data root (:mod:`tessera_downscaling.paths`) and experiment sidecars
fall back to the copies committed under ``scripts/experiments/``
(:func:`resolve_sidecar_path`). Older configs may lack keys that were
added later or carry keys of since-removed options; both are read tolerantly,
so every checkpoint family on the data root evaluates unchanged.

Per target variable the script reports NLL, CRPS, the PIT χ² calibration
test, point-estimate errors (MAE / RMSE / bias / correlation -- on the mean
for Gaussian heads, MAE on the median and the rest on the mean for the
truncated normal), σ coverage, error percentiles and a seasonal MAE
breakdown, then aggregates per station and -- for the station-rollout
experiment -- per station subset (``probe`` / ``always_on`` /
``spatial_test``).

Written to ``--output-dir`` (default: the checkpoint's directory):

* ``test_summary.json`` (= ``test_results.json``) -- aggregate metrics;
* ``test_predictions.npz`` -- per-observation head parameters, targets and
  station indices;
* ``test_station_errors.npz`` -- per-station MAE / RMSE / bias / count with
  station metadata (and ``subset_per_station`` when resolved).
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import chisquare
from torch.utils.data import DataLoader
from tqdm import tqdm

from tessera_downscaling.data.dataset import (
    MultiRegionSnapshotDownscalingDataset,
    downscaling_collate,
)
from tessera_downscaling.model.convcnp import ConvCNPDownscaler
from tessera_downscaling.model.heads import (
    GaussianHead,
    LikelihoodHead,
    TruncatedNormalHead,
)
from tessera_downscaling.paths import resolve

logger = logging.getLogger("evaluate")

SEASONS = ("DJF", "MAM", "JJA", "SON")
SUBSET_NAMES = ("probe", "always_on", "train_stations", "spatial_test")


def date_to_season(date_str: str) -> str:
    """Meteorological season of a ``YYYY-MM-DD[-HH]`` string."""
    month = int(date_str.split("-")[1])
    if month in (12, 1, 2):
        return "DJF"
    if month in (3, 4, 5):
        return "MAM"
    if month in (6, 7, 8):
        return "JJA"
    return "SON"


# ---------------------------------------------------------------------------
# Checkpoint config -> model
# ---------------------------------------------------------------------------


def config_path(config: dict, key: str) -> Path | None:
    """Return a path stored in a training config, resolved onto the data root."""
    value = config.get(key)
    if value is None or str(value) == "None":
        return None
    return resolve(str(value))


def _candidate_repo_roots() -> list[Path]:
    """Directories that look like a checkout of this repository.

    Tried in order: the checkout this module is imported from (when running
    from source), then the working directory and its parents.
    """
    candidates = [Path(__file__).resolve().parents[2]]
    p = Path.cwd().resolve()
    while True:
        candidates.append(p)
        if p == p.parent:
            break
        p = p.parent
    return [c for c in candidates if (c / "scripts" / "experiments").is_dir()]


def resolve_sidecar_path(config: dict, key: str) -> Path | None:
    """Resolve an experiment-sidecar JSON recorded in a training config.

    The stored path is used as-is when it exists -- a fresh run's config
    points at the checkout it was launched from. When it does not (the run
    was trained from another checkout or machine), fall back to the same
    ``scripts/experiments/<folder>/<file>`` inside the current repository,
    where every sidecar is committed.
    """
    stored = config_path(config, key)
    if stored is None or stored.exists():
        return stored
    parts = stored.parts
    for i in range(len(parts) - 1, 0, -1):
        if parts[i - 1 : i + 1] == ("scripts", "experiments"):
            rel = Path(*parts[i - 1 :])
            for root in _candidate_repo_roots():
                candidate = root / rel
                if candidate.exists():
                    logger.info(
                        f"{key}: stored path {stored} not found; using the "
                        f"repo copy {candidate}"
                    )
                    return candidate
            break
    return stored  # missing everywhere; the caller reports it


def precomputed_vector_files(config: dict) -> tuple[Path | None, Path | None]:
    """``(vector .npy, station CSV)`` the run was conditioned on, or ``(None, None)``.

    VAE latents and extra descriptors ride the same precomputed-vector pathway;
    a run that used both trained on their hstack (``precomputed_merged_path``,
    a sibling of the latents file).
    """
    latents = config_path(config, "vae_latents_path")
    descriptors = config_path(config, "extra_descriptors_path")
    if latents is not None and descriptors is not None:
        merged = config_path(config, "precomputed_merged_path")
        if merged is None:
            merged = latents.with_name(f"{latents.stem}_plus_{descriptors.stem}.npy")
        return merged, config_path(config, "vae_latents_station_csv")
    if latents is not None:
        return latents, config_path(config, "vae_latents_station_csv")
    if descriptors is not None:
        return descriptors, config_path(config, "extra_descriptors_station_csv")
    return None, None


def build_model_from_config(config: dict, n_context_channels: int) -> ConvCNPDownscaler:
    """Rebuild the :class:`ConvCNPDownscaler` described by a training config.

    ``config`` is ``ckpt["config"]`` (the ``vars(args)`` of the training run).
    Keys that older configs lack take the value that was the default when
    those runs were trained -- notably ``interpolation`` defaults to
    ``"setconv"`` here, the pre-bilinear default, NOT the constructor's. Keys
    of removed options (``tessera_method``, ``vae_latents_proj_dim``,
    ``decoder_kernel``, ...) are ignored. The per-station vector width is read
    from the vector file itself.
    """
    target_variables = list(config["target_variables"])
    likelihood = config.get("likelihood_per_variable") or dict.fromkeys(
        target_variables, "gaussian"
    )
    vectors_path, _ = precomputed_vector_files(config)
    precomputed_dim = 0
    if vectors_path is not None:
        precomputed_dim = int(np.load(vectors_path, mmap_mode="r").shape[1])
    return ConvCNPDownscaler(
        n_context_channels=n_context_channels,
        cnn_hidden=int(config.get("cnn_hidden", 128)),
        cnn_layers=int(config.get("cnn_layers", 7)),
        cnn_kernel=int(config.get("cnn_kernel", 3)),
        setconv_length_scale=float(config.get("setconv_length_scale", 0.5)),
        interpolation=config.get("interpolation", "setconv"),
        mlp_hidden=int(config.get("mlp_hidden", 128)),
        mlp_n_hidden=int(config.get("mlp_n_hidden", 3)),
        n_elev_features=int(config.get("n_elev_features", 2)),
        include_elevation=bool(config.get("include_elevation", True)),
        target_variables=target_variables,
        likelihood_per_variable=dict(likelihood),
        tessera_injection=config.get("tessera_injection", "concat"),
        tessera_features_precomputed=vectors_path is not None,
        precomputed_tessera_dim=precomputed_dim,
    )


def load_state_dict_compat(model: ConvCNPDownscaler, state_dict: dict) -> None:
    """``load_state_dict(strict=True)`` after renaming legacy ``setconv.`` keys."""
    migrated = {
        (f"interp.{k[len('setconv.') :]}" if k.startswith("setconv.") else k): v
        for k, v in state_dict.items()
    }
    model.load_state_dict(migrated, strict=True)


# ---------------------------------------------------------------------------
# Test dataset
# ---------------------------------------------------------------------------


def load_region_specs(path: Path) -> dict[str, str]:
    """Load a ``{region: "train"|"test"|"all"}`` JSON file."""
    if not path.exists():
        raise FileNotFoundError(f"--region-specs-test-file {path} does not exist.")
    specs = json.loads(path.read_text())
    if not isinstance(specs, dict) or not all(
        isinstance(k, str) and v in ("train", "test", "all") for k, v in specs.items()
    ):
        raise ValueError(
            "--region-specs-test-file: expected a JSON object mapping region name "
            f"to 'train' | 'test' | 'all', got {specs!r}."
        )
    return specs


def build_test_dataset(
    config: dict,
    dataset_dir: Path,
    region_specs_test: dict[str, str] | None,
    station_split: str,
    lead_hours: int | None,
) -> MultiRegionSnapshotDownscalingDataset:
    """Test-split dataset with the station filters the run was trained with."""
    vectors_path, vectors_csv = precomputed_vector_files(config)
    kwargs = {
        "dataset_dir": dataset_dir,
        "split": "test",
        "target_variables": list(config["target_variables"]),
        "tessera_path": config_path(config, "tessera_path"),
        "tessera_station_csv": config_path(config, "tessera_station_csv"),
        "include_static_fields": bool(config.get("include_static_fields", True)),
        "vae_latents_path": vectors_path,
        "vae_latents_station_csv": vectors_csv,
        # Older runs predate the coverage rule; 0.0 reproduces their
        # centre-pixel-only filter.
        "min_patch_coverage": float(config.get("min_tessera_patch_coverage", 0.0)),
        # Lenient: a channel the model dropped at train time may already be
        # absent from this dataset (Aurora has no precipitation).
        "drop_context_channels": config.get("drop_context_channels"),
        "drop_context_strict": False,
        "lead_hours": lead_hours,
    }
    if region_specs_test is not None:
        logger.info(f"Test dataset at {dataset_dir}; region_specs={region_specs_test}")
        return MultiRegionSnapshotDownscalingDataset(
            region_specs=region_specs_test, **kwargs
        )
    regions = config.get("train_regions")
    logger.info(
        f"Test dataset at {dataset_dir}; regions={regions or 'all'}, "
        f"station_split={station_split!r}"
    )
    return MultiRegionSnapshotDownscalingDataset(
        regions=regions, station_split=station_split, **kwargs
    )


# ---------------------------------------------------------------------------
# Station subsets and filters
# ---------------------------------------------------------------------------


def resolve_subset_labels(
    config: dict,
    region_specs_test: dict[str, str] | None,
    dataset_dir: Path,
    station_ids: np.ndarray,
) -> np.ndarray | None:
    """Per-station subset label for the data-efficiency breakdown, or None.

    Resolved when the run hid probe stations (``probe_active_from_file`` in
    the config -- the station-rollout experiment) or when the test spec
    evaluates a region with ``"all"`` stations. Labels: ``probe`` (listed in
    the probe file), ``always_on`` (other spatial-train stations of a probe
    run), ``train_stations`` (spatial-train stations otherwise),
    ``spatial_test`` (held-out stations) and ``unmapped``.
    """
    probe_file = resolve_sidecar_path(config, "probe_active_from_file")
    spec_all = region_specs_test is not None and "all" in region_specs_test.values()
    if probe_file is None and not spec_all:
        return None

    stations_csv = dataset_dir / "stations.csv"
    if not stations_csv.exists():
        logger.warning(f"{stations_csv} not found; skipping the per-subset breakdown.")
        return None
    stations = pd.read_csv(stations_csv)
    split_of = dict(
        zip(stations["station_id"].astype(str), stations["spatial_split"], strict=True)
    )

    probe_ids: set[str] = set()
    if probe_file is not None:
        if probe_file.exists():
            probe_ids = {str(k) for k in json.loads(probe_file.read_text())}
        else:
            logger.warning(
                f"probe_active_from_file {probe_file} not found; spatial-train "
                "stations are labelled train_stations (no probe / always_on split)."
            )

    labels = np.empty(len(station_ids), dtype=object)
    for i, sid in enumerate(str(s) for s in station_ids):
        split = split_of.get(sid)
        if sid in probe_ids:
            labels[i] = "probe"
        elif split == "test":
            labels[i] = "spatial_test"
        elif split == "train":
            labels[i] = "always_on" if probe_ids else "train_stations"
        else:
            labels[i] = "unmapped"
    labels = labels.astype(str)
    counts = Counter(labels.tolist())
    summary = ", ".join(
        f"{name}={counts.get(name, 0)}" for name in (*SUBSET_NAMES, "unmapped")
    )
    logger.info(f"Per-station subset labels: {summary}")
    return labels


def filter_only_station_mask(
    latents_path: Path, station_csv: Path, station_ids: np.ndarray
) -> np.ndarray:
    """Boolean mask over ``station_ids``: True where the station has a non-NaN latent.

    Used to re-evaluate a no-TESSERA checkpoint on the (TESSERA ∩ latent-valid)
    station set of the TESSERA arm without retraining; the model never sees
    the latents.
    """
    csv = pd.read_csv(station_csv)
    latents = np.load(latents_path, mmap_mode="r")
    if latents.shape[0] != len(csv):
        raise ValueError(
            f"--filter-vae-latents-path shape {latents.shape} does not match "
            f"--filter-vae-latents-station-csv ({len(csv)} rows)."
        )
    valid_ids = set(csv["station_id"].astype(str)[~np.isnan(latents).any(axis=1)])
    mask = np.array([str(sid) in valid_ids for sid in station_ids], dtype=bool)
    logger.info(
        f"Filter-only VAE-latent mask: {int(mask.sum())}/{len(mask)} test stations "
        "have a non-NaN latent; predictions at the others are dropped."
    )
    return mask


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------


def run_inference(
    model: ConvCNPDownscaler,
    loader: DataLoader,
    dates: list[str],
    device: torch.device,
    station_mask: np.ndarray | None,
) -> tuple[dict, dict, dict, dict]:
    """Collect per-observation head parameters, targets and station indices.

    Returns ``(params, targets, station_indices, season_errors)``, each keyed
    by variable: ``params[var][param_name]`` / ``targets[var]`` /
    ``station_indices[var]`` are flat lists over observations, and
    ``season_errors[var][season]`` the absolute errors of the predictive mean.
    ``station_mask`` (over the dataset's stations) drops observations at
    stations outside it.
    """
    target_variables = model.target_variables
    params = {
        var: {p: [] for p in model.heads.heads[var].param_names}
        for var in target_variables
    }
    targets = {var: [] for var in target_variables}
    station_indices = {var: [] for var in target_variables}
    season_errors = {var: defaultdict(list) for var in target_variables}
    batch_size = loader.batch_size

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(loader, desc="Evaluating")):
            if batch is None:
                continue
            target_mtpi = batch.get("target_mtpi")
            target_tessera = batch.get("target_tessera")
            params_per_var = model(
                batch["context_grid"].to(device),
                batch["grid_lats"].to(device),
                batch["grid_lons"].to(device),
                batch["target_coords"].to(device),
                batch["target_elev"].to(device),
                batch["target_delta_elev"].to(device),
                batch["target_mask"].to(device),
                None if target_tessera is None else target_tessera.to(device),
                target_mtpi=None if target_mtpi is None else target_mtpi.to(device),
            )
            target_values = batch["target_values"]
            target_mask = batch["target_mask"]
            batch_station_indices = batch["target_station_indices"]

            for vi, var in enumerate(target_variables):
                head = model.heads.heads[var]
                target_var = (
                    target_values[:, :, vi]
                    if target_values.ndim == 3
                    else target_values
                )
                point_est = head.mean(params_per_var[var]).cpu()
                params_cpu = {p: t.cpu() for p, t in params_per_var[var].items()}
                for b in range(target_var.shape[0]):
                    mask = target_mask[b].bool()
                    if not mask.any():
                        continue
                    # Observations at stations outside the filter mask are dropped.
                    sindices = batch_station_indices[b][mask].numpy()
                    keep = np.ones(len(sindices), dtype=bool)
                    if station_mask is not None:
                        keep = station_mask[sindices]
                        if not keep.any():
                            continue
                    sindices = sindices[keep]
                    targets_b = target_var[b][mask].numpy()[keep]
                    errors_b = np.abs(point_est[b][mask].numpy()[keep] - targets_b)
                    for p_name in head.param_names:
                        params[var][p_name].extend(
                            params_cpu[p_name][b][mask].numpy()[keep].tolist()
                        )
                    targets[var].extend(targets_b.tolist())
                    station_indices[var].extend(sindices.tolist())
                    date_idx = batch_idx * batch_size + b
                    if date_idx < len(dates):
                        season_errors[var][date_to_season(dates[date_idx])].extend(
                            errors_b.tolist()
                        )

            if (batch_idx + 1) % 50 == 0:
                first = target_variables[0]
                if targets[first]:
                    head = model.heads.heads[first]
                    params_t = {p: torch.tensor(v) for p, v in params[first].items()}
                    running_mae = (
                        (head.mean(params_t) - torch.tensor(targets[first]))
                        .abs()
                        .mean()
                        .item()
                    )
                    logger.info(
                        f"  {batch_idx + 1} batches, {first} MAE: {running_mae:.3f}"
                    )

    return params, targets, station_indices, season_errors


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def _pit_chi2(cdf_values: np.ndarray, n_bins: int = 10) -> tuple[float, float]:
    """χ² goodness-of-fit of the PIT values against Uniform(0, 1).

    Returns ``(chi2_stat, p_value)``; NaN when fewer than ``5 * n_bins``
    finite PIT values are available.
    """
    cdf_clean = cdf_values[np.isfinite(cdf_values)]
    if len(cdf_clean) < n_bins * 5:
        return float("nan"), float("nan")
    observed, _ = np.histogram(cdf_clean, bins=n_bins, range=(0.0, 1.0))
    expected = np.full(n_bins, len(cdf_clean) / n_bins)
    chi2_stat, p_value = chisquare(observed, expected)
    return float(chi2_stat), float(p_value)


def summarise_variable(
    var: str,
    head: LikelihoodHead,
    params: dict[str, list[float]],
    targets_list: list[float],
    station_indices: list[int],
    season_errors: dict[str, list[float]],
    test_summary: dict,
    npz_data: dict,
) -> None:
    """Aggregate metrics for one variable into ``test_summary`` / ``npz_data``."""
    params_t = {p: torch.tensor(v) for p, v in params.items()}
    targets_t = torch.tensor(targets_list)
    targets = targets_t.numpy()
    n_predictions = len(targets)
    n_test_stations = len(set(station_indices))

    with torch.no_grad():
        nll = head.nll(params_t, targets_t).item()
        crps = head.crps(params_t, targets_t).numpy()
        pit_cdf = head.pit_cdf(params_t, targets_t).numpy()
        means = head.mean(params_t).numpy()
        medians = head.median(params_t).numpy()
    pit_chi2_stat, pit_chi2_p = _pit_chi2(pit_cdf)

    logger.info("=" * 60)
    logger.info(f"TEST RESULTS — {var} ({type(head).__name__})")
    logger.info("=" * 60)
    logger.info(f"  Predictions:  {n_predictions:,}")
    logger.info(f"  Stations:     {n_test_stations}")
    logger.info(f"  NLL:          {nll:.4f}")
    logger.info(f"  CRPS:         {crps.mean():.4f}")
    logger.info(f"  PIT χ²:       stat={pit_chi2_stat:.3f}, p-value={pit_chi2_p:.4f}")

    test_summary[f"{var}_n_predictions"] = n_predictions
    test_summary[f"{var}_n_test_stations"] = n_test_stations
    test_summary[f"{var}_nll"] = float(nll)
    test_summary[f"{var}_crps"] = float(crps.mean())
    test_summary[f"{var}_pit_chi2_stat"] = pit_chi2_stat
    test_summary[f"{var}_pit_chi2_pvalue"] = pit_chi2_p

    # Both heads are (μ, log σ²) parameterised; σ is the underlying scale.
    sigma = np.exp(0.5 * np.clip(params_t["log_var"].numpy(), -10.0, 10.0))
    errors_at_mean = np.abs(means - targets)
    bias = float(np.mean(means - targets))
    correlation = float(np.corrcoef(means, targets)[0, 1]) if n_predictions > 1 else 0.0
    rmse = float(np.sqrt((errors_at_mean**2).mean()))

    if isinstance(head, GaussianHead):
        # Mean == median: MAE, RMSE, bias and correlation all on μ.
        errors = errors_at_mean
        test_summary[f"{var}_mae"] = float(errors.mean())
        test_summary[f"{var}_rmse"] = rmse
        test_summary[f"{var}_bias"] = bias
        test_summary[f"{var}_correlation"] = correlation
        logger.info(f"  MAE:          {errors.mean():.3f}")
        logger.info(f"  RMSE:         {rmse:.3f}")
        logger.info(f"  Bias:         {bias:+.4f}")
        logger.info(f"  Correlation:  {correlation:.4f}")
    elif isinstance(head, TruncatedNormalHead):
        # Right-skewed near calm: MAE on the median (minimises expected |error|),
        # RMSE / bias / correlation on the mean (minimises expected squared
        # error). The σ-coverage and percentile diagnostics below use the
        # median-centred errors.
        errors = np.abs(medians - targets)
        test_summary[f"{var}_mae_at_median"] = float(errors.mean())
        test_summary[f"{var}_rmse_at_mean"] = rmse
        test_summary[f"{var}_bias_at_mean"] = bias
        test_summary[f"{var}_correlation_at_mean"] = correlation
        logger.info(f"  MAE @ median: {errors.mean():.3f}")
        logger.info(f"  RMSE @ mean:  {rmse:.3f}")
        logger.info(f"  Bias @ mean:  {bias:+.4f}")
        logger.info(f"  Corr @ mean:  {correlation:.4f}")
    else:
        raise NotImplementedError(
            f"Metric dispatch not implemented for {type(head).__name__} ({var!r})."
        )

    within_1s = float((errors < sigma).mean() * 100)
    within_2s = float((errors < 2 * sigma).mean() * 100)
    test_summary[f"{var}_mean_pred_std"] = float(sigma.mean())
    test_summary[f"{var}_within_1sigma"] = within_1s
    test_summary[f"{var}_within_2sigma"] = within_2s
    for q in (50, 90, 95, 99):
        test_summary[f"{var}_p{q}"] = float(np.percentile(errors, q))
    logger.info(f"  Mean pred σ:  {sigma.mean():.3f}")
    logger.info(f"  Within 1σ:    {within_1s:.1f}% (Gaussian expectation ~68.3%)")
    logger.info(f"  Within 2σ:    {within_2s:.1f}% (Gaussian expectation ~95.4%)")

    # Seasonal MAE of the predictive mean (collected during inference).
    logger.info(f"  --- Seasonal MAE ({var}, predictive mean) ---")
    seasonal_mae = {}
    for season in SEASONS:
        if season in season_errors:
            se = np.array(season_errors[season])
            seasonal_mae[season] = float(se.mean())
            logger.info(f"    {season}: {se.mean():.3f} (n={len(se):,})")
    test_summary[f"{var}_seasonal_mae"] = seasonal_mae

    for p_name, values in params.items():
        npz_data[f"{var}_param_{p_name}"] = np.array(values)
    npz_data[f"{var}_targets"] = targets
    npz_data[f"{var}_station_indices"] = np.array(station_indices, dtype=np.int64)


def per_station_metrics(
    var: str,
    head: LikelihoodHead,
    params: dict[str, list[float]],
    targets_list: list[float],
    station_indices: list[int],
    test_dataset: MultiRegionSnapshotDownscalingDataset,
    station_npz: dict,
) -> None:
    """Per-station MAE / RMSE / bias / count of the predictive mean.

    Arrays are indexed over the FULL station list of ``test_dataset`` so every
    array in ``test_station_errors.npz`` shares one basis; stations without a
    prediction keep count 0 (consumers mask on ``count > 0``).
    """
    params_t = {p: torch.tensor(v) for p, v in params.items()}
    with torch.no_grad():
        point_est = head.mean(params_t).numpy()
    targets = np.array(targets_list)
    sindices = np.array(station_indices, dtype=np.int64)
    residuals = point_est - targets
    errors = np.abs(residuals)

    n_all = len(test_dataset.station_ids)
    station_mae = np.zeros(n_all)
    station_rmse = np.zeros(n_all)
    station_bias = np.zeros(n_all)
    station_count = np.zeros(n_all, dtype=np.int64)
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

    valid = station_count > 0
    valid_maes = station_mae[valid]
    if len(valid_maes) == 0:
        return
    logger.info(f"  --- {var} per-station MAE distribution ---")
    logger.info(f"  Stations: {len(valid_maes)}")
    logger.info(f"  Mean station MAE: {valid_maes.mean():.3f}")
    logger.info(f"  Median station MAE: {np.median(valid_maes):.3f}")
    logger.info(f"  Std station MAE: {valid_maes.std():.3f}")
    logger.info(f"  P10 (best):  {np.percentile(valid_maes, 10):.3f}")
    logger.info(f"  P90 (worst): {np.percentile(valid_maes, 90):.3f}")
    logger.info("  Worst 5 stations:")
    for idx in np.where(valid)[0][np.argsort(valid_maes)[-5:][::-1]]:
        logger.info(
            f"    {test_dataset.station_ids[idx]}: MAE={station_mae[idx]:.3f}, "
            f"lat={test_dataset.station_lats[idx]:.2f}, "
            f"lon={test_dataset.station_lons[idx]:.2f}, "
            f"elev={test_dataset.station_elevs[idx]:.0f}m, n={station_count[idx]}"
        )


def per_subset_metrics(
    target_variables: list[str],
    subset_per_station: np.ndarray,
    station_npz: dict,
    test_summary: dict,
) -> None:
    """Per-subset MAE / RMSE / bias (``<var>_<subset>_*`` keys in ``test_summary``).

    ``*_mae`` / ``*_rmse`` / ``*_bias`` weight stations by their observation
    count (every observation counts equally, the basis of the top-level
    metrics); ``*_macro`` give every station equal weight. Empty subsets are
    written with zero counts so consumers can look keys up unconditionally.
    """
    subset_names = list(SUBSET_NAMES)
    if np.any(subset_per_station == "unmapped"):
        subset_names.append("unmapped")
    logger.info("-" * 60)
    logger.info("PER-SUBSET METRICS (data-efficiency breakdown)")
    logger.info("-" * 60)
    for var in target_variables:
        station_mae = station_npz[f"{var}_station_mae"]
        station_rmse = station_npz[f"{var}_station_rmse"]
        station_bias = station_npz[f"{var}_station_bias"]
        station_count = station_npz[f"{var}_station_count"]
        for subset in subset_names:
            in_subset = (subset_per_station == subset) & (station_count > 0)
            n_stations = int(in_subset.sum())
            test_summary[f"{var}_{subset}_n_stations"] = n_stations
            test_summary[f"{var}_{subset}_n_predictions"] = int(
                station_count[in_subset].sum()
            )
            if n_stations == 0:
                continue
            weights = station_count[in_subset].astype(np.float64)
            mae = float(np.average(station_mae[in_subset], weights=weights))
            # Weighted RMSE: count-weighted mean of squared station RMSEs, then
            # sqrt -- the per-station cache's best approximation of the
            # all-observation RMSE within the subset.
            rmse = float(
                np.sqrt(np.average(station_rmse[in_subset] ** 2, weights=weights))
            )
            mae_macro = float(station_mae[in_subset].mean())
            test_summary[f"{var}_{subset}_mae_macro"] = mae_macro
            test_summary[f"{var}_{subset}_rmse_macro"] = float(
                station_rmse[in_subset].mean()
            )
            test_summary[f"{var}_{subset}_bias_macro"] = float(
                station_bias[in_subset].mean()
            )
            test_summary[f"{var}_{subset}_mae"] = mae
            test_summary[f"{var}_{subset}_rmse"] = rmse
            test_summary[f"{var}_{subset}_bias"] = float(
                np.average(station_bias[in_subset], weights=weights)
            )
            logger.info(
                f"  {var:8s} {subset:16s} n_stations={n_stations:4d} "
                f"n_predictions={test_summary[f'{var}_{subset}_n_predictions']:7d} "
                f"MAE={mae:.4f} (macro={mae_macro:.4f}) RMSE={rmse:.4f}"
            )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate a tessera-train checkpoint on the test split."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="best_model.pt of a training run.",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--region-specs-test-file",
        type=Path,
        default=None,
        help="JSON file mapping region name to spatial split ('train' | 'test' | "
        '\'all\'), e.g. {"europe": "all"}. Default: the training regions\' '
        "stations selected by --station-split.",
    )
    parser.add_argument(
        "--station-split",
        type=str,
        default="test",
        choices=["test", "train", "all"],
        help="Spatial split to score (always at held-out years): 'test' (default; "
        "held-out stations), 'train' (the training stations), 'all'. Not "
        "combinable with --region-specs-test-file, which carries the split per "
        "region.",
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=None,
        help="Evaluate on this dataset instead of the one in the checkpoint config "
        "(e.g. an Aurora-forecast dataset). Relative to the data root unless "
        "absolute.",
    )
    parser.add_argument(
        "--lead-hours",
        type=int,
        default=None,
        help="Forecast lead of --dataset-dir for a cross-lead checkpoint (0 = ERA5 "
        "analysis, 6/24/72 = Aurora); sets the lead channel. Omit for a "
        "single-lead checkpoint.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Where to write the outputs. Default: the checkpoint's directory; use "
        "a distinct directory per evaluation of one checkpoint (eval_lead6h/, "
        "eval_train_stations/, ...).",
    )
    parser.add_argument(
        "--filter-vae-latents-path",
        type=Path,
        default=None,
        help="Filter-only: drop predictions at stations whose row in this "
        "(n_stations, d) .npy is NaN, so a no-TESSERA checkpoint is scored on "
        "the TESSERA arm's station set. Ignored when the checkpoint already "
        "uses a per-station vector.",
    )
    parser.add_argument(
        "--filter-vae-latents-station-csv",
        type=Path,
        default=None,
        help="CSV row-aligned with --filter-vae-latents-path. Required with it.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    parser = build_parser()
    args = parser.parse_args(argv)
    if (
        args.filter_vae_latents_path is not None
        and args.filter_vae_latents_station_csv is None
    ):
        parser.error(
            "--filter-vae-latents-station-csv is required when "
            "--filter-vae-latents-path is set."
        )
    if args.region_specs_test_file is not None and args.station_split != "test":
        parser.error(
            f"--station-split={args.station_split!r} cannot be combined with "
            "--region-specs-test-file; encode the per-region split in the file."
        )
    region_specs_test = (
        load_region_specs(resolve(args.region_specs_test_file))
        if args.region_specs_test_file is not None
        else None
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = resolve(args.checkpoint)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = ckpt["config"]
    target_variables = list(config["target_variables"])
    likelihood_per_variable = config.get("likelihood_per_variable") or dict.fromkeys(
        target_variables, "gaussian"
    )
    vectors_path, _ = precomputed_vector_files(config)
    dataset_dir = (
        resolve(args.dataset_dir)
        if args.dataset_dir is not None
        else config_path(config, "dataset_dir")
    )
    if dataset_dir is None:
        parser.error("checkpoint config has no dataset_dir; pass --dataset-dir")
    logger.info(
        f"Checkpoint: {checkpoint_path} (epoch {ckpt['epoch']}, "
        f"val loss {ckpt['val_loss']:.4f})"
    )
    mode = (
        f"per-station vector {vectors_path.name}"
        if vectors_path
        else "ERA5-only baseline"
    )
    logger.info(f"Mode: {mode}")
    logger.info(f"Target variables: {target_variables} ({likelihood_per_variable})")

    # ---------------------------------------------------------------
    # Test data, then the model (its input width comes from the dataset)
    # ---------------------------------------------------------------
    test_dataset = build_test_dataset(
        config, dataset_dir, region_specs_test, args.station_split, args.lead_hours
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=downscaling_collate,
    )
    logger.info(f"Test set: {len(test_dataset)} episodes")

    model = build_model_from_config(config, test_dataset.n_context_channels).to(device)
    load_state_dict_compat(model, ckpt["model_state_dict"])
    model.eval()

    subset_per_station = resolve_subset_labels(
        config, region_specs_test, dataset_dir, test_dataset.station_ids
    )
    station_mask = None
    if args.filter_vae_latents_path is not None:
        if vectors_path is not None:
            logger.info(
                "Ignoring --filter-vae-latents-path: the checkpoint already uses a "
                "per-station vector, so its station filter is the strict intersection."
            )
        else:
            station_mask = filter_only_station_mask(
                resolve(args.filter_vae_latents_path),
                resolve(args.filter_vae_latents_station_csv),
                test_dataset.station_ids,
            )

    # ---------------------------------------------------------------
    # Inference and metrics
    # ---------------------------------------------------------------
    params, targets, station_indices, season_errors = run_inference(
        model, test_loader, test_dataset.dates, device, station_mask
    )

    test_summary = {
        "checkpoint_epoch": ckpt["epoch"],
        "best_val_loss": float(ckpt["val_loss"]),
        "target_variables": target_variables,
        "likelihood_per_variable": dict(likelihood_per_variable),
        # Makes test_predictions.npz self-describing.
        "head_spec": {
            var: {
                "distribution": likelihood_per_variable[var],
                "param_names": list(model.heads.heads[var].param_names),
            }
            for var in target_variables
        },
    }
    npz_data: dict = {}
    station_npz: dict = {
        "station_ids": test_dataset.station_ids,
        "station_lats": test_dataset.station_lats,
        "station_lons": test_dataset.station_lons,
        "station_elevs": test_dataset.station_elevs,
        "station_delta_elevs": test_dataset.station_delta_elevs,
    }
    for var in target_variables:
        summarise_variable(
            var,
            model.heads.heads[var],
            params[var],
            targets[var],
            station_indices[var],
            season_errors[var],
            test_summary,
            npz_data,
        )

    logger.info("-" * 60)
    logger.info("PER-STATION ANALYSIS")
    logger.info("-" * 60)
    for var in target_variables:
        per_station_metrics(
            var,
            model.heads.heads[var],
            params[var],
            targets[var],
            station_indices[var],
            test_dataset,
            station_npz,
        )
    if subset_per_station is not None:
        station_npz["subset_per_station"] = subset_per_station
        per_subset_metrics(
            target_variables, subset_per_station, station_npz, test_summary
        )

    # ---------------------------------------------------------------
    # Save
    # ---------------------------------------------------------------
    output_dir = (
        resolve(args.output_dir)
        if args.output_dir is not None
        else checkpoint_path.parent
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    # test_summary.json is the file the submitters check for a completed run;
    # test_results.json is its long-standing alias.
    for name in ("test_summary.json", "test_results.json"):
        with open(output_dir / name, "w") as f:
            json.dump(test_summary, f, indent=2)
    np.savez(output_dir / "test_predictions.npz", **npz_data)
    np.savez(output_dir / "test_station_errors.npz", **station_npz)
    logger.info(f"Saved to {output_dir}")
    logger.info("  test_predictions.npz: raw predictions per observation")
    logger.info("  test_station_errors.npz: per-station aggregated errors + metadata")


if __name__ == "__main__":
    main()
