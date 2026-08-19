r"""Train the ConvCNP station downscaler (console script ``tessera-train``).

One training run fits :class:`~tessera_downscaling.model.convcnp.ConvCNPDownscaler`
on a ``multi_region_snapshot_v1`` dataset (6-hourly ERA5 context, GHCNh
station targets; see :mod:`tessera_downscaling.data.dataset`) by minimising
the per-variable negative log-likelihood of its heads, with Adam, optional
linear learning-rate warm-up, gradient clipping, a non-finite-gradient skip
guard and early stopping on the validation loss.

Two arms share the script and differ only in flags:

* The **TESSERA arm** conditions the decoder on a frozen per-station vector
  (the 16-d latent of a VAE trained on TESSERA patches), served z-scored by
  the dataset and concatenated onto the decoder input::

      tessera-train --dataset-dir dataset_timestamp_global --train-regions europe \
          --tessera-path processed/tessera_global/patch_embeddings_2024.npy \
          --tessera-station-csv processed/tessera_global/station_list_filtered.csv \
          --interpolation bilinear --tessera-injection concat \
          --vae-latents-path processed/vae_tessera_1B-M/station_latents_1B-M_p128_2017_crop64_lat16_grad0.5_auxon.npy \
          --vae-latents-station-csv processed/tessera_global/station_list_filtered.csv \
          --no-static-fields --use-mtpi --weight-decay 1e-4 \
          [--target-variables wind --likelihood wind=truncated_normal] \
          --seed 42 --output-dir training_runs_<folder>/<run_name>

* The **ERA5-only baseline** drops the ``--vae-latents-*`` flags and keeps the
  ERA5 static fields (``--interpolation bilinear --use-mtpi --weight-decay
  1e-4``).

``--tessera-path`` / ``--tessera-station-csv`` are passed to *every* run: they
only filter the stations to those with a valid TESSERA patch, so both arms
train and evaluate on identical station sets. ``--extra-descriptors-path``
feeds hand-crafted surface descriptors through the same precomputed-vector
slot (alone, or hstacked with the latents). ``--lead-datasets`` trains the
lead-conditioned (cross-lead) model on a mix of ERA5 analyses and Aurora
forecast leads. ``--probe-active-from-file`` / ``--train-end-override`` run
the Norway station-rollout data-efficiency experiment.

Relative paths are interpreted relative to the data root
(``$TESSERA_DATA_ROOT``, see :mod:`tessera_downscaling.paths`).

The run directory (``--output-dir``) receives:

* ``config.json`` -- every argument after resolution (paths made absolute,
  derived fields such as ``n_elev_features`` and ``likelihood_per_variable``
  filled in). The same dict is embedded in the checkpoints and is what
  ``tessera-evaluate`` rebuilds the model from.
* ``best_model.pt`` -- lowest validation loss so far (model + optimiser state,
  epoch, val loss / MAE, config); ``latest_model.pt`` every 10 epochs.
* ``training_curves.npz`` -- per-epoch train / val loss, val MAE and
  non-finite-gradient skip counts; ``training_summary.json`` -- the same at a
  glance.

Run ``tessera-evaluate --checkpoint <run>/best_model.pt`` for test metrics.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from tessera_downscaling.data.dataset import (
    MultiLeadDataset,
    MultiRegionSnapshotDownscalingDataset,
    downscaling_collate,
)
from tessera_downscaling.data.helpers import (
    DEFAULT_MIN_TESSERA_PATCH_COVERAGE,
    SUPPORTED_TARGET_VARIABLES,
)
from tessera_downscaling.model.convcnp import ConvCNPDownscaler
from tessera_downscaling.model.heads import HEAD_REGISTRY
from tessera_downscaling.paths import resolve

logger = logging.getLogger("train")

SNAPSHOT_LAYOUT = "multi_region_snapshot_v1"


# ---------------------------------------------------------------------------
# Argument resolution helpers
# ---------------------------------------------------------------------------


def parse_lead_datasets(specs: list[str]) -> list[tuple[int, Path]]:
    """Parse ``--lead-datasets`` ``LEAD_HOURS:DIR`` entries into sorted pairs.

    Each lead must be a non-negative integer and each directory must exist
    (after :func:`~tessera_downscaling.paths.resolve`); duplicate leads are
    rejected. The result is sorted by lead, so the first entry is the lowest
    lead -- conventionally the ERA5 analysis (lead 0) -- which is used as the
    representative dataset for the layout check and metadata.
    """
    pairs: list[tuple[int, Path]] = []
    for spec in specs:
        lead_str, sep, dir_str = spec.partition(":")
        if not sep:
            raise ValueError(
                f"--lead-datasets entry {spec!r} must be 'LEAD_HOURS:DIR'."
            )
        try:
            lead = int(lead_str)
        except ValueError as e:
            raise ValueError(
                f"--lead-datasets entry {spec!r}: {lead_str!r} is not an integer lead."
            ) from e
        if lead < 0:
            raise ValueError(f"--lead-datasets entry {spec!r}: lead must be >= 0.")
        directory = resolve(dir_str)
        if not directory.is_dir():
            raise FileNotFoundError(
                f"--lead-datasets entry {spec!r}: directory not found: {directory}"
            )
        pairs.append((lead, directory))
    leads = [lead for lead, _ in pairs]
    if len(set(leads)) != len(leads):
        raise ValueError(f"--lead-datasets has duplicate leads: {leads}")
    return sorted(pairs, key=lambda p: p[0])


def parse_likelihood(spec: str | None, target_variables: list[str]) -> dict[str, str]:
    """Resolve ``--likelihood var=dist,...`` into ``{var: dist}``.

    Strict 1:1 with ``target_variables`` and every distribution must be a key
    of :data:`~tessera_downscaling.model.heads.HEAD_REGISTRY`. ``None`` means
    Gaussian heads throughout.
    """
    if spec is None:
        return dict.fromkeys(target_variables, "gaussian")
    likelihood: dict[str, str] = {}
    for entry in spec.split(","):
        var, sep, dist = (part.strip() for part in entry.partition("="))
        if not sep or not var or not dist:
            raise ValueError(
                f"--likelihood entry {entry.strip()!r} is malformed; expected "
                "var=dist_name (e.g. wind=truncated_normal)."
            )
        if var in likelihood:
            raise ValueError(f"--likelihood: duplicate entry for variable {var!r}.")
        if dist not in HEAD_REGISTRY:
            raise ValueError(
                f"--likelihood: unknown distribution {dist!r} for {var!r}; "
                f"choose from {sorted(HEAD_REGISTRY)}."
            )
        likelihood[var] = dist
    missing = set(target_variables) - set(likelihood)
    extra = set(likelihood) - set(target_variables)
    if missing or extra:
        raise ValueError(
            "--likelihood does not match --target-variables: "
            f"missing={sorted(missing)}, extra={sorted(extra)}. Each target "
            "variable needs exactly one entry and vice versa."
        )
    return likelihood


def load_json_file(path: Path, flag: str) -> object:
    """Read a JSON sidecar given on the command line, failing with the flag name."""
    if not path.exists():
        raise FileNotFoundError(f"{flag}: {path} does not exist.")
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError as e:
        raise ValueError(f"{flag}: {path} is not valid JSON: {e}") from e


def load_region_specs(path: Path | None, flag: str) -> dict[str, str] | None:
    """Load a ``{region: "train"|"test"|"all"}`` JSON file, or None when unset."""
    if path is None:
        return None
    specs = load_json_file(path, flag)
    if not isinstance(specs, dict) or not all(
        isinstance(k, str) and v in ("train", "test", "all") for k, v in specs.items()
    ):
        raise ValueError(
            f"{flag}: expected a JSON object mapping region name to "
            f"'train' | 'test' | 'all', got {specs!r}."
        )
    return specs


def load_probe_active_from(path: Path | None) -> dict[str, str] | None:
    """Load the ``{station_id: "YYYY-MM-DD-HH"}`` probe-activation map, or None."""
    if path is None:
        return None
    payload = load_json_file(path, "--probe-active-from-file")
    if not isinstance(payload, dict) or not all(
        isinstance(k, str) and isinstance(v, str) for k, v in payload.items()
    ):
        raise ValueError(
            "--probe-active-from-file: expected a JSON object mapping station_id "
            "to an activation timestamp string."
        )
    logger.info(
        f"Probe-station active-from map loaded from {path}: {len(payload)} entries."
    )
    return payload


def resolve_combined_vectors(latents_path: Path, descriptors_path: Path) -> Path:
    """Return (building if absent) the hstacked latents+descriptors ``.npy``.

    Used when ``--vae-latents-path`` and ``--extra-descriptors-path`` are
    given together: the dataset serves ONE precomputed vector per station, so
    the two sources are concatenated into a derived cache file -- a
    deterministic sibling of the latents file -- and that file is what flows
    into the dataset. NaN-row filtering composes into the intersection of
    both sources for free (the loader drops any-NaN rows).

    The write is tmp+rename so concurrent jobs cannot leave a torn file, but
    the z-score stats cache is still built lazily on first use -- pre-build
    both with ``scripts/data/concat_station_vectors.py`` before submitting a
    batch of jobs.
    """
    merged = latents_path.with_name(
        f"{latents_path.stem}_plus_{descriptors_path.stem}.npy"
    )
    if merged.exists():
        logger.info(f"Using existing combined vector file: {merged}")
        return merged
    lat = np.load(latents_path).astype(np.float32)
    desc = np.load(descriptors_path).astype(np.float32)
    if lat.ndim != 2 or desc.ndim != 2 or lat.shape[0] != desc.shape[0]:
        raise ValueError(
            f"Cannot combine {latents_path.name} {lat.shape} with "
            f"{descriptors_path.name} {desc.shape}: need 2-D arrays with "
            "equal row counts (both row-aligned to the same station list)."
        )
    combined = np.hstack([lat, desc])
    tmp = merged.with_name(f"{merged.stem}.tmp{os.getpid()}.npy")
    np.save(tmp, combined)
    tmp.replace(merged)
    logger.info(f"Built combined vector file {merged} shape={combined.shape}")
    return merged


def _stations_csv_has_mtpi(dataset_dir: Path) -> bool:
    """Return True iff the dataset's ``stations.csv`` carries an ``mtpi`` column.

    Reads only the header so it stays cheap. Decides ``n_elev_features``
    (2 vs 3) *before* config.json is written, so the choice round-trips to
    ``tessera-evaluate``.
    """
    stations_csv = dataset_dir / "stations.csv"
    if not stations_csv.exists():
        return False
    with open(stations_csv, newline="") as f:
        header = next(csv.reader(f), [])
    return "mtpi" in header


def _check_snapshot_layout(dataset_dir: Path) -> None:
    """Fail clearly unless ``dataset_dir`` is a ``multi_region_snapshot_v1`` dataset."""
    metadata_path = dataset_dir / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"{metadata_path} not found; is this a dataset directory?"
        )
    with open(metadata_path) as f:
        layout = json.load(f).get("layout_version")
    if layout != SNAPSHOT_LAYOUT:
        raise ValueError(
            f"{dataset_dir} has layout_version={layout!r}; tessera-train only "
            f"supports {SNAPSHOT_LAYOUT!r} (scripts/preprocessing/"
            "preprocess_timestamp_global.py)."
        )


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------


def _build_mr_train_val(
    shared_kwargs: dict,
    train_region_specs: dict[str, str] | None,
    val_region_specs: dict[str, str] | None,
    train_regions: list[str] | None,
) -> tuple[
    MultiRegionSnapshotDownscalingDataset, MultiRegionSnapshotDownscalingDataset
]:
    """Build the (train, val) snapshot datasets for one ``dataset_dir``.

    Regions come from ``--region-specs-train-file`` (per-region spatial split)
    or ``--train-regions`` (every listed region contributes its ``train``
    stations; ``None`` = all regions). Validation uses the held-out (``test``)
    stations of the training regions unless ``--region-specs-val-file`` says
    otherwise. The cross-lead path calls this once per lead.
    """
    if train_region_specs is not None:
        train = MultiRegionSnapshotDownscalingDataset(
            region_specs=train_region_specs, split="train", **shared_kwargs
        )
    else:
        train = MultiRegionSnapshotDownscalingDataset(
            regions=train_regions, split="train", station_split="train", **shared_kwargs
        )
    if val_region_specs is not None:
        val = MultiRegionSnapshotDownscalingDataset(
            region_specs=val_region_specs, split="val", **shared_kwargs
        )
    elif train_region_specs is not None:
        derived_val = {
            r: ("test" if s == "train" else s) for r, s in train_region_specs.items()
        }
        logger.info(
            f"No --region-specs-val-file; derived val specs from train: {derived_val}"
        )
        val = MultiRegionSnapshotDownscalingDataset(
            region_specs=derived_val, split="val", **shared_kwargs
        )
    else:
        val = MultiRegionSnapshotDownscalingDataset(
            regions=train_regions, split="val", station_split="test", **shared_kwargs
        )
    return train, val


def build_datasets(
    args: argparse.Namespace,
    lead_dataset_pairs: list[tuple[int, Path]] | None,
    precomputed_path: Path | None,
    precomputed_station_csv: Path | None,
    probe_active_from: dict[str, str] | None,
    train_region_specs: dict[str, str] | None,
    val_region_specs: dict[str, str] | None,
) -> tuple[Dataset, Dataset]:
    """Build the training and validation datasets described by ``args``."""
    shared_kwargs = {
        "dataset_dir": args.dataset_dir,
        "target_variables": args.target_variables,
        "tessera_path": args.tessera_path,
        "tessera_station_csv": args.tessera_station_csv,
        "include_static_fields": args.include_static_fields,
        # The dataset's vae_latents_* kwargs are the generic precomputed
        # per-station-vector inputs; they also carry the extra descriptors.
        "vae_latents_path": precomputed_path,
        "vae_latents_station_csv": precomputed_station_csv,
        "min_patch_coverage": args.min_tessera_patch_coverage,
        "normalisation_policy": args.normalisation_policy,
        # Uniform across splits: activation dates fall inside the training
        # window, so the mask is a no-op on val/test episodes.
        "probe_active_from": probe_active_from,
        # Moves the train/val boundary for BOTH splits so no timestamp falls
        # between them; normalisation stats still use metadata.train_end.
        "train_end_override": args.train_end_override,
        "drop_context_channels": args.drop_context_channels,
        # Strict when training on one dataset (a misspelt channel name is an
        # error); lenient across leads, where the same list must drop precip
        # from the 20-channel ERA5 analysis and no-op on the 19-channel Aurora
        # forecasts.
        "drop_context_strict": lead_dataset_pairs is None,
    }
    if lead_dataset_pairs is None:
        return _build_mr_train_val(
            shared_kwargs, train_region_specs, val_region_specs, args.train_regions
        )

    # Cross-lead: one (train, val) pair per lead, each on its own directory
    # with lead_hours set so the context grid carries the lead channel, then
    # concatenated so every epoch sees every episode at every lead.
    train_subs, val_subs = [], []
    for lead_hours, directory in lead_dataset_pairs:
        lead_kwargs = {
            **shared_kwargs,
            "dataset_dir": directory,
            "lead_hours": lead_hours,
        }
        tr, va = _build_mr_train_val(
            lead_kwargs, train_region_specs, val_region_specs, args.train_regions
        )
        train_subs.append(tr)
        val_subs.append(va)
    train_dataset = MultiLeadDataset(train_subs)
    val_dataset = MultiLeadDataset(val_subs)
    logger.info(
        f"Cross-lead datasets built: train {len(train_dataset)} "
        f"(= {len(train_subs[0])} episodes x {len(train_subs)} leads), "
        f"val {len(val_dataset)}; "
        f"n_context_channels={train_dataset.n_context_channels}."
    )
    return train_dataset, val_dataset


# ---------------------------------------------------------------------------
# Model helpers
# ---------------------------------------------------------------------------


def initialise_truncated_normal_heads(
    model: ConvCNPDownscaler,
    train_dataset: Dataset,
    batch_size: int,
    max_batches: int = 50,
) -> None:
    """Initialise every truncated-normal head from positive-target climatology.

    Sets the head's ``(μ, log σ²)`` biases to the mean / std of the strictly
    positive training targets seen in the first ``max_batches`` batches (zero
    wind readings are anemometer-threshold reports and are excluded), so the
    head does not spend its first epochs traversing the Mills-ratio region of
    the NLL surface. See ``TruncatedNormalHead.initialise_from_climatology``.
    """
    tn_vars = [
        var
        for var, dist in model.likelihood_per_variable.items()
        if dist == "truncated_normal"
    ]
    if not tn_vars:
        return
    stats_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=downscaling_collate,
    )
    for var in tn_vars:
        vi = model.target_variables.index(var)
        positive_targets: list[torch.Tensor] = []
        for batch_idx, batch in enumerate(stats_loader):
            if batch is None:
                continue
            targets = batch["target_values"]
            mask = batch["target_mask"].bool()
            tgt = targets[:, :, vi] if targets.ndim == 3 else targets
            valid_positive = (tgt > 0) & mask
            if valid_positive.any():
                positive_targets.append(tgt[valid_positive])
            if batch_idx + 1 >= max_batches:
                break
        all_positive = (
            torch.cat(positive_targets) if positive_targets else torch.empty(0)
        )
        if all_positive.numel() < 100:
            raise RuntimeError(
                f"Could not compute climatology for {var!r}: only "
                f"{all_positive.numel()} positive targets in the first "
                f"{max_batches} batches (need >= 100)."
            )
        mean_target = float(all_positive.mean().item())
        std_target = float(all_positive.std().item())
        model.heads.heads[var].initialise_from_climatology(
            mean_target=mean_target, std_target=std_target
        )
        logger.info(
            f"TruncatedNormal head {var!r}: mean ≈ {mean_target:.4f}, "
            f"std ≈ {std_target:.4f} (from {all_positive.numel()} positive "
            f"targets across ≤{max_batches} train batches); (μ, log_var) bias "
            f"initialised to ({mean_target:.4f}, {2 * math.log(std_target):.4f})."
        )


def compute_loss(
    params_per_var: dict[str, dict[str, torch.Tensor]],
    target_values: torch.Tensor,
    target_mask: torch.Tensor,
    model: ConvCNPDownscaler,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Masked NLL summed over target variables, plus the per-variable terms.

    ``target_values`` is ``(B, N)`` for a single variable or ``(B, N, V)`` for
    several; each variable's head computes its own NLL over the unmasked
    entries.
    """
    mask = target_mask.bool()
    per_var: dict[str, torch.Tensor] = {}
    for vi, var in enumerate(model.target_variables):
        target = target_values[:, :, vi] if target_values.ndim == 3 else target_values
        per_var[var] = model.heads.heads[var].nll(
            params_per_var[var], target, mask=mask
        )
    return sum(per_var.values()), per_var


def forward_batch(
    model: ConvCNPDownscaler, batch: dict, device: torch.device
) -> tuple[dict[str, dict[str, torch.Tensor]], torch.Tensor, torch.Tensor]:
    """Move a collated batch to ``device`` and run the model.

    Returns ``(params_per_var, target_values, target_mask)`` on ``device``.
    """
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
    return (
        params_per_var,
        batch["target_values"].to(device),
        batch["target_mask"].to(device),
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train the ConvCNP downscaler (ERA5-only baseline or TESSERA arm)."
    )
    # Data.
    data = parser.add_argument_group("data")
    data.add_argument(
        "--dataset-dir",
        type=Path,
        default=None,
        help="multi_region_snapshot_v1 dataset (relative to the data root unless "
        "absolute). Optional with --lead-datasets, where it is set to the "
        "lowest lead's directory.",
    )
    data.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Run directory for config.json, checkpoints and training curves "
        "(relative to the data root unless absolute).",
    )
    data.add_argument(
        "--train-regions",
        type=str,
        nargs="+",
        default=None,
        help="Regions to train on (their 'train' stations); validation uses their "
        "'test' stations. Default: all regions. Superseded by "
        "--region-specs-train-file.",
    )
    data.add_argument(
        "--region-specs-train-file",
        type=Path,
        default=None,
        help="JSON file mapping region name to spatial split ('train' | 'test' | "
        '\'all\') for the training set, e.g. {"europe": "train", "us": '
        '"all"}. Mutually exclusive with --train-regions.',
    )
    data.add_argument(
        "--region-specs-val-file",
        type=Path,
        default=None,
        help="Same for the validation set. Default: the training specs with "
        "'train' replaced by 'test' (held-out stations of the training regions).",
    )
    data.add_argument(
        "--target-variables",
        type=str,
        nargs="+",
        default=["t2m"],
        choices=sorted(SUPPORTED_TARGET_VARIABLES),
        help="Target variable(s): t2m (2 m temperature) and/or wind (10 m wind "
        "speed). Default: t2m.",
    )
    data.add_argument(
        "--likelihood",
        type=str,
        default=None,
        help="Per-variable likelihood, e.g. 'wind=truncated_normal' or "
        "'t2m=gaussian,wind=truncated_normal'; one entry per target variable. "
        f"Distributions: {sorted(HEAD_REGISTRY)}. Default: Gaussian throughout.",
    )
    data.add_argument(
        "--drop-context-channels",
        type=str,
        nargs="+",
        default=None,
        help="ERA5 dynamic channels to remove from the context grid by name, e.g. "
        "total_precipitation_sum to align the 20-channel ERA5 dataset with the "
        "19-channel Aurora forecasts for cross-lead training.",
    )
    data.add_argument(
        "--lead-datasets",
        type=str,
        nargs="+",
        default=None,
        metavar="LEAD:DIR",
        help="Lead-conditioned (cross-lead) model: one 'LEAD_HOURS:DATASET_DIR' per "
        "forecast lead, e.g. '0:dataset_timestamp_global "
        "6:dataset_timestamp_aurora_lead6h 24:... 72:...'. Each lead becomes "
        "its own dataset carrying a lead/72 context channel; the leads are "
        "concatenated so one epoch sees every episode at every lead. Pass "
        "--drop-context-channels total_precipitation_sum alongside.",
    )
    data.add_argument(
        "--num-workers", type=int, default=4, help="DataLoader workers. Default 4."
    )

    # Station filter (every run).
    filt = parser.add_argument_group("station filter (applied to every run)")
    filt.add_argument(
        "--tessera-path",
        type=Path,
        default=None,
        help="(N, 64, 64, 128) TESSERA patch .npy; stations without a valid patch "
        "are dropped so baseline and TESSERA runs share one station set. "
        "Filter only -- no patch is loaded for training.",
    )
    filt.add_argument(
        "--tessera-station-csv",
        type=Path,
        default=None,
        help="CSV row-aligned with --tessera-path (station_id column).",
    )
    filt.add_argument(
        "--min-tessera-patch-coverage",
        type=float,
        default=DEFAULT_MIN_TESSERA_PATCH_COVERAGE,
        help="Minimum fraction of non-zero pixels in the patch for a station to be "
        "kept (combined with the centre-pixel rule); 0 disables the coverage "
        f"check. Default {DEFAULT_MIN_TESSERA_PATCH_COVERAGE}.",
    )

    # Per-station conditioning vector.
    cond = parser.add_argument_group("per-station conditioning vector (TESSERA arm)")
    cond.add_argument(
        "--vae-latents-path",
        type=Path,
        default=None,
        help="(n_stations, d) precomputed per-station vector .npy (the paper's VAE "
        "latents of TESSERA patches; also shuffled latents or summary "
        "statistics). Served z-scored and concatenated onto the decoder input. "
        "Stations with a NaN row are dropped.",
    )
    cond.add_argument(
        "--vae-latents-station-csv",
        type=Path,
        default=None,
        help="CSV row-aligned with --vae-latents-path. Required with it.",
    )
    cond.add_argument(
        "--extra-descriptors-path",
        type=Path,
        default=None,
        help="(n_stations, d) hand-crafted surface descriptors .npy "
        "(scripts/data/build_extra_descriptors.py). Same pathway as the "
        "latents; when both are given they are hstacked into one vector.",
    )
    cond.add_argument(
        "--extra-descriptors-station-csv",
        type=Path,
        default=None,
        help="CSV row-aligned with --extra-descriptors-path. Required with it; "
        "must equal --vae-latents-station-csv when both sources are used.",
    )
    cond.add_argument(
        "--tessera-injection",
        type=str,
        default="concat",
        choices=["concat", "none"],
        help="'concat' (default) appends the vector to the decoder input; 'none' "
        "loads it for station filtering only (ablation).",
    )

    # Model.
    model = parser.add_argument_group("model")
    model.add_argument(
        "--interpolation",
        type=str,
        default="bilinear",
        choices=["bilinear", "setconv"],
        help="Grid-to-station interpolation: parameter-free 'bilinear' (default; "
        "every paper run) or the vanilla ConvCNP 'setconv' with a learned RBF "
        "length-scale.",
    )
    model.add_argument(
        "--setconv-length-scale",
        type=float,
        default=0.5,
        help="Initial RBF length scale in degrees for --interpolation setconv. "
        "Default 0.5.",
    )
    model.add_argument(
        "--no-elevation",
        action="store_true",
        help="Drop the per-station topographic features (elevation, Δelevation, "
        "mTPI) from the decoder input.",
    )
    model.add_argument(
        "--use-mtpi",
        action="store_true",
        help="Append the station's multi-scale topographic position index as a "
        "third topographic feature (Vaughan et al. 2022). Requires an 'mtpi' "
        "column in the dataset's stations.csv.",
    )
    model.add_argument(
        "--no-static-fields",
        action="store_true",
        help="Exclude the 13 ERA5 static fields (orography, land-sea mask, ...) "
        "from the context grid. The paper's TESSERA arm sets this.",
    )
    model.add_argument("--cnn-hidden", type=int, default=128, help="CNN width.")
    model.add_argument("--cnn-layers", type=int, default=7, help="CNN conv layers.")
    model.add_argument("--cnn-kernel", type=int, default=3, help="CNN kernel size.")
    model.add_argument("--mlp-hidden", type=int, default=128, help="Decoder MLP width.")
    model.add_argument(
        "--mlp-n-hidden", type=int, default=3, help="Decoder MLP hidden layers."
    )

    # Optimisation.
    optim = parser.add_argument_group("optimisation")
    optim.add_argument("--epochs", type=int, default=100)
    optim.add_argument("--batch-size", type=int, default=1)
    optim.add_argument("--lr", type=float, default=2.5e-5)
    optim.add_argument("--weight-decay", type=float, default=0.0)
    optim.add_argument(
        "--lr-warmup-pct",
        type=float,
        default=0.0,
        help="Linear LR warm-up over this fraction of the planned steps "
        "(epochs x batches). Default 0 (none).",
    )
    optim.add_argument(
        "--grad-clip-norm",
        type=float,
        default=1.0,
        help="Clip the total gradient norm before each step; 0 disables. Default 1.",
    )
    optim.add_argument(
        "--patience",
        type=int,
        default=10,
        help="Early-stop after this many epochs without a new best val loss.",
    )
    optim.add_argument("--seed", type=int, default=42)

    # Data-efficiency (Norway rollout) experiment.
    eff = parser.add_argument_group("station-rollout experiment")
    eff.add_argument(
        "--probe-active-from-file",
        type=Path,
        default=None,
        help="JSON {station_id: 'YYYY-MM-DD-HH'}: probe stations contribute no "
        "training targets before their activation timestamp (written by "
        "scripts/experiments/build_rollout_schedule.py).",
    )
    eff.add_argument(
        "--train-end-override",
        type=str,
        default=None,
        help="Move the train/val boundary earlier than metadata.json's train_end "
        "('YYYY-MM-DD[-HH]'). Normalisation statistics are unaffected.",
    )
    return parser


def resolve_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> dict:
    """Validate ``args`` and fill in every derived field, in place.

    All paths are resolved against the data root and every value that the
    run depends on (``n_elev_features``, ``likelihood_per_variable``, ...) is
    set on ``args`` here, BEFORE the config is serialised, so ``config.json``
    and the checkpoints describe the run completely. Returns the side data
    that is not part of the config (lead pairs, loaded JSON sidecars,
    precomputed-vector source).
    """
    for name in (
        "dataset_dir",
        "output_dir",
        "tessera_path",
        "tessera_station_csv",
        "vae_latents_path",
        "vae_latents_station_csv",
        "extra_descriptors_path",
        "extra_descriptors_station_csv",
        "region_specs_train_file",
        "region_specs_val_file",
        "probe_active_from_file",
    ):
        value = getattr(args, name)
        if value is not None:
            setattr(args, name, resolve(value))

    # Cross-lead: the lowest lead (conventionally the ERA5 analysis) is the
    # representative dataset for the layout check, mTPI detection and metadata.
    lead_dataset_pairs = None
    if args.lead_datasets:
        lead_dataset_pairs = parse_lead_datasets(args.lead_datasets)
        args.lead_datasets = [f"{lead}:{d}" for lead, d in lead_dataset_pairs]
        args.dataset_dir = lead_dataset_pairs[0][1]
    if args.dataset_dir is None:
        parser.error("one of --dataset-dir or --lead-datasets is required")
    _check_snapshot_layout(args.dataset_dir)

    if args.region_specs_train_file is not None and args.train_regions is not None:
        parser.error(
            "Pass either --region-specs-train-file OR --train-regions, not both."
        )
    train_region_specs = load_region_specs(
        args.region_specs_train_file, "--region-specs-train-file"
    )
    val_region_specs = load_region_specs(
        args.region_specs_val_file, "--region-specs-val-file"
    )
    probe_active_from = load_probe_active_from(args.probe_active_from_file)

    args.include_elevation = not args.no_elevation
    args.include_static_fields = not args.no_static_fields
    if args.use_mtpi and not _stations_csv_has_mtpi(args.dataset_dir):
        parser.error(
            "--use-mtpi was set but the dataset's stations.csv has no `mtpi` "
            "column. Run scripts/data/backfill_station_mtpi.py (or re-preprocess "
            "with --mtpi-csv) first."
        )
    args.n_elev_features = 3 if args.use_mtpi else 2
    args.likelihood_per_variable = parse_likelihood(
        args.likelihood, args.target_variables
    )
    # ERA5 channels are z-scored with each region's own train-split statistics;
    # recorded for continuity with older configs.
    args.normalisation_policy = "per_region"

    # Precomputed per-station vector: latents and/or extra descriptors ride the
    # same dataset pathway; when both are given they are hstacked into a
    # derived cache file recorded as precomputed_merged_path.
    uses_latents = args.vae_latents_path is not None
    uses_descriptors = args.extra_descriptors_path is not None
    if uses_latents and args.vae_latents_station_csv is None:
        parser.error(
            "--vae-latents-station-csv is required when --vae-latents-path is set"
        )
    if uses_descriptors and args.extra_descriptors_station_csv is None:
        parser.error(
            "--extra-descriptors-station-csv is required when "
            "--extra-descriptors-path is set"
        )
    if (uses_latents or uses_descriptors) and args.tessera_path is None:
        parser.error(
            "--tessera-path is required alongside a per-station vector, so the "
            "station filter is (valid patch) AND (non-NaN vector) as in the "
            "baseline arm."
        )
    if args.tessera_path is not None and args.tessera_station_csv is None:
        parser.error("--tessera-station-csv is required when --tessera-path is set")
    precomputed_path: Path | None = None
    precomputed_station_csv: Path | None = None
    if uses_latents and uses_descriptors:
        if args.extra_descriptors_station_csv != args.vae_latents_station_csv:
            parser.error(
                "--vae-latents-path and --extra-descriptors-path must be row-aligned "
                "to the SAME station CSV when combined."
            )
        precomputed_path = resolve_combined_vectors(
            args.vae_latents_path, args.extra_descriptors_path
        )
        precomputed_station_csv = args.vae_latents_station_csv
        args.precomputed_merged_path = precomputed_path
    elif uses_latents:
        precomputed_path = args.vae_latents_path
        precomputed_station_csv = args.vae_latents_station_csv
    elif uses_descriptors:
        precomputed_path = args.extra_descriptors_path
        precomputed_station_csv = args.extra_descriptors_station_csv

    return {
        "lead_dataset_pairs": lead_dataset_pairs,
        "train_region_specs": train_region_specs,
        "val_region_specs": val_region_specs,
        "probe_active_from": probe_active_from,
        "precomputed_path": precomputed_path,
        "precomputed_station_csv": precomputed_station_csv,
    }


def config_from_args(args: argparse.Namespace) -> dict:
    """JSON-serialisable record of the run (paths as strings)."""
    return {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    parser = build_parser()
    args = parser.parse_args(argv)
    side = resolve_args(parser, args)
    precomputed_path = side["precomputed_path"]
    uses_precomputed = precomputed_path is not None

    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = config_from_args(args)
    with open(args.output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    logger.info(f"Target variables: {args.target_variables}")
    if uses_precomputed:
        logger.info(
            f"Mode: per-station vector {precomputed_path.name} "
            f"(injection={args.tessera_injection})"
        )
    elif args.tessera_path is not None:
        logger.info("Mode: ERA5-only baseline on TESSERA-filtered stations")
    else:
        logger.info("Mode: ERA5-only baseline (all stations)")
    if not args.include_elevation:
        logger.info("Elevation features: EXCLUDED")
    if not args.include_static_fields:
        logger.info("ERA5 static fields: EXCLUDED")
    if args.train_end_override is not None:
        logger.info(f"train_end_override={args.train_end_override!r}")
    if args.drop_context_channels:
        logger.info(f"Dropping context channels: {args.drop_context_channels}")

    # ---------------------------------------------------------------
    # Data
    # ---------------------------------------------------------------
    train_dataset, val_dataset = build_datasets(
        args,
        side["lead_dataset_pairs"],
        precomputed_path,
        side["precomputed_station_csv"],
        side["probe_active_from"],
        side["train_region_specs"],
        side["val_region_specs"],
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=downscaling_collate,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=downscaling_collate,
        pin_memory=True,
    )
    logger.info(
        f"Train: {len(train_dataset)} episodes, Val: {len(val_dataset)} episodes"
    )

    # ---------------------------------------------------------------
    # Model
    # ---------------------------------------------------------------
    precomputed_tessera_dim = train_dataset.vae_latent_dim if uses_precomputed else 0
    if uses_precomputed:
        if val_dataset.vae_latent_dim != precomputed_tessera_dim:
            raise RuntimeError(
                "Precomputed vector dim differs between train "
                f"({precomputed_tessera_dim}) "
                f"and val ({val_dataset.vae_latent_dim})"
            )
        logger.info(f"Precomputed vector dim (from dataset): {precomputed_tessera_dim}")

    model = ConvCNPDownscaler(
        n_context_channels=train_dataset.n_context_channels,
        cnn_hidden=args.cnn_hidden,
        cnn_layers=args.cnn_layers,
        cnn_kernel=args.cnn_kernel,
        setconv_length_scale=args.setconv_length_scale,
        interpolation=args.interpolation,
        mlp_hidden=args.mlp_hidden,
        mlp_n_hidden=args.mlp_n_hidden,
        n_elev_features=args.n_elev_features,
        include_elevation=args.include_elevation,
        target_variables=args.target_variables,
        likelihood_per_variable=args.likelihood_per_variable,
        tessera_injection=args.tessera_injection,
        tessera_features_precomputed=uses_precomputed,
        precomputed_tessera_dim=precomputed_tessera_dim,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {n_params:,}")

    initialise_truncated_normal_heads(model, train_dataset, args.batch_size)

    # ---------------------------------------------------------------
    # Optimiser and LR warm-up
    # ---------------------------------------------------------------
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    # Linear ramp from lr*1e-3 to lr over the first warmup_pct of the planned
    # steps (epochs x batches), then constant. Runs typically early-stop after
    # 5-15 epochs, so a warm-up + plateau fits better than a long decay.
    lr_scheduler = None
    if args.lr_warmup_pct > 0.0:
        total_steps = args.epochs * len(train_loader)
        warmup_steps = max(1, int(args.lr_warmup_pct * total_steps))
        lr_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1e-3, end_factor=1.0, total_iters=warmup_steps
        )
        logger.info(
            f"LR warmup: linear from {args.lr * 1e-3:.2e} to {args.lr:.2e} over "
            f"{warmup_steps} steps ({args.lr_warmup_pct * 100:.1f}% of "
            f"{total_steps} planned steps)"
        )

    # ---------------------------------------------------------------
    # Training loop
    # ---------------------------------------------------------------
    n_target_variables = len(args.target_variables)
    best_val_loss = float("inf")
    epochs_without_improvement = 0
    train_losses: list[float] = []
    val_losses: list[float] = []
    val_maes: list[list[float]] = []
    nonfinite_skips_per_epoch: list[int] = []
    attempted_steps_per_epoch: list[int] = []

    for epoch in range(1, args.epochs + 1):
        # --- Train ---
        model.train()
        epoch_loss = 0.0
        epoch_batches = 0
        nonfinite_skips = 0
        t_start = time.time()

        for batch_idx, batch in enumerate(
            tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs} [train]", leave=False)
        ):
            if batch is None:
                continue
            optimizer.zero_grad()
            params_per_var, target_values, target_mask = forward_batch(
                model, batch, device
            )
            loss, _ = compute_loss(params_per_var, target_values, target_mask, model)
            loss.backward()

            # Skip the step on a non-finite gradient. This MUST precede
            # clip_grad_norm_: the clipper does not sanitise -- a single NaN
            # element makes the global norm NaN, which the clip coefficient
            # then spreads to every parameter in one step.
            if any(
                p.grad is not None and not torch.isfinite(p.grad).all()
                for p in model.parameters()
            ):
                nonfinite_skips += 1
                optimizer.zero_grad()
                if nonfinite_skips <= 5 or nonfinite_skips % 100 == 0:
                    logger.warning(
                        f"Non-finite gradient at epoch={epoch} batch_idx={batch_idx}; "
                        f"skipping optimizer step (total skips so far: "
                        f"{nonfinite_skips})."
                    )
                continue

            if args.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=args.grad_clip_norm
                )
            optimizer.step()
            if lr_scheduler is not None:
                lr_scheduler.step()

            epoch_loss += loss.item()
            epoch_batches += 1

        train_loss = epoch_loss / max(epoch_batches, 1)
        train_losses.append(train_loss)
        nonfinite_skips_per_epoch.append(nonfinite_skips)
        attempted_steps_per_epoch.append(epoch_batches + nonfinite_skips)
        t_train = time.time() - t_start
        if nonfinite_skips > 0:
            pct = 100.0 * nonfinite_skips / max(epoch_batches + nonfinite_skips, 1)
            logger.warning(
                f"Epoch {epoch}: skipped {nonfinite_skips} optimizer steps ({pct:.2f}% "
                "of attempted updates) due to non-finite gradients. Above ~5% the "
                "LR is probably too high for this likelihood."
            )

        # --- Validate ---
        model.eval()
        val_loss = 0.0
        val_batches = 0
        val_mae_sums = [0.0] * n_target_variables
        val_counts = [0.0] * n_target_variables
        with torch.no_grad():
            for batch in tqdm(
                val_loader, desc=f"Epoch {epoch}/{args.epochs} [val]", leave=False
            ):
                if batch is None:
                    continue
                params_per_var, target_values, target_mask = forward_batch(
                    model, batch, device
                )
                loss, _ = compute_loss(
                    params_per_var, target_values, target_mask, model
                )
                val_loss += loss.item()
                val_batches += 1
                # Secondary tracking metric: MAE of the predictive mean. Early
                # stopping uses the val loss above.
                mask_f = target_mask.float()
                for vi, var in enumerate(args.target_variables):
                    point_est = model.heads.heads[var].mean(params_per_var[var])
                    target = (
                        target_values[:, :, vi]
                        if target_values.ndim == 3
                        else target_values
                    )
                    val_mae_sums[vi] += (
                        (torch.abs(point_est - target) * mask_f).sum().item()
                    )
                    val_counts[vi] += mask_f.sum().item()

        val_loss_avg = val_loss / max(val_batches, 1)
        val_losses.append(val_loss_avg)
        epoch_maes = [
            val_mae_sums[vi] / max(val_counts[vi], 1)
            for vi in range(n_target_variables)
        ]
        val_maes.append(epoch_maes)

        mae_str = " | ".join(
            f"{var} MAE: {epoch_maes[vi]:.3f}"
            for vi, var in enumerate(args.target_variables)
        )
        logger.info(
            f"Epoch {epoch:3d}/{args.epochs} | Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss_avg:.4f} | {mae_str} | Time: {t_train:.1f}s"
        )

        # --- Checkpointing + early stopping ---
        save_dict = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss": val_loss_avg,
            "val_maes": dict(zip(args.target_variables, epoch_maes, strict=True)),
            "config": config,
        }
        if val_loss_avg < best_val_loss:
            best_val_loss = val_loss_avg
            epochs_without_improvement = 0
            torch.save(save_dict, args.output_dir / "best_model.pt")
            logger.info(f"  -> New best model (Val Loss: {val_loss_avg:.4f})")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                logger.info(
                    f"Early stopping after {args.patience} epochs without improvement"
                )
                break
        if epoch % 10 == 0:
            torch.save(save_dict, args.output_dir / "latest_model.pt")

    # ---------------------------------------------------------------
    # Training curves and summary
    # ---------------------------------------------------------------
    np.savez(
        args.output_dir / "training_curves.npz",
        train_losses=np.array(train_losses),
        val_losses=np.array(val_losses),
        val_maes=np.array(val_maes),  # (n_epochs, n_vars)
        nonfinite_skips_per_epoch=np.array(nonfinite_skips_per_epoch),
        attempted_steps_per_epoch=np.array(attempted_steps_per_epoch),
    )
    total_skipped = int(sum(nonfinite_skips_per_epoch))
    total_attempted = int(sum(attempted_steps_per_epoch))
    skip_pct_overall = (
        100.0 * total_skipped / total_attempted if total_attempted else 0.0
    )
    training_summary = {
        "n_epochs_run": len(train_losses),
        "best_val_loss": float(best_val_loss),
        "final_train_loss": float(train_losses[-1]) if train_losses else None,
        "final_val_loss": float(val_losses[-1]) if val_losses else None,
        # Non-finite-gradient guard statistics; non-zero on a Gaussian-only
        # run is worth investigating.
        "nonfinite_skips_total": total_skipped,
        "training_steps_attempted_total": total_attempted,
        "nonfinite_skip_pct_overall": skip_pct_overall,
        "nonfinite_skip_pct_per_epoch": [
            100.0 * s / max(a, 1)
            for s, a in zip(
                nonfinite_skips_per_epoch, attempted_steps_per_epoch, strict=True
            )
        ],
        "nonfinite_skips_per_epoch": list(nonfinite_skips_per_epoch),
        # Config echo (subset of run-defining args).
        "target_variables": list(args.target_variables),
        "likelihood_per_variable": dict(args.likelihood_per_variable),
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "grad_clip_norm": args.grad_clip_norm,
        "batch_size": args.batch_size,
        "seed": args.seed,
    }
    with open(args.output_dir / "training_summary.json", "w") as f:
        json.dump(training_summary, f, indent=2)

    logger.info(f"Training complete. Best Val Loss: {best_val_loss:.4f}")
    logger.info(
        f"Total non-finite-gradient skips: {total_skipped}/{total_attempted} "
        f"({skip_pct_overall:.2f}% overall)."
    )
    logger.info(f"Outputs saved to {args.output_dir}")
    logger.info("Run tessera-evaluate on the checkpoint for test metrics.")


if __name__ == "__main__":
    main()
