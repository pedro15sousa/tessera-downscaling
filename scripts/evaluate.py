"""Evaluate a trained ConvCNP downscaler on the test set.

Auto-detects whether the checkpoint is a baseline or TESSERA model, and
whether it's single-task or multi-task, by reading the saved config.
Reports per-variable MAE, RMSE, bias, NLL, calibration, error percentiles,
and seasonal breakdown.

Saves:
  - test_results.json: Per-variable aggregate metrics.
  - test_predictions.npz: Raw predictions, targets, and predicted stds
    for downstream analysis notebooks.
  - test_summary.json: Summary metrics used by submit_parallel.sh to
    detect completed runs.

Usage:
    uv run --group core python projects/tessera_downscaling/scripts/evaluate.py \\
        --checkpoint .tmp_output/training_runs/baseline_seed42/best_model.pt
"""

import argparse
import json
import logging
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("evaluate")


def date_to_season(date_str: str) -> str:
    month = int(date_str.split("-")[1])
    if month in (12, 1, 2):
        return "DJF"
    elif month in (3, 4, 5):
        return "MAM"
    elif month in (6, 7, 8):
        return "JJA"
    return "SON"


def _parse_region_specs_from_args(args) -> dict[str, str]:
    """Parse --region-specs-test (inline JSON) or --region-specs-test-file
    (path) into a dict.

    Returns {} if neither flag was given, letting callers distinguish
    "no spec" from "empty spec" without a separate None check.
    """
    inline = args.region_specs_test
    file_path = args.region_specs_test_file
    if inline is not None and file_path is not None:
        raise ValueError(
            "Pass either --region-specs-test OR --region-specs-test-file, not both."
        )
    if file_path is not None:
        if not file_path.exists():
            raise ValueError(
                f"--region-specs-test-file path {file_path} does not exist."
            )
        raw = file_path.read_text()
    elif inline is not None:
        raw = inline
    else:
        return {}
    try:
        parsed = json.loads(raw)
    except Exception as e:
        raise ValueError(
            f"Could not parse --region-specs-test JSON: {e}"
        )
    if not isinstance(parsed, dict):
        raise ValueError(
            f"--region-specs-test must be a JSON object, got {type(parsed).__name__}"
        )
    for name, s in parsed.items():
        if s not in ("train", "test", "all"):
            raise ValueError(
                f"--region-specs-test['{name}']={s!r} is not one of "
                f"'train', 'test', 'all'."
            )
    return parsed


def main():
    parser = argparse.ArgumentParser(description="Evaluate ConvCNP downscaler")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--test-regions", type=str, nargs="+", default=None,
        help="When the training dataset was multi-region, restrict the "
             "test dataset to these regions. Example: --test-regions europe "
             "after training on US. If omitted, defaults to the training "
             "config's val_regions / train_regions, unless overridden. "
             "Legacy flag — consider --region-specs-test for finer control.",
    )
    parser.add_argument(
        "--region-specs-test", type=str, default=None,
        help="JSON dict mapping region name to spatial split for the test "
             "dataset, e.g. '{\"europe\": \"test\"}' to evaluate on EU "
             "held-out stations after training on {EU train + all of US}. "
             "Supersedes --test-regions. Typical usage: the companion to "
             "a --region-specs-train used at training time. If shell "
             "quoting the JSON is a pain, use --region-specs-test-file.",
    )
    parser.add_argument(
        "--region-specs-test-file", type=Path, default=None,
        help="Path to a JSON file containing the same dict that "
             "--region-specs-test would take inline. Used when shell "
             "quoting is impractical (e.g. via sbatch --wrap). Mutually "
             "exclusive with --region-specs-test.",
    )
    # ---- Filter-only overrides (no model side-effects) ----
    # These let us re-evaluate a trained checkpoint on a stricter
    # station set than its training-time filter, without retraining.
    # Concretely: a no-TESSERA bilinear baseline checkpoint can be
    # re-evaluated against the (TESSERA ∩ VAE-non-NaN) intersection
    # used by VAE variants, so all rows of the headline table land
    # on identical stations. The model never sees the latents — they
    # are passed to the dataset purely to drive the station filter.
    parser.add_argument(
        "--filter-vae-latents-path", type=Path, default=None,
        help="Filter-only override: pass a VAE latents .npy through "
             "to the dataset class so stations are filtered to those "
             "with non-NaN latents, regardless of whether the model "
             "consumes them. Used to re-evaluate trained no-TESSERA "
             "baselines on the matched (TESSERA ∩ VAE) station set "
             "without retraining. Ignored if the checkpoint already "
             "uses VAE latents (the trained filter is already strict).",
    )
    parser.add_argument(
        "--filter-vae-latents-station-csv", type=Path, default=None,
        help="CSV row-aligned with --filter-vae-latents-path. "
             "Required when --filter-vae-latents-path is set.",
    )
    parser.add_argument(
        "--dataset-dir", type=Path, default=None,
        help="Override the dataset directory to evaluate on. Default: the "
             "dataset_dir stored in the checkpoint's training config. Used to "
             "test a model on a different dataset (e.g. the Aurora-forecast "
             "datasets) than it was trained on.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Override where test_summary.json / test_results.json are written. "
             "Default: the checkpoint's parent directory. Use a distinct dir "
             "per test dataset so multiple evals of one checkpoint don't clobber "
             "each other (e.g. eval_era5/, eval_aurora_lead6h/).",
    )
    parser.add_argument(
        "--lead-hours", type=int, default=None,
        help="Forecast lead (hours) of the context grid being evaluated, for a "
             "lead-conditioned (cross-lead) checkpoint: 0 = ERA5 analysis, "
             "6/24/72 = the matching Aurora-forecast dataset. Sets the lead "
             "channel on the test dataset AND the model's input-channel count "
             "so they match the channel the model was trained with; pass the "
             "lead that corresponds to --dataset-dir (e.g. --dataset-dir "
             "…aurora_lead24h --lead-hours 24). Omit for a single-lead "
             "checkpoint (no lead channel).",
    )
    parser.add_argument(
        "--station-split", type=str, default="test",
        choices=["test", "train", "all"],
        help="Which spatial station split to evaluate on. Default 'test' "
             "= the held-out stations the model never saw (the standard "
             "generalisation metric). 'train' = the training stations "
             "(evaluated at the held-out TIME split, i.e. train stations @ "
             "held-out years — measures whether the model's edge also holds "
             "on locations it was fitted at). 'all' = every station. Combine "
             "with --output-dir (e.g. eval_train_stations/) so the resulting "
             "test_summary.json does not clobber the held-out one. Only "
             "affects the flat-snapshot / daily / legacy-multi-region paths; "
             "for a --region-specs-test run the split is carried per-region "
             "by the spec, so a non-default value here is rejected.",
    )
    args = parser.parse_args()

    if (args.filter_vae_latents_path is not None
            and args.filter_vae_latents_station_csv is None):
        parser.error(
            "--filter-vae-latents-station-csv is required when "
            "--filter-vae-latents-path is set."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load config from checkpoint.
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = ckpt.get("config", {})
    dataset_dir = Path(config.get("dataset_dir", ".tmp_output/dataset_daily"))
    if args.dataset_dir is not None:
        dataset_dir = args.dataset_dir
        logger.info(f"Dataset dir overridden to {dataset_dir}")
    # Context channels the model was trained with dropped (e.g. precipitation
    # for a 19-channel Aurora-compatible model). Resolved lenient at eval:
    # absent on a native-reduced dataset (Aurora has no precip) -> skipped.
    drop_context_channels = config.get("drop_context_channels", None)
    tessera_method = config.get("tessera_method", None)
    tessera_path = config.get("tessera_path", None)
    tessera_station_csv = config.get("tessera_station_csv", None)
    tessera_drop_prob = float(config.get("tessera_drop_prob", 0.0))
    include_elevation = config.get("include_elevation", True)
    include_static_fields = config.get("include_static_fields", True)
    # Per-station feature count the checkpoint was trained with. Defaults to 2
    # (elevation, delta_elevation) for pre-mTPI runs whose config.json predates
    # this key; 3 adds mTPI. Must match training so the decoder MLP first-layer
    # shape lines up with the saved weights.
    n_elev_features = int(config.get("n_elev_features", 2))

    # Pre-computed VAE latents (alternative to end-to-end encoder).
    vae_latents_path = config.get("vae_latents_path", None)
    vae_latents_station_csv = config.get("vae_latents_station_csv", None)
    vae_latents_drop_prob = float(config.get("vae_latents_drop_prob", 0.0))
    vae_latents_no_zscore = bool(config.get("vae_latents_no_zscore", False))
    vae_latents_proj_dim = int(config.get("vae_latents_proj_dim", 0))
    # Older checkpoints won't have vae_latents_proj_mlp in their saved
    # config; default to False so they still load.
    vae_latents_proj_mlp = bool(config.get("vae_latents_proj_mlp", False))
    # Same for normalisation_policy — introduced later, default to
    # 'per_region' to preserve the old behaviour for old checkpoints.
    normalisation_policy = config.get("normalisation_policy", "per_region")
    # min_tessera_patch_coverage is also a later addition. Default to 0.0
    # for old checkpoints so their evaluation reproduces what they
    # were trained against (centre-pixel-only filter). New checkpoints
    # will have the value persisted in their config; we only need to
    # support the override flag below, no CLI value to thread in here.
    min_patch_coverage = float(config.get("min_tessera_patch_coverage", 0.0))
    uses_vae_latents = vae_latents_path is not None and str(vae_latents_path) != "None"
    if uses_vae_latents:
        vae_latents_path = Path(vae_latents_path)
        vae_latents_station_csv = Path(vae_latents_station_csv)
    else:
        vae_latents_path = None
        vae_latents_station_csv = None

    # Hand-crafted extra descriptors (train.py --extra-descriptors-path)
    # ride the same precomputed-vector pathway as VAE latents; translate
    # their config keys onto the vae_latents_* locals, which act as the
    # generic precomputed-vector variables from here on.
    extra_descriptors_path = config.get("extra_descriptors_path", None)
    uses_extra_descriptors = (
        extra_descriptors_path is not None
        and str(extra_descriptors_path) != "None"
    )
    if uses_extra_descriptors:
        if uses_vae_latents:
            # Combined run: training concatenated both sources into a
            # derived cache file (see train.py resolve_combined_vectors)
            # and recorded its path. The vae_latents_* station csv /
            # drop / zscore settings already apply to the combined vector
            # (training enforces they match).
            merged = config.get("precomputed_merged_path", None)
            if merged is None or str(merged) == "None":
                lp, dp = Path(str(vae_latents_path)), Path(str(extra_descriptors_path))
                merged = lp.with_name(f"{lp.stem}_plus_{dp.stem}.npy")
            vae_latents_path = Path(merged)
        else:
            vae_latents_path = Path(extra_descriptors_path)
            vae_latents_station_csv = Path(
                config["extra_descriptors_station_csv"]
            )
            vae_latents_drop_prob = float(
                config.get("extra_descriptors_drop_prob", 0.0)
            )
            # Descriptors are always z-scored (no opt-out flag exists), so
            # vae_latents_no_zscore keeps its default False here.
            uses_vae_latents = True

    # Handle target variables: new format (list) or legacy (single string).
    target_variables = config.get("target_variables", None)
    if target_variables is None:
        target_variable = config.get("target_variable", "tmax")
        target_variables = [target_variable]
    n_target_variables = len(target_variables)
    is_multitask = n_target_variables > 1

    # Resolve likelihood_per_variable from config. Legacy configs pre-date
    # the per-variable heads abstraction and were implicitly all-Gaussian;
    # defaulting to that here makes pre-v4 checkpoints (after migration)
    # evaluable without manual config edits.
    likelihood_per_variable = config.get("likelihood_per_variable", None)
    if likelihood_per_variable is None:
        likelihood_per_variable = {var: "gaussian" for var in target_variables}
        logger.info(
            "config.json has no likelihood_per_variable field; assuming "
            "all-Gaussian (legacy default)."
        )
    else:
        # Sanity-check: spec must be 1:1 with target_variables.
        if set(likelihood_per_variable) != set(target_variables):
            raise ValueError(
                f"likelihood_per_variable in config does not match "
                f"target_variables. target_variables={target_variables}, "
                f"likelihood keys={list(likelihood_per_variable)}."
            )

    # Reject hypernet checkpoints — these can't be evaluated under the new
    # heads abstraction (the hypernet generated the entire translator
    # including its output layer, leaving no static weight tensor to split
    # into per-head projections). The migration script also refuses these.
    if config.get("tessera_injection") == "hypernet":
        raise NotImplementedError(
            "This checkpoint was trained with tessera_injection='hypernet', "
            "which is no longer supported. The new code path has no place "
            "for the per-station-generated translator weights — there is "
            "no analogous concept in the per-variable heads abstraction. "
            "If you need to revisit hypernet results, check out the "
            "pre-v4 commit; otherwise discard this checkpoint."
        )

    uses_tessera_features = tessera_method is not None

    logger.info(f"Checkpoint: epoch {ckpt['epoch']}, Val Loss: {ckpt['val_loss']:.4f}")
    if uses_extra_descriptors and str(config.get("vae_latents_path")) != "None" \
            and config.get("vae_latents_path") is not None:
        mode = f"VAE-latent + extra-descriptors ({Path(vae_latents_path).name})"
    elif uses_extra_descriptors:
        mode = f"Extra-descriptors ({Path(vae_latents_path).name})"
    elif uses_vae_latents:
        mode = f"VAE-latent ({Path(vae_latents_path).name})"
    elif uses_tessera_features:
        mode = f"TESSERA {tessera_method}"
    else:
        mode = "Baseline"
    logger.info(f"Mode: {mode}")
    logger.info(f"Target variables: {target_variables}")
    logger.info(f"Include elevation: {include_elevation}")
    if uses_tessera_features:
        logger.info(f"TESSERA drop_prob: {tessera_drop_prob}")
    if uses_vae_latents:
        logger.info(f"VAE latent drop_prob: {vae_latents_drop_prob}")

    # ---------------------------------------------------------------
    # Build model
    # ---------------------------------------------------------------
    from tessera_downscaling.model.convcnp import ConvCNPDownscaler

    with open(dataset_dir / "metadata.json") as f:
        meta = json.load(f)

    # Detect layout from metadata once; used for static-channel resolution,
    # dataset-class dispatch, and the time-channel count in n_context_channels.
    # Four layouts supported:
    #   * multi_region_v1           — daily, layered.
    #   * multi_region_snapshot_v1  — snapshot, layered.
    #   * snapshot_v1               — snapshot, flat.
    #   * (anything else)           — daily, flat (legacy default).
    layout_version = meta.get("layout_version")
    is_multi_region_daily = (
        layout_version == "multi_region_v1"
        and (dataset_dir / "regions").is_dir()
    )
    is_multi_region_snapshot = (
        layout_version == "multi_region_snapshot_v1"
        and (dataset_dir / "regions").is_dir()
    )
    is_multi_region = is_multi_region_daily or is_multi_region_snapshot
    is_snapshot = layout_version == "snapshot_v1" or is_multi_region_snapshot

    if include_static_fields:
        if is_multi_region:
            # Resolve which region we'll evaluate against. Same fallback
            # chain as the dataset construction below: CLI flag, then
            # region_specs-test (new), then training-config val_regions,
            # then train_regions, then first available.
            _eval_region = (
                args.test_regions
                or (list(_parse_region_specs_from_args(args).keys())
                    if (args.region_specs_test or args.region_specs_test_file) else None)
                or config.get("val_regions")
                or config.get("train_regions")
            )
            if isinstance(_eval_region, str):
                _eval_region = [_eval_region]
            if _eval_region is None:
                _eval_region = [next(iter(meta["regions"]))]
            n_static = meta["regions"][_eval_region[0]]["n_static_channels"]
        else:
            n_static = meta["n_static_channels"]
    else:
        n_static = 0

    # Daily layouts have 2 time channels (cos/sin day-of-year) + lat + lon.
    # Snapshot layouts add cos/sin hour-of-day, bringing the count to 4 + 2.
    n_time_and_coord_channels = 6 if is_snapshot else 4
    # Reduce by any context channels dropped at train time. Lenient: a name not
    # present in this dataset's channels is skipped (e.g. precip is already
    # absent from a native-19-channel Aurora dataset), so the same 19-channel
    # checkpoint's n_channels matches whether we eval on ERA5 or on Aurora.
    from tessera_downscaling.data.helpers import resolve_drop_channel_indices
    _drop_indices = resolve_drop_channel_indices(
        drop_context_channels, meta.get("era5_dynamic_channels", []) or [],
        strict=False, logger=logger,
    )
    # +1 for the lead-time channel on a cross-lead checkpoint. build_context_grid
    # appends it as the last (unnormalised) channel when lead_hours is set, and
    # train.py sized the model's input conv to include it — so the rebuilt model
    # here must match, or load_state_dict fails on the first conv. Mirrors the
    # lead_hours threaded into the dataset kwargs below.
    n_lead_channels = 1 if args.lead_hours is not None else 0
    n_channels = (
        meta["n_dynamic_channels"] - len(_drop_indices)
        + n_static + n_time_and_coord_channels + n_lead_channels
    )

    tessera_encoder = None
    if uses_tessera_features:
        from tessera_downscaling.model.tessera_encoder import TesseraPatchEncoder
        tessera_encoder = TesseraPatchEncoder(
            embed_dim=128,
            output_dim=config.get("tessera_output_dim", 64),
            method=tessera_method,
            drop_prob=tessera_drop_prob,
        )

    # For VAE-latent checkpoints, read the latent dim straight from the file.
    precomputed_tessera_dim = 0
    if uses_vae_latents:
        import numpy as _np
        _lat_shape = _np.load(str(vae_latents_path), mmap_mode="r").shape
        precomputed_tessera_dim = int(_lat_shape[1])
        logger.info(f"VAE latent dim: {precomputed_tessera_dim}")

    model = ConvCNPDownscaler(
        n_context_channels=n_channels,
        cnn_hidden=config.get("cnn_hidden", 128),
        cnn_layers=config.get("cnn_layers", 7),
        cnn_kernel=config.get("cnn_kernel", 3),
        setconv_length_scale=config.get("setconv_length_scale", 0.5),
        interpolation=config.get("interpolation", "setconv"),
        mlp_hidden=config.get("mlp_hidden", 128),
        mlp_n_hidden=config.get("mlp_n_hidden", 3),
        n_elev_features=n_elev_features,
        include_elevation=include_elevation,
        target_variables=target_variables,
        likelihood_per_variable=likelihood_per_variable,
        tessera_encoder=tessera_encoder,
        tessera_injection=config.get("tessera_injection", "concat"),
        tessera_features_precomputed=uses_vae_latents,
        precomputed_tessera_dim=precomputed_tessera_dim,
        precomputed_drop_prob=vae_latents_drop_prob,
        precomputed_proj_dim=vae_latents_proj_dim,
        precomputed_proj_mlp=vae_latents_proj_mlp,
        # New embedding-aware mechanisms. Defaults preserve baseline
        # behaviour so checkpoints saved before these flags existed
        # (i.e. config.json without these keys) evaluate identically
        # to how they trained.
        decoder_kernel=config.get("decoder_kernel", "isotropic"),
        use_target_embed_stream=config.get("use_target_embed_stream", False),
        target_embed_attention=config.get("target_embed_attention", "none"),
    ).to(device)

    # Migrate old checkpoint key names (setconv → interp).
    state_dict = ckpt["model_state_dict"]
    migrated = {}
    for k, v in state_dict.items():
        new_k = k.replace("setconv.", "interp.", 1) if k.startswith("setconv.") else k
        migrated[new_k] = v
    model.load_state_dict(migrated)
    model.eval()

    # Extract learned task weights if present (multi-task experiments).
    log_task_weights = ckpt.get("log_task_weights", None)
    if log_task_weights is not None:
        import math
        logger.info("Learned task weights from checkpoint:")
        for vi, var in enumerate(target_variables):
            if vi < len(log_task_weights):
                w = log_task_weights[vi].item()
                sigma = math.exp(w)
                weight_on_loss = math.exp(-2 * w) / 2
                logger.info(
                    f"  {var}: log_w={w:.4f}, σ={sigma:.4f}, "
                    f"weight_on_loss={weight_on_loss:.4f}"
                )

    # ---------------------------------------------------------------
    # Load test data
    # ---------------------------------------------------------------
    from tessera_downscaling.data.dataset import (
        DailyDownscalingDataset,
        MultiRegionDownscalingDataset,
        MultiRegionSnapshotDownscalingDataset,
        SnapshotDownscalingDataset,
        downscaling_collate,
    )

    # Use is_multi_region_{daily,snapshot} / is_snapshot detected above.

    # Parse --region-specs-test if given. Fails loudly on malformed JSON.
    region_specs_test = _parse_region_specs_from_args(args)
    if region_specs_test and args.test_regions is not None:
        parser.error(
            "Pass either --region-specs-test OR --test-regions, not both."
        )
    # --station-split has no effect on the region_specs path (each region
    # carries its own split via the spec dict), so reject the combination
    # rather than silently ignore it.
    if region_specs_test and args.station_split != "test":
        parser.error(
            f"--station-split={args.station_split!r} cannot be combined with "
            f"--region-specs-test; encode the per-region split in the spec "
            f"dict instead (e.g. '{{\"europe\": \"train\"}}')."
        )

    if is_multi_region:
        # Pick the right class for the cadence.
        _MRClass = (
            MultiRegionSnapshotDownscalingDataset
            if is_multi_region_snapshot
            else MultiRegionDownscalingDataset
        )

        if region_specs_test:
            logger.info(
                f"Multi-region test dataset at {dataset_dir}; "
                f"region_specs={region_specs_test}; "
                f"normalisation_policy={normalisation_policy}"
            )
            mr_kwargs = dict(
                dataset_dir=dataset_dir,
                region_specs=region_specs_test,
                target_variables=target_variables,
                split="test",
                tessera_path=tessera_path,
                tessera_station_csv=tessera_station_csv,
                load_tessera_patches=uses_tessera_features,
                include_static_fields=include_static_fields,
                vae_latents_path=vae_latents_path,
                vae_latents_station_csv=vae_latents_station_csv,
                vae_latents_zscore=not vae_latents_no_zscore,
                min_patch_coverage=min_patch_coverage,
            )
            if is_multi_region_snapshot:
                mr_kwargs["normalisation_policy"] = normalisation_policy
                mr_kwargs["drop_context_channels"] = drop_context_channels
                mr_kwargs["drop_context_strict"] = False
                mr_kwargs["lead_hours"] = args.lead_hours
            test_dataset = _MRClass(**mr_kwargs)
        else:
            # Legacy path: resolve test regions from CLI flag or train config.
            test_regions = (
                args.test_regions
                or config.get("val_regions")
                or config.get("train_regions")
            )
            if test_regions is not None and isinstance(test_regions, str):
                test_regions = [test_regions]

            test_mr_kwargs = dict(
                dataset_dir=dataset_dir,
                regions=test_regions,
                target_variables=target_variables,
                split="test",
                station_split=args.station_split,
                tessera_path=tessera_path,
                tessera_station_csv=tessera_station_csv,
                load_tessera_patches=uses_tessera_features,
                include_static_fields=include_static_fields,
                vae_latents_path=vae_latents_path,
                vae_latents_station_csv=vae_latents_station_csv,
                vae_latents_zscore=not vae_latents_no_zscore,
                min_patch_coverage=min_patch_coverage,
            )
            if is_multi_region_snapshot:
                test_mr_kwargs["normalisation_policy"] = normalisation_policy
                test_mr_kwargs["drop_context_channels"] = drop_context_channels
                test_mr_kwargs["drop_context_strict"] = False
                test_mr_kwargs["lead_hours"] = args.lead_hours
            test_dataset = _MRClass(**test_mr_kwargs)
    elif is_snapshot:
        if args.test_regions is not None or region_specs_test:
            parser.error(
                f"Region flags were given but {dataset_dir} is a flat "
                f"snapshot (single-region) dataset."
            )
        logger.info(
            f"Snapshot test dataset at {dataset_dir}; split='test' "
            f"(held-out years), station_split='{args.station_split}'."
        )
        test_dataset = SnapshotDownscalingDataset(
            dataset_dir=dataset_dir,
            target_variables=target_variables,
            split="test",
            station_split=args.station_split,
            tessera_path=tessera_path,
            tessera_station_csv=tessera_station_csv,
            load_tessera_patches=uses_tessera_features,
            include_static_fields=include_static_fields,
            vae_latents_path=vae_latents_path,
            vae_latents_station_csv=vae_latents_station_csv,
            vae_latents_zscore=not vae_latents_no_zscore,
            min_patch_coverage=min_patch_coverage,
            drop_context_channels=drop_context_channels,
            drop_context_strict=False,
            lead_hours=args.lead_hours,
        )
    else:
        if args.test_regions is not None:
            parser.error(
                f"--test-regions was given but {dataset_dir} is not a "
                f"multi-region dataset."
            )
        test_dataset = DailyDownscalingDataset(
            dataset_dir=dataset_dir,
            target_variables=target_variables,
            split="test",
            # Single-region: default to the spatial test holdout, since train
            # and test stations live in the same region and we need the holdout
            # to measure generalisation to new locations. Overridable via
            # --station-split (e.g. 'train' for train-stations @ held-out years).
            station_split=args.station_split,
            tessera_path=tessera_path,
            tessera_station_csv=tessera_station_csv,
            load_tessera_patches=uses_tessera_features,
            include_static_fields=include_static_fields,
            vae_latents_path=vae_latents_path,
            vae_latents_station_csv=vae_latents_station_csv,
            vae_latents_zscore=not vae_latents_no_zscore,
            min_patch_coverage=min_patch_coverage,
        )

    num_workers = args.num_workers
    if uses_tessera_features and num_workers > 0:
        import os
        if tessera_path and os.path.getsize(tessera_path) / (1024 ** 3) > 5.0:
            logger.info("Auto-setting num_workers=0 for large TESSERA patches")
            num_workers = 0

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=downscaling_collate,
    )
    logger.info(f"Test set: {len(test_dataset)} days")

    # ---------------------------------------------------------------
    # Data-efficiency: per-station subset labels
    # ---------------------------------------------------------------
    # Triggered when ANY of the following are present:
    #
    #   (a) probe_active_from_file in saved config
    #       → temporal-axis experiment buckets:
    #         probe / always_on / spatial_test
    #
    #   (b) train_station_allowlist_file in saved config
    #       → station-count experiment buckets (K < Kfull):
    #         train_stations / train_pool_held_out / spatial_test
    #
    #   (c) --region-specs-test contains a region with spec="all"
    #       → generic "evaluate everywhere" buckets:
    #         train_stations / spatial_test
    #
    # (a) and (b) can compose if both files exist; in that case (a)'s
    # probe bucket takes precedence within spatial-train.
    # (c) is also redundant when (a) or (b) is set with region_specs=all
    # (the more common pattern) — the spec=all path here exists so that
    # K=Kfull runs (no allowlist) and any other "evaluate everywhere"
    # one-offs also produce a train_stations / spatial_test breakdown.
    subset_per_station: np.ndarray | None = None
    probe_file = config.get("probe_active_from_file")
    allowlist_file = config.get("train_station_allowlist_file")
    spec_all_present = any(spec == "all" for spec in region_specs_test.values())

    if probe_file or allowlist_file or spec_all_present:
        # Always need stations.csv to look up spatial_split.
        import pandas as _pd
        stations_csv = dataset_dir / "stations.csv"
        if not stations_csv.exists():
            logger.warning(
                f"{stations_csv} not found; can't resolve spatial_split "
                f"per station. Skipping per-subset breakdown."
            )
            split_of = None
        else:
            _df = _pd.read_csv(stations_csv)
            _df["station_id"] = _df["station_id"].astype(str)
            split_of = dict(zip(_df["station_id"], _df["spatial_split"]))

        if split_of is not None:
            probe_ids: set[str] = set()
            if probe_file:
                pf = Path(probe_file)
                if pf.exists():
                    try:
                        probe_map = json.loads(pf.read_text())
                        if isinstance(probe_map, dict):
                            probe_ids = set(str(k) for k in probe_map.keys())
                        else:
                            logger.warning(
                                f"probe_active_from_file JSON not a dict; "
                                f"skipping probe bucket."
                            )
                    except Exception as e:
                        logger.warning(
                            f"Failed to load probe_active_from_file {pf}: {e}"
                        )
                else:
                    logger.warning(
                        f"probe_active_from_file does not exist: {pf}"
                    )

            allowlist_ids: set[str] = set()
            if allowlist_file:
                af = Path(allowlist_file)
                if af.exists():
                    try:
                        payload = json.loads(af.read_text())
                        if isinstance(payload, dict) and "station_ids" in payload:
                            allowlist_ids = set(
                                str(s) for s in payload["station_ids"]
                            )
                        else:
                            logger.warning(
                                f"train_station_allowlist_file JSON missing "
                                f"'station_ids' key; skipping allowlist bucket."
                            )
                    except Exception as e:
                        logger.warning(
                            f"Failed to load train_station_allowlist_file "
                            f"{af}: {e}"
                        )
                else:
                    logger.warning(
                        f"train_station_allowlist_file does not exist: {af}"
                    )

            ds_ids = [str(s) for s in test_dataset.station_ids]
            subset_per_station = np.empty(len(ds_ids), dtype=object)
            for i, sid in enumerate(ds_ids):
                spl = split_of.get(sid)
                if probe_ids and sid in probe_ids:
                    # Probe stations take precedence (they're a subset of
                    # spatial-train but conceptually their own bucket).
                    subset_per_station[i] = "probe"
                elif spl == "test":
                    subset_per_station[i] = "spatial_test"
                elif spl == "train":
                    # Within the spatial-train pool — further sub-divide
                    # based on whichever extra info we have.
                    if allowlist_ids:
                        if sid in allowlist_ids:
                            subset_per_station[i] = "train_stations"
                        else:
                            subset_per_station[i] = "train_pool_held_out"
                    elif probe_ids:
                        # Temporal experiment: in train pool but not a
                        # probe → "always_on" (full training observations).
                        subset_per_station[i] = "always_on"
                    else:
                        # Generic case (c): no allowlist, no probe set.
                        # All spatial-train stations are "training stations".
                        subset_per_station[i] = "train_stations"
                else:
                    # Fallback: not found in stations.csv, or stations.csv
                    # has a spatial_split value other than "train"/"test".
                    subset_per_station[i] = "unmapped"
            subset_per_station = subset_per_station.astype(str)
            from collections import Counter as _Counter
            counts = _Counter(subset_per_station.tolist())
            logger.info(
                f"Per-station subset labels resolved: "
                f"probe={counts.get('probe', 0)}, "
                f"always_on={counts.get('always_on', 0)}, "
                f"train_stations={counts.get('train_stations', 0)}, "
                f"train_pool_held_out={counts.get('train_pool_held_out', 0)}, "
                f"spatial_test={counts.get('spatial_test', 0)}, "
                f"unmapped={counts.get('unmapped', 0)}"
            )

    # ---------------------------------------------------------------
    # Filter-only VAE-latent override (no model side-effects)
    # ---------------------------------------------------------------
    # If the user passed --filter-vae-latents-path AND the checkpoint
    # itself does not already use VAE latents, build a per-station
    # boolean mask aligned with test_dataset.station_ids. Predictions
    # at stations outside this mask are dropped at metric-aggregation
    # time. Used to re-evaluate the trained no-TESSERA baseline on the
    # same (TESSERA ∩ VAE-latent-valid) intersection that VAE-augmented
    # rows are evaluated on, without retraining or rebuilding datasets.
    filter_station_mask: np.ndarray | None = None
    if args.filter_vae_latents_path is not None:
        if uses_vae_latents:
            logger.info(
                "Ignoring --filter-vae-latents-path: this checkpoint "
                "already uses VAE latents, so its dataset filter is "
                "already the strict intersection."
            )
        else:
            import pandas as _pd
            _csv = _pd.read_csv(args.filter_vae_latents_station_csv)
            _csv["station_id"] = _csv["station_id"].astype(str)
            _latents = np.load(str(args.filter_vae_latents_path), mmap_mode="r")
            if _latents.shape[0] != len(_csv):
                raise ValueError(
                    f"Filter-only VAE latents shape {_latents.shape} doesn't "
                    f"match station CSV ({len(_csv)} rows)."
                )
            _valid = ~np.isnan(_latents).any(axis=1)
            _valid_ids = set(_csv.loc[_valid, "station_id"].values.astype(str))
            ds_ids = np.asarray([str(s) for s in test_dataset.station_ids])
            filter_station_mask = np.array(
                [sid in _valid_ids for sid in ds_ids], dtype=bool,
            )
            logger.info(
                f"Filter-only VAE-latent mask: "
                f"{int(filter_station_mask.sum())}/{len(filter_station_mask)} "
                f"test stations have a non-NaN latent. Predictions at the "
                f"other stations will be dropped before metric aggregation."
            )

    # ---------------------------------------------------------------
    # Inference
    # ---------------------------------------------------------------
    # Per-variable collection: parameters dict + target + station indices.
    # We collect each named distribution parameter as its own list — the
    # exact set of names depends on the head type (mu/log_var for Gaussian,
    # k/lam for Weibull, rho/alpha/beta for Bernoulli-Gamma).
    from tessera_downscaling.model.heads import GenerativeHead

    all_params_per_var: dict[str, dict[str, list[float]]] = {
        var: {p: [] for p in model.heads.heads[var].param_names}
        for var in target_variables
    }
    # The implicit GenerativeHead has no scalar params to serialise — its
    # only "param" is the full hidden vector per observation, which would
    # balloon memory and break the scalar-column npz contract. Instead we
    # compute its per-observation diagnostics inline below (each is
    # per-observation independent, so this is identical to recomputing them
    # globally) and store only those scalar arrays.
    generative_vars = {
        var for var in target_variables
        if isinstance(model.heads.heads[var], GenerativeHead)
    }
    gen_diag: dict[str, dict[str, list[float]]] = {
        var: {"crps": [], "cdf": [], "pit_cdf": [], "mean": [], "median": []}
        for var in generative_vars
    }
    all_targets: dict[str, list[float]] = {var: [] for var in target_variables}
    all_station_indices: dict[str, list[int]] = {
        var: [] for var in target_variables
    }
    season_errors: dict[str, dict[str, list[float]]] = {
        var: defaultdict(list) for var in target_variables
    }

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_loader, desc="Evaluating")):
            if batch is None:
                continue

            context_grid = batch["context_grid"].to(device)
            target_coords = batch["target_coords"].to(device)
            target_elev = batch["target_elev"].to(device)
            target_delta_elev = batch["target_delta_elev"].to(device)
            target_mtpi = batch.get("target_mtpi")
            if target_mtpi is not None:
                target_mtpi = target_mtpi.to(device)
            target_values = batch["target_values"].to(device)
            target_mask = batch["target_mask"].to(device)
            grid_lats = batch["grid_lats"].to(device)
            grid_lons = batch["grid_lons"].to(device)
            station_indices = batch["target_station_indices"]  # keep on CPU

            target_tessera = None
            if "target_tessera" in batch and (uses_tessera_features or uses_vae_latents):
                target_tessera = batch["target_tessera"].to(device)

            params_per_var = model(
                context_grid, grid_lats, grid_lons,
                target_coords, target_elev, target_delta_elev,
                target_mask, target_tessera,
                target_mtpi=target_mtpi,
            )

            for vi, var in enumerate(target_variables):
                head = model.heads.heads[var]
                params_var = params_per_var[var]   # dict of tensors
                if is_multitask:
                    target_var = target_values[:, :, vi]
                else:
                    target_var = target_values

                for b in range(target_var.shape[0]):
                    mask = target_mask[b].bool()
                    if not mask.any():
                        continue

                    targets_b = target_var[b][mask].cpu().numpy()
                    mask_cpu = mask.cpu()
                    sindices = station_indices[b][mask_cpu].numpy()

                    # Apply filter-only VAE-latent mask if configured.
                    # Drops predictions whose station is outside the
                    # intersection set; preserves all other behaviour.
                    if filter_station_mask is not None:
                        keep = filter_station_mask[sindices]
                        if not keep.any():
                            continue
                        # Apply station filter to all per-parameter arrays
                        # plus the target and station indices.
                        targets_b = targets_b[keep]
                        sindices = sindices[keep]
                    else:
                        keep = None

                    if var in generative_vars:
                        # Implicit head: store per-observation scalar
                        # diagnostics computed from the live hidden state,
                        # not the hidden vector itself. masked + station-
                        # filtered slice keeps the work minimal.
                        hidden_b = params_var["hidden"][b][mask]
                        target_b_t = target_var[b][mask]
                        if keep is not None:
                            keep_t = torch.as_tensor(keep, device=hidden_b.device)
                            hidden_b = hidden_b[keep_t]
                            target_b_t = target_b_t[keep_t]
                        pb = {"hidden": hidden_b}
                        gen_diag[var]["crps"].extend(
                            head.crps(pb, target_b_t).cpu().numpy().tolist())
                        gen_diag[var]["cdf"].extend(
                            head.cdf(pb, target_b_t).cpu().numpy().tolist())
                        gen_diag[var]["pit_cdf"].extend(
                            head.pit_cdf(pb, target_b_t).cpu().numpy().tolist())
                        gen_diag[var]["mean"].extend(
                            head.mean(pb).cpu().numpy().tolist())
                        gen_diag[var]["median"].extend(
                            head.median(pb).cpu().numpy().tolist())
                    else:
                        # Collect per-parameter arrays (the exact param names
                        # come from the head's class attribute, so this is
                        # type-agnostic — Gaussian, Weibull, B-G all work).
                        for p_name in head.param_names:
                            p_b = params_var[p_name][b][mask].cpu().numpy()
                            if keep is not None:
                                p_b = p_b[keep]
                            all_params_per_var[var][p_name].extend(p_b.tolist())

                    all_targets[var].extend(targets_b.tolist())
                    all_station_indices[var].extend(sindices.tolist())

                    # Seasonal MAE breakdown — uses the head's predictive
                    # mean as the point estimate so it generalises across
                    # distributions (matches legacy Gaussian behaviour
                    # where mean = μ exactly).
                    point_est = head.mean(params_var)[b][mask].cpu().numpy()
                    if keep is not None:
                        point_est = point_est[keep]
                    errors_b = np.abs(point_est - targets_b)
                    date_idx = batch_idx * args.batch_size + b
                    if date_idx < len(test_dataset):
                        season = date_to_season(test_dataset.dates[date_idx])
                        season_errors[var][season].extend(errors_b.tolist())

            if (batch_idx + 1) % 50 == 0:
                # Log running MAE for first variable.
                first_var = target_variables[0]
                first_head = model.heads.heads[first_var]
                if first_var in generative_vars:
                    # Generative head stores per-obs predictive means directly
                    # (no scalar params to rebuild a distribution from).
                    means_so_far = gen_diag[first_var]["mean"]
                    targets_so_far = all_targets[first_var]
                    if len(targets_so_far) > 0:
                        running_mae = float(np.mean(np.abs(
                            np.array(means_so_far) - np.array(targets_so_far))))
                        logger.info(
                            f"  {batch_idx + 1} batches, "
                            f"{first_var} MAE: {running_mae:.3f}"
                        )
                else:
                    # Compute running MAE from collected params + targets via head.mean().
                    params_so_far = {
                        p: torch.tensor(all_params_per_var[first_var][p])
                        for p in first_head.param_names
                    }
                    targets_so_far = torch.tensor(all_targets[first_var])
                    if len(targets_so_far) > 0:
                        point_est = first_head.mean(params_so_far)
                        running_mae = (point_est - targets_so_far).abs().mean().item()
                        logger.info(
                            f"  {batch_idx + 1} batches, "
                            f"{first_var} MAE: {running_mae:.3f}"
                        )

    # ---------------------------------------------------------------
    # Report and save
    # ---------------------------------------------------------------
    output_dir = args.output_dir if args.output_dir is not None else args.checkpoint.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    test_summary = {
        "checkpoint_epoch": ckpt["epoch"],
        "best_val_loss": float(ckpt["val_loss"]),
        "target_variables": target_variables,
        "likelihood_per_variable": dict(likelihood_per_variable),
    }

    # Head-spec metadata: makes downstream consumers of test_predictions.npz
    # self-describing — they can dispatch on this without re-loading config.
    test_summary["head_spec"] = {
        var: {
            "distribution": likelihood_per_variable[var],
            "param_names": list(model.heads.heads[var].param_names),
        }
        for var in target_variables
    }

    # Include learned task weights if available.
    if log_task_weights is not None:
        for vi, var in enumerate(target_variables):
            if vi < len(log_task_weights):
                w = log_task_weights[vi].item()
                test_summary[f"{var}_learned_sigma"] = float(math.exp(w))
                test_summary[f"{var}_learned_log_weight"] = float(w)
                test_summary[f"{var}_learned_loss_weight"] = float(math.exp(-2 * w) / 2)

    # Per-variable arrays for saving to npz.
    npz_data = {}

    # Import head types for isinstance dispatch.
    from tessera_downscaling.model.heads import (
        GaussianHead, WeibullHead, BernoulliGammaHead, TruncatedNormalHead,
        GenerativeHead,
    )

    def _pit_chi2(cdf_values: np.ndarray, n_bins: int = 10) -> tuple[float, float]:
        """Chi-square goodness-of-fit test on PIT values.

        Under perfect calibration, PIT values are Uniform(0, 1). Bins
        them and tests with χ² against the uniform distribution.

        Returns ``(chi2_stat, p_value)``.
        """
        try:
            from scipy.stats import chisquare
        except ImportError:
            return float("nan"), float("nan")
        # Drop any NaN PIT values (can occur from numerical edge cases).
        cdf_clean = cdf_values[np.isfinite(cdf_values)]
        if len(cdf_clean) < n_bins * 5:  # need enough samples per bin
            return float("nan"), float("nan")
        observed, _ = np.histogram(cdf_clean, bins=n_bins, range=(0.0, 1.0))
        expected = np.full(n_bins, len(cdf_clean) / n_bins)
        chi2_stat, p_value = chisquare(observed, expected)
        return float(chi2_stat), float(p_value)

    for vi, var_name in enumerate(target_variables):
        head = model.heads.heads[var_name]
        is_generative = var_name in generative_vars
        targets_t = torch.tensor(all_targets[var_name])
        targets = targets_t.numpy()

        if is_generative:
            # Diagnostics were computed inline during inference (no scalar
            # params to rebuild a distribution from). NLL is undefined for an
            # implicit head (has_density=False), so it is skipped. PIT uses
            # every observation (a continuous implicit predictive — no point
            # mass to mask out, unlike Bernoulli-Gamma).
            params_t = {}
            nll_total = float("nan")
            crps = np.array(gen_diag[var_name]["crps"])
            cdf_vals = np.array(gen_diag[var_name]["cdf"])
            pit_cdf_vals = np.array(gen_diag[var_name]["pit_cdf"])
            pit_mask_t = None
        else:
            # Reconstruct tensors of params + targets for batch ops via head methods.
            params_t = {
                p: torch.tensor(all_params_per_var[var_name][p])
                for p in head.param_names
            }
            # Common: per-variable NLL (proper scoring rule, comparable across
            # distributions) and CRPS.
            with torch.no_grad():
                nll_total = head.nll(params_t, targets_t).item()
                crps = head.crps(params_t, targets_t).numpy()
                # cdf_vals is the unconditional CDF, used by some downstream
                # diagnostic outputs. PIT calibration uses pit_cdf instead —
                # for B-G that's the conditional Gamma CDF, dispatched below.
                cdf_vals = head.cdf(params_t, targets_t).numpy()
                pit_cdf_vals = head.pit_cdf(params_t, targets_t).numpy()
                pit_mask_t = head.pit_mask(targets_t)

        n_predictions = len(targets)
        n_test_stations = int(len(set(all_station_indices[var_name])))

        logger.info("=" * 60)
        logger.info(
            f"TEST RESULTS — {var_name} "
            f"({likelihood_per_variable[var_name]})"
        )
        logger.info("=" * 60)
        logger.info(f"  Predictions:  {n_predictions:,}")
        logger.info(f"  Stations:     {n_test_stations}")

        # Universal fields.
        test_summary[f"{var_name}_n_predictions"] = n_predictions
        test_summary[f"{var_name}_n_test_stations"] = n_test_stations
        test_summary[f"{var_name}_nll"] = float(nll_total)
        test_summary[f"{var_name}_crps"] = float(crps.mean())

        # PIT calibration test. For continuous heads (Gaussian, Weibull)
        # this uses every observation. For hurdle heads (Bernoulli-Gamma)
        # the head's pit_mask filters to wet observations and pit_cdf
        # uses the conditional Gamma CDF, matching Vaughan et al. (2022,
        # Fig. 10) — including the (1−ρ) point mass would pile PIT at 0
        # and inflate the χ² statistic regardless of fit quality.
        if pit_mask_t is not None:
            pit_mask_np = pit_mask_t.numpy().astype(bool)
            pit_inputs = pit_cdf_vals[pit_mask_np]
            n_pit = int(pit_mask_np.sum())
            test_summary[f"{var_name}_pit_n_used"] = n_pit
        else:
            pit_inputs = pit_cdf_vals
            n_pit = n_predictions

        if n_pit > 0:
            pit_chi2_stat, pit_chi2_p = _pit_chi2(pit_inputs)
        else:
            pit_chi2_stat, pit_chi2_p = float("nan"), float("nan")
        test_summary[f"{var_name}_pit_chi2_stat"] = pit_chi2_stat
        test_summary[f"{var_name}_pit_chi2_pvalue"] = pit_chi2_p

        logger.info(f"  NLL:          {nll_total:.4f}")
        logger.info(f"  CRPS:         {crps.mean():.4f}")
        if pit_mask_t is not None:
            logger.info(
                f"  PIT χ²:       stat={pit_chi2_stat:.3f}, "
                f"p-value={pit_chi2_p:.4f}  (wet-only, n={n_pit:,})"
            )
        else:
            logger.info(
                f"  PIT χ²:       stat={pit_chi2_stat:.3f}, "
                f"p-value={pit_chi2_p:.4f}"
            )

        # ----- Distribution-specific metric block -----
        if isinstance(head, GaussianHead):
            mu = params_t["mu"].numpy()
            log_var_clamped = np.clip(params_t["log_var"].numpy(), -10.0, 10.0)
            sigma = np.exp(0.5 * log_var_clamped)
            errors = np.abs(mu - targets)
            bias = float(np.mean(mu - targets))
            correlation = (
                float(np.corrcoef(mu, targets)[0, 1])
                if n_predictions > 1 else 0.0
            )
            within_1s = float((errors < sigma).mean() * 100)
            within_2s = float((errors < 2 * sigma).mean() * 100)

            test_summary[f"{var_name}_mae"] = float(errors.mean())
            test_summary[f"{var_name}_rmse"] = float(np.sqrt((errors ** 2).mean()))
            test_summary[f"{var_name}_bias"] = bias
            test_summary[f"{var_name}_correlation"] = correlation
            test_summary[f"{var_name}_mean_pred_std"] = float(sigma.mean())
            test_summary[f"{var_name}_within_1sigma"] = within_1s
            test_summary[f"{var_name}_within_2sigma"] = within_2s
            test_summary[f"{var_name}_p50"] = float(np.percentile(errors, 50))
            test_summary[f"{var_name}_p90"] = float(np.percentile(errors, 90))
            test_summary[f"{var_name}_p95"] = float(np.percentile(errors, 95))
            test_summary[f"{var_name}_p99"] = float(np.percentile(errors, 99))

            logger.info(f"  MAE:          {errors.mean():.3f}")
            logger.info(f"  RMSE:         {np.sqrt((errors ** 2).mean()):.3f}")
            logger.info(f"  Bias:         {bias:+.4f}")
            logger.info(f"  Correlation:  {correlation:.4f}")
            logger.info(f"  Mean pred σ:  {sigma.mean():.3f}")
            logger.info(f"  Within 1σ:    {within_1s:.1f}% (expected ~68.3%)")
            logger.info(f"  Within 2σ:    {within_2s:.1f}% (expected ~95.4%)")

        elif isinstance(head, WeibullHead):
            # MAE on the median (proper); RMSE/bias/correlation on the mean.
            with torch.no_grad():
                means = head.mean(params_t).numpy()
                medians = head.median(params_t).numpy()
            errors_at_mean = np.abs(means - targets)
            errors_at_median = np.abs(medians - targets)
            bias_at_mean = float(np.mean(means - targets))
            correlation_at_mean = (
                float(np.corrcoef(means, targets)[0, 1])
                if n_predictions > 1 else 0.0
            )

            test_summary[f"{var_name}_mae_at_median"] = float(errors_at_median.mean())
            test_summary[f"{var_name}_rmse_at_mean"] = float(
                np.sqrt((errors_at_mean ** 2).mean())
            )
            test_summary[f"{var_name}_bias_at_mean"] = bias_at_mean
            test_summary[f"{var_name}_correlation_at_mean"] = correlation_at_mean

            logger.info(f"  MAE @ median:    {errors_at_median.mean():.3f}")
            logger.info(f"  RMSE @ mean:     {np.sqrt((errors_at_mean**2).mean()):.3f}")
            logger.info(f"  Bias @ mean:     {bias_at_mean:+.4f}")
            logger.info(f"  Corr @ mean:     {correlation_at_mean:.4f}")

        elif isinstance(head, BernoulliGammaHead):
            # ρ ≥ 0.5 ⇒ predict wet. PoD/FaR/Brier on the binary wet/dry
            # decision; amount metrics on wet days only.
            rho = params_t["rho"].numpy()
            with torch.no_grad():
                means = head.mean(params_t).numpy()
                medians = head.median(params_t).numpy()
            wet_obs = (targets > 0)
            wet_pred = (rho >= 0.5)
            n_wet_obs = int(wet_obs.sum())
            n_dry_obs = int((~wet_obs).sum())

            # Binary classification metrics (PoD/FaR/Brier).
            tp = int((wet_pred & wet_obs).sum())
            fp = int((wet_pred & ~wet_obs).sum())
            fn = int((~wet_pred & wet_obs).sum())
            pod = float(tp / max(tp + fn, 1))    # hit rate
            far = float(fp / max(fp + tp, 1))    # false alarm ratio
            brier = float(np.mean((rho - wet_obs.astype(float)) ** 2))
            test_summary[f"{var_name}_pod"] = pod
            test_summary[f"{var_name}_far"] = far
            test_summary[f"{var_name}_brier"] = brier

            # Threshold-frequency metrics. ``R01`` (≥ 0.1mm) is the
            # standard wet-event-frequency threshold and translates
            # directly across cadences. ``R05`` (≥ 5mm) replaces the
            # daily-cadence ``R10`` (≥ 10mm) used in Vaughan et al.
            # (2022) — at the 6h cadence we use here, 10mm/6h is rare
            # enough that the metric is dominated by sampling noise,
            # whereas 5mm/6h captures moderate-to-heavy precip events
            # in temperate regimes (≈ 20mm/day equivalent).
            r01_obs = float((targets >= 0.1).mean())
            r01_pred = float((medians >= 0.1).mean())
            r05_obs = float((targets >= 5.0).mean())
            r05_pred = float((medians >= 5.0).mean())
            test_summary[f"{var_name}_r01_obs"] = r01_obs
            test_summary[f"{var_name}_r01_pred"] = r01_pred
            test_summary[f"{var_name}_r05_obs"] = r05_obs
            test_summary[f"{var_name}_r05_pred"] = r05_pred

            # Wet-day amount metrics (skewed, so MAE@median + RMSE@mean,
            # like Weibull).
            if n_wet_obs > 0:
                wet_means = means[wet_obs]
                wet_medians = medians[wet_obs]
                wet_targets = targets[wet_obs]
                wet_err_mean = wet_means - wet_targets
                wet_err_median = np.abs(wet_medians - wet_targets)

                test_summary[f"{var_name}_wet_mae_at_median"] = float(wet_err_median.mean())
                test_summary[f"{var_name}_wet_rmse_at_mean"] = float(
                    np.sqrt((wet_err_mean ** 2).mean())
                )
                test_summary[f"{var_name}_wet_bias_at_mean"] = float(wet_err_mean.mean())
                if n_wet_obs > 1:
                    test_summary[f"{var_name}_wet_correlation_at_mean"] = float(
                        np.corrcoef(wet_means, wet_targets)[0, 1]
                    )
                # P98 of wet-day amounts — heavy-tail bias indicator.
                p98_obs = float(np.percentile(wet_targets, 98))
                p98_pred = float(np.percentile(wet_means, 98))
                test_summary[f"{var_name}_p98_obs_wet_days"] = p98_obs
                test_summary[f"{var_name}_p98_pred_wet_days"] = p98_pred
                test_summary[f"{var_name}_p98_bias_wet_days"] = p98_pred - p98_obs

            logger.info(f"  Obs wet days: {n_wet_obs:,} / {n_predictions:,}  ({n_wet_obs/max(n_predictions,1)*100:.1f}%)")
            logger.info(f"  PoD:          {pod:.3f}")
            logger.info(f"  FaR:          {far:.3f}")
            logger.info(f"  Brier:        {brier:.4f}")
            logger.info(f"  R01 obs/pred: {r01_obs:.3f} / {r01_pred:.3f}")
            logger.info(f"  R05 obs/pred: {r05_obs:.3f} / {r05_pred:.3f}")
            if n_wet_obs > 0:
                logger.info(f"  Wet MAE@med:  {test_summary[f'{var_name}_wet_mae_at_median']:.3f}")
                logger.info(f"  Wet RMSE@mn:  {test_summary[f'{var_name}_wet_rmse_at_mean']:.3f}")

        elif isinstance(head, TruncatedNormalHead):
            # TruncNormal is right-skewed near calm (σ ~ μ), so we follow the
            # same proper-scoring split as the Weibull block: MAE on the
            # *median* (minimises expected absolute error) and RMSE / bias /
            # correlation on the *mean* (minimises expected squared error).
            # Field names mirror Weibull (_mae_at_median / _rmse_at_mean /
            # _bias_at_mean / _correlation_at_mean) so the two wind heads are
            # directly comparable. In the high-μ/σ regime typical for wind the
            # mean and median nearly coincide, so this only diverges from the
            # legacy mean-only numbers near calm — exactly where this head was
            # chosen to matter. The σ-coverage / percentile diagnostics keep
            # their distribution-agnostic names (unchanged).
            with torch.no_grad():
                means = head.mean(params_t).numpy()
                medians = head.median(params_t).numpy()
            log_var_clamped = np.clip(params_t["log_var"].numpy(), -10.0, 10.0)
            sigma = np.exp(0.5 * log_var_clamped)

            errors_at_mean = np.abs(means - targets)
            errors_at_median = np.abs(medians - targets)
            bias_at_mean = float(np.mean(means - targets))
            correlation_at_mean = (
                float(np.corrcoef(means, targets)[0, 1])
                if n_predictions > 1 else 0.0
            )
            # Calibration check using underlying σ as a proxy for the truncated
            # std. Exact within-Xσ expected fractions for TruncNormal depend
            # on μ/σ; PIT χ² (computed above, universally) is the proper
            # calibration diagnostic. Computed on median-centred errors so the
            # percentiles below describe the MAE-relevant error distribution.
            within_1s = float((errors_at_median < sigma).mean() * 100)
            within_2s = float((errors_at_median < 2 * sigma).mean() * 100)

            test_summary[f"{var_name}_mae_at_median"] = float(errors_at_median.mean())
            test_summary[f"{var_name}_rmse_at_mean"] = float(
                np.sqrt((errors_at_mean ** 2).mean())
            )
            test_summary[f"{var_name}_bias_at_mean"] = bias_at_mean
            test_summary[f"{var_name}_correlation_at_mean"] = correlation_at_mean
            test_summary[f"{var_name}_mean_pred_std"] = float(sigma.mean())
            test_summary[f"{var_name}_within_1sigma"] = within_1s
            test_summary[f"{var_name}_within_2sigma"] = within_2s
            test_summary[f"{var_name}_p50"] = float(np.percentile(errors_at_median, 50))
            test_summary[f"{var_name}_p90"] = float(np.percentile(errors_at_median, 90))
            test_summary[f"{var_name}_p95"] = float(np.percentile(errors_at_median, 95))
            test_summary[f"{var_name}_p99"] = float(np.percentile(errors_at_median, 99))

            logger.info(f"  MAE @ median:    {errors_at_median.mean():.3f}")
            logger.info(f"  RMSE @ mean:     {np.sqrt((errors_at_mean ** 2).mean()):.3f}")
            logger.info(f"  Bias @ mean:     {bias_at_mean:+.4f}")
            logger.info(f"  Corr @ mean:     {correlation_at_mean:.4f}")
            logger.info(f"  Mean pred σ:     {sigma.mean():.3f}")
            logger.info(f"  Within 1σ:       {within_1s:.1f}%")
            logger.info(f"  Within 2σ:       {within_2s:.1f}%")

        elif isinstance(head, GenerativeHead):
            # Implicit generative head — potentially skewed/heavy-tailed, so
            # we report the same proper-scoring split as the skewed parametric
            # heads: MAE on the *median* (minimises expected absolute error)
            # and RMSE / bias / correlation on the *mean* (minimises expected
            # squared error). Point estimates were computed inline during
            # inference (ensemble mean / median of the generator). PIT χ²
            # (computed above, universally) is the calibration diagnostic;
            # there is no σ here, so the within-Xσ coverage fields are omitted.
            means = np.array(gen_diag[var_name]["mean"])
            medians = np.array(gen_diag[var_name]["median"])
            errors_at_mean = np.abs(means - targets)
            errors_at_median = np.abs(medians - targets)
            bias_at_mean = float(np.mean(means - targets))
            correlation_at_mean = (
                float(np.corrcoef(means, targets)[0, 1])
                if n_predictions > 1 else 0.0
            )

            test_summary[f"{var_name}_mae_at_median"] = float(errors_at_median.mean())
            test_summary[f"{var_name}_rmse_at_mean"] = float(
                np.sqrt((errors_at_mean ** 2).mean())
            )
            test_summary[f"{var_name}_bias_at_mean"] = bias_at_mean
            test_summary[f"{var_name}_correlation_at_mean"] = correlation_at_mean
            test_summary[f"{var_name}_p50"] = float(np.percentile(errors_at_median, 50))
            test_summary[f"{var_name}_p90"] = float(np.percentile(errors_at_median, 90))
            test_summary[f"{var_name}_p95"] = float(np.percentile(errors_at_median, 95))
            test_summary[f"{var_name}_p99"] = float(np.percentile(errors_at_median, 99))

            logger.info(f"  MAE @ median:    {errors_at_median.mean():.3f}")
            logger.info(f"  RMSE @ mean:     {np.sqrt((errors_at_mean ** 2).mean()):.3f}")
            logger.info(f"  Bias @ mean:     {bias_at_mean:+.4f}")
            logger.info(f"  Corr @ mean:     {correlation_at_mean:.4f}")

        else:
            raise NotImplementedError(
                f"Metric dispatch not implemented for head type "
                f"{type(head).__name__} on variable {var_name!r}."
            )

        # Seasonal MAE (computed inline during inference, so we just
        # surface what's collected). Uses head.mean() as point estimate.
        logger.info(f"  --- Seasonal MAE ({var_name}, point estimate = predictive mean) ---")
        seasonal_mae = {}
        for season in ["DJF", "MAM", "JJA", "SON"]:
            if season in season_errors[var_name]:
                se = np.array(season_errors[var_name][season])
                seasonal_mae[season] = float(se.mean())
                logger.info(f"    {season}: {se.mean():.3f} (n={len(se):,})")
        test_summary[f"{var_name}_seasonal_mae"] = seasonal_mae

        # NPZ arrays — per-parameter, distribution-agnostic schema. The
        # implicit head has no scalar params; it saves the per-observation
        # diagnostics computed inline (crps / cdf / point estimates) instead.
        if is_generative:
            npz_data[f"{var_name}_crps"] = crps
            npz_data[f"{var_name}_cdf"] = cdf_vals
            npz_data[f"{var_name}_mean"] = np.array(gen_diag[var_name]["mean"])
            npz_data[f"{var_name}_median"] = np.array(gen_diag[var_name]["median"])
        else:
            for p_name in head.param_names:
                npz_data[f"{var_name}_param_{p_name}"] = np.array(
                    all_params_per_var[var_name][p_name]
                )
        npz_data[f"{var_name}_targets"] = targets
        npz_data[f"{var_name}_station_indices"] = np.array(
            all_station_indices[var_name], dtype=np.int64,
        )

    # ---------------------------------------------------------------
    # Per-station aggregation
    # ---------------------------------------------------------------
    logger.info("-" * 60)
    logger.info("PER-STATION ANALYSIS")
    logger.info("-" * 60)

    station_meta = {
        "station_ids": test_dataset.station_ids,
        "station_lats": test_dataset.station_lats,
        "station_lons": test_dataset.station_lons,
        "station_elevs": test_dataset.station_elevs,
        "station_delta_elevs": test_dataset.station_delta_elevs,
    }
    station_npz = dict(station_meta)

    for vi, var_name in enumerate(target_variables):
        head = model.heads.heads[var_name]
        # Point estimate is the predictive mean (matches Gaussian legacy
        # behaviour exactly for vars with isinstance(head, GaussianHead);
        # for skewed dists this is the per-distribution analogue). The
        # implicit head stored its per-obs ensemble mean inline.
        if var_name in generative_vars:
            point_est = np.array(gen_diag[var_name]["mean"])
        else:
            params_t = {
                p: torch.tensor(all_params_per_var[var_name][p])
                for p in head.param_names
            }
            with torch.no_grad():
                point_est = head.mean(params_t).numpy()
        targets = np.array(all_targets[var_name])
        sindices = np.array(all_station_indices[var_name], dtype=np.int64)
        errors = np.abs(point_est - targets)

        # Aggregate per station.
        unique_stations = np.unique(sindices)
        n_stations = len(unique_stations)
        station_mae = np.zeros(len(test_dataset.station_ids))
        station_rmse = np.zeros(len(test_dataset.station_ids))
        station_bias = np.zeros(len(test_dataset.station_ids))
        station_count = np.zeros(len(test_dataset.station_ids), dtype=np.int64)

        for si in unique_stations:
            mask = sindices == si
            station_errors = errors[mask]
            station_residuals = point_est[mask] - targets[mask]
            station_mae[si] = station_errors.mean()
            station_rmse[si] = np.sqrt((station_errors ** 2).mean())
            station_bias[si] = station_residuals.mean()
            station_count[si] = mask.sum()

        station_npz[f"{var_name}_station_mae"] = station_mae
        station_npz[f"{var_name}_station_rmse"] = station_rmse
        station_npz[f"{var_name}_station_bias"] = station_bias
        station_npz[f"{var_name}_station_count"] = station_count

        # Log tail analysis.
        valid_mask = station_count > 0
        valid_maes = station_mae[valid_mask]

        if len(valid_maes) > 0:
            logger.info(f"  --- {var_name} per-station MAE distribution ---")
            logger.info(f"  Stations: {len(valid_maes)}")
            logger.info(f"  Mean station MAE: {valid_maes.mean():.3f}")
            logger.info(f"  Median station MAE: {np.median(valid_maes):.3f}")
            logger.info(f"  Std station MAE: {valid_maes.std():.3f}")
            logger.info(f"  P10 (best):  {np.percentile(valid_maes, 10):.3f}")
            logger.info(f"  P90 (worst): {np.percentile(valid_maes, 90):.3f}")

            # Worst stations.
            worst_idx = np.argsort(valid_maes)[-5:][::-1]
            actual_indices = np.where(valid_mask)[0][worst_idx]
            logger.info(f"  Worst 5 stations:")
            for idx in actual_indices:
                sid = test_dataset.station_ids[idx]
                lat = test_dataset.station_lats[idx]
                lon = test_dataset.station_lons[idx]
                elev = test_dataset.station_elevs[idx]
                logger.info(
                    f"    {sid}: MAE={station_mae[idx]:.3f}, "
                    f"lat={lat:.2f}, lon={lon:.2f}, elev={elev:.0f}m, "
                    f"n={station_count[idx]}"
                )

    # ---------------------------------------------------------------
    # Per-subset breakdown (data-efficiency experiments)
    # ---------------------------------------------------------------
    # When `subset_per_station` was resolved at the top of this script,
    # add an extra block of {subset}_{metric} keys into test_summary so
    # the analysis notebook can read each subset's numbers directly
    # without recomputing from per-station arrays.
    #
    # The five subset names cover every data-efficiency experiment:
    #
    #   probe              — temporal experiment (probe stations)
    #   always_on          — temporal experiment (in-train non-probes)
    #   train_stations     — station-count experiment (in K-allowlist)
    #                        OR generic "evaluate everywhere" (full
    #                        spatial-train pool when no allowlist is
    #                        configured, e.g. K=Kfull runs)
    #   train_pool_held_out— station-count experiment (in spatial-train
    #                        pool but excluded by this run's allowlist)
    #   spatial_test       — the held-out 15% spatial-test stations
    #
    # For any given run, only some of these will have non-zero
    # n_stations (e.g. a station-count K<Kfull run has train_stations
    # + train_pool_held_out + spatial_test; a temporal run has probe +
    # always_on + spatial_test). Empty subsets are still emitted with
    # n_stations=0 / n_predictions=0 so downstream consumers can do a
    # straightforward dict lookup without contains-checks.
    #
    # The breakdown uses the same point-estimate-of-mean basis as the
    # existing top-level metrics: a per-station MAE/RMSE/bias mean +
    # weighted-mean MAE per variable. NLL is intentionally NOT broken
    # down here — it's per-observation rather than per-station, so a
    # per-station mean would mis-weight it; the per-subset NLL is left
    # for the notebook to compute from the saved (params, targets,
    # station_indices) arrays if needed.
    if subset_per_station is not None:
        station_npz["subset_per_station"] = subset_per_station
        subset_names = [
            "probe",
            "always_on",
            "train_stations",
            "train_pool_held_out",
            "spatial_test",
        ]
        unmapped_count = int(np.sum(subset_per_station == "unmapped"))
        if unmapped_count:
            subset_names.append("unmapped")
        logger.info("-" * 60)
        logger.info("PER-SUBSET METRICS (data-efficiency breakdown)")
        logger.info("-" * 60)
        for var_name in target_variables:
            station_mae_var = station_npz[f"{var_name}_station_mae"]
            station_rmse_var = station_npz[f"{var_name}_station_rmse"]
            station_bias_var = station_npz[f"{var_name}_station_bias"]
            station_count_var = station_npz[f"{var_name}_station_count"]
            valid_mask_var = station_count_var > 0
            for subset in subset_names:
                in_subset = (subset_per_station == subset) & valid_mask_var
                n_stations_in_subset = int(in_subset.sum())
                if n_stations_in_subset == 0:
                    # Persist explicit zeros so the notebook can detect
                    # the "no stations in subset" case without an extra
                    # contains-check on the json dict.
                    test_summary[f"{var_name}_{subset}_n_stations"] = 0
                    test_summary[f"{var_name}_{subset}_n_predictions"] = 0
                    continue
                n_predictions_in_subset = int(station_count_var[in_subset].sum())
                # Two MAE/RMSE numbers per subset:
                #   - station_mean: unweighted mean over the subset's
                #     stations, matching the "macro" framing where each
                #     station counts equally.
                #   - weighted: weighted by station_count, matching the
                #     "micro" framing where each observation counts
                #     equally (same basis as the top-level <var>_mae).
                station_macro_mae = float(station_mae_var[in_subset].mean())
                station_macro_rmse = float(station_rmse_var[in_subset].mean())
                station_macro_bias = float(station_bias_var[in_subset].mean())
                weights = station_count_var[in_subset].astype(np.float64)
                weighted_mae = float(
                    np.average(station_mae_var[in_subset], weights=weights)
                )
                # Weighted RMSE: average squared station-RMSE across the
                # subset weighted by count, then sqrt. Approximates the
                # all-observation RMSE in the subset (exact would need
                # the raw squared residuals; this is the per-station
                # cache's best approximation).
                weighted_rmse = float(np.sqrt(
                    np.average(station_rmse_var[in_subset] ** 2, weights=weights)
                ))
                weighted_bias = float(
                    np.average(station_bias_var[in_subset], weights=weights)
                )
                test_summary[f"{var_name}_{subset}_n_stations"] = n_stations_in_subset
                test_summary[f"{var_name}_{subset}_n_predictions"] = n_predictions_in_subset
                test_summary[f"{var_name}_{subset}_mae_macro"] = station_macro_mae
                test_summary[f"{var_name}_{subset}_rmse_macro"] = station_macro_rmse
                test_summary[f"{var_name}_{subset}_bias_macro"] = station_macro_bias
                test_summary[f"{var_name}_{subset}_mae"] = weighted_mae
                test_summary[f"{var_name}_{subset}_rmse"] = weighted_rmse
                test_summary[f"{var_name}_{subset}_bias"] = weighted_bias
                logger.info(
                    f"  {var_name:8s} {subset:20s} "
                    f"n_stations={n_stations_in_subset:4d} "
                    f"n_predictions={n_predictions_in_subset:7d} "
                    f"MAE={weighted_mae:.4f} (macro={station_macro_mae:.4f}) "
                    f"RMSE={weighted_rmse:.4f}"
                )

    # Save results.
    with open(output_dir / "test_results.json", "w") as f:
        json.dump(test_summary, f, indent=2)

    # Also save as test_summary.json for submit_parallel.sh skip detection.
    with open(output_dir / "test_summary.json", "w") as f:
        json.dump(test_summary, f, indent=2)

    np.savez(output_dir / "test_predictions.npz", **npz_data)
    np.savez(output_dir / "test_station_errors.npz", **station_npz)

    logger.info(f"\nSaved to {output_dir}")
    logger.info(f"  test_predictions.npz: raw predictions per observation")
    logger.info(f"  test_station_errors.npz: per-station aggregated errors + metadata")


if __name__ == "__main__":
    main()