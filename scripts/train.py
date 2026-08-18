"""Train the ConvCNP downscaler (baseline or TESSERA-augmented).

A single script handles both variants and supports single-task or multi-task:

  --tessera-path / --tessera-station-csv:
      Provided for ALL runs (baseline and TESSERA alike). Filters stations
      to those with valid TESSERA patches so both models train on the same
      station set.

  --tessera-method:
      Omitted  -> baseline (no TESSERA features, patches not loaded).
      Provided -> TESSERA model. Patches read on-the-fly via mmap and
      encoded in chunks on the GPU.

  --target-variables:
      One or more of: tmax, wind_mean. When multiple are given, the model
      predicts all variables jointly (multi-task) with learned task weights.

Usage (baseline on TESSERA-filtered stations):
    python scripts/train.py \\
        --dataset-dir .tmp_output/dataset_daily \\
        --tessera-path .tmp_output/processed/tessera/patch_embeddings_2024.npy \\
        --tessera-station-csv .tmp_output/processed/tessera/station_list_filtered.csv

Usage (TESSERA with embedding dropout, no elevation, multi-task):
    python scripts/train.py \\
        --dataset-dir .tmp_output/dataset_daily \\
        --tessera-path .tmp_output/processed/tessera/patch16_embeddings_2024.npy \\
        --tessera-station-csv .tmp_output/processed/tessera/station_list_filtered.csv \\
        --tessera-method cnn \\
        --tessera-output-dim 16 \\
        --tessera-drop-prob 0.3 \\
        --no-elevation \\
        --target-variables tmax wind_mean
"""

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
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("train")


def build_output_dir_name(args: argparse.Namespace) -> str:
    """Generate a descriptive output directory name from the config."""
    parts = []
    if args.vae_latents_path is not None:
        parts.append("vae_latents")
        parts.append(Path(args.vae_latents_path).stem)
        if args.extra_descriptors_path is not None:
            parts.append("plus")
            parts.append(Path(args.extra_descriptors_path).stem)
        parts.append(f"inj_{args.tessera_injection}")
        if args.vae_latents_drop_prob > 0:
            parts.append(f"drop{args.vae_latents_drop_prob}")
    elif args.extra_descriptors_path is not None:
        parts.append(Path(args.extra_descriptors_path).stem)
        parts.append(f"inj_{args.tessera_injection}")
        if args.extra_descriptors_drop_prob > 0:
            parts.append(f"drop{args.extra_descriptors_drop_prob}")
    elif args.tessera_method:
        parts.append(f"tessera_{args.tessera_method}")
        if args.tessera_method != "meanpool":
            parts.append(f"dim{args.tessera_output_dim}")
        if args.tessera_drop_prob > 0:
            parts.append(f"drop{args.tessera_drop_prob}")
    elif args.tessera_path:
        parts.append("baseline_tessera_stations")
    else:
        parts.append("baseline")
    if not args.include_elevation:
        parts.append("no_elev")
    # Target variables.
    parts.append("_".join(args.target_variables))
    parts.append(f"cnn{args.cnn_hidden}x{args.cnn_layers}")
    parts.append(f"mlp{args.mlp_hidden}x{args.mlp_n_hidden}")
    parts.append(f"lr{args.lr}")
    parts.append(f"bs{args.batch_size}")
    parts.append(f"seed{args.seed}")
    return "_".join(parts)


def resolve_combined_vectors(latents_path: Path, descriptors_path: Path) -> Path:
    """Return (building if absent) the hstacked latents+descriptors ``.npy``.

    Used when ``--vae-latents-path`` and ``--extra-descriptors-path`` are
    given together: the dataset serves ONE precomputed vector per station, so
    the two sources are concatenated into a derived cache file — a
    deterministic sibling of the latents file — and that file is what flows
    into the dataset. NaN-row filtering composes into the intersection of
    both sources for free (the loader drops any-NaN rows).

    The write is tmp+rename so concurrent jobs cannot leave a torn file, but
    the z-score stats cache is still built lazily on first use — pre-build
    both with scripts/preprocessing/concat_station_vectors.py before
    submitting a batch of jobs.
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
    os.replace(tmp, merged)
    logger.info(f"Built combined vector file {merged} shape={combined.shape}")
    return merged


def compute_loss_via_heads(
    params_per_var: dict,
    target_values: torch.Tensor,
    target_mask: torch.Tensor,
    target_variables: list,
    heads,
    is_multitask: bool,
    log_task_weights: torch.Tensor | None,
    loss_function: str = "nll",
) -> tuple[torch.Tensor, dict]:
    """Aggregate per-variable loss via the heads dispatcher.

    Replaces the legacy ``multitask_nll_loss`` / ``gaussian_nll_loss``
    pair: each variable's head computes its own loss on its own
    parameters, and Kendall weighting is applied at the aggregation
    layer (so it works for any mix of distributions, not just Gaussian).

    Two loss functions are supported:

    * ``"nll"`` — negative log-likelihood (the standard ML objective for
      probabilistic regression; ``head.nll`` handles per-distribution
      details and applies the target mask internally).
    * ``"crps"`` — continuous ranked probability score, a strictly proper
      scoring rule with closed-form expressions for Gaussian and
      truncated-normal heads. CRPS is bounded for any single observation
      (NLL is not), so it tends to produce sharper, better-calibrated
      forecasts in finite samples — at a typically-small cost to the
      point-estimate MAE. For CRPS, ``head.crps`` returns per-element
      values; the target mask is applied here (means over the masked
      elements, same convention as ``head.nll``).

    Args:
        params_per_var: ``{var: {param_name: tensor}}`` from
            ``model(...)``.
        target_values: Shape ``(B, N)`` for single-task or ``(B, N, V)``
            for multi-task.
        target_mask: ``(B, N)`` bool — shared across variables.
        target_variables: Ordered list, defines variable indexing.
        heads: ``LikelihoodHeadDict`` (typically ``model.heads``).
        is_multitask: Whether ``target_values`` has the per-variable
            third dimension.
        log_task_weights: Per-variable Kendall weights, or None for
            unweighted sum.
        loss_function: ``"nll"`` (default) or ``"crps"``.

    Returns:
        ``(total_loss, per_var_losses_dict)``. The per-variable dict is
        useful for logging / monitoring even though only the total is
        backpropagated.
    """
    target_mask_bool = target_mask.bool()
    per_var_losses = {}
    for vi, var in enumerate(target_variables):
        target_var = (
            target_values[:, :, vi] if is_multitask else target_values
        )
        head = heads.heads[var]
        if loss_function == "nll":
            per_var_losses[var] = head.nll(
                params_per_var[var], target_var, mask=target_mask_bool,
            )
        elif loss_function == "crps":
            # head.crps returns per-element CRPS (no internal masking);
            # apply the target mask to match the head.nll convention of
            # returning a scalar mean over masked elements.
            crps_elementwise = head.crps(params_per_var[var], target_var)
            mask_f = target_mask_bool.to(crps_elementwise.dtype)
            per_var_losses[var] = (
                (crps_elementwise * mask_f).sum() / mask_f.sum().clamp(min=1.0)
            )
        else:
            raise ValueError(
                f"Unknown loss_function {loss_function!r}; expected 'nll' or 'crps'."
            )

    if log_task_weights is not None:
        # Kendall et al. 2018: nll_i * exp(-2 w_i) / 2 + w_i
        total_loss = sum(
            per_var_losses[var] * torch.exp(-2 * log_task_weights[vi]) / 2
            + log_task_weights[vi]
            for vi, var in enumerate(target_variables)
        )
    else:
        total_loss = sum(per_var_losses.values())
    return total_loss, per_var_losses


def build_region_balanced_sampler(dataset, logger):
    """A WeightedRandomSampler that draws episodes uniformly per region.

    Multi-region datasets concatenate per-region episode lists in
    ``region_order`` and expose cumulative lengths in ``_cum_lengths``; a flat
    index ``idx`` belongs to the region whose cumulative length first exceeds
    it. Plain ``shuffle=True`` therefore samples proportionally to each
    region's episode count, letting a station-dense / long-window region
    dominate. We instead weight every episode by ``1 / (episodes in its
    region)`` so each region's total sampling mass is equal — the §3.2.2
    "region random per episode" mechanism.

    Returns ``None`` (caller falls back to shuffling) when ``dataset`` is not a
    multi-region dataset or has a single region, where balancing is a no-op.
    """
    region_order = getattr(dataset, "region_order", None)
    cum_lengths = getattr(dataset, "_cum_lengths", None)
    if not region_order or not cum_lengths or len(region_order) < 2:
        return None

    # Per-region episode counts from consecutive cumulative lengths.
    counts, prev = [], 0
    for c in cum_lengths:
        counts.append(c - prev)
        prev = c

    logger.info("Region-balanced sampling — per-region episode counts:")
    for name, n in zip(region_order, counts):
        logger.info(f"    {name:<18} {n} episodes")
    total = sum(counts)
    skew = max(counts) / max(min(counts), 1)
    logger.info(
        f"    (total {total}; max/min region ratio {skew:.2f} — "
        f"{'≈balanced, sampler is light-touch' if skew < 1.25 else 'imbalanced, sampler matters'})"
    )

    import torch
    from torch.utils.data import WeightedRandomSampler

    weights = []
    for n in counts:
        weights.extend([1.0 / max(n, 1)] * n)
    return WeightedRandomSampler(
        torch.as_tensor(weights, dtype=torch.double),
        num_samples=total,
        replacement=True,
    )


def parse_lead_datasets(specs: "list[str]") -> "list[tuple[int, Path]]":
    """Parse ``--lead-datasets`` 'LEAD_HOURS:DIR' entries into sorted pairs.

    Validates each lead is a non-negative int and each directory exists, and
    rejects duplicate leads. Returns ``(lead_hours, dir)`` sorted by lead, so the
    first entry is the lowest lead — conventionally the ERA5 analysis (lead 0) —
    which is used as the representative dataset for layout detection + metadata.
    """
    pairs: "list[tuple[int, Path]]" = []
    for spec in specs:
        if ":" not in spec:
            raise ValueError(f"--lead-datasets entry {spec!r} must be 'LEAD_HOURS:DIR'.")
        lead_str, _, dir_str = spec.partition(":")
        try:
            lead = int(lead_str)
        except ValueError as e:
            raise ValueError(
                f"--lead-datasets entry {spec!r}: {lead_str!r} is not an integer lead."
            ) from e
        if lead < 0:
            raise ValueError(f"--lead-datasets entry {spec!r}: lead must be >= 0.")
        d = Path(dir_str)
        if not d.is_dir():
            raise FileNotFoundError(f"--lead-datasets entry {spec!r}: directory not found: {d}")
        pairs.append((lead, d))
    leads = [lead for lead, _ in pairs]
    if len(set(leads)) != len(leads):
        raise ValueError(f"--lead-datasets has duplicate leads: {leads}")
    return sorted(pairs, key=lambda p: p[0])


def _stations_csv_has_mtpi(dataset_dir: "Path | None") -> bool:
    """True iff the dataset's ``stations.csv`` carries an ``mtpi`` column.

    Reads only the header (no pandas) so it stays cheap. Used to decide
    ``n_elev_features`` (2 vs 3) *before* config.json is written, so the
    choice round-trips to evaluate.py. Returns False when the file is absent
    — the dataset layer then serves the 2-feature (elevation, delta_elevation)
    layout that pre-mTPI checkpoints expect.
    """
    if dataset_dir is None:
        return False
    stations_csv = Path(dataset_dir) / "stations.csv"
    if not stations_csv.exists():
        return False
    with open(stations_csv, newline="") as f:
        header = next(csv.reader(f), [])
    return "mtpi" in header


def _build_mr_train_val(
    _MRClass, shared_kwargs, train_only_kwargs,
    train_region_specs, val_region_specs, args, logger,
):
    """Build (train, val) multi-region snapshot datasets for one dataset_dir.

    Extracted verbatim from the inline construction so the cross-lead path can
    reuse the exact same region-resolution + val-derivation logic per lead. The
    val split defaults to held-out stations within the training regions when no
    explicit val regions/specs are given.
    """
    if train_region_specs is not None:
        train = _MRClass(
            region_specs=train_region_specs, split="train",
            **shared_kwargs, **train_only_kwargs,
        )
    else:
        train = _MRClass(
            regions=args.train_regions, split="train", station_split="train",
            **shared_kwargs, **train_only_kwargs,
        )
    if val_region_specs is not None:
        val = _MRClass(region_specs=val_region_specs, split="val", **shared_kwargs)
    elif train_region_specs is not None:
        derived_val = {
            r: ("test" if s == "train" else s)
            for r, s in train_region_specs.items()
        }
        logger.info(
            f"No --region-specs-val given; derived val specs from train: {derived_val}"
        )
        val = _MRClass(region_specs=derived_val, split="val", **shared_kwargs)
    else:
        val_regions = args.val_regions or args.train_regions
        val = _MRClass(
            regions=val_regions, split="val", station_split="test", **shared_kwargs,
        )
    return train, val


def main():
    parser = argparse.ArgumentParser(
        description="Train ConvCNP downscaler (baseline or TESSERA)",
    )
    # Data.
    parser.add_argument(
        "--dataset-dir", type=Path, required=False, default=None,
        help="Path to preprocessed dataset from preprocess_daily.py. Optional "
             "when --lead-datasets is given (it is then set to the first lead's "
             "directory, used for layout detection + metadata).",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Output directory. If omitted, auto-generated from config.",
    )
    # TESSERA options.
    parser.add_argument(
        "--tessera-path", type=Path, default=None,
        help="Path to TESSERA patches .npy. Provided for ALL runs to "
             "filter stations to those with valid patches.",
    )
    parser.add_argument(
        "--tessera-station-csv", type=Path, default=None,
        help="CSV matching the TESSERA file's row ordering.",
    )
    parser.add_argument(
        "--min-tessera-patch-coverage", type=float, default=0.5,
        help="Minimum fraction of pixels in the 64x64 patch that must "
             "have any non-zero channel for a station to be kept. "
             "Combined (AND) with the centre-pixel-non-zero rule to "
             "produce the dataset's TESSERA filter. 0.0 disables the "
             "coverage check (legacy centre-pixel-only behaviour). "
             "Default 0.5.",
    )
    parser.add_argument(
        "--tessera-method", type=str, default=None,
        choices=["meanpool", "linear", "cnn"],
        help="TESSERA encoder method. Omit for baseline.",
    )
    parser.add_argument(
        "--tessera-output-dim", type=int, default=64,
        help="Output dim of TESSERA encoder (ignored for meanpool).",
    )
    parser.add_argument(
        "--tessera-chunk-size", type=int, default=128,
        help="Number of TESSERA patches to encode per GPU chunk.",
    )
    parser.add_argument(
        "--tessera-drop-prob", type=float, default=0.0,
        help="Full-embedding dropout probability for TESSERA encoder. "
             "During training, the entire TESSERA vector is zeroed out "
             "for this fraction of stations, forcing the model to learn "
             "generalisable patterns. Default: 0.0 (no dropout).",
    )
    parser.add_argument(
        "--tessera-injection", type=str, default="concat",
        choices=["concat", "film", "none"],
        help="How TESSERA features are injected into the decoder. "
             "'concat' (default): appended to MLP input alongside weather "
             "features. 'film': TESSERA generates per-layer scale and shift "
             "parameters that modulate MLP hidden activations (FiLM "
             "conditioning). 'none': skip the per-target injection entirely "
             "(useful for ablations that test the new embedding-aware "
             "mechanisms below standalone). The legacy 'hypernet' mode "
             "is no longer supported.",
    )
    # --- Embedding-aware decoder mechanisms (methodological extensions) ---
    parser.add_argument(
        "--decoder-kernel", type=str, default="isotropic",
        choices=["isotropic", "embedding_conditioned"],
        help="Choice of decoder SetConv kernel. 'isotropic' (default): "
             "standard RBFSetConv with a single learnable lengthscale. "
             "'embedding_conditioned' (§5.1): per-target lengthscale "
             "produced by a small shape MLP that reads the target's "
             "embedding. Requires an embedding source. Only valid with "
             "--interpolation=setconv.",
    )
    parser.add_argument(
        "--use-target-embed-stream", action="store_true",
        help="Enable §5.2 — a station→grid SetConv that aggregates "
             "target-station embeddings onto the internal CNN grid; the "
             "output is concatenated channel-wise with the CNN's F before "
             "the decoder SetConv reads it. Requires an embedding source.",
    )
    parser.add_argument(
        "--target-embed-attention", type=str, default="none",
        choices=["none", "embedding", "hybrid"],
        help="Self-attention over the target set aggregating embeddings. "
             "'none' (default): off. 'embedding': pure cosine-similarity "
             "weights. 'hybrid': learnable mix of cosine similarity and a "
             "Gaussian spatial term. Requires an embedding source.",
    )
    parser.add_argument(
        "--detach-attn-embed", action="store_true",
        help="Ablation flag: detach the per-target embedding at the attention "
             "module's input. Forward pass is numerically identical to the "
             "live version, but gradients no longer flow back to the projection "
             "/ encoder via the attention path. Use to test whether the "
             "attention gain comes from its forward-pass output content (gain "
             "preserved when set) or from extra gradient flow on the projection "
             "(gain disappears when set). Requires --target-embed-attention != none.",
    )
    # --- Pre-computed VAE latents (alternative to end-to-end encoder) ---
    parser.add_argument(
        "--vae-latents-path", type=Path, default=None,
        help="Path to a pre-computed VAE latents .npy of shape "
             "(n_global_stations, d). When set, the dataset serves these "
             "latents as the land surface features, bypassing the end-to-end "
             "TESSERA encoder. Incompatible with --tessera-method. Station "
             "filtering becomes the intersection of valid patches AND "
             "non-NaN latents.",
    )
    parser.add_argument(
        "--vae-latents-station-csv", type=Path, default=None,
        help="CSV row-aligned with --vae-latents-path, containing a "
             "'station_id' column. Required when --vae-latents-path is set.",
    )
    parser.add_argument(
        "--vae-latents-drop-prob", type=float, default=0.0,
        help="Full-embedding dropout probability applied to pre-computed "
             "VAE latents during training. Mirrors --tessera-drop-prob for "
             "the end-to-end encoder. Default: 0.0.",
    )
    parser.add_argument(
        "--vae-latents-no-zscore", action="store_true",
        help="Disable per-dim z-scoring of VAE latents. Default: z-score "
             "using stats computed over training-split stations.",
    )
    # --- Pre-computed hand-crafted surface descriptors ---
    parser.add_argument(
        "--extra-descriptors-path", type=Path, default=None,
        help="Path to a hand-crafted surface-descriptor .npy of shape "
             "(n_global_stations, d) built by preprocessing/"
             "build_extra_descriptors.py (land-cover fractions, canopy "
             "height, soil texture, topographic neighbourhood stats — after "
             "Bakketun et al. 2026). Served through the same precomputed-"
             "vector pathway as --vae-latents-path (NaN-row filtering, "
             "global z-score stats cached next to the .npy, injection via "
             "--tessera-injection). Named 'extra' to distinguish from the "
             "elevation/delta-elevation/mTPI descriptors already in the "
             "auxiliary vector. Incompatible with --tessera-method and "
             "--vae-latents-path.",
    )
    parser.add_argument(
        "--extra-descriptors-station-csv", type=Path, default=None,
        help="CSV row-aligned with --extra-descriptors-path, containing a "
             "'station_id' column (normally the same global station list "
             "used for --vae-latents-station-csv). Required when "
             "--extra-descriptors-path is set.",
    )
    parser.add_argument(
        "--extra-descriptors-drop-prob", type=float, default=0.0,
        help="Full-vector dropout probability applied to the extra "
             "descriptors during training. Mirrors --vae-latents-drop-prob. "
             "Default: 0.0.",
    )
    # The descriptors are stored as raw physical values and ALWAYS z-scored
    # at load (global stats cached next to the .npy) — the same convention as
    # every other precomputed vector in this slot; no opt-out flag.
    # --- Multi-region dataset support ---
    parser.add_argument(
        "--train-regions", type=str, nargs="+", default=None,
        help="When --dataset-dir points at a multi-region dataset root "
             "(contains a `regions/` subdir), train on these regions only. "
             "Example: --train-regions us. If omitted, all available "
             "regions are used. Legacy flag — consider --region-specs-train "
             "for more control (per-region spatial split).",
    )
    parser.add_argument(
        "--val-regions", type=str, nargs="+", default=None,
        help="Regions to validate on. Defaults to --train-regions (same "
             "regions as training, held-out stations within those regions). "
             "Legacy flag — consider --region-specs-val for finer control.",
    )
    parser.add_argument(
        "--region-specs-train", type=str, default=None,
        help="JSON dict mapping region name to spatial split, e.g. "
             "'{\"europe\": \"train\", \"us\": \"all\"}'. Supersedes "
             "--train-regions when given. Enables held-out-within-training "
             "experiments: train on {EU train + all of US}, test on EU test. "
             "If shell quoting is a pain (hello Slurm), use "
             "--region-specs-train-file instead.",
    )
    parser.add_argument(
        "--region-specs-val", type=str, default=None,
        help="JSON dict for validation region+split selection. See "
             "--region-specs-train. Typical pairing: train-specs use "
             "'train'/'all'; val-specs use 'test'/'all' to monitor held-out "
             "performance during training.",
    )
    parser.add_argument(
        "--region-specs-train-file", type=Path, default=None,
        help="Path to a JSON file containing the same dict that "
             "--region-specs-train would take inline. Used when shell "
             "quoting the JSON is impractical (e.g. via sbatch --wrap). "
             "Mutually exclusive with --region-specs-train.",
    )
    parser.add_argument(
        "--region-specs-val-file", type=Path, default=None,
        help="Path to a JSON file for validation region specs. Mutually "
             "exclusive with --region-specs-val. See "
             "--region-specs-train-file.",
    )
    parser.add_argument(
        "--vae-latents-proj-dim", type=int, default=0,
        help="If > 0, insert a learnable projection head taking the raw "
             "latent_dim and producing proj_dim between the frozen VAE "
             "latent and the decoder's injection mechanism. The head is "
             "a single nn.Linear by default; pass --vae-latents-proj-mlp "
             "to use a 2-layer MLP instead. Default: 0 (no projection, "
             "use raw latent directly).",
    )
    parser.add_argument(
        "--vae-latents-proj-mlp", action="store_true",
        help="When --vae-latents-proj-dim > 0, use a 2-layer MLP as the "
             "projection head (Linear(d_in, 2*proj_dim) → ReLU → "
             "Linear(2*proj_dim, proj_dim)) instead of a single nn.Linear. "
             "Tests whether a small non-linear transform of the VAE "
             "latent helps downstream prediction.",
    )
    parser.add_argument(
        "--target-variables", type=str, nargs="+", default=["tmax"],
        choices=["tmax", "wind_mean", "t2m", "wind", "precip"],
        help="Target variable(s) to predict. For multi-task, pass multiple. "
             "Daily-cadence datasets accept tmax and wind_mean; "
             "snapshot-cadence datasets accept t2m, wind and precip. "
             "Default: tmax.",
    )
    parser.add_argument(
        "--likelihood", type=str, default=None,
        help="Per-variable likelihood spec, e.g. "
             "`--likelihood t2m=gaussian,wind=weibull,precip=bernoulli_gamma`. "
             "Strict 1:1 with --target-variables: every variable must "
             "appear exactly once. Available distributions: gaussian "
             "(t2m default), weibull (wind), bernoulli_gamma (precip). "
             "When omitted, defaults to all-Gaussian — preserving the "
             "implicit Gaussian-everywhere behaviour of the legacy code.",
    )
    parser.add_argument(
        "--loss-function", type=str, default="nll", choices=["nll", "crps"],
        help="Training loss objective. 'nll' (default, legacy behaviour) "
             "minimises the negative log-likelihood of the per-variable head. "
             "'crps' minimises the continuous ranked probability score — a "
             "strictly proper scoring rule with closed-form expressions for "
             "Gaussian and truncated-normal heads (Thorarinsdottir & Gneiting "
             "2010; Gneiting & Raftery 2007). CRPS is bounded for any single "
             "observation (NLL is not), so it tends to produce sharper, "
             "better-calibrated forecasts in finite samples — typically at a "
             "small cost to point-estimate MAE. Evaluation metrics are "
             "unchanged regardless of the training loss: evaluate.py always "
             "emits MAE, RMSE, NLL, CRPS, PIT-calibration, and coverage "
             "statistics so NLL-trained and CRPS-trained models are "
             "directly comparable along every axis.",
    )
    # Elevation.
    parser.add_argument(
        "--no-elevation", action="store_true",
        help="Exclude elevation features from the MLP input. Forces the "
             "model to rely on other features (e.g. TESSERA) for local "
             "terrain information.",
    )
    parser.add_argument(
        "--use-mtpi", action="store_true",
        help="Add per-station mTPI (multi-scale topographic position index) as "
             "a third per-station feature, alongside elevation and "
             "delta_elevation, matching the auxiliary vector of Vaughan et al. "
             "(2022) (n_elev_features 2->3). OPT-IN: without this flag the model "
             "uses elevation+delta_elevation only, regardless of whether the "
             "dataset carries an `mtpi` column — so an elev-only run and an "
             "elev+mTPI run are distinguished explicitly in the experiment "
             "config, not implied by the data. Requires an `mtpi` column in the "
             "dataset's stations.csv (run backfill_station_mtpi.py or "
             "re-preprocess with --mtpi-csv first). No effect when "
             "--no-elevation is set (that drops all per-station features).",
    )
    parser.add_argument(
        "--no-static-fields", action="store_true",
        help="Exclude ERA5 static fields (orography, land-sea mask, soil "
             "type, etc.) from the CNN input grid. Combined with "
             "--no-elevation, this removes all surface information from "
             "the weather pathway, forcing the model to rely entirely on "
             "TESSERA for local surface context.",
    )
    parser.add_argument(
        "--normalisation-policy", type=str, default="per_region",
        choices=["per_region", "global"],
        help="ERA5 normalisation policy for multi-region snapshot datasets. "
             "'per_region' (default, legacy) loads each region's own train-"
             "timestamp-only stats — correct for single-region experiments "
             "but mismatches scales between train/test when regions differ. "
             "'global' loads stats computed across all regions' training "
             "timestamps (written by preprocess_timestamp_global.py) — "
             "recommended for transfer and joint-source experiments.",
    )
    parser.add_argument(
        "--lr-warmup-pct", type=float, default=0.0,
        help="Linear LR warmup: fraction of total training steps over which "
             "the learning rate linearly ramps from 0 to the target LR. "
             "Typical values 0.05-0.10 for transformer-style training. "
             "Default 0.0 (no warmup). Helps stabilise early gradient "
             "dynamics especially when randomly-initialised heads (FiLM, "
             "projection) produce noisy outputs at step 1.",
    )
    # Model hyperparameters.
    parser.add_argument("--cnn-hidden", type=int, default=128)
    parser.add_argument("--cnn-layers", type=int, default=7)
    parser.add_argument("--cnn-kernel", type=int, default=3)
    parser.add_argument(
        "--setconv-length-scale", type=float, default=0.5,
        help="Initial RBF length scale for the decoder SetConv (degrees). "
             "Default 0.5° biases the decoder toward local aggregation at "
             "init; the parameter is learnable and can grow if the loss "
             "rewards wider kernels. Also used as the initial value for "
             "the §5.2 embedding-stream SetConv and the hybrid attention's "
             "spatial scale.",
    )
    parser.add_argument(
        "--interpolation", type=str, default="setconv",
        choices=["setconv", "bilinear"],
        help="Grid-to-station interpolation method. 'setconv' uses a learned "
             "RBF kernel (default). 'bilinear' uses fixed bilinear interpolation "
             "at native grid resolution, forcing local corrections onto TESSERA.",
    )
    parser.add_argument("--mlp-hidden", type=int, default=128)
    parser.add_argument("--mlp-n-hidden", type=int, default=3)
    # Training hyperparameters.
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=2.5e-5)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument(
        "--drop-context-channels", type=str, nargs="+", default=None,
        help="Names of ERA5 dynamic context channels to drop from the context "
             "grid (e.g. 'total_precipitation_sum' to train a 19-channel model "
             "for evaluation on Aurora forecasts, which have no precip). "
             "Resolved by name against the dataset's era5_dynamic_channels: a "
             "listed channel that is present is dropped, one that is absent is "
             "skipped (logged), so the same flag works on a full-channel grid "
             "and an already-reduced one (e.g. the 20-channel ERA5 lead-0 "
             "dataset and the 19-channel Aurora leads in cross-lead training). "
             "Snapshot datasets only. Default: none (full channel set).",
    )
    parser.add_argument(
        "--lead-datasets", type=str, nargs="+", default=None, metavar="LEAD:DIR",
        help="Enable the lead-conditioned (cross-lead) model. Each entry is "
             "'LEAD_HOURS:DATASET_DIR', e.g. '0:/path/dataset_timestamp_global "
             "6:/path/dataset_timestamp_aurora_lead6h "
             "24:/path/dataset_timestamp_aurora_lead24h "
             "72:/path/dataset_timestamp_aurora_lead72h'. Each lead becomes its "
             "own snapshot dataset carrying a normalised lead/72 context channel; "
             "the leads are concatenated so one epoch sees every episode at every "
             "lead. Add or drop a lead just by editing this list (e.g. omit "
             "'0:...' to exclude the ERA5 analysis). Precip is dropped leniently "
             "across all leads so ERA5 (20ch) lines up with Aurora (19ch). "
             "Snapshot multi-region datasets only.",
    )
    parser.add_argument(
        "--grad-clip-norm", type=float, default=1.0,
        help="Clip total gradient norm to this value before each "
             "optimizer step (set 0 to disable). Necessary for the "
             "Weibull NLL: its log-prob has a (y/lam)^k term whose "
             "gradients can explode for extreme parameter values, "
             "blowing up training within ~1k steps without clipping. "
             "Helpful but not critical for Gaussian / Bernoulli-Gamma.",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument(
        "--region-balanced-sampling", action="store_true",
        help="Sample training episodes uniformly PER REGION rather than "
             "proportionally to each region's episode count. For a multi-"
             "region (globally-trained) run this makes the region effectively "
             "random per episode (scoping doc §3.2.2), so a station-dense "
             "region (e.g. europe, ~20x southern_africa's stations) does not "
             "dominate the gradient. No-op for single-region datasets. "
             "Implemented as a WeightedRandomSampler with per-episode weight "
             "1/(episodes in that region); per-region counts are logged.",
    )
    parser.add_argument("--seed", type=int, default=42)
    # --- Data-efficiency experiments (probe + station-count axes) ---
    # Both flags are file-based (paths to JSON sidecars) rather than
    # inline strings, mirroring the --region-specs-{train,val}-file
    # pattern. Shell-quoting a dict through `sbatch --wrap` is
    # impractical, and the sidecar files are also a useful audit
    # artefact next to the run directory.
    parser.add_argument(
        "--train-station-allowlist-file", type=Path, default=None,
        help="Path to a JSON file with shape "
             "{\"station_ids\": [\"GHCNH:USW00012345\", ...]} restricting "
             "the train-split stations of EACH train-spec region to "
             "those listed. Applied *after* the spatial split + "
             "TESSERA + VAE filters, so the listed count matches the "
             "model's actual training set size. Used by the "
             "station-count data-efficiency experiment; ignored for "
             "stations whose region's station_split is not 'train'. "
             "Persisted into config.json by file path so evaluate.py "
             "and downstream analysis can re-locate it.",
    )
    parser.add_argument(
        "--probe-active-from-file", type=Path, default=None,
        help="Path to a JSON file with shape "
             "{\"GHCNH:USW00012345\": \"2020-01-01-00\", ...} mapping "
             "probe-station IDs to their earliest usable episode "
             "timestamp during training. Stations not in the dict are "
             "never filtered; stations in the dict contribute zero "
             "training observations from episodes earlier than their "
             "active-from value. Used by the temporal-axis data-"
             "efficiency experiment to simulate a newly-deployed "
             "station that has accumulated d months of observations. "
             "Persisted into config.json by file path.",
    )
    parser.add_argument(
        "--train-end-override", type=str, default=None,
        help="Override the dataset's metadata-derived train_end "
             "timestamp ('YYYY-MM-DD-HH' or 'YYYY-MM-DD') for the train "
             "split. Train timestamps strictly later than this value "
             "are excluded from training; they fall into the val "
             "split's range instead. Used by the rollout data-"
             "efficiency experiment to simulate a 'snapshot at time T' "
             "of an in-progress network deployment: combined with "
             "--probe-active-from-file, this yields per-station "
             "training data of [t_open[i], train_end_override] for "
             "each online station. ERA5 normalisation stats are NOT "
             "affected — they continue to be computed over the full "
             "metadata.train_end window so cross-sweep comparability "
             "is preserved. Applies only to the train dataset "
             "construction; val/test datasets ignore this flag.",
    )
    args = parser.parse_args()

    # Cross-lead: parse the LEAD:DIR pairs once and use the lowest lead's dir
    # (conventionally ERA5 lead-0) as the representative for layout detection,
    # metadata and norm-stats. The per-lead dirs are applied during dataset
    # construction.
    args.lead_dataset_pairs = None
    if args.lead_datasets:
        args.lead_dataset_pairs = parse_lead_datasets(args.lead_datasets)
        args.dataset_dir = args.lead_dataset_pairs[0][1]
    if args.dataset_dir is None:
        parser.error("one of --dataset-dir or --lead-datasets is required")

    args.include_elevation = not args.no_elevation
    args.include_static_fields = not args.no_static_fields
    # mTPI is an OPT-IN third per-station feature: Vaughan et al. (2022)'s
    # per-station vector is elevation + delta_elevation + mTPI. Whether it is
    # used is controlled explicitly by --use-mtpi (NOT by whether the dataset
    # happens to carry an `mtpi` column), so an elev-only baseline and an
    # elev+mTPI variant are distinguished in the experiment config rather than
    # implied by the data. Resolved here — before config.json is written below
    # — so n_elev_features is recorded and reconstructed identically by
    # evaluate.py. Fail loudly if mTPI is requested but the column is absent.
    if args.use_mtpi and not _stations_csv_has_mtpi(args.dataset_dir):
        parser.error(
            "--use-mtpi was set but the dataset's stations.csv has no `mtpi` "
            "column. Run projects/tessera_downscaling/scripts/preprocessing/"
            "backfill_station_mtpi.py (or re-preprocess with --mtpi-csv) first."
        )
    args.n_elev_features = 3 if args.use_mtpi else 2
    n_target_variables = len(args.target_variables)
    is_multitask = n_target_variables > 1

    # --- Resolve data-efficiency JSON sidecars ----------------------------
    # Both files are simple JSON; this block loads them once at startup
    # and stashes the resolved structures on `args` so the dataset
    # construction below can pass them through without re-reading.
    # File paths are kept on args (string-cast at config.json save time)
    # so the run record links back to the canonical sidecars.
    train_station_allowlist: set[str] | None = None
    if args.train_station_allowlist_file is not None:
        path = args.train_station_allowlist_file
        if not path.exists():
            parser.error(
                f"--train-station-allowlist-file does not exist: {path}"
            )
        try:
            payload = json.loads(path.read_text())
        except Exception as e:
            parser.error(
                f"--train-station-allowlist-file is not valid JSON: {e}"
            )
        if not isinstance(payload, dict) or "station_ids" not in payload:
            parser.error(
                f"--train-station-allowlist-file payload must be a JSON "
                f"object with a 'station_ids' list; got "
                f"{type(payload).__name__} with keys "
                f"{list(payload) if isinstance(payload, dict) else 'n/a'}."
            )
        ids = payload["station_ids"]
        if not isinstance(ids, list) or not all(isinstance(s, str) for s in ids):
            parser.error(
                f"--train-station-allowlist-file 'station_ids' must be a "
                f"list of strings."
            )
        train_station_allowlist = set(ids)
        logger.info(
            f"Train-station allowlist loaded from {path}: "
            f"{len(train_station_allowlist)} unique station_ids."
        )

    probe_active_from: dict[str, str] | None = None
    if args.probe_active_from_file is not None:
        path = args.probe_active_from_file
        if not path.exists():
            parser.error(
                f"--probe-active-from-file does not exist: {path}"
            )
        try:
            payload = json.loads(path.read_text())
        except Exception as e:
            parser.error(
                f"--probe-active-from-file is not valid JSON: {e}"
            )
        if not isinstance(payload, dict):
            parser.error(
                f"--probe-active-from-file payload must be a JSON object; "
                f"got {type(payload).__name__}."
            )
        bad_values = [
            (k, v) for k, v in payload.items()
            if not isinstance(k, str) or not isinstance(v, str)
        ]
        if bad_values:
            parser.error(
                f"--probe-active-from-file entries must be "
                f"<str station_id>: <str timestamp>; got bad entries: "
                f"{bad_values[:3]}"
            )
        probe_active_from = payload
        logger.info(
            f"Probe-station active-from map loaded from {path}: "
            f"{len(probe_active_from)} entries."
        )

    # --- Resolve --likelihood into a {var: dist_name} dict -----------------
    # Strict 1:1 validation against --target-variables. Defaults to
    # all-Gaussian when --likelihood is omitted (legacy behaviour).
    if args.likelihood is None:
        likelihood_per_variable = {var: "gaussian" for var in args.target_variables}
    else:
        likelihood_per_variable = {}
        for entry in args.likelihood.split(","):
            entry = entry.strip()
            if "=" not in entry:
                parser.error(
                    f"--likelihood entry {entry!r} is malformed; expected "
                    "var=dist_name (e.g. t2m=gaussian)."
                )
            var, dist = entry.split("=", 1)
            var, dist = var.strip(), dist.strip()
            if var in likelihood_per_variable:
                parser.error(
                    f"--likelihood: duplicate entry for variable {var!r}."
                )
            likelihood_per_variable[var] = dist
        spec_vars = set(likelihood_per_variable)
        target_vars_set = set(args.target_variables)
        missing = target_vars_set - spec_vars
        extra = spec_vars - target_vars_set
        if missing or extra:
            parser.error(
                f"--likelihood does not match --target-variables. "
                f"target_variables={sorted(target_vars_set)}, "
                f"likelihood keys={sorted(spec_vars)}. "
                f"Missing={sorted(missing)}, extra={sorted(extra)}. "
                "Each target variable must have exactly one likelihood "
                "entry and vice versa."
            )
    # Persisted onto args so config.json captures the resolved spec.
    args.likelihood_per_variable = likelihood_per_variable

    # --- Validate VAE latent args ---
    uses_vae_latents = args.vae_latents_path is not None
    if uses_vae_latents:
        if args.vae_latents_station_csv is None:
            parser.error(
                "--vae-latents-station-csv is required when "
                "--vae-latents-path is set"
            )
        if args.tessera_method is not None:
            parser.error(
                "--vae-latents-path is incompatible with --tessera-method. "
                "Choose one source of TESSERA features (end-to-end encoder "
                "OR pre-computed latents)."
            )
        if args.tessera_path is None:
            parser.error(
                "--tessera-path is still required when using VAE latents, "
                "so station filtering can intersect (valid patch) AND "
                "(non-NaN latent). Pass the same patch file you'd use for "
                "the baseline run."
            )

    # --- Validate extra-descriptor args ---
    uses_extra_descriptors = args.extra_descriptors_path is not None
    if uses_extra_descriptors:
        if args.extra_descriptors_station_csv is None:
            parser.error(
                "--extra-descriptors-station-csv is required when "
                "--extra-descriptors-path is set"
            )
        if args.tessera_method is not None:
            parser.error(
                "--extra-descriptors-path is incompatible with "
                "--tessera-method. Choose one source of land-surface "
                "features (end-to-end encoder OR hand-crafted descriptors)."
            )
        if args.tessera_path is None:
            parser.error(
                "--tessera-path is still required when using extra "
                "descriptors, so station filtering matches the TESSERA and "
                "baseline arms (valid patch AND non-NaN descriptors). Pass "
                "the same patch file you'd use for the baseline run."
            )
        if uses_vae_latents:
            # Combining both sources: the dataset serves one vector per
            # station, so both files must share row alignment and load-time
            # settings — the vectors are concatenated below.
            if str(args.extra_descriptors_station_csv) != str(
                    args.vae_latents_station_csv):
                parser.error(
                    "when combining --vae-latents-path with "
                    "--extra-descriptors-path, both must be row-aligned to "
                    "the SAME station CSV (got different "
                    "--vae-latents-station-csv / "
                    "--extra-descriptors-station-csv)."
                )
            if args.extra_descriptors_drop_prob not in (
                    0.0, args.vae_latents_drop_prob):
                parser.error(
                    "when combining latents and extra descriptors, "
                    "--extra-descriptors-drop-prob must equal "
                    "--vae-latents-drop-prob (full-vector dropout applies "
                    "to the combined vector) or be left at 0.0."
                )

    # VAE latents and extra descriptors ride the same precomputed-vector
    # pathway (dataset kwargs, model slot, z-score machinery); only the CLI
    # naming and provenance differ. Unify here so downstream code has a
    # single source. When BOTH are given, they are concatenated into a
    # derived cache file (recorded in config.json for evaluate.py).
    uses_precomputed = uses_vae_latents or uses_extra_descriptors
    if uses_vae_latents and uses_extra_descriptors:
        precomputed_path = resolve_combined_vectors(
            args.vae_latents_path, args.extra_descriptors_path,
        )
        args.precomputed_merged_path = str(precomputed_path)
        precomputed_station_csv = args.vae_latents_station_csv
        precomputed_zscore = not args.vae_latents_no_zscore
        precomputed_drop_prob = args.vae_latents_drop_prob
    elif uses_vae_latents:
        precomputed_path = args.vae_latents_path
        precomputed_station_csv = args.vae_latents_station_csv
        precomputed_zscore = not args.vae_latents_no_zscore
        precomputed_drop_prob = args.vae_latents_drop_prob
    else:
        precomputed_path = args.extra_descriptors_path
        precomputed_station_csv = args.extra_descriptors_station_csv
        # Descriptors are stored raw and always z-scored at load.
        precomputed_zscore = True
        precomputed_drop_prob = args.extra_descriptors_drop_prob

    # Auto-generate output directory if not specified.
    if args.output_dir is None:
        name = build_output_dir_name(args)
        args.output_dir = Path(".tmp_output/training_runs") / name
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Save config.
    with open(args.output_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2, default=str)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    logger.info(f"Target variables: {args.target_variables}")
    if is_multitask:
        logger.info("Multi-task mode with learned task weights")

    uses_tessera_features = args.tessera_method is not None

    if uses_vae_latents and uses_extra_descriptors:
        logger.info(
            f"Mode: VAE-latent TESSERA + extra descriptors "
            f"(latents={args.vae_latents_path.name}, "
            f"descriptors={args.extra_descriptors_path.name}, "
            f"combined={precomputed_path.name}, "
            f"injection={args.tessera_injection}, "
            f"drop_prob={precomputed_drop_prob}, "
            f"zscore={precomputed_zscore})"
        )
    elif uses_vae_latents:
        logger.info(
            f"Mode: VAE-latent TESSERA "
            f"(path={args.vae_latents_path.name}, "
            f"injection={args.tessera_injection}, "
            f"drop_prob={args.vae_latents_drop_prob}, "
            f"zscore={not args.vae_latents_no_zscore})"
        )
    elif uses_extra_descriptors:
        logger.info(
            f"Mode: extra descriptors (hand-crafted) "
            f"(path={args.extra_descriptors_path.name}, "
            f"injection={args.tessera_injection}, "
            f"drop_prob={args.extra_descriptors_drop_prob}, zscore=True)"
        )
    elif uses_tessera_features:
        logger.info(
            f"Mode: TESSERA-augmented ({args.tessera_method}, "
            f"drop_prob={args.tessera_drop_prob})"
        )
    elif args.tessera_path:
        logger.info("Mode: Baseline on TESSERA-filtered stations")
    else:
        logger.info("Mode: Baseline (all stations)")

    if not args.include_elevation:
        logger.info("Elevation features: EXCLUDED")
    if not args.include_static_fields:
        logger.info("ERA5 static fields: EXCLUDED")

    # ---------------------------------------------------------------
    # Data
    # ---------------------------------------------------------------
    from tessera_downscaling.data.dataset import (
        DailyDownscalingDataset,
        MultiLeadDataset,
        MultiRegionDownscalingDataset,
        MultiRegionSnapshotDownscalingDataset,
        SnapshotDownscalingDataset,
        downscaling_collate,
    )

    # Detect dataset layout from metadata. Four layouts are supported:
    #   * multi_region_v1            — daily, layered (daily_global).
    #   * multi_region_snapshot_v1   — snapshot, layered (timestamp_global).
    #   * snapshot_v1                — snapshot, flat (single-region snapshot).
    #   * default                    — daily, flat (original daily pipeline).
    # Detection is a pure metadata read — no file-existence guessing.
    is_multi_region_daily = False
    is_multi_region_snapshot = False
    is_snapshot = False
    _top_md_path = args.dataset_dir / "metadata.json"
    if _top_md_path.exists():
        try:
            with open(_top_md_path) as _f:
                _md = json.load(_f)
            layout = _md.get("layout_version")
            is_multi_region_daily = (
                layout == "multi_region_v1"
                and (args.dataset_dir / "regions").is_dir()
            )
            is_multi_region_snapshot = (
                layout == "multi_region_snapshot_v1"
                and (args.dataset_dir / "regions").is_dir()
            )
            is_snapshot = layout == "snapshot_v1"
        except Exception:
            pass

    is_multi_region = is_multi_region_daily or is_multi_region_snapshot

    # Parse the JSON region-specs flags once up front.
    # --region-specs-{train,val} supersede --train-regions / --val-regions
    # for the relevant dataset. If neither is given for a dataset that
    # IS multi-region, fall back to the legacy flags.
    #
    # There are two ways to pass a region-specs dict:
    #   * --region-specs-train <json>       (inline JSON string)
    #   * --region-specs-train-file <path>  (path to a JSON file)
    # The inline form works fine when invoked directly but is a nightmare
    # to shell-quote through `sbatch --wrap`. Submission scripts should
    # use the file form; the inline form is kept for manual runs.
    def _resolve_region_specs(
        inline: str | None,
        file_path: Path | None,
        flag_name: str,
    ) -> str | None:
        """Resolve the JSON string to parse, from either inline or file."""
        if inline is not None and file_path is not None:
            parser.error(
                f"Pass either --{flag_name} OR --{flag_name}-file, not both."
            )
        if file_path is not None:
            if not file_path.exists():
                parser.error(f"--{flag_name}-file path {file_path} does not exist.")
            return file_path.read_text()
        return inline

    def _parse_region_specs(raw: str | None) -> dict[str, str] | None:
        if raw is None:
            return None
        try:
            parsed = json.loads(raw)
        except Exception as e:
            parser.error(f"Could not parse region-specs JSON: {e}")
        if not isinstance(parsed, dict):
            parser.error(f"region-specs must be a JSON object, got {type(parsed).__name__}")
        return parsed

    train_specs_raw = _resolve_region_specs(
        args.region_specs_train, args.region_specs_train_file, "region-specs-train",
    )
    val_specs_raw = _resolve_region_specs(
        args.region_specs_val, args.region_specs_val_file, "region-specs-val",
    )
    train_region_specs = _parse_region_specs(train_specs_raw)
    val_region_specs = _parse_region_specs(val_specs_raw)

    # Sanity-check that the old and new flags aren't mixed on the same split.
    if train_region_specs is not None and args.train_regions is not None:
        parser.error(
            "Pass either --region-specs-train OR --train-regions, not both."
        )
    if val_region_specs is not None and args.val_regions is not None:
        parser.error(
            "Pass either --region-specs-val OR --val-regions, not both."
        )

    if is_multi_region:
        layout_label = "daily" if is_multi_region_daily else "snapshot"
        if train_region_specs is not None:
            logger.info(
                f"Detected multi-region {layout_label} dataset at {args.dataset_dir}. "
                f"train_specs={train_region_specs}, val_specs={val_region_specs}"
            )
        else:
            logger.info(
                f"Detected multi-region {layout_label} dataset at {args.dataset_dir}. "
                f"train_regions={args.train_regions}, val_regions={args.val_regions}"
            )
    elif is_snapshot:
        logger.info(
            f"Detected snapshot (timestamp-cadence) dataset at {args.dataset_dir}."
        )
        if (args.train_regions is not None or args.val_regions is not None
                or train_region_specs is not None or val_region_specs is not None):
            parser.error(
                "Region flags are only valid for multi-region datasets. "
                "The flat snapshot dataset class serves a single region."
            )
    else:
        if (args.train_regions is not None or args.val_regions is not None
                or train_region_specs is not None or val_region_specs is not None):
            parser.error(
                "Region flags were given but "
                f"{args.dataset_dir} is not a multi-region dataset "
                "(no regions/ subdir or unrecognised layout_version)."
            )

    # Gate the new data-efficiency flags on snapshot-cadence datasets:
    # the daily-cadence classes don't accept these kwargs and we'd just
    # silently mis-construct. Easier to error early with a clear message.
    if (
        (train_station_allowlist is not None or probe_active_from is not None)
        and not (is_snapshot or is_multi_region_snapshot)
    ):
        parser.error(
            "--train-station-allowlist-file / --probe-active-from-file "
            "require a snapshot-cadence dataset. The current dataset at "
            f"{args.dataset_dir} is not snapshot — the daily-cadence "
            "dataset classes don't implement these filters."
        )

    shared_kwargs = {
        "dataset_dir": args.dataset_dir,
        "target_variables": args.target_variables,
        "tessera_path": args.tessera_path,
        "tessera_station_csv": args.tessera_station_csv,
        "load_tessera_patches": uses_tessera_features,
        "include_static_fields": args.include_static_fields,
        # The dataset's vae_latents_* kwargs are the generic precomputed
        # per-station-vector inputs; they also carry the extra descriptors.
        "vae_latents_path": precomputed_path,
        "vae_latents_station_csv": precomputed_station_csv,
        "vae_latents_zscore": precomputed_zscore,
        "min_patch_coverage": args.min_tessera_patch_coverage,
    }
    # normalisation_policy only applies to multi-region snapshot
    # dataset (where global cross-region stats are a new feature).
    # Add it to shared_kwargs only for MR snapshot — other dataset
    # classes don't accept it and would error.
    if is_multi_region and is_multi_region_snapshot:
        shared_kwargs["normalisation_policy"] = args.normalisation_policy

    # probe_active_from is uniform across train/val/test for snapshot
    # datasets — the dataset class is idempotent on val/test episodes
    # because typical active_from values fall inside the training
    # window and lexicographically compare less than every val/test
    # timestamp. Adding it to shared_kwargs keeps the val/test
    # constructions self-consistent (and lets the audit logging in
    # the dataset's __init__ print the same map for every split).
    if probe_active_from is not None and (is_snapshot or is_multi_region_snapshot):
        shared_kwargs["probe_active_from"] = probe_active_from

    # train_end_override moves the train/val boundary earlier (rollout
    # data-efficiency experiment). Goes into shared_kwargs because BOTH
    # the train and val dataset constructions need to agree on the
    # boundary — otherwise timestamps in [override, original_train_end]
    # fall into neither split. Norm-stats are computed from the
    # metadata train_end value (not the override) so cross-sweep
    # comparability is preserved — see the comment on the kwarg in
    # SnapshotDownscalingDataset.__init__.
    if args.train_end_override is not None:
        if not (is_snapshot or is_multi_region_snapshot):
            parser.error(
                "--train-end-override is only supported for snapshot "
                "datasets (snapshot_v1, multi_region_snapshot_v1). The "
                "daily-cadence dataset classes don't implement this flag."
            )
        shared_kwargs["train_end_override"] = args.train_end_override
        logger.info(
            f"train_end_override set to {args.train_end_override!r}; "
            f"train/val boundary shifted from "
            f"metadata.train_end to this value."
        )

    # Context channels to drop (e.g. precipitation -> 19-channel model). Only
    # the snapshot dataset classes implement it. Strict resolution at train
    # time (default in the dataset class): a name absent from the dataset is an
    # error, since training always runs against the full-channel dataset.
    if args.drop_context_channels:
        if not (is_snapshot or is_multi_region_snapshot):
            parser.error(
                "--drop-context-channels is only supported for snapshot "
                "datasets (snapshot_v1, multi_region_snapshot_v1)."
            )
        shared_kwargs["drop_context_channels"] = args.drop_context_channels
        logger.info(f"Dropping context channels: {args.drop_context_channels}")

    if is_multi_region:
        # Pick the right class for the cadence.
        _MRClass = (
            MultiRegionSnapshotDownscalingDataset
            if is_multi_region_snapshot
            else MultiRegionDownscalingDataset
        )
        # train_station_allowlist applies to the TRAIN dataset only. The
        # MR snapshot class silently ignores it for non-"train" specs,
        # but we still want to be explicit about not passing it to val
        # — both for safety (val/test on the full held-out set) and
        # because the daily-cadence MR class doesn't accept the kwarg.
        train_only_kwargs: dict = {}
        if (
            train_station_allowlist is not None
            and is_multi_region_snapshot
        ):
            train_only_kwargs["train_station_allowlist"] = train_station_allowlist
        # Resolve train args: prefer region_specs-train if given, else
        # fall back to --train-regions + station_split="train".
        if args.lead_dataset_pairs is not None:
            # Cross-lead: build one MR train/val dataset per lead (each on its
            # own dir, with lead_hours set so the lead channel is present), then
            # concatenate. The precip drop that lines up the 20-channel ERA5
            # lead-0 dataset with the 19-channel Aurora leads is supplied by the
            # caller via --drop-context-channels total_precipitation_sum (handled
            # in the shared_kwargs block above, so config.json records it and
            # evaluate.py rebuilds the model identically). The drop is by name
            # and lenient by default, so it drops precip from the ERA5 lead-0
            # dataset and no-ops on the Aurora leads where it is already absent.
            # If the flag is omitted, MultiLeadDataset raises a clear
            # n_context_channels mismatch naming the missing drop.
            train_subs, val_subs = [], []
            for lead_h, dir_ in args.lead_dataset_pairs:
                sk = {**shared_kwargs, "dataset_dir": dir_, "lead_hours": lead_h}
                tr, va = _build_mr_train_val(
                    _MRClass, sk, train_only_kwargs,
                    train_region_specs, val_region_specs, args, logger,
                )
                train_subs.append(tr)
                val_subs.append(va)
            train_dataset = MultiLeadDataset(train_subs)
            val_dataset = MultiLeadDataset(val_subs)
            logger.info(
                f"Cross-lead datasets built: train {len(train_dataset)} "
                f"(= {len(train_subs[0])} episodes x {len(train_subs)} leads), "
                f"val {len(val_dataset)}; n_context_channels={train_dataset.n_context_channels}."
            )
        else:
            train_dataset, val_dataset = _build_mr_train_val(
                _MRClass, shared_kwargs, train_only_kwargs,
                train_region_specs, val_region_specs, args, logger,
            )
    elif is_snapshot:
        train_only_kwargs: dict = {}
        if train_station_allowlist is not None:
            train_only_kwargs["train_station_allowlist"] = train_station_allowlist
        train_dataset = SnapshotDownscalingDataset(
            split="train", station_split="train",
            **shared_kwargs, **train_only_kwargs,
        )
        val_dataset = SnapshotDownscalingDataset(
            split="val", station_split="test", **shared_kwargs,
        )
    else:
        train_dataset = DailyDownscalingDataset(
            split="train", station_split="train", **shared_kwargs,
        )
        val_dataset = DailyDownscalingDataset(
            split="val", station_split="test", **shared_kwargs,
        )

    num_workers = args.num_workers
    if uses_tessera_features and num_workers > 0:
        import os
        patch_size_gb = os.path.getsize(args.tessera_path) / (1024 ** 3)
        if patch_size_gb > 5.0:
            logger.info(
                f"Auto-setting num_workers=0 for large TESSERA patches "
                f"({patch_size_gb:.1f}GB)"
            )
            num_workers = 0

    # Optional region-balanced sampling (scoping doc §3.2.2): draw episodes
    # uniformly per region instead of proportionally to per-region episode
    # counts. No-op (sampler stays None → standard shuffle) for single-region
    # datasets.
    train_sampler = None
    if getattr(args, "region_balanced_sampling", False):
        train_sampler = build_region_balanced_sampler(train_dataset, logger)
        if train_sampler is None:
            logger.info(
                "--region-balanced-sampling set but dataset is single-region; "
                "using standard shuffling."
            )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=num_workers,
        collate_fn=downscaling_collate,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=downscaling_collate,
        pin_memory=True,
    )

    logger.info(f"Train: {len(train_dataset)} days, Val: {len(val_dataset)} days")

    # ---------------------------------------------------------------
    # Model
    # ---------------------------------------------------------------
    from tessera_downscaling.model.convcnp import ConvCNPDownscaler

    with open(args.dataset_dir / "metadata.json") as f:
        meta = json.load(f)

    # Channel count comes from the dataset, which accounts for
    # whether static fields are included or excluded.
    n_context_channels = train_dataset.n_context_channels

    # Build TESSERA encoder if requested (end-to-end path only).
    tessera_encoder = None
    if uses_tessera_features:
        from tessera_downscaling.model.tessera_encoder import TesseraPatchEncoder
        tessera_encoder = TesseraPatchEncoder(
            embed_dim=128,
            output_dim=args.tessera_output_dim,
            method=args.tessera_method,
            drop_prob=args.tessera_drop_prob,
        )

    # Pre-computed vector dim (VAE latents or extra descriptors) comes from
    # the dataset after loading.
    precomputed_tessera_dim = 0
    if uses_precomputed:
        precomputed_tessera_dim = train_dataset.vae_latent_dim
        # Sanity check: val dataset should agree.
        assert val_dataset.vae_latent_dim == precomputed_tessera_dim, (
            f"Precomputed vector dim mismatch between train "
            f"({precomputed_tessera_dim}) and val "
            f"({val_dataset.vae_latent_dim})"
        )
        logger.info(
            f"Precomputed vector dim (from dataset): {precomputed_tessera_dim}"
        )

    model = ConvCNPDownscaler(
        n_context_channels=n_context_channels,
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
        likelihood_per_variable=likelihood_per_variable,
        tessera_encoder=tessera_encoder,
        tessera_injection=args.tessera_injection,
        tessera_features_precomputed=uses_precomputed,
        precomputed_tessera_dim=precomputed_tessera_dim,
        precomputed_drop_prob=precomputed_drop_prob,
        precomputed_proj_dim=args.vae_latents_proj_dim,
        precomputed_proj_mlp=args.vae_latents_proj_mlp,
        decoder_kernel=args.decoder_kernel,
        use_target_embed_stream=args.use_target_embed_stream,
        target_embed_attention=args.target_embed_attention,
        detach_attn_embed=args.detach_attn_embed,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {n_params:,}")
    if uses_tessera_features:
        logger.info(
            f"TESSERA encoder: {args.tessera_method} "
            f"(output_dim={model.tessera_encoder.output_dim}, "
            f"drop_prob={args.tessera_drop_prob})"
        )

    # Bernoulli-Gamma heads need ρ-bias initialised to logit(p_wet) where
    # p_wet is the climatological wet-day frequency, otherwise the early
    # training loss is dominated by the dry-day branch in dry regions.
    # We compute p_wet by iterating the training loader for a bounded
    # number of batches — exact precision isn't needed for an init.
    bg_vars = [
        var for var, dist in likelihood_per_variable.items()
        if dist == "bernoulli_gamma"
    ]
    if bg_vars:
        # We re-use the same collate but cap batches; this is just for
        # statistics. Use args.batch_size and num_workers=0 to avoid the
        # overhead of spinning up workers for ~50 batches.
        stats_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=0,
            collate_fn=downscaling_collate,
        )
        max_stats_batches = 50
        for var in bg_vars:
            vi = args.target_variables.index(var)
            n_wet, n_valid = 0, 0
            for batch_idx, batch in enumerate(stats_loader):
                if batch is None:
                    continue
                targets = batch["target_values"]
                mask = batch["target_mask"].bool()
                if is_multitask:
                    precip = targets[:, :, vi]
                else:
                    precip = targets
                # Wet ↔ y > 0 ; dry ↔ y == 0 (NaNs are excluded by the mask)
                wet_mask = (precip > 0) & mask
                n_wet += wet_mask.sum().item()
                n_valid += mask.sum().item()
                if batch_idx + 1 >= max_stats_batches:
                    break
            if n_valid == 0:
                raise RuntimeError(
                    f"Could not compute climatology for {var!r}: no valid "
                    f"targets in the first {max_stats_batches} batches."
                )
            p_wet = n_wet / n_valid
            # Clamp into a sensible range to avoid logit explosion if a
            # region happens to be entirely dry / wet in the sample.
            p_wet = max(0.005, min(0.995, p_wet))
            model.heads.heads[var].initialise_rho_bias_from_climatology(p_wet)
            logger.info(
                f"BernoulliGamma head {var!r}: p_wet ≈ {p_wet:.4f} "
                f"(estimated from {n_valid} valid targets across "
                f"≤{max_stats_batches} train batches); ρ-bias initialised "
                f"to logit(p_wet) = {math.log(p_wet/(1-p_wet)):.4f}."
            )

    # TruncatedNormal heads need (μ, log_var) initialised from positive-
    # target climatology so they don't waste ~5 epochs of warmup
    # traversing the Mills-ratio region of the NLL surface. Mirrors the
    # Bernoulli-Gamma pattern above; the gradient-conditioning argument
    # is documented on TruncatedNormalHead.initialise_from_climatology.
    tn_vars = [
        var for var, dist in likelihood_per_variable.items()
        if dist == "truncated_normal"
    ]
    if tn_vars:
        stats_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=0,
            collate_fn=downscaling_collate,
        )
        max_stats_batches = 50
        for var in tn_vars:
            vi = args.target_variables.index(var)
            positive_targets: list[torch.Tensor] = []
            for batch_idx, batch in enumerate(stats_loader):
                if batch is None:
                    continue
                targets = batch["target_values"]
                mask = batch["target_mask"].bool()
                tgt = targets[:, :, vi] if is_multitask else targets
                # Restrict to strictly positive observations. Zero-wind
                # readings (anemometer threshold) are excluded because
                # they bias σ downward and the head's NLL clamp at
                # _POSITIVE_MIN handles them at runtime regardless.
                valid_positive = (tgt > 0) & mask
                if valid_positive.any():
                    positive_targets.append(tgt[valid_positive])
                if batch_idx + 1 >= max_stats_batches:
                    break
            all_positive = (
                torch.cat(positive_targets) if positive_targets
                else torch.empty(0)
            )
            if all_positive.numel() < 100:
                raise RuntimeError(
                    f"Could not compute climatology for {var!r}: only "
                    f"{all_positive.numel()} positive targets in the first "
                    f"{max_stats_batches} batches (need >= 100)."
                )
            mean_target = float(all_positive.mean().item())
            std_target = float(all_positive.std().item())
            model.heads.heads[var].initialise_from_climatology(
                mean_target=mean_target, std_target=std_target,
            )
            logger.info(
                f"TruncatedNormal head {var!r}: "
                f"mean ≈ {mean_target:.4f}, std ≈ {std_target:.4f} "
                f"(estimated from {all_positive.numel()} positive targets "
                f"across ≤{max_stats_batches} train batches); "
                f"(μ, log_var) bias initialised to "
                f"({mean_target:.4f}, {2 * math.log(std_target):.4f})."
            )

    # Weibull heads need (k, λ) initialised to a region-appropriate Rayleigh
    # so training doesn't spend many epochs scaling from the Linear default
    # (~0.7) to a physically reasonable wind regime. Empirical mean wind is
    # computed from the same bounded loader pass used for B-G p_wet.
    weibull_vars = [
        var for var, dist in likelihood_per_variable.items()
        if dist == "weibull"
    ]
    if weibull_vars:
        stats_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=0,
            collate_fn=downscaling_collate,
        )
        max_stats_batches = 50
        for var in weibull_vars:
            vi = args.target_variables.index(var)
            sum_y, n_valid = 0.0, 0
            for batch_idx, batch in enumerate(stats_loader):
                if batch is None:
                    continue
                if batch_idx >= max_stats_batches:
                    break
                tv = batch["target_values"]
                tm = batch["target_mask"].bool()
                if tv.ndim == 3:
                    tv = tv[..., vi]
                vals = tv[tm].numpy()
                vals = vals[vals > 0]  # skip exact zeros
                if vals.size:
                    sum_y += float(vals.sum())
                    n_valid += vals.size
            if n_valid == 0:
                logger.warning(
                    f"Weibull init: no positive {var} samples in "
                    f"{max_stats_batches} batches; leaving head at default init."
                )
                continue
            mean_wind = sum_y / n_valid
            logger.info(
                f"Weibull init for {var}: empirical mean = {mean_wind:.3f} "
                f"({n_valid:,} samples). Setting k→2.0, λ→{mean_wind/0.886:.3f}."
            )
            model.heads.heads[var].initialise_from_climatology(
                mean_wind=mean_wind, target_k=2.0,
            )

    # Learned task weights for multi-task (Kendall et al. 2018).
    log_task_weights = None
    if is_multitask:
        log_task_weights = nn.Parameter(
            torch.zeros(n_target_variables, device=device)
        )
        logger.info(
            f"Learned task weights initialised: "
            f"{['%.2f' % w for w in log_task_weights.tolist()]}"
        )

    # ---------------------------------------------------------------
    # Optimiser
    # ---------------------------------------------------------------
    params = list(model.parameters())
    if log_task_weights is not None:
        params.append(log_task_weights)
    optimizer = torch.optim.Adam(
        params, lr=args.lr, weight_decay=args.weight_decay,
    )

    # ---------------------------------------------------------------
    # LR warmup (optional). Linear ramp from lr*1e-3 → lr over
    # warmup_steps, then constant at lr thereafter. We don't chain a
    # decay schedule here because your snapshot runs typically early-
    # stop after 5-15 epochs; a warmup + plateau is a better match
    # than warmup + long cosine decay.
    #
    # Warmup step count is derived from an estimate of total steps
    # (epochs × batches_per_epoch). We use args.epochs as the upper
    # bound — if the run early-stops before warmup completes, the LR
    # will just not yet have reached the target, which is fine (it
    # was heading up the whole time).
    # ---------------------------------------------------------------
    lr_scheduler = None
    if args.lr_warmup_pct > 0.0:
        # batches per epoch is len(train_loader); we don't have it yet,
        # so compute it here.
        n_batches_per_epoch = len(train_loader)
        total_steps = args.epochs * n_batches_per_epoch
        warmup_steps = max(1, int(args.lr_warmup_pct * total_steps))
        lr_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1e-3,  # start at 0.1% of target LR
            end_factor=1.0,      # ramp to 100%
            total_iters=warmup_steps,
        )
        logger.info(
            f"LR warmup: linear from {args.lr * 1e-3:.2e} to {args.lr:.2e} "
            f"over {warmup_steps} steps "
            f"({args.lr_warmup_pct * 100:.1f}% of {total_steps} total steps = "
            f"{args.epochs} epochs × {n_batches_per_epoch} batches)"
        )

    # ---------------------------------------------------------------
    # Training loop
    # ---------------------------------------------------------------
    best_val_loss = float("inf")
    epochs_without_improvement = 0
    train_losses = []
    val_losses = []
    val_maes = []  # per-variable MAEs for logging
    nonfinite_skips_per_epoch = []     # # of skipped optim steps per epoch
    attempted_steps_per_epoch = []     # train batches attempted (for % computation)

    for epoch in range(1, args.epochs + 1):
        # --- Train ---
        model.train()
        epoch_loss = 0.0
        epoch_batches = 0
        nonfinite_skips = 0  # count of optimizer steps skipped due to NaN/Inf grad
        t_start = time.time()

        for batch_idx, batch in enumerate(tqdm(
            train_loader,
            desc=f"Epoch {epoch}/{args.epochs} [train]",
            leave=False,
        )):
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

            target_tessera = None
            if "target_tessera" in batch and (uses_tessera_features or uses_precomputed):
                target_tessera = batch["target_tessera"].to(device)

            # ============ DIAGNOSTIC: NaN INPUT CHECK ============
            # Localizes NaN at the training-loop boundary so we can tell
            # whether bad data comes from the dataloader, the encoder, the
            # interp, or the head. Remove this block once the source is
            # identified.
            def _nan_check(name, t):
                if t is None:
                    return
                if torch.isnan(t).any():
                    n_nan = int(torch.isnan(t).sum())
                    n_tot = t.numel()
                    per_row = "n/a"
                    if t.dim() >= 2:
                        per_row = int(torch.isnan(t).any(dim=tuple(range(1, t.dim()))).sum())
                    raise RuntimeError(
                        f"[NaN_DIAG] epoch={epoch} batch_idx={batch_idx} "
                        f"input '{name}' has {n_nan}/{n_tot} NaN entries "
                        f"(shape={tuple(t.shape)}, rows-with-any-NaN={per_row})"
                    )
            _nan_check("context_grid", context_grid)
            _nan_check("target_coords", target_coords)
            _nan_check("target_elev", target_elev)
            _nan_check("target_delta_elev", target_delta_elev)
            _nan_check("target_mtpi", target_mtpi)
            _nan_check("target_values", target_values)
            _nan_check("target_tessera", target_tessera)
            # =====================================================

            optimizer.zero_grad()
            params_per_var = model(
                context_grid, grid_lats, grid_lons,
                target_coords, target_elev, target_delta_elev,
                target_mask, target_tessera,
                target_mtpi=target_mtpi,
            )

            # ============ DIAGNOSTIC: NaN POST-FORWARD CHECK =====
            # If inputs are clean but params have NaN, the model itself
            # introduced it (init issue, numerical instability in CNN /
            # interp / MLP body). Reports which parameter went NaN and
            # whether weights themselves are NaN (gradient explosion in
            # a previous step).
            for _v, _ph in params_per_var.items():
                for _pn, _pt in _ph.items():
                    if torch.isnan(_pt).any() or torch.isinf(_pt).any():
                        n_nan = int(torch.isnan(_pt).sum())
                        n_inf = int(torch.isinf(_pt).sum())
                        # Weight NaN audit — narrows root cause.
                        weight_diag = []
                        for _name, _p in model.named_parameters():
                            if torch.isnan(_p).any() or torch.isinf(_p).any():
                                weight_diag.append(
                                    f"{_name}(nan={int(torch.isnan(_p).sum())},"
                                    f"inf={int(torch.isinf(_p).sum())})"
                                )
                        # Sibling-param stats (the non-NaN side, if any)
                        sibling_stats = {}
                        for _spn, _spt in _ph.items():
                            finite = _spt[torch.isfinite(_spt)]
                            sibling_stats[_spn] = {
                                "shape": tuple(_spt.shape),
                                "n_nan": int(torch.isnan(_spt).sum()),
                                "n_inf": int(torch.isinf(_spt).sum()),
                                "min_finite": float(finite.min()) if finite.numel() else None,
                                "max_finite": float(finite.max()) if finite.numel() else None,
                            }
                        raise RuntimeError(
                            f"[NaN_DIAG] epoch={epoch} batch_idx={batch_idx}\n"
                            f"  head output {_v}/{_pn}: nan={n_nan}, inf={n_inf}, "
                            f"shape={tuple(_pt.shape)}\n"
                            f"  inputs were clean → bug is inside model.forward.\n"
                            f"  all head-output stats this batch: {sibling_stats}\n"
                            f"  model weights with NaN/Inf: "
                            f"{weight_diag if weight_diag else '(none — weights still finite)'}"
                        )
            # =====================================================

            loss, _ = compute_loss_via_heads(
                params_per_var, target_values, target_mask,
                args.target_variables, model.heads,
                is_multitask, log_task_weights,
                loss_function=args.loss_function,
            )

            loss.backward()

            # Detect non-finite gradients (NaN or Inf) and skip the step.
            # This MUST happen before clip_grad_norm_ — the clipper does
            # not sanitize: it computes a global norm that becomes NaN
            # if any single gradient element is NaN, then propagates that
            # NaN to every parameter via the clip coefficient, turning
            # the entire model into NaN in one step.
            nonfinite_grad = False
            for p in model.parameters():
                if p.grad is not None and not torch.isfinite(p.grad).all():
                    nonfinite_grad = True
                    break
            if nonfinite_grad:
                nonfinite_skips += 1
                optimizer.zero_grad()
                if nonfinite_skips <= 5 or nonfinite_skips % 100 == 0:
                    logger.warning(
                        f"Non-finite gradient at epoch={epoch} "
                        f"batch_idx={batch_idx}; skipping optimizer step "
                        f"(total skips so far: {nonfinite_skips})."
                    )
                continue

            # Clip total gradient norm BEFORE optimizer.step. Without this
            # the Weibull NLL can blow up training: its log-prob includes
            # a (y/lam)^k term whose gradient grows exponentially when
            # the optimizer pushes lam small or k large for any subset
            # of stations. Empirically, wind-Weibull crashes at ~770 steps
            # without clipping (every weight goes NaN). With max_norm=1.0
            # the explosion is bounded; same flag also protects Gaussian
            # and B-G runs at no measurable cost.
            if args.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=args.grad_clip_norm,
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
                f"Epoch {epoch}: skipped {nonfinite_skips} optimizer steps "
                f"({pct:.2f}% of attempted updates) due to non-finite gradients. "
                f"If this is >5% of updates, the head's parameter clamps may "
                f"be too loose or the LR is too high for this likelihood."
            )

        # --- Validate ---
        model.eval()
        val_loss = 0.0
        val_batches = 0
        # Per-variable MAE tracking.
        val_mae_sums = [0.0] * n_target_variables
        val_counts = [0] * n_target_variables

        with torch.no_grad():
            for batch in tqdm(
                val_loader,
                desc=f"Epoch {epoch}/{args.epochs} [val]",
                leave=False,
            ):
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

                target_tessera = None
                if "target_tessera" in batch and (uses_tessera_features or uses_precomputed):
                    target_tessera = batch["target_tessera"].to(device)

                params_per_var = model(
                    context_grid, grid_lats, grid_lons,
                    target_coords, target_elev, target_delta_elev,
                    target_mask, target_tessera,
                    target_mtpi=target_mtpi,
                )

                loss, _ = compute_loss_via_heads(
                    params_per_var, target_values, target_mask,
                    args.target_variables, model.heads,
                    is_multitask, log_task_weights,
                    loss_function=args.loss_function,
                )

                # Per-variable validation MAE — point estimate is
                # head.mean(params). For Gaussian this is just μ
                # (matches legacy behaviour exactly); for skewed
                # distributions this is the predictive mean, which is
                # the right per-distribution analogue of "predicted
                # value" for early-stopping consistency. The principled
                # metric for early stopping is val NLL (the loss we just
                # computed); the MAE is a secondary tracking metric.
                target_mask_float = target_mask.float()
                for vi, var in enumerate(args.target_variables):
                    head = model.heads.heads[var]
                    point_est = head.mean(params_per_var[var])
                    target_var = (
                        target_values[:, :, vi] if is_multitask
                        else target_values
                    )
                    abs_err = torch.abs(point_est - target_var) * target_mask_float
                    val_mae_sums[vi] += abs_err.sum().item()
                    val_counts[vi] += target_mask_float.sum().item()

                val_loss += loss.item()
                val_batches += 1

        val_loss_avg = val_loss / max(val_batches, 1)
        val_losses.append(val_loss_avg)

        # Compute per-variable MAEs.
        epoch_maes = []
        for vi in range(n_target_variables):
            mae_vi = val_mae_sums[vi] / max(val_counts[vi], 1)
            epoch_maes.append(mae_vi)
        val_maes.append(epoch_maes)

        # Log.
        mae_str = " | ".join(
            f"{args.target_variables[vi]} MAE: {epoch_maes[vi]:.3f}"
            for vi in range(n_target_variables)
        )
        task_weight_str = ""
        if log_task_weights is not None:
            weights = [f"{torch.exp(w).item():.3f}" for w in log_task_weights]
            task_weight_str = f" | σ={weights}"

        logger.info(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss_avg:.4f} | "
            f"{mae_str}{task_weight_str} | "
            f"Time: {t_train:.1f}s"
        )

        # --- Checkpointing + early stopping ---
        if val_loss_avg < best_val_loss:
            best_val_loss = val_loss_avg
            epochs_without_improvement = 0
            save_dict = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss_avg,
                "val_maes": {
                    args.target_variables[vi]: epoch_maes[vi]
                    for vi in range(n_target_variables)
                },
                "tessera_method": args.tessera_method,
                "config": {k: str(v) if isinstance(v, Path) else v
                           for k, v in vars(args).items()},
            }
            if log_task_weights is not None:
                save_dict["log_task_weights"] = log_task_weights.detach().cpu()
            torch.save(save_dict, args.output_dir / "best_model.pt")
            logger.info(f"  -> New best model (Val Loss: {val_loss_avg:.4f})")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                logger.info(
                    f"Early stopping after {args.patience} epochs "
                    "without improvement"
                )
                break

        if epoch % 10 == 0:
            save_dict = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss_avg,
            }
            if log_task_weights is not None:
                save_dict["log_task_weights"] = log_task_weights.detach().cpu()
            torch.save(save_dict, args.output_dir / "latest_model.pt")

    # Save training curves.
    np.savez(
        args.output_dir / "training_curves.npz",
        train_losses=np.array(train_losses),
        val_losses=np.array(val_losses),
        val_maes=np.array(val_maes),  # (n_epochs, n_vars)
        nonfinite_skips_per_epoch=np.array(nonfinite_skips_per_epoch),
        attempted_steps_per_epoch=np.array(attempted_steps_per_epoch),
    )

    # Persist a small training_summary.json for at-a-glance triage across
    # many runs without having to load training_curves.npz. Captures
    # exactly the metadata that's useful when a sweep produces hundreds
    # of test_summary.json files and you need to flag the ones that had
    # noteworthy training-time behaviour.
    total_skipped = int(sum(nonfinite_skips_per_epoch))
    total_attempted = int(sum(attempted_steps_per_epoch))
    skip_pct_overall = (
        100.0 * total_skipped / total_attempted if total_attempted else 0.0
    )
    skip_pct_per_epoch = [
        100.0 * s / max(a, 1)
        for s, a in zip(nonfinite_skips_per_epoch, attempted_steps_per_epoch)
    ]
    training_summary = {
        "n_epochs_run": len(train_losses),
        "best_val_loss": float(best_val_loss),
        "final_train_loss": float(train_losses[-1]) if train_losses else None,
        "final_val_loss": float(val_losses[-1]) if val_losses else None,
        # Numerical-stability flags — populated by the non-finite-gradient
        # guard in the training loop. Above 5% suggests the head's
        # parameter clamps are too loose or the LR is too high for the
        # chosen likelihood; non-zero on Gaussian-only runs is suspicious
        # and worth investigating.
        "nonfinite_skips_total": total_skipped,
        "training_steps_attempted_total": total_attempted,
        "nonfinite_skip_pct_overall": skip_pct_overall,
        "nonfinite_skip_pct_per_epoch": skip_pct_per_epoch,
        "nonfinite_skips_per_epoch": list(nonfinite_skips_per_epoch),
        # Config echo (subset of run-defining args).
        "target_variables": list(args.target_variables),
        "likelihood_per_variable": {
            v: likelihood_per_variable[v] for v in args.target_variables
        },
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
    logger.info("Run evaluate.py on the checkpoint for detailed test metrics.")


if __name__ == "__main__":
    main()