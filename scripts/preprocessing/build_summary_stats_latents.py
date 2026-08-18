"""Build hand-crafted summary-statistics ‘latents’ from TESSERA patches.

Produces a station-aligned ``.npy`` file (one per requested dim) matching the
shape and load-time contract of the existing VAE-latent files, so they can be
swapped in via ``--vae-latents-path`` without any model-side changes.

For each station's ``(64, 64, 128)`` TESSERA patch:

  1. Identify the top-K most cross-station-variant channels, using
     ``spatial_split == 'train'`` stations from the dataset's ``stations.csv``
     to compute per-channel variance (test stations are EXCLUDED to avoid
     contamination of the channel selection step).
  2. For each station and each selected channel, compute four statistics:
     mean, std, p10, p90 over the 64×64 spatial dimensions, ignoring NaN
     pixels (gracefully handles partial-coverage patches).
  3. Concatenate into a ``(n_stations, 4*K)`` array. K = dim // 4.

Output column layout is stat-blocked:

  [mean_c0, mean_c1, ..., mean_cK-1,
   std_c0,  std_c1,  ..., std_cK-1,
   p10_c0,  p10_c1,  ..., p10_cK-1,
   p90_c0,  p90_c1,  ..., p90_cK-1]

The trainer applies its standard global per-column z-score on load (same as
for the VAE latents), so the raw scale of the columns doesn't matter.

Memory: streams the ``(38870, 64, 64, 128)`` ~76 GB patch file in chunks via
``np.load(..., mmap_mode='r')``. Peak RAM is ``chunk_size * 2 MB``.

Example:
    python scripts/preprocessing/build_summary_stats_latents.py \\
        --tessera-path  ${BASE_DIR}/processed/tessera_global/patch_embeddings_2024.npy \\
        --tessera-station-csv ${BASE_DIR}/processed/tessera_global/station_list_filtered.csv \\
        --dataset-stations-csv ${BASE_DIR}/dataset_timestamp_global/stations.csv \\
        --output-dir    ${BASE_DIR}/processed/ \\
        --dims          16,64
"""

from __future__ import annotations

import argparse
import json
import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--tessera-path", required=True, type=Path,
                   help="Path to patch_embeddings_*.npy of shape (N, 64, 64, 128).")
    p.add_argument("--tessera-station-csv", required=True, type=Path,
                   help="CSV row-aligned with the patches file (station_id column).")
    p.add_argument("--dataset-stations-csv", required=True, type=Path,
                   help="Dataset's stations.csv with station_id + spatial_split columns. "
                        "Only spatial_split=='train' rows contribute to channel-selection variance.")
    p.add_argument("--output-dir", required=True, type=Path,
                   help="Where to write station_summary_stats_dim{N}.npy + meta JSON.")
    p.add_argument("--dims", default="16,64",
                   help="Comma-separated output dims. Each must be a multiple of 4 "
                        "(4 stats per channel: mean, std, p10, p90). Default: 16,64.")
    p.add_argument("--chunk-size", type=int, default=128,
                   help="Stations per chunk for streaming. Peak RAM ≈ chunk_size * 2 MB.")
    p.add_argument("--force", action="store_true",
                   help="Overwrite existing output files.")
    p.add_argument(
        "--outlier-threshold", type=float, default=1000.0,
        help="Patch-level outlier filter (mirrors TESSERA patch-encoder repo "
             "at src/common/dataset.py:66). A patch is flagged outlier and its "
             "output row set to NaN if any of its 64*64*128 values is "
             "non-finite OR exceeds this threshold in absolute value. "
             "Default 1000.0 matches the upstream VAE training pipeline.",
    )
    p.add_argument(
        "--center-crop", type=int, default=0,
        help="If > 0, statistics (and the outlier filter) are computed over "
             "the central crop x crop spatial window of each patch instead of "
             "its full extent. Use 64 on the p128 extractions so the stats "
             "see exactly the crop64 VAE encoder's input window (~640 m).",
    )
    p.add_argument(
        "--align-nan-mask-to", type=Path, default=None,
        help="Optional station-aligned VAE-latents .npy. After the stats are "
             "computed, every station whose row in this file contains NaN is "
             "also set to NaN in the output, so the dataset loader's NaN "
             "filter yields the same station set as the reference latent arm "
             "(plus any rows the stats pass itself flagged as outliers).",
    )
    p.add_argument(
        "--output-prefix", default="station_summary_stats",
        help="Output basename prefix; files are written as "
             "<prefix>_dim{N}.npy with a sibling _meta.json.",
    )
    return p.parse_args()


def build_train_mask(
    tessera_csv: Path,
    dataset_csv: Path,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (train_mask, tessera_station_ids).

    train_mask is over the TESSERA rows. True iff the station is present in
    the dataset stations.csv AND its spatial_split is 'train'. All other
    stations (test, or not in the dataset at all) are False.
    """
    tessera = pd.read_csv(tessera_csv)
    dataset = pd.read_csv(dataset_csv)

    if "station_id" not in tessera.columns:
        raise ValueError(f"Missing 'station_id' column in {tessera_csv}")
    if "station_id" not in dataset.columns:
        raise ValueError(f"Missing 'station_id' column in {dataset_csv}")
    if "spatial_split" not in dataset.columns:
        raise ValueError(
            f"Missing 'spatial_split' column in {dataset_csv}. "
            f"Expected one of {{train, test}} per row."
        )

    # Map station_id -> spatial_split via dataset (subset of tessera).
    id_to_split = dict(zip(
        dataset["station_id"].astype(str),
        dataset["spatial_split"].astype(str),
    ))
    train_mask = np.array(
        [id_to_split.get(str(sid), "") == "train"
         for sid in tessera["station_id"].values],
        dtype=bool,
    )
    return train_mask, tessera["station_id"].values


def compute_per_station_channel_means(
    patches: np.ndarray,
    n_stations: int,
    n_channels: int,
    chunk_size: int,
    outlier_threshold: float,
    crop: tuple[slice, slice] | None = None,
) -> tuple[np.ndarray, int]:
    """Pass 1: per-station per-channel mean, with VAE-style outlier filtering.

    Mirrors the TESSERA patch-encoder repo's training filter
    (src/common/dataset.py:66): a patch is flagged outlier iff any of its
    64*64*128 values is non-finite OR exceeds ``outlier_threshold`` in
    absolute value. Outlier patches contribute NaN to every channel of
    per_station_means; the final output file will then carry NaN rows for
    those patches, matching the NaN-row convention of the VAE-latents files.

    Returns ``(per_station_means_float64, n_outliers_detected)``.
    """
    out = np.full((n_stations, n_channels), np.nan, dtype=np.float64)
    n_outliers = 0
    log_every = max(1, (n_stations // chunk_size) // 10)
    chunk_count = 0
    for start in range(0, n_stations, chunk_size):
        end = min(start + chunk_size, n_stations)
        if crop is not None:
            chunk = np.asarray(patches[start:end, crop[0], crop[1], :])
        else:
            chunk = np.asarray(patches[start:end])

        # Per-patch flag. np.abs(NaN)=NaN, np.abs(Inf)=Inf, both propagate
        # through .max(...) so ~np.isfinite catches them together with the
        # explicit magnitude check.
        max_abs_per_patch = np.abs(chunk).max(axis=(1, 2, 3))
        is_outlier = (
            ~np.isfinite(max_abs_per_patch)
            | (max_abs_per_patch > outlier_threshold)
        )
        n_outliers += int(is_outlier.sum())

        keep_local = np.where(~is_outlier)[0]
        if keep_local.size > 0:
            kept = chunk[keep_local]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                out[start + keep_local] = np.nanmean(
                    kept, axis=(1, 2), dtype=np.float64
                )
        chunk_count += 1
        if chunk_count % log_every == 0 or end == n_stations:
            pct = 100 * end // n_stations
            logger.info(f"  Pass 1: {end}/{n_stations} ({pct}%)")
    return out, n_outliers


def select_top_k_channels(
    per_station_means: np.ndarray,
    train_mask: np.ndarray,
    k: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Top-K channels by cross-train-station variance.

    With per-patch outlier filtering done upstream, channel-level pathology is
    already handled — corrupt patches contributed NaN rows that ``nanvar``
    correctly skips. NaN-variance channels (e.g., where every train station's
    patch was an outlier in that channel) are pushed to the end via an
    -inf sort key for determinism.
    """
    train_means = per_station_means[train_mask]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        channel_variances = np.nanvar(train_means, axis=0, dtype=np.float64)
    sort_key = np.where(np.isnan(channel_variances), -np.inf, channel_variances)
    sorted_indices = np.argsort(-sort_key, kind="stable")
    return sorted_indices[:k], channel_variances


def compute_stats_for_selected(
    patches: np.ndarray,
    n_stations: int,
    selected_channels: np.ndarray,
    chunk_size: int,
    outlier_threshold: float,
    crop: tuple[slice, slice] | None = None,
) -> np.ndarray:
    """Pass 2: per-station (mean, std, p10, p90) for selected channels.

    Same per-patch outlier filter as Pass 1: outlier patches get NaN output
    rows so the file is row-aligned with the VAE-latents convention.
    Float64 accumulators on mean and std to avoid float32 overflow on the
    surviving extreme-but-legal values.
    """
    k = len(selected_channels)
    dim = 4 * k
    out = np.full((n_stations, dim), np.nan, dtype=np.float32)
    log_every = max(1, (n_stations // chunk_size) // 10)
    chunk_count = 0
    for start in range(0, n_stations, chunk_size):
        end = min(start + chunk_size, n_stations)
        if crop is not None:
            chunk = np.asarray(patches[start:end, crop[0], crop[1], :])
        else:
            chunk = np.asarray(patches[start:end])

        max_abs_per_patch = np.abs(chunk).max(axis=(1, 2, 3))
        is_outlier = (
            ~np.isfinite(max_abs_per_patch)
            | (max_abs_per_patch > outlier_threshold)
        )
        keep_local = np.where(~is_outlier)[0]
        if keep_local.size > 0:
            kept = chunk[keep_local]
            kept_sel = kept[..., selected_channels]
            kept_flat = kept_sel.reshape(kept_sel.shape[0], -1, k)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                means = np.nanmean(kept_flat, axis=1, dtype=np.float64).astype(np.float32)
                stds  = np.nanstd(kept_flat,  axis=1, dtype=np.float64).astype(np.float32)
                p10s  = np.nanpercentile(kept_flat, 10, axis=1).astype(np.float32)
                p90s  = np.nanpercentile(kept_flat, 90, axis=1).astype(np.float32)
            tgt_rows = start + keep_local
            out[tgt_rows, 0*k:1*k] = means
            out[tgt_rows, 1*k:2*k] = stds
            out[tgt_rows, 2*k:3*k] = p10s
            out[tgt_rows, 3*k:4*k] = p90s
        chunk_count += 1
        if chunk_count % log_every == 0 or end == n_stations:
            pct = 100 * end // n_stations
            logger.info(f"  Pass 2 (dim={dim}): {end}/{n_stations} ({pct}%)")
    return out


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    dims = [int(d.strip()) for d in args.dims.split(",")]
    for d in dims:
        if d % 4 != 0 or d <= 0:
            raise ValueError(f"--dims values must be positive multiples of 4; got {d}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    target_paths = {
        d: args.output_dir / f"{args.output_prefix}_dim{d}.npy"
        for d in dims
    }
    if not args.force:
        existing = [p for p in target_paths.values() if p.exists()]
        if existing:
            raise FileExistsError(
                f"Output files already exist: {[str(p) for p in existing]}. "
                f"Pass --force to overwrite."
            )

    # ---- Step 1: train mask + sanity ----
    logger.info(f"Reading station CSVs:")
    logger.info(f"  tessera : {args.tessera_station_csv}")
    logger.info(f"  dataset : {args.dataset_stations_csv}")
    train_mask, tessera_ids = build_train_mask(
        args.tessera_station_csv, args.dataset_stations_csv
    )
    n_stations = len(tessera_ids)
    n_train = int(train_mask.sum())
    logger.info(
        f"  Total tessera rows: {n_stations:,}; "
        f"train stations in dataset: {n_train:,} ({100*n_train/n_stations:.1f}%)"
    )
    if n_train == 0:
        raise RuntimeError(
            "No tessera rows matched a train station in the dataset CSV. "
            "Check that station_id values align between the two CSVs."
        )

    # ---- Step 2: open patches with mmap ----
    logger.info(f"Opening patches (mmap): {args.tessera_path}")
    patches = np.load(args.tessera_path, mmap_mode="r")
    if patches.ndim != 4:
        raise ValueError(f"Expected 4D patches (N, H, W, C); got shape {patches.shape}")
    if patches.shape[0] != n_stations:
        raise ValueError(
            f"Row mismatch: patches has {patches.shape[0]} rows but tessera CSV "
            f"lists {n_stations} stations."
        )
    n_channels = patches.shape[-1]
    logger.info(
        f"  Patches shape: {patches.shape} ({patches.dtype}); "
        f"file size: {patches.nbytes / 1e9:.1f} GB"
    )

    # Optional central spatial crop, matching the VAE encoder's input window.
    crop = None
    if args.center_crop > 0:
        h, w = patches.shape[1], patches.shape[2]
        c = args.center_crop
        if c > h or c > w:
            raise ValueError(
                f"--center-crop {c} exceeds patch spatial extent ({h}x{w})."
            )
        h0, w0 = (h - c) // 2, (w - c) // 2
        crop = (slice(h0, h0 + c), slice(w0, w0 + c))
        logger.info(
            f"  Central crop: [{h0}:{h0+c}, {w0}:{w0+c}] of {h}x{w} "
            f"({c}x{c} window)"
        )

    # ---- Step 3: Pass 1 — per-station per-channel means ----
    logger.info(f"Pass 1: computing per-station means across {n_channels} channels...")
    per_station_means, n_outliers = compute_per_station_channel_means(
       patches, n_stations, n_channels, args.chunk_size, args.outlier_threshold,
       crop=crop,
    )
    logger.info(
       f"  Outlier patches (|x|>{args.outlier_threshold:.0f} or non-finite): "
       f"{n_outliers:,}/{n_stations:,} ({100*n_outliers/n_stations:.2f}%)"
    )

    # ---- Step 4: channel selection (do once for the max-K, slice prefixes for smaller dims) ----
    max_k = max(dims) // 4
    logger.info(
        f"Selecting top-{max_k} channels by cross-train-station variance "
        f"(n_train={n_train:,})..."
    )
    all_selected, channel_variances = select_top_k_channels(
        per_station_means, train_mask, max_k,
    )
    logger.info(
        f"  Variance (top-{max_k}): "
        f"{[float(channel_variances[c]) for c in all_selected]}"
    )
    logger.info(f"  Channel indices: {all_selected.tolist()}")

    # ---- Step 5: Pass 2 per requested dim, save outputs ----
    for d in dims:
        k = d // 4
        selected = all_selected[:k]
        logger.info(
            f"\nDim {d}: top-{k} channels = {selected.tolist()} "
            f"-> stats {{mean, std, p10, p90}} -> {d}-d output"
        )
        out = compute_stats_for_selected(
            patches, n_stations, selected, args.chunk_size, args.outlier_threshold,
            crop=crop,
        )

        # Optional NaN-mask alignment: adopt the reference latent file's
        # invalid rows so the loader filters to the same station set as the
        # reference arm (union with rows the stats pass itself NaN'd).
        n_nan_pre_align = int(np.isnan(out).any(axis=1).sum())
        if args.align_nan_mask_to is not None:
            ref = np.load(args.align_nan_mask_to)
            if ref.shape[0] != n_stations:
                raise ValueError(
                    f"--align-nan-mask-to row count {ref.shape[0]} does not "
                    f"match {n_stations} stations."
                )
            ref_nan = np.isnan(ref).any(axis=1)
            out[ref_nan] = np.nan
            logger.info(
                f"  NaN-mask alignment to {args.align_nan_mask_to.name}: "
                f"{n_nan_pre_align} own NaN rows -> "
                f"{int(np.isnan(out).any(axis=1).sum())} after union with "
                f"{int(ref_nan.sum())} reference NaN rows"
            )

        nan_rows = int(np.isnan(out).any(axis=1).sum())
        valid = out[~np.isnan(out).any(axis=1)]
        out_path = target_paths[d]

        non_finite = (~np.isfinite(out)).sum()
        n_finite_per_row = np.isfinite(out).all(axis=1).sum()
        if non_finite > 0:
            finite_only = out[np.isfinite(out)]
            logger.warning(
                f"  WARNING: {non_finite} non-finite cells in output "
                f"({n_finite_per_row}/{n_stations} rows fully finite). "
                f"Finite-only stats: range=[{finite_only.min():.3f}, {finite_only.max():.3f}], "
                f"mean={finite_only.mean():.3f}, std={finite_only.std():.3f}"
            )

        np.save(out_path, out)
        logger.info(
            f"  Saved {out_path}  shape={out.shape}, "
            f"NaN rows={nan_rows}/{n_stations}, "
            f"value range over valid rows: [{valid.min():.3f}, {valid.max():.3f}], "
            f"mean={valid.mean():.3f}, std={valid.std():.3f}"
        )

        meta_path = out_path.with_name(out_path.stem + "_meta.json")
        meta = {
            "dim": d,
            "stats": ["mean", "std", "p10", "p90"],
            "stats_order": (
                "stat-blocked: cols [0:K]=means, [K:2K]=stds, "
                "[2K:3K]=p10s, [3K:4K]=p90s"
            ),
            "n_channels_selected": int(k),
            "selected_channel_indices": [int(c) for c in selected.tolist()],
            "selected_channel_variances": [float(channel_variances[c]) for c in selected],
            "channel_selection_rule": (
                "top-K by cross-train-station variance of per-station "
                "per-channel means (over patch spatial dims), after applying "
                "the per-patch outlier filter"
            ),
            "outlier_threshold": float(args.outlier_threshold),
            "n_outlier_patches": int(n_outliers),
            "n_train_stations_in_variance": int(n_train),
            "n_total_stations": int(n_stations),
            "n_nan_rows_in_output": nan_rows,
            "tessera_path": str(args.tessera_path),
            "tessera_station_csv": str(args.tessera_station_csv),
            "dataset_stations_csv": str(args.dataset_stations_csv),
            "center_crop": int(args.center_crop) if args.center_crop > 0 else None,
            "n_nan_rows_pre_vae_alignment": n_nan_pre_align,
            "nan_mask_aligned_to_vae": (
                str(args.align_nan_mask_to)
                if args.align_nan_mask_to is not None else None
            ),
        }
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)
        logger.info(f"  Saved {meta_path}")

    logger.info("\nDone.")


if __name__ == "__main__":
    main()