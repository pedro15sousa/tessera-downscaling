"""Station-patch dataset for the VAE: cache, station filter, normalisation, crop.

The encoder trains on the per-station TESSERA patch file written by
``scripts/data/extract_tessera_patches_local.py``:
``processed/tessera_station_patches/patch_embeddings_<year>_p128.npy``, shape
``(38870, 128, 128, 128)`` = (station, row, column, channel), row-aligned with
``station_list_filtered.csv`` next to it. Patches are north-up and centred on
the station.

The same machinery serves the two foundation-model arms of the benchmark,
whose patches are written by ``scripts/patch_encoder/extract/`` against the
same station list: AlphaEarth, ``(38870, 128, 128, 64)``, and OlmoEarth,
``(38870, 16, 16, 768)`` -- a token grid rather than a 10 m raster, used whole
instead of cropped. Nothing here is specific to a channel count or a patch
size: both come from the file, and the model is built to match
(``train_vae.py``).

Three things happen between the file and the model:

1. *Validity.* :func:`prepare_data` scans the file once, flagging stations
   whose patch is all zero (no TESSERA coverage) or holds non-finite/extreme
   values, and computes the per-channel mean/std used to z-score the input.
   Both live in ``<cache_dir>/<patch-file stem>/cache.npz``, namespaced by
   patch filename because indices and statistics belong to one extraction.
   :func:`filter_elevation_sentinels` then drops the stations whose GHCNh
   elevation is a missing-data sentinel, which is what the auxiliary elevation
   head would otherwise be trained on.
2. *Cropping.* Patches are stored at 128x128 pixels; ``data.crop_size`` cuts a
   centred window out of them at read time (64 in the paper, a 640 m window at
   10 m resolution). Since the statistics are per-channel over the full stored
   patch, every crop size of one extraction shares a single cache.
3. *Normalisation.* Patches are z-scored per channel; auxiliary targets are
   z-scored per target over the usable stations.

The patch file is memory-mapped and read one patch at a time, so memory stays
flat whatever its size (326 GB for the paper's extraction).
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, Subset

from ..paths import patch_encoder_dir, processed_dir, resolve

logger = logging.getLogger(__name__)

# A patch with any |value| above this is a corrupted extraction, not a landscape.
OUTLIER_THRESHOLD = 1000.0
# GHCNh elevations outside this range are missing-data sentinels (-999.9, 9999).
ELEV_SENTINEL_LOW = -900
ELEV_SENTINEL_HIGH = 8848

# Directories under ``processed/`` holding station patch files, one per
# embedding source. ``eval_vae.py`` searches them in this order when the patch
# file recorded in a checkpoint no longer resolves.
STATION_PATCH_DIRS = (
    "tessera_station_patches",
    "alphaearth_station_patches",
    "olmoearth_station_patches",
)
STATION_PATCH_DIR = STATION_PATCH_DIRS[0]


def default_patches_path(year: int = 2017) -> Path:
    """Station patch file for ``year`` under the data root (2017 in the paper)."""
    return processed_dir(STATION_PATCH_DIR, f"patch_embeddings_{year}_p128.npy")


def default_stations_path() -> Path:
    """Station CSV row-aligned with the patch files."""
    return processed_dir(STATION_PATCH_DIR, "station_list_filtered.csv")


def default_cache_dir() -> Path:
    """Where dataset caches live: ``<root>/tessera_patch_encoder/outputs/dataset_cache``."""
    return patch_encoder_dir("outputs", "dataset_cache")


def _scan_patches(patches_path: Path) -> dict[str, np.ndarray]:
    """One pass over the patch file, splitting rows into valid/zero/outlier."""
    mmap = np.load(str(patches_path), mmap_mode="r")
    n_patches = mmap.shape[0]
    logger.info(f"Scanning {n_patches} patches for invalid values...")

    is_zero = np.zeros(n_patches, dtype=bool)
    is_outlier = np.zeros(n_patches, dtype=bool)

    for i in range(n_patches):
        max_abs = np.abs(mmap[i]).max()
        if not np.isfinite(max_abs) or max_abs > OUTLIER_THRESHOLD:
            is_outlier[i] = True
        elif max_abs == 0:
            is_zero[i] = True
        if (i + 1) % 5000 == 0:
            logger.info(f"  {i + 1}/{n_patches}")

    valid_idx = np.where(~(is_zero | is_outlier))[0]
    logger.info(
        f"  valid {len(valid_idx)} ({100 * len(valid_idx) / n_patches:.1f}%), "
        f"zero {int(is_zero.sum())}, outlier {int(is_outlier.sum())}"
    )
    return {
        "valid_indices": valid_idx,
        "zero_indices": np.where(is_zero)[0],
        "outlier_indices": np.where(is_outlier)[0],
    }


def _compute_channel_stats(
    mmap: np.ndarray, valid_idx: np.ndarray, n_sample: int = 2000
) -> tuple[np.ndarray, np.ndarray]:
    """Per-channel mean/std over a fixed random sample of valid patches.

    Chained Welford updates over one patch at a time, so the full file is never
    materialised. The sample is drawn with a fixed seed: the statistics are
    part of the cache and must be reproducible.
    """
    rng = np.random.RandomState(42)
    sample = rng.choice(valid_idx, min(n_sample, len(valid_idx)), replace=False)
    n_channels = mmap.shape[-1]
    logger.info(f"Computing channel stats from {len(sample)} patches...")

    count = 0
    ch_mean = np.zeros(n_channels, dtype=np.float64)
    ch_m2 = np.zeros(n_channels, dtype=np.float64)

    for i, idx in enumerate(sample):
        patch = mmap[idx].astype(np.float32).reshape(-1, n_channels)
        batch_mean = patch.mean(axis=0)
        batch_var = patch.var(axis=0)
        batch_n = patch.shape[0]
        new_count = count + batch_n
        delta = batch_mean - ch_mean
        ch_mean = ch_mean + delta * batch_n / new_count
        ch_m2 = ch_m2 + batch_var * batch_n + delta**2 * count * batch_n / new_count
        count = new_count
        if (i + 1) % 500 == 0:
            logger.info(f"  {i + 1}/{len(sample)}")

    ch_std = np.sqrt(ch_m2 / count)
    logger.info(
        f"  mean in [{ch_mean.min():.2f}, {ch_mean.max():.2f}], "
        f"std in [{ch_std.min():.4f}, {ch_std.max():.4f}]"
    )
    return ch_mean.astype(np.float32), ch_std.astype(np.float32)


def prepare_data(
    patches_path: str | Path,
    cache_dir: str | Path | None = None,
    rebuild: bool = False,
) -> dict[str, np.ndarray]:
    """Build (or load) the cache of valid indices and normalisation statistics.

    The scan takes minutes over a 326 GB file, so a sweep pre-builds the cache
    once per patch file (``scripts/patch_encoder/prebuild_cache.py``) instead
    of letting every job scan the same file concurrently.

    Args:
        patches_path: Patch ``.npy``; relative paths resolve under the data root.
        cache_dir: Root of the cache tree (default :func:`default_cache_dir`).
            The cache itself is written to ``<cache_dir>/<patch stem>/cache.npz``.
        rebuild: Rescan even if the cache exists.

    Returns:
        ``valid_indices``, ``zero_indices``, ``outlier_indices``,
        ``channel_mean`` and ``channel_std``.
    """
    patches_path = resolve(patches_path)
    cache_root = Path(cache_dir) if cache_dir is not None else default_cache_dir()
    cache_file = resolve(cache_root) / patches_path.stem / "cache.npz"

    if cache_file.exists() and not rebuild:
        npz = np.load(cache_file)
        cache = {key: npz[key] for key in npz.files}
        logger.info(
            f"Loaded dataset cache {cache_file} "
            f"({len(cache['valid_indices'])} valid patches)"
        )
        return cache

    logger.info(f"Building dataset cache for {patches_path} (one-time scan)")
    cache = _scan_patches(patches_path)
    mmap = np.load(str(patches_path), mmap_mode="r")
    ch_mean, ch_std = _compute_channel_stats(mmap, cache["valid_indices"])
    cache["channel_mean"] = ch_mean
    cache["channel_std"] = ch_std

    cache_file.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache_file, **cache)
    logger.info(f"Cache saved to {cache_file}")
    return cache


def filter_elevation_sentinels(
    valid_indices: np.ndarray, stations_path: str | Path
) -> np.ndarray:
    """Drop stations whose GHCNh elevation is a missing-data sentinel.

    The second half of the station filter: :func:`prepare_data` removes patches
    the extraction could not fill, this removes stations whose metadata cannot
    supervise the auxiliary elevation head. Training and evaluation must apply
    both, in this order, to see the same station set.
    """
    stations = pd.read_csv(resolve(stations_path))
    elevation = stations.iloc[valid_indices]["elevation"].to_numpy()
    usable = (elevation > ELEV_SENTINEL_LOW) & (elevation <= ELEV_SENTINEL_HIGH)
    return valid_indices[usable]


class TesseraPatchDataset(Dataset):
    """Memory-mapped station patches with their auxiliary targets.

    Each item is ``{"patch": (C, S, S) float32 tensor}`` plus one entry per
    auxiliary target, z-scored and NaN where the station's value is a sentinel
    (the auxiliary loss masks those out).

    Args:
        patches_path: Patch ``.npy`` of shape ``(N, S_stored, S_stored, C)``.
        stations_path: Station CSV row-aligned with the patch file.
        valid_indices: Rows to serve, in order -- the output of
            :func:`prepare_data` narrowed by :func:`filter_elevation_sentinels`.
        channel_mean, channel_std: ``(C,)`` normalisation statistics from the
            cache. Never recompute these at inference time; latents are only
            comparable across runs if they share the training statistics.
        aux_targets: Station-CSV columns to serve as auxiliary targets.
        target_stats: ``{column: {"mean": float, "std": float}}`` for z-scoring
            the targets; computed from the served stations when omitted.
        crop_size: Side of the centred window taken out of each stored patch.
            Patches are station-centred, so the crop keeps the station in the
            middle. ``None`` (or a value at least the stored size) uses the
            whole patch.
    """

    def __init__(
        self,
        patches_path: str | Path,
        stations_path: str | Path,
        valid_indices: np.ndarray,
        channel_mean: np.ndarray,
        channel_std: np.ndarray,
        aux_targets: list[str] | None = None,
        target_stats: dict[str, dict[str, float]] | None = None,
        crop_size: int | None = None,
    ) -> None:
        self.mmap = np.load(str(resolve(patches_path)), mmap_mode="r")
        self.valid_indices = np.asarray(valid_indices)
        self.channel_mean = channel_mean.astype(np.float32)
        self.channel_std = np.maximum(channel_std.astype(np.float32), 1e-6)

        self.stored_size = int(self.mmap.shape[1])
        if crop_size is None or int(crop_size) >= self.stored_size:
            self.crop_size = None
            self.spatial_size = self.stored_size
            self._crop_offset = 0
        else:
            self.crop_size = int(crop_size)
            self.spatial_size = int(crop_size)
            self._crop_offset = (self.stored_size - self.crop_size) // 2

        stations = pd.read_csv(resolve(stations_path))
        self.metadata = stations.iloc[self.valid_indices].reset_index(drop=True)

        self.aux_targets = aux_targets or []
        self.target_stats = (
            target_stats if target_stats is not None else self._compute_target_stats()
        )

    def _compute_target_stats(self) -> dict[str, dict[str, float]]:
        stats: dict[str, dict[str, float]] = {}
        for col in self.aux_targets:
            if col not in self.metadata.columns:
                continue
            values = self.metadata[col].to_numpy().astype(np.float32)
            valid = np.isfinite(values)
            if col == "elevation":
                valid &= (values > ELEV_SENTINEL_LOW) & (values <= ELEV_SENTINEL_HIGH)
            stats[col] = {
                "mean": float(values[valid].mean()),
                "std": float(values[valid].std()) + 1e-8,
            }
        return stats

    def __len__(self) -> int:
        return len(self.valid_indices)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        patch = self.mmap[self.valid_indices[idx]].astype(np.float32)  # (S, S, C)
        if self.crop_size is not None:
            off, size = self._crop_offset, self.crop_size
            patch = patch[off : off + size, off : off + size, :]
        patch = (patch - self.channel_mean) / self.channel_std
        patch = np.transpose(patch, (2, 0, 1))  # -> (C, S, S)

        item = {"patch": torch.from_numpy(patch.copy())}
        for col in self.aux_targets:
            if col not in self.metadata.columns:
                continue
            value = float(self.metadata.iloc[idx][col])
            if col == "elevation" and not (
                ELEV_SENTINEL_LOW < value <= ELEV_SENTINEL_HIGH
            ):
                value = float("nan")  # masked by the auxiliary loss
            if col in self.target_stats and np.isfinite(value):
                stats = self.target_stats[col]
                value = (value - stats["mean"]) / stats["std"]
            item[col] = torch.tensor(value, dtype=torch.float32)
        return item


def create_dataloaders(
    cfg: dict,
    cache_dir: str | Path | None = None,
    rebuild_cache: bool = False,
) -> tuple[DataLoader, DataLoader, TesseraPatchDataset]:
    """Build the train/validation loaders described by a run config.

    Applies both halves of the station filter, then splits the usable stations
    into training and validation sets with a permutation seeded by
    ``training.seed`` -- so the split depends only on the config, not on the
    machine or the run order.

    Returns:
        ``(train_loader, val_loader, dataset)``. The dataset carries the
        spatial size and channel count the model must be built for.
    """
    data_cfg = cfg["data"]
    train_cfg = cfg["training"]
    aux_cfg = cfg.get("auxiliary", {})

    cache = prepare_data(
        data_cfg["patches_path"], cache_dir=cache_dir, rebuild=rebuild_cache
    )
    usable_idx = filter_elevation_sentinels(
        cache["valid_indices"], data_cfg["stations_path"]
    )

    dataset = TesseraPatchDataset(
        patches_path=data_cfg["patches_path"],
        stations_path=data_cfg["stations_path"],
        valid_indices=usable_idx,
        channel_mean=cache["channel_mean"],
        channel_std=cache["channel_std"],
        aux_targets=aux_cfg.get("targets", []) if aux_cfg.get("enable", False) else [],
        crop_size=data_cfg.get("crop_size"),
    )

    n_total = len(dataset)
    perm = np.random.RandomState(train_cfg["seed"]).permutation(n_total)
    n_val = int(n_total * train_cfg["val_split"])

    train_loader = DataLoader(
        Subset(dataset, perm[n_val:]),
        batch_size=train_cfg["batch_size"],
        shuffle=True,
        num_workers=train_cfg["num_workers"],
        pin_memory=train_cfg["pin_memory"],
        drop_last=True,
    )
    val_loader = DataLoader(
        Subset(dataset, perm[:n_val]),
        batch_size=train_cfg["batch_size"],
        shuffle=False,
        num_workers=train_cfg["num_workers"],
        pin_memory=train_cfg["pin_memory"],
    )

    crop_note = (
        f"centre-cropped {dataset.stored_size} -> {dataset.spatial_size}"
        if dataset.crop_size
        else f"patch {dataset.spatial_size}px"
    )
    logger.info(
        f"Dataset: {n_total} usable patches "
        f"({n_total - n_val} train, {n_val} val), {crop_note}"
    )
    return train_loader, val_loader, dataset
