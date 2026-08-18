"""Shared helpers behind :mod:`tessera_downscaling.data.dataset`.

The dataset classes delegate the non-trivial, cross-cutting data plumbing to
the functions in this module:

    * filter stations by valid TESSERA patches / non-NaN precomputed vectors
    * load (or compute) ERA5 channel z-score statistics
    * build one episode's context grid (ERA5 + static + lat/lon + time [+ lead])
    * select which target stations are observed at a timestamp
    * stitch the per-episode dict returned by ``__getitem__``
    * partition a sorted list of timestamps by temporal split
    * collate a batch of variable-size target sets

The helpers know nothing about *which* Dataset class is calling them: they
receive plain arrays / dicts / paths and return the same, so the dataset
classes stay focused on init, indexing and orchestration.

Targets are 6-hourly UTC snapshots of GHCNh observations: ``t2m`` is the
instantaneous 2 m temperature and ``wind`` the 10 m wind speed at the synoptic
hour (00/06/12/18 UTC).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch

SUPPORTED_TARGET_VARIABLES = {"t2m", "wind"}

# A TESSERA patch is valid iff its centre pixel is non-zero AND at least this
# fraction of all pixels in the 64x64 patch have any non-zero channel. The
# threshold lives at the dataset level (rather than the preprocessor) so it can
# be changed without re-extracting patches: re-running train.py / evaluate.py
# with --min-tessera-patch-coverage is enough. 0.0 recovers the centre-only rule.
DEFAULT_MIN_TESSERA_PATCH_COVERAGE: float = 0.5

# Scale for the lead-time context channel (0 = ERA5 analysis, up to 72h Aurora
# forecast). lead_hours / MAX_LEAD_HOURS keeps the channel in [0, 1] and lets the
# lead-conditioned model interpolate σ(lead) to unseen horizons.
MAX_LEAD_HOURS = 72.0


# ---------------------------------------------------------------------------
# Target-variable validation
# ---------------------------------------------------------------------------


def validate_target_variables(target_variables: list[str] | None) -> list[str]:
    """Validate the target-variable list; default ``["t2m"]``.

    Supported: ``t2m`` (2 m temperature) and ``wind`` (10 m wind speed).
    """
    tv = list(target_variables) if target_variables else ["t2m"]
    for name in tv:
        if name not in SUPPORTED_TARGET_VARIABLES:
            raise ValueError(
                f"Unknown target variable '{name}'. "
                f"Supported: {sorted(SUPPORTED_TARGET_VARIABLES)}"
            )
    return tv


# ---------------------------------------------------------------------------
# Station filtering — TESSERA patches
# ---------------------------------------------------------------------------


@dataclass
class TesseraFilterResult:
    """Output of :func:`filter_stations_by_tessera_patches`."""

    kept_mask: np.ndarray  # bool, len == len(spatial_indices_in)
    tessera_row_indices: np.ndarray  # int64, patch-file row per kept station


def filter_stations_by_tessera_patches(
    stations_df: pd.DataFrame,
    spatial_indices: np.ndarray,
    tessera_path: Path,
    tessera_station_csv: Path,
    min_patch_coverage: float = DEFAULT_MIN_TESSERA_PATCH_COVERAGE,
) -> TesseraFilterResult:
    """Filter stations to those with a valid TESSERA patch.

    ``tessera_path`` is an ``(N, H, W, C)`` patch array (memory-mapped, never
    materialised) row-aligned with ``tessera_station_csv``. A station's patch
    is valid iff BOTH:

      * the centre pixel has at least one non-zero channel (the target's own
        location carries real signal), AND
      * the fraction of pixels with at least one non-zero channel is
        ``>= min_patch_coverage`` (the surrounding context carries enough real
        data).

    The centre rule alone occasionally let through patches that were
    near-empty everywhere except the centre pixel — coastal-edge stations
    where the patch happened to land on a single intact land cell surrounded
    by water. Conversely, it also rejected patches with a one-pixel hole
    exactly at the station, despite >99% of the patch being real data.
    Combining the two rules removes both pathologies. Pass 0.0 to disable the
    coverage check and recover the centre-pixel-only rule.

    Stations absent from ``tessera_station_csv`` are dropped. Applied for BOTH
    baseline and TESSERA-enabled models, so they always operate on identical
    station sets.
    """
    tessera_stations = pd.read_csv(tessera_station_csv)
    patches_mmap = np.load(str(tessera_path), mmap_mode="r")

    n_tessera = patches_mmap.shape[0]
    h = patches_mmap.shape[1]
    w = patches_mmap.shape[2]
    centre = h // 2
    # Per-row centre-pixel-non-zero check.
    centre_nonzero = np.any(patches_mmap[:, centre, centre, :] != 0, axis=1)
    # Per-row patch-coverage check.
    if min_patch_coverage > 0.0:
        # Fold over channels first to get a per-pixel "any non-zero" mask,
        # then average over the spatial extent. Done one patch at a time to
        # avoid materialising the full (N, 64, 64, 128) tensor in memory.
        n_pix = h * w
        coverage = np.empty(n_tessera, dtype=np.float32)
        for i in range(n_tessera):
            pix_nonzero = np.any(patches_mmap[i] != 0, axis=-1)
            coverage[i] = pix_nonzero.sum() / n_pix
        coverage_ok = coverage >= min_patch_coverage
        valid_patches = centre_nonzero & coverage_ok
        print(
            f"TESSERA filtering: centre-pixel-nonzero "
            f"{int(centre_nonzero.sum())}/{n_tessera}, "
            f"coverage>={min_patch_coverage:.2f} "
            f"{int(coverage_ok.sum())}/{n_tessera}, "
            f"both {int(valid_patches.sum())}/{n_tessera}"
        )
    else:
        valid_patches = centre_nonzero
        print(
            f"TESSERA filtering (centre-only): "
            f"{int(centre_nonzero.sum())}/{n_tessera} stations pass"
        )
    del patches_mmap

    tessera_id_to_idx = {
        sid: i for i, sid in enumerate(tessera_stations["station_id"].values)
    }

    kept_mask = np.zeros(len(spatial_indices), dtype=bool)
    tessera_row_indices: list[int] = []
    for local_i, sidx in enumerate(spatial_indices):
        station_id = stations_df["station_id"].values[sidx]
        t_idx = tessera_id_to_idx.get(station_id)
        if t_idx is None or t_idx >= n_tessera or not valid_patches[t_idx]:
            continue
        kept_mask[local_i] = True
        tessera_row_indices.append(t_idx)

    return TesseraFilterResult(
        kept_mask=kept_mask,
        tessera_row_indices=np.array(tessera_row_indices, dtype=np.int64),
    )


# ---------------------------------------------------------------------------
# Station filtering — precomputed per-station vectors (VAE latents etc.)
# ---------------------------------------------------------------------------


@dataclass
class VaeLatentFilterResult:
    """Output of :func:`filter_stations_by_vae_latents`."""

    kept_mask: np.ndarray  # bool, len == len(spatial_indices_in)
    latents: np.ndarray  # (n_kept, d) float32, z-scored if requested
    latent_dim: int


def filter_stations_by_vae_latents(
    stations_df: pd.DataFrame,
    spatial_indices: np.ndarray,
    vae_latents_path: Path,
    vae_latents_station_csv: Path,
    zscore: bool,
) -> VaeLatentFilterResult:
    """Filter stations to those with a non-NaN precomputed vector and z-score.

    ``vae_latents_path`` is an ``(N, d)`` array row-aligned with
    ``vae_latents_station_csv`` — VAE latents, shuffled latents, summary
    statistics or extra descriptors; the dataset treats them all the same.
    Stations absent from the CSV, or whose row contains a NaN, are dropped.

    Z-scoring uses global stats computed once across all valid rows of the
    file and cached alongside the ``.npy``
    (:func:`tessera_downscaling.data.vae_latents.compute_or_load_global_vae_stats`):
    a single normalisation applied across every region, split and experiment.
    """
    from tessera_downscaling.data.vae_latents import (
        compute_or_load_global_vae_stats,
        load_vae_latents,
        zscore_latents,
    )

    latents_arr, id_to_row, valid_mask_global = load_vae_latents(
        vae_latents_path,
        vae_latents_station_csv,
    )
    print(
        f"Precomputed station vectors loaded ({vae_latents_path.name}): "
        f"shape={latents_arr.shape}, "
        f"{int(valid_mask_global.sum())}/{latents_arr.shape[0]} non-NaN"
    )

    kept_mask = np.zeros(len(spatial_indices), dtype=bool)
    vae_rows: list[int] = []
    for local_i, sidx in enumerate(spatial_indices):
        station_id = stations_df["station_id"].values[sidx]
        row = id_to_row.get(station_id)
        if row is None or not valid_mask_global[row]:
            continue
        kept_mask[local_i] = True
        vae_rows.append(row)

    if not vae_rows:
        raise RuntimeError(
            "VAE latent filtering produced zero stations. Check that "
            "vae_latents_station_csv matches the station IDs in your dataset."
        )

    vae_rows_arr = np.array(vae_rows, dtype=np.int64)
    latents_kept = latents_arr[vae_rows_arr]

    if zscore:
        mean, std = compute_or_load_global_vae_stats(vae_latents_path)
        latents_final = zscore_latents(latents_kept, mean, std)
    else:
        latents_final = latents_kept.astype(np.float32)

    return VaeLatentFilterResult(
        kept_mask=kept_mask,
        latents=latents_final,
        latent_dim=latents_final.shape[1],
    )


# ---------------------------------------------------------------------------
# ERA5 channel normalisation
# ---------------------------------------------------------------------------


def load_or_compute_era5_norm_stats(
    cache_path: Path,
    train_episode_ids: list[str],
    era5_dir: Path,
    grid_lats: np.ndarray,
    grid_lons: np.ndarray,
    static_fields: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(era5_mean, era5_std)`` for the normalised context-grid channels.

    Channels covered, in grid order: ERA5 dynamic + (static if given) + lat +
    lon. Cos/sin time channels and the lead channel are NOT normalised.

    ``train_episode_ids`` lists the filename stems (``"YYYY-MM-DD-HH"``) of the
    ``era5_dir/<id>.npy`` files to accumulate pixel-wise moments over. Stats
    are loaded from ``cache_path`` when it exists (the preprocessor writes it)
    and written there otherwise.
    """
    if cache_path.exists():
        data = np.load(cache_path)
        return data["era5_mean"].astype(np.float32), data["era5_std"].astype(np.float32)

    n_static = static_fields.shape[0] if static_fields is not None else 0
    first = np.load(era5_dir / f"{train_episode_ids[0]}.npy")
    n_dynamic = first.shape[0]

    n_channels = n_dynamic + n_static + 2
    running_sum = np.zeros(n_channels, dtype=np.float64)
    running_sq_sum = np.zeros(n_channels, dtype=np.float64)
    n_pixels = 0

    lat_grid = grid_lats[:, None] * np.ones((1, len(grid_lons)), dtype=np.float32)
    lon_grid = np.ones((len(grid_lats), 1), dtype=np.float32) * grid_lons[None, :]

    for episode_id in train_episode_ids:
        path = era5_dir / f"{episode_id}.npy"
        if not path.exists():
            continue
        era5 = np.load(path)
        parts = [era5]
        if static_fields is not None:
            parts.append(static_fields)
        parts.extend([lat_grid[None, :, :], lon_grid[None, :, :]])
        combined = np.concatenate(parts, axis=0)
        running_sum += combined.reshape(n_channels, -1).sum(axis=1)
        running_sq_sum += (combined.reshape(n_channels, -1) ** 2).sum(axis=1)
        n_pixels += combined.shape[1] * combined.shape[2]

    mean = (running_sum / n_pixels).astype(np.float32)
    std = np.sqrt(running_sq_sum / n_pixels - mean**2)
    std = np.maximum(std, 1e-8).astype(np.float32)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache_path, era5_mean=mean, era5_std=std)
    return mean, std


# ---------------------------------------------------------------------------
# Context grid assembly
# ---------------------------------------------------------------------------


def build_context_grid(
    era5_path: Path,
    static_fields: np.ndarray | None,
    grid_lats: np.ndarray,
    grid_lons: np.ndarray,
    date_str: str,
    hour: int,
    era5_mean: np.ndarray | None,
    era5_std: np.ndarray | None,
    drop_dynamic_indices: list[int] | None = None,
    lead_hours: int | None = None,
) -> np.ndarray:
    """Assemble one episode's ``(C, H, W)`` context grid.

    Channel order::

        ERA5 dynamic (minus ``drop_dynamic_indices``)   z-scored
        static fields (if given)                        z-scored
        lat, lon                                        z-scored
        cos(DoY), sin(DoY), cos(HoD), sin(HoD)          raw, in [-1, 1]
        [lead_hours / MAX_LEAD_HOURS]                   raw, only if lead_hours given

    ``date_str`` is ``"YYYY-MM-DD"`` and ``hour`` the UTC snapshot hour
    (0, 6, 12, 18); together they define the day-of-year and hour-of-day
    channels.

    ``era5_mean``/``era5_std`` cover the z-scored block in the order above
    (see :func:`load_or_compute_era5_norm_stats`); ``None`` skips
    normalisation. ``drop_dynamic_indices`` removes channels from the ERA5
    dynamic block (e.g. precipitation, to match a 19-channel model) and the
    same indices from the stats, so both stay aligned.

    ``lead_hours`` is the forecast horizon of the coarse fields (0 = ERA5
    analysis, 6/24/72 = Aurora forecast). When set, a single broadcast
    channel holding ``lead_hours / MAX_LEAD_HOURS`` is appended LAST and is
    NOT normalised — the only signal the lead-conditioned model needs to
    express a lead-dependent predictive spread. ``None`` appends nothing, so
    the grid is byte-identical to the single-lead behaviour.
    """
    era5 = np.load(era5_path)
    if drop_dynamic_indices:
        era5 = np.delete(era5, drop_dynamic_indices, axis=0)
    parts = [era5]
    if static_fields is not None:
        parts.append(static_fields)

    lat_grid = grid_lats[:, None] * np.ones((1, len(grid_lons)), dtype=np.float32)
    lon_grid = np.ones((len(grid_lats), 1), dtype=np.float32) * grid_lons[None, :]
    parts.extend([lat_grid[None, :, :], lon_grid[None, :, :]])
    context_grid = np.concatenate(parts, axis=0)

    dt = datetime.strptime(date_str, "%Y-%m-%d")
    day_of_year = dt.timetuple().tm_yday
    H, W = context_grid.shape[1], context_grid.shape[2]
    cos_doy = np.full(
        (1, H, W), np.cos(2 * np.pi * day_of_year / 365), dtype=np.float32
    )
    sin_doy = np.full(
        (1, H, W), np.sin(2 * np.pi * day_of_year / 365), dtype=np.float32
    )
    cos_hod = np.full((1, H, W), np.cos(2 * np.pi * hour / 24), dtype=np.float32)
    sin_hod = np.full((1, H, W), np.sin(2 * np.pi * hour / 24), dtype=np.float32)
    context_grid = np.concatenate(
        [context_grid, cos_doy, sin_doy, cos_hod, sin_hod], axis=0
    )

    if era5_mean is not None:
        mean, std = era5_mean, era5_std
        if drop_dynamic_indices:
            mean = np.delete(era5_mean, drop_dynamic_indices)
            std = np.delete(era5_std, drop_dynamic_indices)
        n_norm = len(mean)
        context_grid[:n_norm] = (context_grid[:n_norm] - mean[:, None, None]) / std[
            :, None, None
        ]

    # Lead channel last, AFTER normalisation (metadata, not a physical field).
    if lead_hours is not None:
        lead_ch = np.full((1, H, W), lead_hours / MAX_LEAD_HOURS, dtype=np.float32)
        context_grid = np.concatenate([context_grid, lead_ch], axis=0)

    return context_grid


def resolve_drop_channel_indices(
    drop_names: list[str] | None,
    channel_names: list[str],
    strict: bool,
    logger: logging.Logger | None = None,
) -> list[int]:
    """Map channel names to drop into indices within the dynamic block.

    Dropping is by name, so an absent name is unambiguous and safe to no-op:
    the post-condition ("this channel is not in the grid") already holds. The
    default is therefore lenient — a requested name not present is skipped and
    logged (if a logger is given), so the same ``--drop-context-channels`` works
    on a full-channel grid (precip dropped) and an already-reduced one (precip
    absent, e.g. native-19-channel Aurora). Pass ``strict=True`` to instead
    raise on an absent name. Returns a sorted list of integer indices (empty if
    nothing to drop), so default/empty behaviour is a no-op.
    """
    if not drop_names:
        return []
    indices: list[int] = []
    for name in drop_names:
        if name in channel_names:
            indices.append(channel_names.index(name))
        elif strict:
            raise ValueError(
                f"--drop-context-channels: channel '{name}' not found in dataset "
                f"dynamic channels {channel_names}."
            )
        elif logger is not None:
            logger.info(
                f"drop-context-channel '{name}' not present in this dataset; "
                "skipping (already absent)."
            )
    return sorted(indices)


# ---------------------------------------------------------------------------
# Target selection and episode assembly
# ---------------------------------------------------------------------------


def select_valid_targets(
    ghcnh_path: Path,
    ghcnh_index_for_station: np.ndarray,
    target_variables: list[str],
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Load target variables for one timestamp and compute the valid mask.

    A station contributes iff all requested targets are non-NaN (and above
    the ``-100`` sentinel used for missing values).

    Args:
        ghcnh_path: ``ghcnh_snapshot/<ts>.npz`` for this timestamp.
        ghcnh_index_for_station: Array of length n_stations, where the i-th
            entry is the row in the ghcnh .npz arrays for station i in the
            dataset's final filtered station order.
        target_variables: Which keys to read.

    Returns:
        (valid_indices, per_var_values), where ``valid_indices`` selects
        rows in the dataset's filtered station order and each element of
        ``per_var_values`` has length n_stations (pre-filter).

    """
    ghcnh = np.load(ghcnh_path)
    per_var_values = []
    for tv in target_variables:
        target_all = ghcnh[tv]
        per_var_values.append(target_all[ghcnh_index_for_station])

    valid_mask = np.ones(len(ghcnh_index_for_station), dtype=bool)
    for vals in per_var_values:
        valid_mask &= ~np.isnan(vals) & (vals > -100.0)
    valid_indices = np.where(valid_mask)[0]
    return valid_indices, per_var_values


def filter_valid_indices_by_probe_active_from(
    valid_indices: np.ndarray,
    station_ids: np.ndarray,
    timestamp: str,
    probe_active_from: dict[str, str],
) -> np.ndarray:
    """Drop probe-station rows whose ``probe_active_from`` is later than ``timestamp``.

    Implements the temporal-axis data-efficiency experiment: a small subset
    of training stations ("probes") are revealed to the model only from a
    per-station start timestamp onward, simulating a newly-deployed station
    that has accumulated d months of observations.

    Only rows whose station_id appears as a key in ``probe_active_from`` are
    subject to filtering; rows for any other station are passed through
    unchanged, so the mask has no effect on the always-on training stations
    or on the held-out spatial-test stations.

    Comparison is lexicographic on ``"YYYY-MM-DD-HH"`` strings. To exclude a
    probe station from training entirely (the x=0 case), set its value to
    any string greater than ``train_end`` (e.g. ``"9999-12-31"``); to include
    it for the full training window (the x=full case), use any string less
    than the dataset's earliest timestamp (e.g. ``"0000-01-01"``).

    Args:
        valid_indices: Local station indices (output of
            :func:`select_valid_targets`) into the dataset's filtered
            station array.
        station_ids: 1-D string array, length == n_stations in the
            dataset's filtered station array. ``station_ids[valid_indices[i]]``
            is the station_id of the i-th candidate target.
        timestamp: This episode's identifier, ``"YYYY-MM-DD-HH"``.
        probe_active_from: Mapping of station_id → earliest-usable
            timestamp string. Stations absent from this dict are never
            filtered.

    Returns:
        A subset of ``valid_indices`` (possibly empty) excluding probe
        stations whose active-from is greater than ``timestamp``.

    """
    if not probe_active_from or len(valid_indices) == 0:
        return valid_indices
    keep = np.ones(len(valid_indices), dtype=bool)
    for i, local_idx in enumerate(valid_indices):
        sid = str(station_ids[local_idx])
        active_from = probe_active_from.get(sid)
        if active_from is not None and timestamp < active_from:
            keep[i] = False
    if keep.all():
        return valid_indices
    return valid_indices[keep]


def empty_episode_result(
    context_grid: np.ndarray,
    grid_lats: np.ndarray,
    grid_lons: np.ndarray,
    n_target_variables: int,
    date_str: str,
    tessera_dim: int = 0,
) -> dict:
    """Zero-target result dict (the collate function filters these out).

    ``tessera_dim > 0`` adds a ``(0, tessera_dim)`` ``target_tessera``
    placeholder so the dict has the same keys as non-empty episodes served
    with precomputed vectors.
    """
    result = {
        "context_grid": torch.from_numpy(context_grid),
        "target_coords": torch.zeros(0, 2),
        "target_elev": torch.zeros(0),
        "target_delta_elev": torch.zeros(0),
        "target_values": torch.zeros(0, n_target_variables),
        "target_station_indices": torch.zeros(0, dtype=torch.long),
        "target_mask": torch.zeros(0, dtype=torch.bool),
        "grid_lats": torch.from_numpy(grid_lats),
        "grid_lons": torch.from_numpy(grid_lons),
        "n_targets": 0,
        "date": date_str,
    }
    if tessera_dim > 0:
        result["target_tessera"] = torch.zeros(0, tessera_dim)
    return result


def assemble_episode_result(
    context_grid: np.ndarray,
    grid_lats: np.ndarray,
    grid_lons: np.ndarray,
    date_str: str,
    valid_indices: np.ndarray,
    per_var_values: list[np.ndarray],
    station_lats: np.ndarray,
    station_lons: np.ndarray,
    station_elevs: np.ndarray,
    station_delta_elevs: np.ndarray,
    n_target_variables: int,
    station_mtpi: np.ndarray | None = None,
) -> dict:
    """Build the non-TESSERA portion of a non-empty episode result dict.

    ``station_mtpi`` (per-station multi-scale topographic position index) is
    optional: when provided, a ``target_mtpi`` tensor is added to the result
    and the collate function pads/stacks it alongside the other per-station
    features. Datasets built from a stations table without an ``mtpi`` column
    pass ``None`` and the key is simply omitted, so pre-mTPI models keep
    working unchanged.
    """
    target_coords = np.stack(
        [
            station_lats[valid_indices],
            station_lons[valid_indices],
        ],
        axis=1,
    )

    if n_target_variables == 1:
        target_values_np = per_var_values[0][valid_indices]
    else:
        target_values_np = np.stack(
            [v[valid_indices] for v in per_var_values],
            axis=1,
        )

    result = {
        "context_grid": torch.from_numpy(context_grid),
        "target_coords": torch.from_numpy(target_coords),
        "target_elev": torch.from_numpy(station_elevs[valid_indices]),
        "target_delta_elev": torch.from_numpy(station_delta_elevs[valid_indices]),
        "target_values": torch.from_numpy(target_values_np),
        "target_station_indices": torch.from_numpy(valid_indices.astype(np.int64)),
        "grid_lats": torch.from_numpy(grid_lats),
        "grid_lons": torch.from_numpy(grid_lons),
        "n_targets": len(valid_indices),
        "date": date_str,
    }
    if station_mtpi is not None:
        result["target_mtpi"] = torch.from_numpy(station_mtpi[valid_indices])
    return result


# ---------------------------------------------------------------------------
# Temporal split partitioning
# ---------------------------------------------------------------------------


def episodes_for_split(
    valid_episode_ids: list[str],
    split: str,
    train_end: str,
    val_end: str,
) -> list[str]:
    """Partition sorted ``"YYYY-MM-DD-HH"`` identifiers by temporal split.

    ``train_end`` / ``val_end`` are bare dates (``"YYYY-MM-DD"``) and the
    comparison is lexicographic. A timestamp on a boundary date compares
    *greater* than the bare date (``"2020-12-31-00" > "2020-12-31"``), so the
    four snapshots of the ``train_end`` date open the val split and the four
    of the ``val_end`` date open the test split — e.g. with
    ``train_end="2020-12-31"``, ``val_end="2021-12-31"``: train ends at
    2020-12-30 18Z, val is 2020-12-31 00Z .. 2021-12-30 18Z, test starts at
    2021-12-31 00Z. This is the convention the paper's splits use; do not
    "fix" it.
    """
    if split == "train":
        return [d for d in valid_episode_ids if d <= train_end]
    if split == "val":
        return [d for d in valid_episode_ids if train_end < d <= val_end]
    if split == "test":
        return [d for d in valid_episode_ids if d > val_end]
    raise ValueError(f"Unknown split: {split}")


# ---------------------------------------------------------------------------
# Batch collation
# ---------------------------------------------------------------------------


def downscaling_collate(batch: list[dict]) -> dict | None:
    """Custom collate for variable-size target sets.

    Drops zero-target episodes (returns ``None`` if none remain). Pads
    target-side tensors to the max ``n_targets`` in the batch and returns a
    ``target_mask`` indicating which positions are real; padded
    ``target_station_indices`` are ``-1``.
    """
    batch = [b for b in batch if b["n_targets"] > 0]
    if not batch:
        return None

    max_targets = max(b["n_targets"] for b in batch)
    batch_size = len(batch)
    has_tessera = "target_tessera" in batch[0]
    # Require mTPI from EVERY episode, not just batch[0]. A MultiLeadDataset
    # can concatenate sub-datasets that disagree on whether their stations.csv
    # carries an `mtpi` column (e.g. cross-lead: the ERA5 lead-0 dataset was
    # mTPI-backfilled but the Aurora lead datasets were not), so a single batch
    # can mix episodes that have `target_mtpi` with ones that don't. Keying off
    # batch[0] alone then either KeyErrors (batch[0] has it, a later episode
    # doesn't) or silently drops it. Serving mTPI only when it is uniformly
    # present is correct for every consumer: models built without --use-mtpi
    # (n_elev_features=2) ignore the field regardless, and a mixed batch cannot
    # feed a 3-feature model anyway.
    has_mtpi = all("target_mtpi" in b for b in batch)

    sample_tv = batch[0]["target_values"]
    if sample_tv.ndim == 1:
        target_values = torch.zeros(batch_size, max_targets)
    else:
        n_vars = sample_tv.shape[-1]
        target_values = torch.zeros(batch_size, max_targets, n_vars)

    context_grids = torch.stack([b["context_grid"] for b in batch])
    target_coords = torch.zeros(batch_size, max_targets, 2)
    target_elev = torch.zeros(batch_size, max_targets)
    target_delta_elev = torch.zeros(batch_size, max_targets)
    if has_mtpi:
        target_mtpi = torch.zeros(batch_size, max_targets)
    target_mask = torch.zeros(batch_size, max_targets, dtype=torch.bool)
    target_station_indices = torch.full(
        (batch_size, max_targets),
        -1,
        dtype=torch.long,
    )

    if has_tessera:
        t_shape = batch[0]["target_tessera"].shape[1:]
        target_tessera = torch.zeros(batch_size, max_targets, *t_shape)

    for i, b in enumerate(batch):
        n = b["n_targets"]
        target_coords[i, :n] = b["target_coords"]
        target_elev[i, :n] = b["target_elev"]
        target_delta_elev[i, :n] = b["target_delta_elev"]
        if has_mtpi:
            target_mtpi[i, :n] = b["target_mtpi"]
        target_values[i, :n] = b["target_values"]
        target_mask[i, :n] = True
        target_station_indices[i, :n] = b["target_station_indices"]
        if has_tessera:
            target_tessera[i, :n] = b["target_tessera"]

    result = {
        "context_grid": context_grids,
        "target_coords": target_coords,
        "target_elev": target_elev,
        "target_delta_elev": target_delta_elev,
        "target_values": target_values,
        "target_mask": target_mask,
        "target_station_indices": target_station_indices,
        "grid_lats": batch[0]["grid_lats"],
        "grid_lons": batch[0]["grid_lons"],
    }
    if has_mtpi:
        result["target_mtpi"] = target_mtpi
    if has_tessera:
        result["target_tessera"] = target_tessera
    return result
