"""Pre-computed per-station vector loading (TESSERA VAE latents and friends).

The model consumes one *frozen* vector per station rather than raw TESSERA
patches. In the paper that vector is the latent of a VAE trained (in the
``tessera_patch_encoder`` repository) on the global 64x64 TESSERA patch set;
the same loader also serves the control inputs built with the same
``(n_stations, d)`` + station-CSV convention — shuffled latents, patch summary
statistics and the extra terrain/land-cover descriptors.

At training time:

  1. We load the latents and their station CSV (both global — produced once
     from the full 38,870-station GHCNh set).
  2. We build a ``station_id -> latent_row_index`` lookup.
  3. The dataset uses the lookup to fetch the correct latent vector for each
     target station. Stations whose latent is NaN (the VAE couldn't encode
     their patch) are excluded.
  4. Z-scoring statistics are computed once across all valid rows in the
     latents file, cached next to the ``.npy``, and reused globally — same
     stats for every region, split, and experiment. Matches the WeatherBench2
     convention of a single normalisation applied uniformly.

Design notes:

  - The latents file is tiny relative to the raw patches (~2.4 MB for
    38,870 × 16 × float32), so we load it fully into RAM rather than mmap.
  - Z-scoring is done inside this module; the dataset and model see
    already-normalised latents. This keeps the normalisation policy in one
    place and matches how ERA5 channels are handled.
  - Stats cache files are named from the latents filename stem to avoid
    collisions when iterating on different VAE variants.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def load_vae_latents(
    latents_path: str | Path,
    station_csv_path: str | Path,
) -> tuple[np.ndarray, dict[str, int], np.ndarray]:
    """Load VAE latents and build a station_id -> row lookup.

    Args:
        latents_path: Path to ``.npy`` file of shape ``(n_stations, d)``.
        station_csv_path: Path to CSV row-aligned with the latents file.
            Must contain a ``station_id`` column.

    Returns:
        Tuple of:
          - ``latents``: ``(n_stations, d)`` float32 array (NaN rows kept;
            the caller filters by ``valid_mask``).
          - ``id_to_row``: dict mapping station_id string to row index.
          - ``valid_mask``: boolean array of length ``n_stations``, True
            where the latent row has no NaNs.

    """
    latents_path = Path(latents_path)
    station_csv_path = Path(station_csv_path)

    latents = np.load(latents_path).astype(np.float32)
    stations = pd.read_csv(station_csv_path)

    if "station_id" not in stations.columns:
        raise ValueError(
            f"Expected a 'station_id' column in {station_csv_path}. "
            f"Got columns: {list(stations.columns)}"
        )

    if len(stations) != latents.shape[0]:
        raise ValueError(
            f"Row count mismatch: latents {latents.shape[0]} vs "
            f"station CSV {len(stations)} ({station_csv_path})"
        )

    valid_mask = ~np.isnan(latents).any(axis=1)

    id_to_row = {sid: i for i, sid in enumerate(stations["station_id"].values)}

    return latents, id_to_row, valid_mask


def zscore_latents(
    latents: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
) -> np.ndarray:
    """Apply per-dim z-scoring.

    Args:
        latents: ``(..., d)`` array.
        mean: ``(d,)``.
        std: ``(d,)``.

    Returns:
        Normalised array with same shape as input.

    """
    return ((latents - mean) / std).astype(np.float32)


def compute_or_load_global_vae_stats(
    latents_path: str | Path,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute or load global VAE latent z-score stats.

    Uses EVERY valid (non-NaN) row in the latents file to compute
    (mean, std). Independent of any split or region: a single set of
    stats is used across all regions, train and test.

    This includes test-split station rows in the stats. Minor mean/std
    shift since test stations are a small fraction of the total, and
    the stats don't carry predictive information (climatological
    normalisation constants). Matches WeatherBench2-style convention.

    Stats are cached alongside the latents file so every caller gets
    the same numbers without recomputing.

    Args:
        latents_path: Path to the ``.npy`` file with latents.

    Returns:
        Tuple ``(mean, std)`` of float32 arrays, each shape ``(d,)``.

    """
    latents_path = Path(latents_path)
    cache_path = latents_path.with_name(latents_path.stem + "_global_stats.npz")

    if cache_path.exists():
        cached = np.load(cache_path)
        return cached["mean"].astype(np.float32), cached["std"].astype(np.float32)

    # Compute from all valid rows.
    latents = np.load(latents_path)
    valid_mask = ~np.isnan(latents).any(axis=1)
    if not valid_mask.any():
        raise RuntimeError(
            f"No valid (non-NaN) rows in {latents_path}; cannot compute stats."
        )
    valid = latents[valid_mask]
    mean = valid.mean(axis=0).astype(np.float32)
    std = np.maximum(valid.std(axis=0), 1e-6).astype(np.float32)

    np.savez(cache_path, mean=mean, std=std)
    print(
        f"Computed global VAE stats from {valid_mask.sum()}/{len(latents)} "
        f"valid rows, cached to {cache_path.name}"
    )
    return mean, std
