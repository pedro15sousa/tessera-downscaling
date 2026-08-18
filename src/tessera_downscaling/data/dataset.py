"""PyTorch Datasets for downscaling episodes.

Exposes Dataset classes for the different preprocessed layouts:

  - :class:`DailyDownscalingDataset` serves daily episodes from a single
    flat preprocessed directory (``dataset_daily/`` from
    ``preprocess_daily.py``). One episode = one date.
  - :class:`MultiRegionDownscalingDataset` serves daily episodes from a
    layered multi-region directory (``dataset_daily_global/`` from
    ``preprocess_daily_global.py``), where each episode is scoped to one
    region's ERA5 grid and only contains target stations inside that
    region.

All Dataset classes delegate the substantive work (TESSERA patch
filtering, VAE latent loading, ERA5 normalisation, per-episode context
grid assembly, per-episode target slicing) to module-level helpers in
:mod:`tessera_downscaling.data.helpers`. The classes themselves just
orchestrate init, indexing, and ``__getitem__``.

Each episode ``dict`` returned by ``__getitem__`` contains:
    context_grid          (C, H, W)
    target_coords         (N, 2)
    target_elev           (N,)
    target_delta_elev     (N,)
    target_values         (N,) or (N, n_vars)
    target_station_indices (N,)
    grid_lats             (H,)
    grid_lons             (W,)
    n_targets             int
    date                  str (either "YYYY-MM-DD" or "YYYY-MM-DD-HH")
    [target_tessera]      (N, 128, h, w) for patches, or (N, d) for VAE latents

``n_targets == 0`` means the episode has no valid stations today, and
:func:`downscaling_collate` drops it before it reaches the model.

The collate function and helpers are re-exported at the bottom so that
existing callers can still write ``from tessera_downscaling.data.dataset
import downscaling_collate`` without knowing we moved it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from tessera_downscaling.data.helpers import (
    build_context_grid,
    downscaling_collate,
    empty_episode_result,
    episodes_for_split,
    assemble_episode_result,
    filter_stations_by_tessera_patches,
    filter_stations_by_vae_latents,
    filter_valid_indices_by_probe_active_from,
    load_or_compute_era5_norm_stats,
    resolve_drop_channel_indices,
    select_valid_targets,
    slice_tessera_patches,
    validate_target_variables,
)


# ---------------------------------------------------------------------------
# Default for the patch-coverage filter.
# ---------------------------------------------------------------------------
# A patch is considered valid iff the centre pixel is non-zero AND at
# least this fraction of all pixels in the 64x64 patch have any non-zero
# channel. The threshold lives at the dataset level (rather than the
# preprocessor) so it can be tuned without re-preprocessing — re-running
# train.py / evaluate.py with --min-tessera-patch-coverage is enough to
# explore other values. Setting to 0.0 recovers legacy centre-only behaviour.
DEFAULT_MIN_TESSERA_PATCH_COVERAGE: float = 0.5


def _load_optional_station_mtpi(
    stations: "pd.DataFrame", spatial_indices: np.ndarray,
) -> np.ndarray | None:
    """Per-station mTPI for the selected stations, or None if unavailable.

    Returns the ``mtpi`` column sliced to ``spatial_indices`` (float32) when
    the stations table carries it; otherwise ``None``. Keeping this optional
    lets datasets built from older stations.csv files (no ``mtpi`` column)
    continue to serve the 2-feature (elevation, delta_elevation) layout that
    pre-mTPI checkpoints expect, while newer tables transparently enable the
    3-feature (elevation, delta_elevation, mTPI) layout of Vaughan et al.
    (2022).
    """
    if "mtpi" not in stations.columns:
        return None
    return stations["mtpi"].values[spatial_indices].astype(np.float32)


# ---------------------------------------------------------------------------
# Flat-layout daily dataset
# ---------------------------------------------------------------------------


class DailyDownscalingDataset(Dataset):
    """Dataset of daily downscaling episodes (flat preprocessed layout).

    Args:
        dataset_dir: Path to the preprocessed dataset directory (output of
            ``preprocess_daily.py``).
        split: Temporal split — ``"train"``, ``"val"``, or ``"test"``.
        station_split: Spatial split — ``"train"`` or ``"test"``.
        target_variables: Variables to predict. A station contributes to
            an episode only if all requested variables are non-NaN that
            day. Default: ``["tmax"]``.
        tessera_path: Patches ``.npy``. When set, stations are filtered to
            those with non-zero centre pixels — applied for both baseline
            and TESSERA runs so they see the same station set.
        tessera_station_csv: CSV row-aligned with ``tessera_path``.
        load_tessera_patches: Whether to serve patches in ``__getitem__``.
            True for end-to-end TESSERA models, False for baseline or VAE.
        normalise_era5: Whether to z-score normalise ERA5 channels.
        include_static_fields: Whether to include ERA5 static fields in
            the context grid.
        vae_latents_path: Pre-computed VAE latents ``.npy``. When set,
            serves the (z-scored) latent vector for each station via
            ``target_tessera``, bypassing the end-to-end encoder. Station
            set becomes (valid patch) ∧ (non-NaN latent).
        vae_latents_station_csv: CSV row-aligned with the latents file.
        vae_latents_zscore: Whether to z-score the latents per-dim.
    """

    def __init__(
        self,
        dataset_dir: str | Path,
        split: str = "train",
        station_split: str = "train",
        target_variables: list[str] | None = None,
        tessera_path: str | Path | None = None,
        tessera_station_csv: str | Path | None = None,
        load_tessera_patches: bool = False,
        normalise_era5: bool = True,
        include_static_fields: bool = True,
        vae_latents_path: str | Path | None = None,
        vae_latents_station_csv: str | Path | None = None,
        vae_latents_zscore: bool = True,
        min_patch_coverage: float = DEFAULT_MIN_TESSERA_PATCH_COVERAGE,
    ):
        self.dataset_dir = Path(dataset_dir)
        self.split = split
        self.station_split = station_split
        self.target_variables = validate_target_variables(target_variables)
        self.n_target_variables = len(self.target_variables)
        self.include_static_fields = include_static_fields
        self.normalise_era5 = normalise_era5
        self.load_tessera_patches = load_tessera_patches

        # --- Metadata + grid coords ---
        with open(self.dataset_dir / "metadata.json") as f:
            self.metadata = json.load(f)
        self.grid_lats = np.load(self.dataset_dir / "lats.npy").astype(np.float32)
        self.grid_lons = np.load(self.dataset_dir / "lons.npy").astype(np.float32)

        if include_static_fields:
            self.static_fields = np.load(
                self.dataset_dir / "static_fields.npy"
            ).astype(np.float32)
            self.n_static = self.metadata["n_static_channels"]
        else:
            self.static_fields = None
            self.n_static = 0

        self.n_context_channels = (
            self.metadata["n_dynamic_channels"] + self.n_static + 4
        )

        # --- Stations: spatial split → TESSERA → VAE latents ---
        stations = pd.read_csv(self.dataset_dir / "stations.csv")
        valid_station_indices = np.load(
            self.dataset_dir / "valid_station_indices.npy"
        )
        # station_split='all' bypasses the per-station train/test filter,
        # serving every preprocessed station. Used for evaluation in the
        # multi-region transfer setting where the held-out test region
        # was never seen during training, so further holdout within that
        # region adds no statistical guarantee — only noise.
        if station_split == "all":
            spatial_mask = np.ones(len(stations), dtype=bool)
        else:
            spatial_mask = stations["spatial_split"].values == station_split
        spatial_indices = np.where(spatial_mask)[0]
        # Precompute the ghcnh-row lookup for each station in spatial_indices.
        # ghcnh[tv] has length == unfiltered_station_count; valid_station_indices[i]
        # gives the ghcnh row for stations.csv row i.
        ghcnh_index_for_station = valid_station_indices[spatial_indices]

        self._tessera_indices: np.ndarray | None = None
        self._patches_mmap: np.memmap | None = None

        if tessera_path is not None:
            if tessera_station_csv is None:
                raise ValueError(
                    "tessera_station_csv is required when tessera_path is set"
                )
            t_result = filter_stations_by_tessera_patches(
                stations_df=stations,
                spatial_indices=spatial_indices,
                tessera_path=Path(tessera_path),
                tessera_station_csv=Path(tessera_station_csv),
                keep_mmap_alive=load_tessera_patches,
                min_patch_coverage=min_patch_coverage,
            )
            n_before = len(spatial_indices)
            spatial_indices = spatial_indices[t_result.kept_mask]
            ghcnh_index_for_station = ghcnh_index_for_station[t_result.kept_mask]
            self._tessera_indices = t_result.tessera_row_indices
            self._patches_mmap = t_result.patches_mmap
            print(
                f"DailyDownscalingDataset({split}/{station_split}): "
                f"{len(spatial_indices)}/{n_before} stations after TESSERA filtering"
            )
        elif load_tessera_patches:
            raise ValueError("load_tessera_patches=True requires tessera_path")

        self._vae_latents: np.ndarray | None = None
        self._use_vae_latents = False
        self.vae_latent_dim = 0

        if vae_latents_path is not None:
            if vae_latents_station_csv is None:
                raise ValueError(
                    "vae_latents_station_csv is required when vae_latents_path is set"
                )
            v_result = filter_stations_by_vae_latents(
                stations_df=stations,
                spatial_indices=spatial_indices,
                vae_latents_path=Path(vae_latents_path),
                vae_latents_station_csv=Path(vae_latents_station_csv),
                zscore=vae_latents_zscore,
            )
            n_before = len(spatial_indices)
            spatial_indices = spatial_indices[v_result.kept_mask]
            ghcnh_index_for_station = ghcnh_index_for_station[v_result.kept_mask]
            if self._tessera_indices is not None:
                self._tessera_indices = self._tessera_indices[v_result.kept_mask]
            self._vae_latents = v_result.latents
            self._use_vae_latents = True
            self.vae_latent_dim = v_result.latent_dim
            print(
                f"DailyDownscalingDataset({split}/{station_split}): "
                f"{len(spatial_indices)}/{n_before} stations after VAE latent filtering"
            )
            # VAE latents replace raw patches.
            if self.load_tessera_patches:
                print(
                    "VAE latents active: raw TESSERA patches will NOT be "
                    "served during __getitem__ (latents replace them)."
                )
                self.load_tessera_patches = False
                self._patches_mmap = None

        # Final station arrays.
        self._ghcnh_index_for_station = ghcnh_index_for_station
        self.station_lats = stations["latitude"].values[spatial_indices].astype(np.float32)
        self.station_lons = stations["longitude"].values[spatial_indices].astype(np.float32)
        self.station_elevs = stations["elevation"].values[spatial_indices].astype(np.float32)
        self.station_delta_elevs = (
            stations["delta_elevation"].values[spatial_indices].astype(np.float32)
        )
        self.station_mtpi = _load_optional_station_mtpi(stations, spatial_indices)
        self.station_ids = stations["station_id"].values[spatial_indices]

        # Temporal split.
        self.dates = episodes_for_split(
            self.metadata["valid_dates"],
            split=split,
            train_end=self.metadata["temporal_split"]["train_end"],
            val_end=self.metadata["temporal_split"]["val_end"],
        )

        # Normalisation stats.
        self.era5_mean: np.ndarray | None = None
        self.era5_std: np.ndarray | None = None
        if normalise_era5:
            stats_name = (
                "normalisation_stats.npz" if include_static_fields
                else "normalisation_stats_no_static.npz"
            )
            train_dates = [
                d for d in self.metadata["valid_dates"]
                if d <= self.metadata["temporal_split"]["train_end"]
            ]
            self.era5_mean, self.era5_std = load_or_compute_era5_norm_stats(
                cache_path=self.dataset_dir / stats_name,
                train_episode_ids=train_dates,
                era5_daily_dir=self.dataset_dir / "era5_daily",
                grid_lats=self.grid_lats,
                grid_lons=self.grid_lons,
                static_fields=self.static_fields,
            )

        mode_label = (
            f"vae latents (dim={self.vae_latent_dim})" if self._use_vae_latents
            else "serving patches" if self.load_tessera_patches
            else "filter-only" if tessera_path
            else "off"
        )
        print(
            f"DailyDownscalingDataset({split}/{station_split}): "
            f"{len(self.dates)} days, {len(spatial_indices)} stations, "
            f"targets={self.target_variables}, tessera={mode_label}"
        )

    def __len__(self) -> int:
        return len(self.dates)

    def __getitem__(self, idx: int) -> dict:
        date_str = self.dates[idx]

        context_grid = build_context_grid(
            era5_daily_path=self.dataset_dir / "era5_daily" / f"{date_str}.npy",
            static_fields=self.static_fields,
            grid_lats=self.grid_lats,
            grid_lons=self.grid_lons,
            date_str=date_str,
            era5_mean=self.era5_mean,
            era5_std=self.era5_std,
        )

        valid_indices, per_var_values = select_valid_targets(
            ghcnh_daily_path=self.dataset_dir / "ghcnh_daily" / f"{date_str}.npz",
            ghcnh_index_for_station=self._ghcnh_index_for_station,
            target_variables=self.target_variables,
        )

        tessera_mode = (
            "patches" if self.load_tessera_patches and self._patches_mmap is not None
            else "latents" if self._use_vae_latents
            else "off"
        )

        if len(valid_indices) == 0:
            return empty_episode_result(
                context_grid=context_grid,
                grid_lats=self.grid_lats,
                grid_lons=self.grid_lons,
                n_target_variables=self.n_target_variables,
                date_str=date_str,
                tessera_mode=tessera_mode,
                tessera_dim=self.vae_latent_dim,
            )

        result = assemble_episode_result(
            context_grid=context_grid,
            grid_lats=self.grid_lats,
            grid_lons=self.grid_lons,
            date_str=date_str,
            valid_indices=valid_indices,
            per_var_values=per_var_values,
            station_lats=self.station_lats,
            station_lons=self.station_lons,
            station_elevs=self.station_elevs,
            station_delta_elevs=self.station_delta_elevs,
            station_mtpi=self.station_mtpi,
            n_target_variables=self.n_target_variables,
        )

        if tessera_mode == "patches":
            patches_np = slice_tessera_patches(
                self._patches_mmap, self._tessera_indices, valid_indices,
            )
            result["target_tessera"] = torch.from_numpy(patches_np)
        elif tessera_mode == "latents":
            result["target_tessera"] = torch.from_numpy(
                self._vae_latents[valid_indices]
            )

        return result


# ---------------------------------------------------------------------------
# Multi-region daily dataset
# ---------------------------------------------------------------------------


@dataclass
class RegionState:
    """All per-region state needed to serve one region's episodes.

    Populated once at init and read-only thereafter.
    """
    name: str
    region_dir: Path
    grid_lats: np.ndarray
    grid_lons: np.ndarray
    static_fields: np.ndarray | None
    n_static: int
    era5_mean: np.ndarray | None
    era5_std: np.ndarray | None
    station_ids: np.ndarray
    station_lats: np.ndarray
    station_lons: np.ndarray
    station_elevs: np.ndarray
    station_delta_elevs: np.ndarray
    station_mtpi: np.ndarray | None
    tessera_row_indices: np.ndarray | None
    patches_mmap: np.memmap | None
    vae_latents: np.ndarray | None
    ghcnh_index_for_station: np.ndarray  # per-station row into ghcnh .npz
    dates: list[str]
    # Position of this region's first station in the flat per-region-
    # concatenated arrays (self.station_ids etc. on the parent dataset).
    # Per-region local station indices i become flat global indices
    # (i + flat_offset).
    flat_offset: int


class MultiRegionDownscalingDataset(Dataset):
    """Dataset composed of one grid+station view per region.

    Each episode is a ``(region, date)`` pair: the context grid comes from
    that region's ERA5 crop and only stations inside the region are
    eligible targets. PyTorch's ``DataLoader`` shuffles across regions and
    dates via the flat episode index, which :meth:`_dispatch` maps to
    ``(region, local_idx)``.

    Args:
        dataset_dir: Root of the multi-region dataset (output of
            ``preprocess_daily_global.py``).
        regions: Which regions to include. ``None`` = all. Use this to hold
            regions out: e.g. ``regions=["us"]`` for training, a separate
            instance with ``regions=["europe"]`` for testing.
        Other args: same semantics as :class:`DailyDownscalingDataset`.
    """

    def __init__(
        self,
        dataset_dir: str | Path,
        regions: list[str] | None = None,
        split: str = "train",
        station_split: str = "train",
        target_variables: list[str] | None = None,
        tessera_path: str | Path | None = None,
        tessera_station_csv: str | Path | None = None,
        load_tessera_patches: bool = False,
        normalise_era5: bool = True,
        include_static_fields: bool = True,
        vae_latents_path: str | Path | None = None,
        vae_latents_station_csv: str | Path | None = None,
        vae_latents_zscore: bool = True,
        region_specs: dict[str, str] | None = None,
        min_patch_coverage: float = DEFAULT_MIN_TESSERA_PATCH_COVERAGE,
    ):
        """Multi-region daily dataset.

        Two ways to declare which regions+splits to include:

        Old-style (``regions=[...]``, ``station_split="train"``):
            Every listed region contributes stations from the given
            spatial split. This is how transfer experiments were set
            up — e.g. ``regions=["us"], station_split="train"`` for
            training, and a separate dataset instance with
            ``regions=["europe"], station_split="all"`` for testing.

        New-style (``region_specs={"us": "all", "europe": "train"}``):
            Per-region spatial split. Enables held-out-within-training
            experiments — e.g. train on {EU train stations + all of US},
            then test on {EU test stations} by instantiating a new
            dataset with ``region_specs={"europe": "test"}``.

        The two forms are mutually exclusive. If both are given it's a
        config error. If neither is given, defaults to old-style with
        all available regions and ``station_split="train"``.

        Valid per-region splits in ``region_specs``: ``"train"``,
        ``"test"``, ``"all"`` (same semantics as ``station_split`` in
        the old form).
        """
        self.dataset_dir = Path(dataset_dir)
        self.split = split
        self.target_variables = validate_target_variables(target_variables)
        self.n_target_variables = len(self.target_variables)
        self.include_static_fields = include_static_fields
        self.normalise_era5 = normalise_era5
        self.load_tessera_patches = load_tessera_patches
        self._use_vae_latents = vae_latents_path is not None

        # Top-level manifest.
        top_md_path = self.dataset_dir / "metadata.json"
        if not top_md_path.exists():
            raise FileNotFoundError(
                f"Expected top-level metadata.json at {top_md_path}. "
                f"Is {self.dataset_dir} really a multi-region dataset?"
            )
        with open(top_md_path) as f:
            self.top_metadata = json.load(f)
        if self.top_metadata.get("layout_version") != "multi_region_v1":
            raise ValueError(
                f"Dataset at {self.dataset_dir} is not a multi_region_v1 layout."
            )

        available = list(self.top_metadata["regions"].keys())

        # Resolve region_specs from whichever form the caller used.
        # Canonical internal representation is always a dict
        # {region_name: spatial_split_str} — the rest of __init__ only
        # looks at self.region_specs, never the raw parameters.
        if region_specs is not None and regions is not None:
            raise ValueError(
                "Pass either `region_specs` OR `regions`+`station_split`, "
                "not both. region_specs is the preferred form for new "
                "experiments."
            )
        if region_specs is not None:
            # New-style. Validate names and splits.
            for name in region_specs:
                if name not in available:
                    raise ValueError(
                        f"region_specs includes '{name}' but dataset "
                        f"only has regions {available}."
                    )
            for name, s in region_specs.items():
                if s not in ("train", "test", "all"):
                    raise ValueError(
                        f"region_specs['{name}']='{s}' is not one of "
                        f"'train', 'test', 'all'."
                    )
            self.region_specs = dict(region_specs)
        else:
            # Old-style. regions defaults to all available; every region
            # uses the same station_split.
            resolved_regions = list(regions) if regions is not None else available
            missing = [r for r in resolved_regions if r not in available]
            if missing:
                raise ValueError(
                    f"Requested regions {missing} not found. Available: {available}"
                )
            if station_split not in ("train", "test", "all"):
                raise ValueError(
                    f"station_split='{station_split}' is not one of "
                    f"'train', 'test', 'all'."
                )
            self.region_specs = {r: station_split for r in resolved_regions}

        self.region_order: list[str] = list(self.region_specs.keys())
        # Back-compat attribute: some callers (and evaluate.py on older
        # checkpoints) read `self.station_split`. Keep it populated with
        # the majority split, or "mixed" if the region_specs are mixed.
        split_values = set(self.region_specs.values())
        self.station_split = (
            next(iter(split_values)) if len(split_values) == 1 else "mixed"
        )

        self.n_dynamic_channels = self.top_metadata["n_dynamic_channels"]
        temporal = self.top_metadata["temporal_split"]
        valid_dates = self.top_metadata["valid_dates"]

        global_stations = pd.read_csv(self.dataset_dir / "stations.csv")
        global_valid_indices = np.load(
            self.dataset_dir / "valid_station_indices.npy"
        )
        self._ghcnh_daily_dir = self.dataset_dir / "ghcnh_daily"

        # VAE latent z-score stats (if used) are computed lazily inside
        # filter_stations_by_vae_latents from the first region state
        # built and cached alongside the .npy for all subsequent users.
        # No precompute needed here.

        self.per_region: dict[str, RegionState] = {}
        flat_offset = 0
        for name in self.region_order:
            region_dir = self.dataset_dir / "regions" / name
            if not region_dir.exists():
                raise FileNotFoundError(
                    f"Region subdir {region_dir} does not exist. "
                    f"Did preprocessing include '{name}'?"
                )
            state = self._build_region_state(
                name=name,
                region_dir=region_dir,
                global_stations=global_stations,
                global_valid_indices=global_valid_indices,
                station_split=self.region_specs[name],
                split=split,
                valid_dates=valid_dates,
                train_end=temporal["train_end"],
                val_end=temporal["val_end"],
                tessera_path=Path(tessera_path) if tessera_path else None,
                tessera_station_csv=(
                    Path(tessera_station_csv) if tessera_station_csv else None
                ),
                load_tessera_patches=load_tessera_patches,
                vae_latents_path=Path(vae_latents_path) if vae_latents_path else None,
                vae_latents_station_csv=(
                    Path(vae_latents_station_csv) if vae_latents_station_csv else None
                ),
                vae_latents_zscore=vae_latents_zscore,
                min_patch_coverage=min_patch_coverage,
                flat_offset=flat_offset,
            )
            self.per_region[name] = state
            flat_offset += len(state.station_ids)

        # Flat station arrays: concatenate per-region arrays in region_order.
        # Used by evaluate.py and any other code that wants to address stations
        # by a single integer index across the whole dataset. Per-region local
        # indices i in episode dicts become flat indices (i + region.flat_offset).
        self.station_ids = np.concatenate([
            self.per_region[n].station_ids for n in self.region_order
        ]) if self.region_order else np.array([], dtype=object)
        self.station_lats = np.concatenate([
            self.per_region[n].station_lats for n in self.region_order
        ]) if self.region_order else np.array([], dtype=np.float32)
        self.station_lons = np.concatenate([
            self.per_region[n].station_lons for n in self.region_order
        ]) if self.region_order else np.array([], dtype=np.float32)
        self.station_elevs = np.concatenate([
            self.per_region[n].station_elevs for n in self.region_order
        ]) if self.region_order else np.array([], dtype=np.float32)
        self.station_delta_elevs = np.concatenate([
            self.per_region[n].station_delta_elevs for n in self.region_order
        ]) if self.region_order else np.array([], dtype=np.float32)
        # mTPI is optional (present only when stations.csv carried an `mtpi`
        # column). All regions share one global stations table, so it is
        # either available for every region or for none; concatenate only
        # when uniformly present, else expose None so the model falls back
        # to the 2-feature (elevation, delta_elevation) layout.
        if self.region_order and all(
            self.per_region[n].station_mtpi is not None for n in self.region_order
        ):
            self.station_mtpi = np.concatenate([
                self.per_region[n].station_mtpi for n in self.region_order
            ])
        else:
            self.station_mtpi = None

        self._cum_lengths: list[int] = []
        running = 0
        for name in self.region_order:
            running += len(self.per_region[name].dates)
            self._cum_lengths.append(running)

        ch_counts = {
            name: (st.n_static + self.n_dynamic_channels + 4)
            for name, st in self.per_region.items()
        }
        if len(set(ch_counts.values())) != 1:
            raise ValueError(
                f"Regions disagree on n_context_channels: {ch_counts}."
            )
        self.n_context_channels = next(iter(ch_counts.values()))

        if self._use_vae_latents:
            dims = {
                name: st.vae_latents.shape[1]
                for name, st in self.per_region.items()
                if st.vae_latents is not None
            }
            if len(set(dims.values())) != 1:
                raise ValueError(f"Regions disagree on VAE latent dim: {dims}")
            self.vae_latent_dim = next(iter(dims.values()))
        else:
            self.vae_latent_dim = 0

        total = self._cum_lengths[-1] if self._cum_lengths else 0
        specs_str = ", ".join(f"{n}:{s}" for n, s in self.region_specs.items())
        print(
            f"MultiRegionDownscalingDataset({split}): "
            f"{total} total episodes across {len(self.region_order)} regions "
            f"[{specs_str}]"
        )

    def _build_region_state(
        self,
        name: str,
        region_dir: Path,
        global_stations: pd.DataFrame,
        global_valid_indices: np.ndarray,
        station_split: str,
        split: str,
        valid_dates: list[str],
        train_end: str,
        val_end: str,
        tessera_path: Path | None,
        tessera_station_csv: Path | None,
        load_tessera_patches: bool,
        vae_latents_path: Path | None,
        vae_latents_station_csv: Path | None,
        vae_latents_zscore: bool,
        min_patch_coverage: float,
        flat_offset: int,
    ) -> RegionState:
        """Assemble all state needed to serve episodes for one region."""
        grid_lats = np.load(region_dir / "lats.npy").astype(np.float32)
        grid_lons = np.load(region_dir / "lons.npy").astype(np.float32)

        if self.include_static_fields:
            static_fields = np.load(region_dir / "static_fields.npy").astype(np.float32)
            with open(region_dir / "region_metadata.json") as f:
                n_static = json.load(f)["n_static_channels"]
        else:
            static_fields = None
            n_static = 0

        # station_split='all' includes every station in this region
        # regardless of train/test spatial split. See DailyDownscalingDataset
        # for rationale.
        in_region = global_stations["region"].values == name
        if station_split == "all":
            region_station_mask = in_region
        else:
            region_station_mask = in_region & (
                global_stations["spatial_split"].values == station_split
            )
        spatial_indices = np.where(region_station_mask)[0]
        ghcnh_index_for_station = global_valid_indices[spatial_indices]

        tessera_row_indices: np.ndarray | None = None
        patches_mmap: np.memmap | None = None
        if tessera_path is not None:
            if tessera_station_csv is None:
                raise ValueError(
                    "tessera_station_csv is required when tessera_path is set"
                )
            t_result = filter_stations_by_tessera_patches(
                stations_df=global_stations,
                spatial_indices=spatial_indices,
                tessera_path=tessera_path,
                tessera_station_csv=tessera_station_csv,
                keep_mmap_alive=load_tessera_patches,
                min_patch_coverage=min_patch_coverage,
            )
            n_before = len(spatial_indices)
            spatial_indices = spatial_indices[t_result.kept_mask]
            ghcnh_index_for_station = ghcnh_index_for_station[t_result.kept_mask]
            tessera_row_indices = t_result.tessera_row_indices
            patches_mmap = t_result.patches_mmap
            print(
                f"[{name}] {len(spatial_indices)}/{n_before} stations "
                f"after TESSERA filtering"
            )
        elif load_tessera_patches:
            raise ValueError("load_tessera_patches=True requires tessera_path")

        vae_latents: np.ndarray | None = None
        if vae_latents_path is not None:
            if vae_latents_station_csv is None:
                raise ValueError(
                    "vae_latents_station_csv is required when vae_latents_path is set"
                )
            v_result = filter_stations_by_vae_latents(
                stations_df=global_stations,
                spatial_indices=spatial_indices,
                vae_latents_path=vae_latents_path,
                vae_latents_station_csv=vae_latents_station_csv,
                zscore=vae_latents_zscore,
            )
            n_before = len(spatial_indices)
            spatial_indices = spatial_indices[v_result.kept_mask]
            ghcnh_index_for_station = ghcnh_index_for_station[v_result.kept_mask]
            if tessera_row_indices is not None:
                tessera_row_indices = tessera_row_indices[v_result.kept_mask]
            vae_latents = v_result.latents
            print(
                f"[{name}] {len(spatial_indices)}/{n_before} stations "
                f"after VAE latent filtering"
            )
            if load_tessera_patches:
                patches_mmap = None

        era5_mean = None
        era5_std = None
        if self.normalise_era5:
            stats_name = (
                "normalisation_stats.npz" if self.include_static_fields
                else "normalisation_stats_no_static.npz"
            )
            train_dates = [d for d in valid_dates if d <= train_end]
            era5_mean, era5_std = load_or_compute_era5_norm_stats(
                cache_path=region_dir / stats_name,
                train_episode_ids=train_dates,
                era5_daily_dir=region_dir / "era5_daily",
                grid_lats=grid_lats,
                grid_lons=grid_lons,
                static_fields=static_fields,
            )

        station_ids = global_stations["station_id"].values[spatial_indices]
        station_lats = global_stations["latitude"].values[spatial_indices].astype(np.float32)
        station_lons = global_stations["longitude"].values[spatial_indices].astype(np.float32)
        station_elevs = global_stations["elevation"].values[spatial_indices].astype(np.float32)
        station_delta_elevs = (
            global_stations["delta_elevation"].values[spatial_indices].astype(np.float32)
        )
        station_mtpi = _load_optional_station_mtpi(global_stations, spatial_indices)

        dates = episodes_for_split(
            valid_dates, split=split, train_end=train_end, val_end=val_end,
        )

        return RegionState(
            name=name,
            region_dir=region_dir,
            grid_lats=grid_lats,
            grid_lons=grid_lons,
            static_fields=static_fields,
            n_static=n_static,
            era5_mean=era5_mean,
            era5_std=era5_std,
            station_ids=station_ids,
            station_lats=station_lats,
            station_lons=station_lons,
            station_elevs=station_elevs,
            station_delta_elevs=station_delta_elevs,
            station_mtpi=station_mtpi,
            tessera_row_indices=tessera_row_indices,
            patches_mmap=patches_mmap,
            vae_latents=vae_latents,
            ghcnh_index_for_station=ghcnh_index_for_station,
            dates=dates,
            flat_offset=flat_offset,
        )

    @property
    def dates(self) -> list[str]:
        """Flat list of dates aligned with episode indices.

        Concatenates per-region date lists in ``region_order``, so
        ``self.dates[k]`` is the date for the k-th episode regardless of
        which region the episode falls in. Provided for compatibility
        with code that expects the same ``.dates`` attribute as
        :class:`DailyDownscalingDataset`.
        """
        out: list[str] = []
        for name in self.region_order:
            out.extend(self.per_region[name].dates)
        return out

    def __len__(self) -> int:
        return self._cum_lengths[-1] if self._cum_lengths else 0

    def _dispatch(self, idx: int) -> tuple[str, int]:
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Episode index {idx} out of range [0, {len(self)})")
        prev_cum = 0
        for name, cum in zip(self.region_order, self._cum_lengths):
            if idx < cum:
                return name, idx - prev_cum
            prev_cum = cum
        raise RuntimeError("unreachable")  # pragma: no cover

    def __getitem__(self, idx: int) -> dict:
        region_name, local_idx = self._dispatch(idx)
        st = self.per_region[region_name]
        date_str = st.dates[local_idx]

        context_grid = build_context_grid(
            era5_daily_path=st.region_dir / "era5_daily" / f"{date_str}.npy",
            static_fields=st.static_fields,
            grid_lats=st.grid_lats,
            grid_lons=st.grid_lons,
            date_str=date_str,
            era5_mean=st.era5_mean,
            era5_std=st.era5_std,
        )

        valid_indices, per_var_values = select_valid_targets(
            ghcnh_daily_path=self._ghcnh_daily_dir / f"{date_str}.npz",
            ghcnh_index_for_station=st.ghcnh_index_for_station,
            target_variables=self.target_variables,
        )

        tessera_mode = (
            "patches" if (self.load_tessera_patches and st.patches_mmap is not None)
            else "latents" if st.vae_latents is not None
            else "off"
        )

        if len(valid_indices) == 0:
            return empty_episode_result(
                context_grid=context_grid,
                grid_lats=st.grid_lats,
                grid_lons=st.grid_lons,
                n_target_variables=self.n_target_variables,
                date_str=date_str,
                tessera_mode=tessera_mode,
                tessera_dim=self.vae_latent_dim,
            )

        result = assemble_episode_result(
            context_grid=context_grid,
            grid_lats=st.grid_lats,
            grid_lons=st.grid_lons,
            date_str=date_str,
            valid_indices=valid_indices,
            per_var_values=per_var_values,
            station_lats=st.station_lats,
            station_lons=st.station_lons,
            station_elevs=st.station_elevs,
            station_delta_elevs=st.station_delta_elevs,
            station_mtpi=st.station_mtpi,
            n_target_variables=self.n_target_variables,
        )

        # Translate per-region local station indices to flat global indices,
        # so callers (e.g. evaluate.py) can address stations via a single
        # integer index into self.station_ids etc.
        result["target_station_indices"] = (
            result["target_station_indices"] + st.flat_offset
        )

        if tessera_mode == "patches":
            patches_np = slice_tessera_patches(
                st.patches_mmap, st.tessera_row_indices, valid_indices,
            )
            result["target_tessera"] = torch.from_numpy(patches_np)
        elif tessera_mode == "latents":
            result["target_tessera"] = torch.from_numpy(
                st.vae_latents[valid_indices]
            )

        return result


# ---------------------------------------------------------------------------
# Flat-layout snapshot (timestamp-cadence) dataset
# ---------------------------------------------------------------------------


class SnapshotDownscalingDataset(Dataset):
    """Dataset of timestamp-cadence downscaling episodes (single region).

    Each episode is one ``(date, hour)`` pair. Compared to
    :class:`DailyDownscalingDataset`:

      * Episode identifiers look like ``"YYYY-MM-DD-HH"`` rather than
        ``"YYYY-MM-DD"``. There are 4× more of them over the same date
        range (one per ERA5 timestamp: 00, 06, 12, 18 UTC).
      * Target variables must be drawn from ``{"t2m", "wind", "precip"}`` —
        the instantaneous counterparts of the daily ``tmax`` / ``wind_mean``,
        plus the 6h-accumulated precipitation ending at the synoptic
        timestamp (read from the GHCNh ``precipitation_6_hour`` column).
      * The context grid has 4 temporal channels (cos/sin day-of-year
        plus cos/sin hour-of-day) rather than 2, giving the model a
        signal for time-of-day in addition to day-of-year.
      * Files live under ``era5_snapshot/`` and ``ghcnh_snapshot/``
        rather than ``era5_daily/`` and ``ghcnh_daily/``.

    **Two supported layouts**, auto-detected from the top-level
    ``metadata.json``:

      * ``snapshot_v1`` (from ``preprocess_timestamp.py``): single-region
        flat layout. Every file lives at the top level of ``dataset_dir``.
        The ``region`` kwarg must be ``None``.

      * ``multi_region_snapshot_v1`` (from
        ``preprocess_timestamp_global.py``): multi-region layered layout.
        Region-specific files (ERA5 grid, static fields, normalisation
        stats) live under ``regions/<region>/``; GHCNh is shared at the
        top level. Use the ``region`` kwarg to pick which region this
        dataset serves — the class will then behave identically to a
        flat snapshot dataset for that region.

    Args:
        dataset_dir: Path to the preprocessed dataset directory. Either
            ``preprocess_timestamp.py``'s output (flat) or
            ``preprocess_timestamp_global.py``'s output (multi-region).
        region: For multi-region datasets, which region to read. Must
            be one of the regions present in the preprocessed tree.
            Ignored / must be ``None`` for flat datasets.
        split: Temporal split — ``"train"``, ``"val"``, or ``"test"``.
        station_split: Spatial split — ``"train"``, ``"test"``, or
            ``"all"`` (bypass the filter, return every station in the
            selected region).
        target_variables: Snapshot variables. Default: ``["t2m"]``.
        tessera_path, tessera_station_csv, load_tessera_patches: As in
            :class:`DailyDownscalingDataset`.
        normalise_era5: Z-score the ERA5 channels using training-split
            statistics (cached alongside the ERA5 grid files).
        include_static_fields: Include ERA5 static fields in the context
            grid.
        vae_latents_path, vae_latents_station_csv, vae_latents_zscore:
            As in :class:`DailyDownscalingDataset`.
        train_station_allowlist: Optional further filter applied to the
            station set *after* the spatial split and TESSERA/VAE
            filters. When set, only stations whose station_id appears
            in this iterable are kept. Used by the station-count data-
            efficiency experiment to train on a random sub-sample of
            the train-split stations. ``None`` (default) keeps every
            station that passes the other filters.
        probe_active_from: Optional per-station "first usable timestamp"
            map for the temporal-axis data-efficiency experiment. When
            set, each ``__getitem__`` drops rows whose
            ``probe_active_from[station_id]`` is later than the
            episode's timestamp — simulating a station that has been
            in the field for only the last d months of the training
            window. Stations not in this dict are never filtered. The
            mask applies to whatever split this dataset serves, but
            because typical values fall inside the training window
            it only affects training episodes in practice. ``None``
            (default) disables the temporal mask entirely.
    """

    SNAPSHOT_TARGET_VARIABLES = {"t2m", "wind", "precip"}

    def __init__(
        self,
        dataset_dir: str | Path,
        split: str = "train",
        station_split: str = "train",
        target_variables: list[str] | None = None,
        tessera_path: str | Path | None = None,
        tessera_station_csv: str | Path | None = None,
        load_tessera_patches: bool = False,
        normalise_era5: bool = True,
        include_static_fields: bool = True,
        vae_latents_path: str | Path | None = None,
        vae_latents_station_csv: str | Path | None = None,
        vae_latents_zscore: bool = True,
        region: str | None = None,
        min_patch_coverage: float = DEFAULT_MIN_TESSERA_PATCH_COVERAGE,
        train_station_allowlist: "set[str] | list[str] | None" = None,
        probe_active_from: dict[str, str] | None = None,
        train_end_override: str | None = None,
        drop_context_channels: list[str] | None = None,
        drop_context_strict: bool = False,
        lead_hours: int | None = None,
    ):
        self.dataset_dir = Path(dataset_dir)
        self.split = split
        self.station_split = station_split
        self._drop_context_channels = drop_context_channels
        self._drop_context_strict = drop_context_strict
        self.lead_hours = lead_hours

        # Train-end override (data-efficiency rollout experiment). When set,
        # the boundary between the train and val splits is moved earlier
        # to this timestamp; observations after it are excluded from the
        # train split (and pushed into the val split, since the dataset's
        # full valid_timestamps list is the same). NOT applied to
        # normalisation-stats computation: those statistics are a property
        # of the dataset's ERA5 distribution, not of any one training
        # subset, and recomputing per sweep point would (a) destroy
        # cross-sweep comparability and (b) invalidate the cached stats
        # file shared with v1/v2 runs.
        self.train_end_override = train_end_override

        # Default to t2m only, and reject any request that mixes
        # daily-cadence variable names into a snapshot dataset. The
        # generic validate_target_variables only checks membership in the
        # union set; it can't know that "tmax" makes no sense here.
        requested = list(target_variables) if target_variables else ["t2m"]
        bad = [v for v in requested if v not in self.SNAPSHOT_TARGET_VARIABLES]
        if bad:
            raise ValueError(
                f"SnapshotDownscalingDataset only supports "
                f"{sorted(self.SNAPSHOT_TARGET_VARIABLES)}; got unsupported "
                f"variables {bad}. Did you mean to use DailyDownscalingDataset?"
            )
        self.target_variables = validate_target_variables(requested)
        self.n_target_variables = len(self.target_variables)
        self.include_static_fields = include_static_fields
        self.normalise_era5 = normalise_era5
        self.load_tessera_patches = load_tessera_patches

        # --- Metadata + layout resolution ---
        #
        # Two layouts supported:
        #   snapshot_v1               — flat, single region.
        #   multi_region_snapshot_v1  — layered, one subdir per region.
        #
        # Resolve which ERA5 grid directory to read from (self._grid_root)
        # and where GHCNh lives (self._ghcnh_root). For the flat layout
        # both point at dataset_dir; for multi-region, _grid_root points
        # at the region subdir and _ghcnh_root stays at the top level.
        with open(self.dataset_dir / "metadata.json") as f:
            self.metadata = json.load(f)
        layout = self.metadata.get("layout_version")

        if layout == "snapshot_v1":
            if region is not None:
                raise ValueError(
                    f"Dataset at {self.dataset_dir} is flat (layout_version="
                    f"'snapshot_v1'); do not pass region={region!r}."
                )
            self.region = None
            self._grid_root = self.dataset_dir
            self._ghcnh_root = self.dataset_dir
            self._n_static_meta = self.metadata.get("n_static_channels", 0)
        elif layout == "multi_region_snapshot_v1":
            if region is None:
                raise ValueError(
                    f"Dataset at {self.dataset_dir} is multi-region (layout_"
                    f"version='multi_region_snapshot_v1'); must pass region="
                    f"<name>. Available: {list(self.metadata['regions'])}"
                )
            if region not in self.metadata["regions"]:
                raise ValueError(
                    f"Region {region!r} not found in {self.dataset_dir}. "
                    f"Available: {list(self.metadata['regions'])}"
                )
            self.region = region
            self._grid_root = self.dataset_dir / "regions" / region
            self._ghcnh_root = self.dataset_dir
            # Per-region static field count lives inside the region entry
            # of the top-level manifest.
            self._n_static_meta = self.metadata["regions"][region].get(
                "n_static_channels", 0,
            )
        else:
            raise ValueError(
                f"Dataset at {self.dataset_dir} has layout_version={layout!r}; "
                f"expected 'snapshot_v1' or 'multi_region_snapshot_v1'. "
                f"Point this class at a snapshot or snapshot-global preprocessing "
                f"output."
            )

        self.grid_lats = np.load(self._grid_root / "lats.npy").astype(np.float32)
        self.grid_lons = np.load(self._grid_root / "lons.npy").astype(np.float32)

        if include_static_fields:
            self.static_fields = np.load(
                self._grid_root / "static_fields.npy"
            ).astype(np.float32)
            self.n_static = self._n_static_meta
        else:
            self.static_fields = None
            self.n_static = 0

        # Resolve any context channels to drop (e.g. precipitation, to match a
        # 19-channel model) into dynamic-block indices, against this dataset's
        # own channel names. Strict at train (typo -> error); lenient at eval
        # (a name already absent -> skipped). Empty default -> no-op.
        _chan_names = self.metadata.get("era5_dynamic_channels")
        if self._drop_context_channels and not _chan_names:
            raise ValueError(
                "drop_context_channels requested but metadata has no "
                "'era5_dynamic_channels' to resolve names against."
            )
        self.drop_dynamic_indices = resolve_drop_channel_indices(
            self._drop_context_channels, _chan_names or [],
            strict=self._drop_context_strict,
        )

        # Snapshot context grid has 4 time channels (cos/sin DoY +
        # cos/sin HoD) rather than the 2 used by the daily layout.
        self.n_context_channels = (
            self.metadata["n_dynamic_channels"] - len(self.drop_dynamic_indices)
            + self.n_static + 2 + 4
            + (1 if self.lead_hours is not None else 0)
        )

        # --- Stations: read top-level stations.csv, filter by region
        # (if multi-region), then apply the spatial split. ---
        stations_all = pd.read_csv(self.dataset_dir / "stations.csv")
        valid_station_indices = np.load(
            self.dataset_dir / "valid_station_indices.npy"
        )
        if self.region is not None:
            region_mask = (stations_all["region"].values == self.region)
        else:
            region_mask = np.ones(len(stations_all), dtype=bool)

        if station_split == "all":
            spatial_mask = region_mask
        else:
            spatial_mask = region_mask & (
                stations_all["spatial_split"].values == station_split
            )
        spatial_indices = np.where(spatial_mask)[0]
        ghcnh_index_for_station = valid_station_indices[spatial_indices]
        # Cache stations_all's filtered view for later slicing.
        stations = stations_all

        self._tessera_indices: np.ndarray | None = None
        self._patches_mmap: np.memmap | None = None

        if tessera_path is not None:
            if tessera_station_csv is None:
                raise ValueError(
                    "tessera_station_csv is required when tessera_path is set"
                )
            t_result = filter_stations_by_tessera_patches(
                stations_df=stations,
                spatial_indices=spatial_indices,
                tessera_path=Path(tessera_path),
                tessera_station_csv=Path(tessera_station_csv),
                keep_mmap_alive=load_tessera_patches,
                min_patch_coverage=min_patch_coverage,
            )
            n_before = len(spatial_indices)
            spatial_indices = spatial_indices[t_result.kept_mask]
            ghcnh_index_for_station = ghcnh_index_for_station[t_result.kept_mask]
            self._tessera_indices = t_result.tessera_row_indices
            self._patches_mmap = t_result.patches_mmap
            print(
                f"SnapshotDownscalingDataset({split}/{station_split}"
                f"{f'/{self.region}' if self.region else ''}): "
                f"{len(spatial_indices)}/{n_before} stations after TESSERA filtering"
            )
        elif load_tessera_patches:
            raise ValueError("load_tessera_patches=True requires tessera_path")

        self._vae_latents: np.ndarray | None = None
        self._use_vae_latents = False
        self.vae_latent_dim = 0

        if vae_latents_path is not None:
            if vae_latents_station_csv is None:
                raise ValueError(
                    "vae_latents_station_csv is required when vae_latents_path is set"
                )
            v_result = filter_stations_by_vae_latents(
                stations_df=stations,
                spatial_indices=spatial_indices,
                vae_latents_path=Path(vae_latents_path),
                vae_latents_station_csv=Path(vae_latents_station_csv),
                zscore=vae_latents_zscore,
            )
            n_before = len(spatial_indices)
            spatial_indices = spatial_indices[v_result.kept_mask]
            ghcnh_index_for_station = ghcnh_index_for_station[v_result.kept_mask]
            if self._tessera_indices is not None:
                self._tessera_indices = self._tessera_indices[v_result.kept_mask]
            self._vae_latents = v_result.latents
            self._use_vae_latents = True
            self.vae_latent_dim = v_result.latent_dim
            print(
                f"SnapshotDownscalingDataset({split}/{station_split}"
                f"{f'/{self.region}' if self.region else ''}): "
                f"{len(spatial_indices)}/{n_before} stations after VAE latent filtering"
            )
            if self.load_tessera_patches:
                print(
                    "VAE latents active: raw TESSERA patches will NOT be "
                    "served during __getitem__ (latents replace them)."
                )
                self.load_tessera_patches = False
                self._patches_mmap = None

        self._ghcnh_index_for_station = ghcnh_index_for_station
        self.station_lats = stations["latitude"].values[spatial_indices].astype(np.float32)
        self.station_lons = stations["longitude"].values[spatial_indices].astype(np.float32)
        self.station_elevs = stations["elevation"].values[spatial_indices].astype(np.float32)
        self.station_delta_elevs = (
            stations["delta_elevation"].values[spatial_indices].astype(np.float32)
        )
        self.station_mtpi = _load_optional_station_mtpi(stations, spatial_indices)
        self.station_ids = stations["station_id"].values[spatial_indices]

        # Train-station allowlist (station-count data-efficiency experiment).
        # Applied AFTER the TESSERA/VAE filters so that the requested K is the
        # final count the model trains on. Only applied when station_split is
        # "train" — for val/test we always evaluate on the full held-out set
        # regardless of how the train set was sub-sampled.
        self.train_station_allowlist: set[str] | None = None
        if train_station_allowlist is not None:
            if station_split != "train":
                # Silently ignore for val/test to keep the API symmetric:
                # the caller can pass this kwarg unconditionally to both
                # train and val/test datasets without it accidentally
                # shrinking the held-out evaluation set.
                pass
            else:
                allowlist = set(str(s) for s in train_station_allowlist)
                if not allowlist:
                    raise ValueError(
                        "train_station_allowlist is empty; this would leave "
                        "the train dataset with zero stations."
                    )
                keep = np.array(
                    [str(sid) in allowlist for sid in self.station_ids],
                    dtype=bool,
                )
                n_before = len(self.station_ids)
                if not keep.any():
                    raise ValueError(
                        f"train_station_allowlist filtered out every station "
                        f"in the dataset (allowlist had {len(allowlist)} "
                        f"entries, none of which matched any of the {n_before} "
                        f"stations remaining after spatial+TESSERA+VAE filtering). "
                        f"Check that the allowlist's station_ids come from the "
                        f"same canonical station list this dataset uses."
                    )
                self._apply_post_filter_mask(keep)
                self.train_station_allowlist = allowlist
                print(
                    f"SnapshotDownscalingDataset({split}/{station_split}"
                    f"{f'/{self.region}' if self.region else ''}): "
                    f"{int(keep.sum())}/{n_before} stations after "
                    f"train_station_allowlist filtering"
                )

        # Probe-station temporal mask (temporal-axis data-efficiency
        # experiment). Stored verbatim and applied per-episode in
        # __getitem__; the mask is a no-op for stations not in this dict.
        # Persisting None vs {} matters: None means "no mask configured",
        # {} means "mask configured but currently empty" — both behave
        # identically in __getitem__ but the distinction is useful for
        # config-echo / debugging.
        self.probe_active_from: dict[str, str] | None = None
        if probe_active_from is not None:
            self.probe_active_from = {
                str(sid): str(ts) for sid, ts in probe_active_from.items()
            }
            n_in_dataset = sum(
                1 for sid in self.station_ids
                if str(sid) in self.probe_active_from
            )
            print(
                f"SnapshotDownscalingDataset({split}/{station_split}"
                f"{f'/{self.region}' if self.region else ''}): "
                f"probe_active_from configured with "
                f"{len(self.probe_active_from)} entries; "
                f"{n_in_dataset} match a station in this dataset"
            )

        # Temporal split — uses valid_timestamps, which the snapshot
        # preprocessor writes. The daily 'valid_dates' alias also present
        # in metadata is intentionally NOT used here so that this class
        # never reads daily-cadence state by accident.
        effective_train_end = (
            self.train_end_override
            if self.train_end_override is not None
            else self.metadata["temporal_split"]["train_end"]
        )
        self.timestamps = episodes_for_split(
            self.metadata["valid_timestamps"],
            split=split,
            train_end=effective_train_end,
            val_end=self.metadata["temporal_split"]["val_end"],
        )
        # Daily-cadence back-compat: ``.dates`` is a well-known attribute
        # on DailyDownscalingDataset, and evaluate.py reads it to drive
        # season-of-year analysis. We expose the same name here pointing
        # at the timestamp list so callers don't need to branch.
        self.dates = self.timestamps

        self.era5_mean: np.ndarray | None = None
        self.era5_std: np.ndarray | None = None
        if normalise_era5:
            stats_name = (
                "normalisation_stats.npz" if include_static_fields
                else "normalisation_stats_no_static.npz"
            )
            # NOTE: norm-stats train_end is intentionally the metadata
            # value, NOT the override — see the comment on
            # ``train_end_override`` in __init__.
            train_timestamps = [
                t for t in self.metadata["valid_timestamps"]
                if t <= self.metadata["temporal_split"]["train_end"]
            ]
            self.era5_mean, self.era5_std = load_or_compute_era5_norm_stats(
                cache_path=self._grid_root / stats_name,
                train_episode_ids=train_timestamps,
                era5_daily_dir=self._grid_root / "era5_snapshot",
                grid_lats=self.grid_lats,
                grid_lons=self.grid_lons,
                static_fields=self.static_fields,
            )

        mode_label = (
            f"vae latents (dim={self.vae_latent_dim})" if self._use_vae_latents
            else "serving patches" if self.load_tessera_patches
            else "filter-only" if tessera_path
            else "off"
        )
        print(
            f"SnapshotDownscalingDataset({split}/{station_split}"
            f"{f'/{self.region}' if self.region else ''}): "
            f"{len(self.timestamps)} episodes, {len(spatial_indices)} stations, "
            f"targets={self.target_variables}, tessera={mode_label}"
        )

    def __len__(self) -> int:
        return len(self.timestamps)

    def _apply_post_filter_mask(self, keep: np.ndarray) -> None:
        """Apply a boolean mask to every station-aligned array.

        Used by the ``train_station_allowlist`` filter, which runs after
        the spatial split + TESSERA + VAE filtering chain and so needs
        to subset each per-station array in lockstep. ``keep`` is
        length-equal to the current self.station_ids and selects which
        stations survive.
        """
        self._ghcnh_index_for_station = self._ghcnh_index_for_station[keep]
        self.station_lats = self.station_lats[keep]
        self.station_lons = self.station_lons[keep]
        self.station_elevs = self.station_elevs[keep]
        self.station_delta_elevs = self.station_delta_elevs[keep]
        if self.station_mtpi is not None:
            self.station_mtpi = self.station_mtpi[keep]
        self.station_ids = self.station_ids[keep]
        if self._tessera_indices is not None:
            self._tessera_indices = self._tessera_indices[keep]
        if self._vae_latents is not None:
            self._vae_latents = self._vae_latents[keep]

    def __getitem__(self, idx: int) -> dict:
        ts = self.timestamps[idx]
        # ts is "YYYY-MM-DD-HH"; split into the date portion (passed to
        # build_context_grid for day-of-year) and the integer hour.
        date_str = ts[:10]
        hour = int(ts[11:13])

        context_grid = build_context_grid(
            era5_daily_path=self._grid_root / "era5_snapshot" / f"{ts}.npy",
            static_fields=self.static_fields,
            grid_lats=self.grid_lats,
            grid_lons=self.grid_lons,
            date_str=date_str,
            era5_mean=self.era5_mean,
            era5_std=self.era5_std,
            hour=hour,
            drop_dynamic_indices=self.drop_dynamic_indices,
            lead_hours=self.lead_hours,
        )

        valid_indices, per_var_values = select_valid_targets(
            ghcnh_daily_path=self._ghcnh_root / "ghcnh_snapshot" / f"{ts}.npz",
            ghcnh_index_for_station=self._ghcnh_index_for_station,
            target_variables=self.target_variables,
        )

        # Probe-station temporal mask (no-op if probe_active_from is None
        # or empty, or if the episode's timestamp is later than every
        # probe station's active_from). Applied before the early-empty
        # short-circuit below so that an entirely-masked episode falls
        # through into ``empty_episode_result`` and is dropped by the
        # collate function, same as if no observations were valid.
        if self.probe_active_from:
            valid_indices = filter_valid_indices_by_probe_active_from(
                valid_indices=valid_indices,
                station_ids=self.station_ids,
                timestamp=ts,
                probe_active_from=self.probe_active_from,
            )

        tessera_mode = (
            "patches" if self.load_tessera_patches and self._patches_mmap is not None
            else "latents" if self._use_vae_latents
            else "off"
        )

        if len(valid_indices) == 0:
            # The 'date' field in the result dict stores the full
            # timestamp for snapshot episodes — callers that want just
            # the date can trivially slice it.
            return empty_episode_result(
                context_grid=context_grid,
                grid_lats=self.grid_lats,
                grid_lons=self.grid_lons,
                n_target_variables=self.n_target_variables,
                date_str=ts,
                tessera_mode=tessera_mode,
                tessera_dim=self.vae_latent_dim,
            )

        result = assemble_episode_result(
            context_grid=context_grid,
            grid_lats=self.grid_lats,
            grid_lons=self.grid_lons,
            date_str=ts,
            valid_indices=valid_indices,
            per_var_values=per_var_values,
            station_lats=self.station_lats,
            station_lons=self.station_lons,
            station_elevs=self.station_elevs,
            station_delta_elevs=self.station_delta_elevs,
            station_mtpi=self.station_mtpi,
            n_target_variables=self.n_target_variables,
        )

        if tessera_mode == "patches":
            patches_np = slice_tessera_patches(
                self._patches_mmap, self._tessera_indices, valid_indices,
            )
            result["target_tessera"] = torch.from_numpy(patches_np)
        elif tessera_mode == "latents":
            result["target_tessera"] = torch.from_numpy(
                self._vae_latents[valid_indices]
            )

        return result


# ---------------------------------------------------------------------------
# Multi-region snapshot (timestamp-cadence) dataset
# ---------------------------------------------------------------------------


@dataclass
class SnapshotRegionState:
    """All per-region state needed to serve one region's snapshot episodes.

    Structurally identical to :class:`RegionState` but points at the
    snapshot-layout paths (``era5_snapshot/`` rather than
    ``era5_daily/``) and uses timestamp identifiers. Kept as a distinct
    dataclass so the two dataset classes don't accidentally share
    assumptions.
    """
    name: str
    region_dir: Path
    grid_lats: np.ndarray
    grid_lons: np.ndarray
    static_fields: np.ndarray | None
    n_static: int
    era5_mean: np.ndarray | None
    era5_std: np.ndarray | None
    station_ids: np.ndarray
    station_lats: np.ndarray
    station_lons: np.ndarray
    station_elevs: np.ndarray
    station_delta_elevs: np.ndarray
    station_mtpi: np.ndarray | None
    tessera_row_indices: np.ndarray | None
    patches_mmap: np.memmap | None
    vae_latents: np.ndarray | None
    ghcnh_index_for_station: np.ndarray
    # Episode identifiers as "YYYY-MM-DD-HH" strings — same length for
    # every region under a given (split, temporal_split), since the
    # temporal split is global in the preprocessor.
    timestamps: list[str]
    flat_offset: int


class MultiRegionSnapshotDownscalingDataset(Dataset):
    """Multi-region snapshot dataset with per-region spatial splits.

    Counterpart to :class:`MultiRegionDownscalingDataset` for the 6-hourly
    snapshot layout. Episodes are ``(region, date, hour)`` triples.

    Supports both API forms from its daily sibling:

      * Old-style: ``regions=[...], station_split="train"`` — every
        listed region contributes stations from the given spatial split.
      * New-style: ``region_specs={"europe": "train", "us": "all"}`` —
        per-region spatial split, enabling held-out-within-training
        experiments.

    Only the new-style form supports the interesting multi-region
    experiments (train on EU+US, test on EU-test); old-style is kept
    for backward-compat with the existing code paths.

    Layout expectations (from :mod:`preprocess_timestamp_global`):

      * Top-level ``metadata.json`` with ``layout_version ==
        "multi_region_snapshot_v1"`` and ``cadence == "6h"``.
      * ``regions/<n>/`` containing the per-region grid, static
        fields, and ``era5_snapshot/<ts>.npy`` files.
      * ``ghcnh_snapshot/<ts>.npz`` shared at the top level.
      * ``stations.csv`` at the top level with a ``region`` column.

    Target variables: ``{"t2m", "wind", "precip"}`` only — same constraint as
    :class:`SnapshotDownscalingDataset`.
    """

    SNAPSHOT_TARGET_VARIABLES = {"t2m", "wind", "precip"}

    def __init__(
        self,
        dataset_dir: str | Path,
        regions: list[str] | None = None,
        split: str = "train",
        station_split: str = "train",
        target_variables: list[str] | None = None,
        tessera_path: str | Path | None = None,
        tessera_station_csv: str | Path | None = None,
        load_tessera_patches: bool = False,
        normalise_era5: bool = True,
        include_static_fields: bool = True,
        vae_latents_path: str | Path | None = None,
        vae_latents_station_csv: str | Path | None = None,
        vae_latents_zscore: bool = True,
        region_specs: dict[str, str] | None = None,
        normalisation_policy: str = "per_region",
        min_patch_coverage: float = DEFAULT_MIN_TESSERA_PATCH_COVERAGE,
        train_station_allowlist: "set[str] | list[str] | None" = None,
        probe_active_from: dict[str, str] | None = None,
        train_end_override: str | None = None,
        drop_context_channels: list[str] | None = None,
        drop_context_strict: bool = False,
        lead_hours: int | None = None,
    ):
        self.dataset_dir = Path(dataset_dir)
        self.split = split
        self._drop_context_channels = drop_context_channels
        self._drop_context_strict = drop_context_strict
        self.lead_hours = lead_hours
        # See SnapshotDownscalingDataset.__init__ for the rationale on
        # why this override applies to the train/val split boundary
        # but not to ERA5 normalisation-stats computation.
        self.train_end_override = train_end_override
        if normalisation_policy not in ("per_region", "global"):
            raise ValueError(
                f"normalisation_policy must be 'per_region' or 'global', "
                f"got {normalisation_policy!r}"
            )
        self.normalisation_policy = normalisation_policy

        # Target-variable validation — reject daily names explicitly.
        requested = list(target_variables) if target_variables else ["t2m"]
        bad = [v for v in requested if v not in self.SNAPSHOT_TARGET_VARIABLES]
        if bad:
            raise ValueError(
                f"MultiRegionSnapshotDownscalingDataset only supports "
                f"{sorted(self.SNAPSHOT_TARGET_VARIABLES)}; got unsupported "
                f"variables {bad}. Did you mean to use "
                f"MultiRegionDownscalingDataset?"
            )
        self.target_variables = validate_target_variables(requested)
        self.n_target_variables = len(self.target_variables)
        self.include_static_fields = include_static_fields
        self.normalise_era5 = normalise_era5
        self.load_tessera_patches = load_tessera_patches
        self._use_vae_latents = vae_latents_path is not None

        # --- Top-level manifest + layout check ---
        top_md_path = self.dataset_dir / "metadata.json"
        if not top_md_path.exists():
            raise FileNotFoundError(
                f"Expected top-level metadata.json at {top_md_path}. "
                f"Is {self.dataset_dir} really a multi-region snapshot dataset?"
            )
        with open(top_md_path) as f:
            self.top_metadata = json.load(f)
        if self.top_metadata.get("layout_version") != "multi_region_snapshot_v1":
            raise ValueError(
                f"Dataset at {self.dataset_dir} is not a "
                f"multi_region_snapshot_v1 layout (got layout_version="
                f"{self.top_metadata.get('layout_version')!r})."
            )

        available = list(self.top_metadata["regions"].keys())

        # --- Resolve region_specs (same logic as MultiRegionDownscalingDataset) ---
        if region_specs is not None and regions is not None:
            raise ValueError(
                "Pass either `region_specs` OR `regions`+`station_split`, "
                "not both. region_specs is the preferred form."
            )
        if region_specs is not None:
            for name in region_specs:
                if name not in available:
                    raise ValueError(
                        f"region_specs includes '{name}' but dataset "
                        f"only has regions {available}."
                    )
            for name, s in region_specs.items():
                if s not in ("train", "test", "all"):
                    raise ValueError(
                        f"region_specs['{name}']='{s}' is not one of "
                        f"'train', 'test', 'all'."
                    )
            self.region_specs = dict(region_specs)
        else:
            resolved_regions = list(regions) if regions is not None else available
            missing = [r for r in resolved_regions if r not in available]
            if missing:
                raise ValueError(
                    f"Requested regions {missing} not found. Available: {available}"
                )
            if station_split not in ("train", "test", "all"):
                raise ValueError(
                    f"station_split='{station_split}' is not one of "
                    f"'train', 'test', 'all'."
                )
            self.region_specs = {r: station_split for r in resolved_regions}

        self.region_order: list[str] = list(self.region_specs.keys())
        split_values = set(self.region_specs.values())
        self.station_split = (
            next(iter(split_values)) if len(split_values) == 1 else "mixed"
        )

        self.n_dynamic_channels = self.top_metadata["n_dynamic_channels"]
        temporal = self.top_metadata["temporal_split"]
        valid_timestamps = self.top_metadata["valid_timestamps"]

        global_stations = pd.read_csv(self.dataset_dir / "stations.csv")
        global_valid_indices = np.load(
            self.dataset_dir / "valid_station_indices.npy"
        )
        # GHCNh is shared at the top level — every region's episodes
        # read from the same directory, keyed by station index.
        self._ghcnh_snapshot_dir = self.dataset_dir / "ghcnh_snapshot"

        # VAE latent z-score stats (if used) are computed lazily inside
        # filter_stations_by_vae_latents from the first region state
        # built and cached alongside the .npy for all subsequent users.
        # No precompute needed here.

        # Normalise the train-station allowlist to a set of strings up
        # front. Stored on the instance for visibility / debugging; the
        # actual filtering happens inside _build_region_state for each
        # region whose spec is "train".
        self.train_station_allowlist: set[str] | None = None
        if train_station_allowlist is not None:
            self.train_station_allowlist = set(
                str(s) for s in train_station_allowlist
            )
            if not self.train_station_allowlist:
                raise ValueError(
                    "train_station_allowlist is empty; this would leave "
                    "every train-spec region with zero stations."
                )

        self.per_region: dict[str, SnapshotRegionState] = {}
        flat_offset = 0
        for name in self.region_order:
            region_dir = self.dataset_dir / "regions" / name
            if not region_dir.exists():
                raise FileNotFoundError(
                    f"Region subdir {region_dir} does not exist. "
                    f"Did preprocessing include '{name}'?"
                )
            state = self._build_region_state(
                name=name,
                region_dir=region_dir,
                global_stations=global_stations,
                global_valid_indices=global_valid_indices,
                station_split=self.region_specs[name],
                split=split,
                valid_timestamps=valid_timestamps,
                train_end=temporal["train_end"],
                val_end=temporal["val_end"],
                tessera_path=Path(tessera_path) if tessera_path else None,
                tessera_station_csv=(
                    Path(tessera_station_csv) if tessera_station_csv else None
                ),
                load_tessera_patches=load_tessera_patches,
                vae_latents_path=Path(vae_latents_path) if vae_latents_path else None,
                vae_latents_station_csv=(
                    Path(vae_latents_station_csv) if vae_latents_station_csv else None
                ),
                vae_latents_zscore=vae_latents_zscore,
                min_patch_coverage=min_patch_coverage,
                flat_offset=flat_offset,
                train_station_allowlist=self.train_station_allowlist,
                train_end_override=self.train_end_override,
            )
            self.per_region[name] = state
            flat_offset += len(state.station_ids)

        # Flat station arrays across all regions, in region_order. Same
        # semantics as MultiRegionDownscalingDataset — per-region local
        # station indices i become flat (i + flat_offset) in result dicts.
        self.station_ids = np.concatenate([
            self.per_region[n].station_ids for n in self.region_order
        ]) if self.region_order else np.array([], dtype=object)
        self.station_lats = np.concatenate([
            self.per_region[n].station_lats for n in self.region_order
        ]) if self.region_order else np.array([], dtype=np.float32)
        self.station_lons = np.concatenate([
            self.per_region[n].station_lons for n in self.region_order
        ]) if self.region_order else np.array([], dtype=np.float32)
        self.station_elevs = np.concatenate([
            self.per_region[n].station_elevs for n in self.region_order
        ]) if self.region_order else np.array([], dtype=np.float32)
        self.station_delta_elevs = np.concatenate([
            self.per_region[n].station_delta_elevs for n in self.region_order
        ]) if self.region_order else np.array([], dtype=np.float32)
        # mTPI is optional (present only when stations.csv carried an `mtpi`
        # column). All regions share one global stations table, so it is
        # either available for every region or for none; concatenate only
        # when uniformly present, else expose None so the model falls back
        # to the 2-feature (elevation, delta_elevation) layout.
        if self.region_order and all(
            self.per_region[n].station_mtpi is not None for n in self.region_order
        ):
            self.station_mtpi = np.concatenate([
                self.per_region[n].station_mtpi for n in self.region_order
            ])
        else:
            self.station_mtpi = None

        # Episode dispatch: cumulative lengths across regions, so
        # idx < cum_lengths[k] means episode k comes from region_order[k].
        self._cum_lengths: list[int] = []
        running = 0
        for name in self.region_order:
            running += len(self.per_region[name].timestamps)
            self._cum_lengths.append(running)

        # Resolve context channels to drop into dynamic-block indices (see
        # SnapshotDownscalingDataset). Strict at train, lenient at eval.
        _chan_names = self.top_metadata.get("era5_dynamic_channels")
        if self._drop_context_channels and not _chan_names:
            raise ValueError(
                "drop_context_channels requested but top metadata has no "
                "'era5_dynamic_channels' to resolve names against."
            )
        self.drop_dynamic_indices = resolve_drop_channel_indices(
            self._drop_context_channels, _chan_names or [],
            strict=self._drop_context_strict,
        )

        # Snapshot context grid: 4 time channels (cos/sin DoY + cos/sin HoD)
        # + 2 lat/lon coord channels.
        _n_dyn = self.n_dynamic_channels - len(self.drop_dynamic_indices)
        _lead = 1 if self.lead_hours is not None else 0
        ch_counts = {
            name: (st.n_static + _n_dyn + 2 + 4 + _lead)
            for name, st in self.per_region.items()
        }
        if len(set(ch_counts.values())) != 1:
            raise ValueError(
                f"Regions disagree on n_context_channels: {ch_counts}."
            )
        self.n_context_channels = next(iter(ch_counts.values()))

        if self._use_vae_latents:
            dims = {
                name: st.vae_latents.shape[1]
                for name, st in self.per_region.items()
                if st.vae_latents is not None
            }
            if len(set(dims.values())) != 1:
                raise ValueError(f"Regions disagree on VAE latent dim: {dims}")
            self.vae_latent_dim = next(iter(dims.values()))
        else:
            self.vae_latent_dim = 0

        total = self._cum_lengths[-1] if self._cum_lengths else 0
        specs_str = ", ".join(f"{n}:{s}" for n, s in self.region_specs.items())
        print(
            f"MultiRegionSnapshotDownscalingDataset({split}): "
            f"{total} total episodes across {len(self.region_order)} regions "
            f"[{specs_str}]"
        )

        # Probe-station temporal mask (temporal-axis data-efficiency
        # experiment). Stored verbatim and applied per-episode in
        # __getitem__; the mask is a no-op for any station not in this
        # dict. Same caveats as on SnapshotDownscalingDataset: persisting
        # None vs {} matters for downstream config-echo / debugging.
        self.probe_active_from: dict[str, str] | None = None
        if probe_active_from is not None:
            self.probe_active_from = {
                str(sid): str(ts) for sid, ts in probe_active_from.items()
            }
            n_in_dataset = sum(
                1 for sid in self.station_ids
                if str(sid) in self.probe_active_from
            )
            print(
                f"MultiRegionSnapshotDownscalingDataset({split}): "
                f"probe_active_from configured with "
                f"{len(self.probe_active_from)} entries; "
                f"{n_in_dataset} match a station in this dataset"
            )

    def _build_region_state(
        self,
        name: str,
        region_dir: Path,
        global_stations: pd.DataFrame,
        global_valid_indices: np.ndarray,
        station_split: str,
        split: str,
        valid_timestamps: list[str],
        train_end: str,
        val_end: str,
        tessera_path: Path | None,
        tessera_station_csv: Path | None,
        load_tessera_patches: bool,
        vae_latents_path: Path | None,
        vae_latents_station_csv: Path | None,
        vae_latents_zscore: bool,
        min_patch_coverage: float,
        flat_offset: int,
        train_station_allowlist: set[str] | None = None,
        train_end_override: str | None = None,
    ) -> SnapshotRegionState:
        """Build one region's state for snapshot episodes."""
        grid_lats = np.load(region_dir / "lats.npy").astype(np.float32)
        grid_lons = np.load(region_dir / "lons.npy").astype(np.float32)

        if self.include_static_fields:
            static_fields = np.load(region_dir / "static_fields.npy").astype(np.float32)
            with open(region_dir / "region_metadata.json") as f:
                n_static = json.load(f)["n_static_channels"]
        else:
            static_fields = None
            n_static = 0

        in_region = global_stations["region"].values == name
        if station_split == "all":
            region_station_mask = in_region
        else:
            region_station_mask = in_region & (
                global_stations["spatial_split"].values == station_split
            )
        spatial_indices = np.where(region_station_mask)[0]
        ghcnh_index_for_station = global_valid_indices[spatial_indices]

        tessera_row_indices: np.ndarray | None = None
        patches_mmap: np.memmap | None = None
        if tessera_path is not None:
            if tessera_station_csv is None:
                raise ValueError(
                    "tessera_station_csv is required when tessera_path is set"
                )
            t_result = filter_stations_by_tessera_patches(
                stations_df=global_stations,
                spatial_indices=spatial_indices,
                tessera_path=tessera_path,
                tessera_station_csv=tessera_station_csv,
                keep_mmap_alive=load_tessera_patches,
                min_patch_coverage=min_patch_coverage,
            )
            n_before = len(spatial_indices)
            spatial_indices = spatial_indices[t_result.kept_mask]
            ghcnh_index_for_station = ghcnh_index_for_station[t_result.kept_mask]
            tessera_row_indices = t_result.tessera_row_indices
            patches_mmap = t_result.patches_mmap
            print(
                f"[{name}] {len(spatial_indices)}/{n_before} stations "
                f"after TESSERA filtering"
            )
        elif load_tessera_patches:
            raise ValueError("load_tessera_patches=True requires tessera_path")

        vae_latents: np.ndarray | None = None
        if vae_latents_path is not None:
            if vae_latents_station_csv is None:
                raise ValueError(
                    "vae_latents_station_csv is required when vae_latents_path is set"
                )
            v_result = filter_stations_by_vae_latents(
                stations_df=global_stations,
                spatial_indices=spatial_indices,
                vae_latents_path=vae_latents_path,
                vae_latents_station_csv=vae_latents_station_csv,
                zscore=vae_latents_zscore,
            )
            n_before = len(spatial_indices)
            spatial_indices = spatial_indices[v_result.kept_mask]
            ghcnh_index_for_station = ghcnh_index_for_station[v_result.kept_mask]
            if tessera_row_indices is not None:
                tessera_row_indices = tessera_row_indices[v_result.kept_mask]
            vae_latents = v_result.latents
            print(
                f"[{name}] {len(spatial_indices)}/{n_before} stations "
                f"after VAE latent filtering"
            )
            if load_tessera_patches:
                patches_mmap = None

        # Train-station allowlist (station-count data-efficiency experiment).
        # Applied AFTER spatial split + TESSERA + VAE filtering so the
        # requested K equals the final count the model trains on. Only
        # applied when this region's spec is "train" — for val/test we
        # never sub-sample, so the held-out evaluation set is unchanged.
        # ``train_station_allowlist`` is a global allowlist applied across
        # all regions; stations from other regions whose IDs happen to be
        # in the allowlist (extremely unlikely given GHCNh IDs) are just
        # kept, which is the intended behaviour for a multi-region run.
        if train_station_allowlist is not None and station_split == "train":
            station_ids_pre = global_stations["station_id"].values[spatial_indices]
            keep_mask = np.array(
                [str(sid) in train_station_allowlist for sid in station_ids_pre],
                dtype=bool,
            )
            n_before = len(spatial_indices)
            if not keep_mask.any():
                raise ValueError(
                    f"[{name}] train_station_allowlist filtered out every "
                    f"station after TESSERA+VAE filtering. Either the "
                    f"allowlist is wrong for this region or it was sized "
                    f"too aggressively at submit time."
                )
            spatial_indices = spatial_indices[keep_mask]
            ghcnh_index_for_station = ghcnh_index_for_station[keep_mask]
            if tessera_row_indices is not None:
                tessera_row_indices = tessera_row_indices[keep_mask]
            if vae_latents is not None:
                vae_latents = vae_latents[keep_mask]
            print(
                f"[{name}] {len(spatial_indices)}/{n_before} stations "
                f"after train_station_allowlist filtering "
                f"(allowlist size: {len(train_station_allowlist)})"
            )

        # Per-region ERA5 normalisation stats, read from / cached at
        # region_dir. Preprocessing also writes these stats proactively,
        # so in practice the cache exists and we just load.
        #
        # When normalisation_policy == "global", load the cross-region
        # stats from the top-level dataset_dir instead. Used for
        # transfer / joint-source experiments where train-region and
        # test-region would otherwise have mismatched z-score scales.
        era5_mean = None
        era5_std = None
        if self.normalise_era5:
            stats_name = (
                "normalisation_stats.npz" if self.include_static_fields
                else "normalisation_stats_no_static.npz"
            )
            if self.normalisation_policy == "global":
                global_name = (
                    "normalisation_stats_global.npz" if self.include_static_fields
                    else "normalisation_stats_no_static_global.npz"
                )
                global_stats_path = self.dataset_dir / global_name
                if not global_stats_path.exists():
                    raise FileNotFoundError(
                        f"normalisation_policy='global' but {global_stats_path} "
                        f"does not exist. Re-run preprocess_timestamp_global.py to "
                        f"generate global stats."
                    )
                loaded = np.load(global_stats_path)
                era5_mean = loaded["era5_mean"]
                era5_std = loaded["era5_std"]
            else:
                train_ts_list = [t for t in valid_timestamps if t <= train_end]
                era5_mean, era5_std = load_or_compute_era5_norm_stats(
                    cache_path=region_dir / stats_name,
                    train_episode_ids=train_ts_list,
                    era5_daily_dir=region_dir / "era5_snapshot",
                    grid_lats=grid_lats,
                    grid_lons=grid_lons,
                    static_fields=static_fields,
                )

        station_ids = global_stations["station_id"].values[spatial_indices]
        station_lats = global_stations["latitude"].values[spatial_indices].astype(np.float32)
        station_lons = global_stations["longitude"].values[spatial_indices].astype(np.float32)
        station_elevs = global_stations["elevation"].values[spatial_indices].astype(np.float32)
        station_delta_elevs = (
            global_stations["delta_elevation"].values[spatial_indices].astype(np.float32)
        )
        station_mtpi = _load_optional_station_mtpi(global_stations, spatial_indices)

        # Temporal split — same logic as SnapshotDownscalingDataset, and
        # identical for every region because the temporal split is global
        # in the preprocessor. train_end_override (used by the rollout
        # data-efficiency experiment) shifts the train/val boundary earlier
        # while leaving norm-stats / val_end / test_end unchanged.
        effective_train_end = (
            train_end_override if train_end_override is not None else train_end
        )
        timestamps = episodes_for_split(
            valid_timestamps, split=split,
            train_end=effective_train_end, val_end=val_end,
        )

        return SnapshotRegionState(
            name=name,
            region_dir=region_dir,
            grid_lats=grid_lats,
            grid_lons=grid_lons,
            static_fields=static_fields,
            n_static=n_static,
            era5_mean=era5_mean,
            era5_std=era5_std,
            station_ids=station_ids,
            station_lats=station_lats,
            station_lons=station_lons,
            station_elevs=station_elevs,
            station_delta_elevs=station_delta_elevs,
            station_mtpi=station_mtpi,
            tessera_row_indices=tessera_row_indices,
            patches_mmap=patches_mmap,
            vae_latents=vae_latents,
            ghcnh_index_for_station=ghcnh_index_for_station,
            timestamps=timestamps,
            flat_offset=flat_offset,
        )

    @property
    def dates(self) -> list[str]:
        """Flat list of timestamps aligned with episode indices.

        Provided for back-compat with :class:`DailyDownscalingDataset`'s
        ``.dates`` attribute — evaluate.py uses it to drive
        season-of-year analysis and expects the same interface.
        """
        out: list[str] = []
        for name in self.region_order:
            out.extend(self.per_region[name].timestamps)
        return out

    # Exposed under a snapshot-appropriate name too, for callers that
    # want to make the cadence explicit.
    @property
    def timestamps(self) -> list[str]:
        return self.dates

    def __len__(self) -> int:
        return self._cum_lengths[-1] if self._cum_lengths else 0

    def _dispatch(self, idx: int) -> tuple[str, int]:
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Episode index {idx} out of range [0, {len(self)})")
        prev_cum = 0
        for name, cum in zip(self.region_order, self._cum_lengths):
            if idx < cum:
                return name, idx - prev_cum
            prev_cum = cum
        raise RuntimeError("unreachable")  # pragma: no cover

    def __getitem__(self, idx: int) -> dict:
        region_name, local_idx = self._dispatch(idx)
        st = self.per_region[region_name]
        ts = st.timestamps[local_idx]
        # ts is "YYYY-MM-DD-HH".
        date_str = ts[:10]
        hour = int(ts[11:13])

        context_grid = build_context_grid(
            era5_daily_path=st.region_dir / "era5_snapshot" / f"{ts}.npy",
            static_fields=st.static_fields,
            grid_lats=st.grid_lats,
            grid_lons=st.grid_lons,
            date_str=date_str,
            era5_mean=st.era5_mean,
            era5_std=st.era5_std,
            hour=hour,
            drop_dynamic_indices=self.drop_dynamic_indices,
            lead_hours=self.lead_hours,
        )

        valid_indices, per_var_values = select_valid_targets(
            ghcnh_daily_path=self._ghcnh_snapshot_dir / f"{ts}.npz",
            ghcnh_index_for_station=st.ghcnh_index_for_station,
            target_variables=self.target_variables,
        )

        # Probe-station temporal mask (no-op if probe_active_from is None
        # or empty). Same semantics as on SnapshotDownscalingDataset: rows
        # are dropped where the station_id is in probe_active_from AND the
        # episode's timestamp is earlier than that station's active_from.
        # st.station_ids is the per-region filtered station array, aligned
        # with the per-region local valid_indices returned above.
        if self.probe_active_from:
            valid_indices = filter_valid_indices_by_probe_active_from(
                valid_indices=valid_indices,
                station_ids=st.station_ids,
                timestamp=ts,
                probe_active_from=self.probe_active_from,
            )

        tessera_mode = (
            "patches" if (self.load_tessera_patches and st.patches_mmap is not None)
            else "latents" if st.vae_latents is not None
            else "off"
        )

        if len(valid_indices) == 0:
            return empty_episode_result(
                context_grid=context_grid,
                grid_lats=st.grid_lats,
                grid_lons=st.grid_lons,
                n_target_variables=self.n_target_variables,
                date_str=ts,
                tessera_mode=tessera_mode,
                tessera_dim=self.vae_latent_dim,
            )

        result = assemble_episode_result(
            context_grid=context_grid,
            grid_lats=st.grid_lats,
            grid_lons=st.grid_lons,
            date_str=ts,
            valid_indices=valid_indices,
            per_var_values=per_var_values,
            station_lats=st.station_lats,
            station_lons=st.station_lons,
            station_elevs=st.station_elevs,
            station_delta_elevs=st.station_delta_elevs,
            station_mtpi=st.station_mtpi,
            n_target_variables=self.n_target_variables,
        )

        # Translate per-region local station indices to flat global
        # indices, so callers can address stations via a single integer
        # index into self.station_ids etc.
        result["target_station_indices"] = (
            result["target_station_indices"] + st.flat_offset
        )

        if tessera_mode == "patches":
            patches_np = slice_tessera_patches(
                st.patches_mmap, st.tessera_row_indices, valid_indices,
            )
            result["target_tessera"] = torch.from_numpy(patches_np)
        elif tessera_mode == "latents":
            result["target_tessera"] = torch.from_numpy(
                st.vae_latents[valid_indices]
            )

        return result


# ---------------------------------------------------------------------------
# Lead-conditioned (cross-lead) dataset
# ---------------------------------------------------------------------------


class MultiLeadDataset(Dataset):
    """Concatenate per-lead snapshot datasets for the lead-conditioned model.

    Each element of ``sub_datasets`` is a fully-built
    ``MultiRegionSnapshotDownscalingDataset`` for one forecast lead — same
    split, regions, stations, targets and TESSERA config — differing only in
    its ``dataset_dir`` (which lead's coarse context it reads) and its
    ``lead_hours`` (so its context grids carry the lead channel). Because the
    downscaling *target* is the station observation at the valid time, it is
    identical across leads; the leads are just alternative inputs for the same
    supervision. Concatenating them means one epoch sees every episode at every
    lead (uniform mixing), which is what lets the heads learn a lead-dependent
    spread σ(lead).

    Indexing is ConcatDataset-style (cumulative offsets). ``n_context_channels``
    and ``vae_latent_dim`` must agree across leads (they will, given identical
    channel + TESSERA config — the +1 lead channel is present in every sub); a
    mismatch is raised loudly since it almost always means a lead was built
    without the precip drop (lead-0 ERA5 is 20-channel, Aurora is 19). Any other
    attribute access forwards to the first sub-dataset, so the training loop can
    read dataset metadata transparently.
    """

    def __init__(self, sub_datasets: list):
        if not sub_datasets:
            raise ValueError("MultiLeadDataset requires at least one sub-dataset.")
        self.subs = list(sub_datasets)
        self._lens = [len(s) for s in self.subs]
        self._cum = np.cumsum([0, *self._lens])
        self.leads = [getattr(s, "lead_hours", None) for s in self.subs]

        nc = {s.n_context_channels for s in self.subs}
        if len(nc) != 1:
            raise ValueError(
                "Leads disagree on n_context_channels: "
                f"{list(zip(self.leads, (s.n_context_channels for s in self.subs)))}. "
                "Most likely a lead was built without the precipitation drop "
                "(ERA5 lead-0 is 20-channel, Aurora is 19) — pass "
                "--drop-context-channels total_precipitation_sum."
            )
        self.n_context_channels = next(iter(nc))

        vd = {getattr(s, "vae_latent_dim", None) for s in self.subs}
        if len(vd) != 1:
            raise ValueError(f"Leads disagree on vae_latent_dim: {vd}.")
        self.vae_latent_dim = next(iter(vd))

    def __len__(self) -> int:
        return int(self._cum[-1])

    def __getitem__(self, idx: int):
        if idx < 0:
            idx += len(self)
        if idx < 0 or idx >= len(self):
            raise IndexError(idx)
        j = int(np.searchsorted(self._cum, idx, side="right") - 1)
        return self.subs[j][idx - int(self._cum[j])]

    def __getattr__(self, name):
        # Only invoked when normal lookup fails, so the explicit attributes
        # above (n_context_channels, vae_latent_dim, ...) take priority. Forward
        # metadata-like reads to the first sub-dataset.
        subs = self.__dict__.get("subs")
        if subs:
            return getattr(subs[0], name)
        raise AttributeError(name)


# ---------------------------------------------------------------------------
# Backwards-compat re-exports
# ---------------------------------------------------------------------------
# Keep `from tessera_downscaling.data.dataset import downscaling_collate`
# working for callers that don't know the function has moved to helpers.py.

__all__ = [
    "DailyDownscalingDataset",
    "MultiLeadDataset",
    "MultiRegionDownscalingDataset",
    "MultiRegionSnapshotDownscalingDataset",
    "RegionState",
    "SnapshotDownscalingDataset",
    "SnapshotRegionState",
    "downscaling_collate",
]