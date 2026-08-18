"""PyTorch Datasets for 6-hourly station downscaling episodes.

Every run in the paper trains and evaluates on a *multi-region snapshot*
dataset — the layout written by ``scripts/preprocessing/preprocess_timestamp_global.py``
for ERA5 analyses and by ``scripts/preprocessing/preprocess_aurora.py`` for the
Aurora forecast leads (``layout_version == "multi_region_snapshot_v1"``)::

    <dataset_dir>/
        metadata.json                 layout_version, cadence "6h",
                                      era5_dynamic_channels, n_dynamic_channels,
                                      regions {name: {bbox_lat_lon, grid_shape,
                                      n_static_channels}}, valid_timestamps
                                      ["YYYY-MM-DD-HH", ...], temporal_split
                                      {train_end, val_end}, ...
        stations.csv                  one row per station: station_id, latitude,
                                      longitude, elevation, delta_elevation,
                                      region, spatial_split ("train"/"test"),
                                      [mtpi]
        valid_station_indices.npy     row of each stations.csv row inside the
                                      ghcnh_snapshot arrays
        ghcnh_snapshot/<ts>.npz       shared by all regions; keys ``t2m``
                                      (2 m temperature), ``wind`` (10 m wind
                                      speed), ``obs_count``; one entry per
                                      ghcnh row, NaN where unobserved
        regions/<name>/
            era5_snapshot/<ts>.npy    (C_dyn, H, W) float32 coarse fields
                                      (C_dyn = 20 for ERA5, 19 for Aurora)
            lats.npy, lons.npy        (H,), (W,) grid coordinates
            static_fields.npy         (13, H, W) ERA5 invariant fields
            region_metadata.json      static_channels, n_static_channels, ...
            normalisation_stats.npz   train-split ``era5_mean``/``era5_std``
            normalisation_stats_no_static.npz
                                      over dynamic (+ static) + lat + lon channels

An episode is one ``(region, timestamp)`` pair, ``ts = "YYYY-MM-DD-HH"`` at
00/06/12/18 UTC. :class:`MultiRegionSnapshotDownscalingDataset` serves them;
:class:`MultiLeadDataset` concatenates one such dataset per forecast lead for
the lead-conditioned (cross-lead) model.

The classes orchestrate init, indexing and ``__getitem__`` and delegate the
substantive work (station filtering, ERA5 normalisation, context-grid assembly,
target selection, collation) to :mod:`tessera_downscaling.data.helpers`.

Each episode ``dict`` returned by ``__getitem__`` contains::

    context_grid            (C, H, W)  float32
    target_coords           (N, 2)     [lat, lon] per station
    target_elev             (N,)       station elevation (m)
    target_delta_elev       (N,)       station elevation - ERA5 orography (m)
    [target_mtpi]           (N,)       when stations.csv carries ``mtpi``
    target_values           (N,) or (N, n_vars)
    target_station_indices  (N,)       flat indices into ``dataset.station_ids``
    grid_lats               (H,)
    grid_lons               (W,)
    n_targets               int
    date                    "YYYY-MM-DD-HH"
    [target_tessera]        (N, d)     z-scored precomputed per-station vector
                                       when ``vae_latents_path`` is set

``n_targets == 0`` means no station has a valid observation at that timestamp;
:func:`downscaling_collate` drops such episodes before they reach the model.
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
    DEFAULT_MIN_TESSERA_PATCH_COVERAGE,
    assemble_episode_result,
    build_context_grid,
    downscaling_collate,
    empty_episode_result,
    episodes_for_split,
    filter_stations_by_tessera_patches,
    filter_stations_by_vae_latents,
    filter_valid_indices_by_probe_active_from,
    load_or_compute_era5_norm_stats,
    resolve_drop_channel_indices,
    select_valid_targets,
    validate_target_variables,
)


def _load_optional_station_mtpi(
    stations: pd.DataFrame,
    spatial_indices: np.ndarray,
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
# Multi-region snapshot dataset
# ---------------------------------------------------------------------------


@dataclass
class SnapshotRegionState:
    """Everything needed to serve one region's episodes.

    Station arrays are already filtered (spatial split, TESSERA validity,
    precomputed-vector availability) and aligned with each other; local
    station index ``i`` corresponds to flat index ``i + flat_offset`` in the
    parent dataset's concatenated station arrays.
    """

    name: str
    region_dir: Path
    grid_lats: np.ndarray
    grid_lons: np.ndarray
    static_fields: np.ndarray | None
    n_static: int
    era5_mean: np.ndarray
    era5_std: np.ndarray
    station_ids: np.ndarray
    station_lats: np.ndarray
    station_lons: np.ndarray
    station_elevs: np.ndarray
    station_delta_elevs: np.ndarray
    station_mtpi: np.ndarray | None
    vae_latents: np.ndarray | None
    ghcnh_index_for_station: np.ndarray
    # Episode identifiers as "YYYY-MM-DD-HH" strings — the same list for every
    # region under a given split, since the temporal split is global.
    timestamps: list[str]
    flat_offset: int


class MultiRegionSnapshotDownscalingDataset(Dataset):
    """Multi-region 6-hourly downscaling dataset with per-region spatial splits.

    Episodes are ``(region, timestamp)`` pairs drawn from a
    ``multi_region_snapshot_v1`` dataset (see the module docstring for the
    on-disk layout). Regions are listed in ``region_order`` and episodes are
    concatenated region by region.

    Two ways to say which stations each region contributes:

      * ``region_specs={"europe": "train", "us": "all"}`` — per-region
        spatial split (preferred; used by the multi-region experiments).
      * ``regions=[...], station_split="train"`` — every listed region uses
        the same spatial split (``regions=None`` means all regions).

    A spatial split is ``"train"`` or ``"test"`` (the 85/15 per-region station
    holdout in ``stations.csv``) or ``"all"`` (every station in the region —
    used to evaluate on a region that was never seen in training).

    Station filtering, per region, in this order:

      1. region membership and spatial split;
      2. TESSERA patch validity (``tessera_path``): centre pixel non-zero
         and at least ``min_patch_coverage`` of the 64x64 patch non-zero.
         Applied to baseline and TESSERA runs alike so both see the same
         stations;
      3. availability of a non-NaN precomputed per-station vector
         (``vae_latents_path``).

    Normalisation is per region: ERA5 dynamic (+ static) + lat/lon channels
    are z-scored with the region's own train-split statistics
    (``regions/<name>/normalisation_stats*.npz``). Precomputed vectors are
    z-scored with global statistics cached next to the ``.npy``.

    Args:
        dataset_dir: Root of the ``multi_region_snapshot_v1`` dataset.
        regions: Region names (legacy form, with ``station_split``).
        split: Temporal split — ``"train"``, ``"val"`` or ``"test"``. Boundaries
            come from ``metadata.json["temporal_split"]``.
        station_split: Spatial split for every region in ``regions``.
        target_variables: ``["t2m"]`` (default), ``["wind"]`` or both. A
            station is a target at a timestamp only if all requested
            variables are observed.
        tessera_path: ``(N, 64, 64, 128)`` TESSERA patch ``.npy`` (memory-mapped;
            used for the station-validity filter only).
        tessera_station_csv: CSV row-aligned with ``tessera_path``.
        include_static_fields: Append the 13 ERA5 static fields to the
            context grid (and use the with-static normalisation stats).
        vae_latents_path: ``(N, d)`` precomputed per-station vectors ``.npy``
            (VAE latents, shuffled latents, summary statistics or extra
            descriptors). Served z-scored as ``target_tessera``.
        vae_latents_station_csv: CSV row-aligned with ``vae_latents_path``.
        vae_latents_zscore: Z-score the vectors per dimension with global
            statistics over all non-NaN rows.
        region_specs: Per-region spatial split (preferred form).
        normalisation_policy: Only ``"per_region"`` is supported; kept as a
            kwarg because training configs record it.
        min_patch_coverage: Patch coverage threshold for the TESSERA filter.
        probe_active_from: ``{station_id: "YYYY-MM-DD-HH"}`` — hide the
            listed stations at timestamps earlier than their value
            (temporal data-efficiency probes). Other stations are unaffected.
        train_end_override: Move the train/val boundary earlier than
            ``metadata.json``'s ``train_end`` (Norway rollout experiment).
            Normalisation statistics still come from the full training
            window so every sweep point shares the same cached stats.
        drop_context_channels: ERA5 dynamic channel names to remove from the
            context grid (e.g. ``["total_precipitation_sum"]`` to align the
            20-channel ERA5 dataset with the 19-channel Aurora datasets).
        drop_context_strict: Raise if a name to drop is absent (training);
            ``False`` skips absent names (evaluation on a reduced dataset).
        lead_hours: Forecast lead of the coarse fields. When set, a constant
            ``lead_hours / MAX_LEAD_HOURS`` channel is appended to the grid
            (cross-lead model).

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
        include_static_fields: bool = True,
        vae_latents_path: str | Path | None = None,
        vae_latents_station_csv: str | Path | None = None,
        vae_latents_zscore: bool = True,
        region_specs: dict[str, str] | None = None,
        normalisation_policy: str = "per_region",
        min_patch_coverage: float = DEFAULT_MIN_TESSERA_PATCH_COVERAGE,
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
        self.train_end_override = train_end_override
        if normalisation_policy != "per_region":
            raise ValueError(
                "Only normalisation_policy='per_region' is supported, "
                f"got {normalisation_policy!r}"
            )
        self.normalisation_policy = normalisation_policy

        self.target_variables = validate_target_variables(target_variables)
        self.n_target_variables = len(self.target_variables)
        self.include_static_fields = include_static_fields
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

        # --- Resolve region_specs ---
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
            self.region_specs = dict.fromkeys(resolved_regions, station_split)

        self.region_order: list[str] = list(self.region_specs.keys())
        split_values = set(self.region_specs.values())
        self.station_split = (
            next(iter(split_values)) if len(split_values) == 1 else "mixed"
        )

        self.n_dynamic_channels = self.top_metadata["n_dynamic_channels"]
        temporal = self.top_metadata["temporal_split"]
        valid_timestamps = self.top_metadata["valid_timestamps"]

        global_stations = pd.read_csv(self.dataset_dir / "stations.csv")
        global_valid_indices = np.load(self.dataset_dir / "valid_station_indices.npy")
        # GHCNh is shared at the top level — every region's episodes read
        # from the same directory, keyed by station index.
        self._ghcnh_snapshot_dir = self.dataset_dir / "ghcnh_snapshot"

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
                vae_latents_path=Path(vae_latents_path) if vae_latents_path else None,
                vae_latents_station_csv=(
                    Path(vae_latents_station_csv) if vae_latents_station_csv else None
                ),
                vae_latents_zscore=vae_latents_zscore,
                min_patch_coverage=min_patch_coverage,
                flat_offset=flat_offset,
                train_end_override=self.train_end_override,
            )
            self.per_region[name] = state
            flat_offset += len(state.station_ids)

        # Flat station arrays across all regions, in region_order. Per-region
        # local station index i becomes flat index (i + flat_offset) in the
        # result dicts.
        self.station_ids = self._concat_regions(
            "station_ids", np.array([], dtype=object)
        )
        self.station_lats = self._concat_regions("station_lats")
        self.station_lons = self._concat_regions("station_lons")
        self.station_elevs = self._concat_regions("station_elevs")
        self.station_delta_elevs = self._concat_regions("station_delta_elevs")
        # mTPI is optional (present only when stations.csv carried an `mtpi`
        # column). All regions share one global stations table, so it is
        # either available for every region or for none; expose None otherwise
        # so the model falls back to the 2-feature (elevation, delta_elevation)
        # layout.
        if self.region_order and all(
            self.per_region[n].station_mtpi is not None for n in self.region_order
        ):
            self.station_mtpi = self._concat_regions("station_mtpi")
        else:
            self.station_mtpi = None

        # Episode dispatch: cumulative lengths across regions, so
        # idx < cum_lengths[k] means episode k comes from region_order[k].
        self._cum_lengths: list[int] = []
        running = 0
        for name in self.region_order:
            running += len(self.per_region[name].timestamps)
            self._cum_lengths.append(running)

        # Resolve context channels to drop into dynamic-block indices.
        # Strict at train, lenient at eval.
        _chan_names = self.top_metadata.get("era5_dynamic_channels")
        if self._drop_context_channels and not _chan_names:
            raise ValueError(
                "drop_context_channels requested but top metadata has no "
                "'era5_dynamic_channels' to resolve names against."
            )
        self.drop_dynamic_indices = resolve_drop_channel_indices(
            self._drop_context_channels,
            _chan_names or [],
            strict=self._drop_context_strict,
        )

        # Context grid: dynamic + static + 2 lat/lon channels + 4 time
        # channels (cos/sin DoY, cos/sin HoD) [+ 1 lead channel].
        _n_dyn = self.n_dynamic_channels - len(self.drop_dynamic_indices)
        _lead = 1 if self.lead_hours is not None else 0
        ch_counts = {
            name: (st.n_static + _n_dyn + 2 + 4 + _lead)
            for name, st in self.per_region.items()
        }
        if len(set(ch_counts.values())) != 1:
            raise ValueError(f"Regions disagree on n_context_channels: {ch_counts}.")
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

        # Probe-station temporal mask, applied per episode in __getitem__; a
        # no-op for any station not in the dict. None means "no mask
        # configured", {} "configured but empty" — the distinction is kept for
        # config echo / debugging.
        self.probe_active_from: dict[str, str] | None = None
        if probe_active_from is not None:
            self.probe_active_from = {
                str(sid): str(ts) for sid, ts in probe_active_from.items()
            }
            n_in_dataset = sum(
                1 for sid in self.station_ids if str(sid) in self.probe_active_from
            )
            print(
                f"MultiRegionSnapshotDownscalingDataset({split}): "
                f"probe_active_from configured with "
                f"{len(self.probe_active_from)} entries; "
                f"{n_in_dataset} match a station in this dataset"
            )

    def _concat_regions(
        self,
        field: str,
        empty: np.ndarray | None = None,
    ) -> np.ndarray:
        """Concatenate one per-region station array over ``region_order``."""
        if not self.region_order:
            return empty if empty is not None else np.array([], dtype=np.float32)
        return np.concatenate(
            [getattr(self.per_region[n], field) for n in self.region_order]
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
        vae_latents_path: Path | None,
        vae_latents_station_csv: Path | None,
        vae_latents_zscore: bool,
        min_patch_coverage: float,
        flat_offset: int,
        train_end_override: str | None = None,
    ) -> SnapshotRegionState:
        """Build one region's state: grid, filtered stations, stats, episodes."""
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
                min_patch_coverage=min_patch_coverage,
            )
            n_before = len(spatial_indices)
            spatial_indices = spatial_indices[t_result.kept_mask]
            ghcnh_index_for_station = ghcnh_index_for_station[t_result.kept_mask]
            print(
                f"[{name}] {len(spatial_indices)}/{n_before} stations "
                f"after TESSERA filtering"
            )

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
            vae_latents = v_result.latents
            print(
                f"[{name}] {len(spatial_indices)}/{n_before} stations "
                f"after VAE latent filtering"
            )

        # Per-region ERA5 normalisation stats, read from / cached at
        # region_dir. Preprocessing writes these proactively, so in practice
        # the cache exists and we just load. Always computed over the
        # metadata train window, not train_end_override (see class docstring).
        stats_name = (
            "normalisation_stats.npz"
            if self.include_static_fields
            else "normalisation_stats_no_static.npz"
        )
        train_ts_list = [t for t in valid_timestamps if t <= train_end]
        era5_mean, era5_std = load_or_compute_era5_norm_stats(
            cache_path=region_dir / stats_name,
            train_episode_ids=train_ts_list,
            era5_dir=region_dir / "era5_snapshot",
            grid_lats=grid_lats,
            grid_lons=grid_lons,
            static_fields=static_fields,
        )

        station_ids = global_stations["station_id"].values[spatial_indices]
        station_lats = (
            global_stations["latitude"].values[spatial_indices].astype(np.float32)
        )
        station_lons = (
            global_stations["longitude"].values[spatial_indices].astype(np.float32)
        )
        station_elevs = (
            global_stations["elevation"].values[spatial_indices].astype(np.float32)
        )
        station_delta_elevs = (
            global_stations["delta_elevation"]
            .values[spatial_indices]
            .astype(np.float32)
        )
        station_mtpi = _load_optional_station_mtpi(global_stations, spatial_indices)

        # Temporal split — identical for every region because the split is
        # global in the preprocessor. train_end_override shifts the train/val
        # boundary earlier while leaving val_end unchanged.
        effective_train_end = (
            train_end_override if train_end_override is not None else train_end
        )
        timestamps = episodes_for_split(
            valid_timestamps,
            split=split,
            train_end=effective_train_end,
            val_end=val_end,
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
            vae_latents=vae_latents,
            ghcnh_index_for_station=ghcnh_index_for_station,
            timestamps=timestamps,
            flat_offset=flat_offset,
        )

    @property
    def timestamps(self) -> list[str]:
        """Flat list of ``"YYYY-MM-DD-HH"`` identifiers aligned with episode indices."""
        out: list[str] = []
        for name in self.region_order:
            out.extend(self.per_region[name].timestamps)
        return out

    @property
    def dates(self) -> list[str]:
        """Alias of :attr:`timestamps` (evaluate.py reads it for seasonal analysis)."""
        return self.timestamps

    def __len__(self) -> int:
        return self._cum_lengths[-1] if self._cum_lengths else 0

    def _dispatch(self, idx: int) -> tuple[str, int]:
        """Map a flat episode index to ``(region_name, local_index)``."""
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Episode index {idx} out of range [0, {len(self)})")
        prev_cum = 0
        for name, cum in zip(self.region_order, self._cum_lengths, strict=True):
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
            era5_path=st.region_dir / "era5_snapshot" / f"{ts}.npy",
            static_fields=st.static_fields,
            grid_lats=st.grid_lats,
            grid_lons=st.grid_lons,
            date_str=date_str,
            hour=hour,
            era5_mean=st.era5_mean,
            era5_std=st.era5_std,
            drop_dynamic_indices=self.drop_dynamic_indices,
            lead_hours=self.lead_hours,
        )

        valid_indices, per_var_values = select_valid_targets(
            ghcnh_path=self._ghcnh_snapshot_dir / f"{ts}.npz",
            ghcnh_index_for_station=st.ghcnh_index_for_station,
            target_variables=self.target_variables,
        )

        # Probe-station temporal mask (no-op if probe_active_from is None or
        # empty): rows are dropped where the station_id is in
        # probe_active_from AND the timestamp is earlier than that station's
        # active_from. st.station_ids is aligned with the local valid_indices.
        if self.probe_active_from:
            valid_indices = filter_valid_indices_by_probe_active_from(
                valid_indices=valid_indices,
                station_ids=st.station_ids,
                timestamp=ts,
                probe_active_from=self.probe_active_from,
            )

        if len(valid_indices) == 0:
            return empty_episode_result(
                context_grid=context_grid,
                grid_lats=st.grid_lats,
                grid_lons=st.grid_lons,
                n_target_variables=self.n_target_variables,
                date_str=ts,
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

        # Translate per-region local station indices to flat global indices,
        # so callers can address stations via a single integer index into
        # self.station_ids etc.
        result["target_station_indices"] = (
            result["target_station_indices"] + st.flat_offset
        )

        if st.vae_latents is not None:
            result["target_tessera"] = torch.from_numpy(st.vae_latents[valid_indices])

        return result


# ---------------------------------------------------------------------------
# Lead-conditioned (cross-lead) dataset
# ---------------------------------------------------------------------------


class MultiLeadDataset(Dataset):
    """Concatenate per-lead snapshot datasets for the lead-conditioned model.

    Each element of ``sub_datasets`` is a fully-built
    :class:`MultiRegionSnapshotDownscalingDataset` for one forecast lead — same
    split, regions, stations, targets and TESSERA config — differing only in
    its ``dataset_dir`` (which lead's coarse context it reads) and its
    ``lead_hours`` (so its context grids carry the lead channel). Because the
    downscaling *target* is the station observation at the valid time, it is
    identical across leads; the leads are just alternative inputs for the same
    supervision. Concatenating them means one epoch sees every episode at every
    lead (uniform mixing), which is what lets the heads learn a lead-dependent
    spread σ(lead).

    Indexing is ConcatDataset-style (cumulative offsets). ``n_context_channels``
    and ``vae_latent_dim`` must agree across leads; a mismatch is raised loudly
    since it almost always means a lead was built without the precipitation
    drop (lead-0 ERA5 is 20-channel, Aurora is 19). Any other attribute access
    forwards to the first sub-dataset, so the training loop can read dataset
    metadata transparently.
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
            per_lead = list(
                zip(self.leads, [s.n_context_channels for s in self.subs], strict=True)
            )
            raise ValueError(
                f"Leads disagree on n_context_channels: {per_lead}. "
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

    def __getattr__(self, name: str):
        # Only invoked when normal lookup fails, so the explicit attributes
        # above (n_context_channels, vae_latent_dim, ...) take priority. Forward
        # metadata-like reads to the first sub-dataset.
        subs = self.__dict__.get("subs")
        if subs:
            return getattr(subs[0], name)
        raise AttributeError(name)


__all__ = [
    "DEFAULT_MIN_TESSERA_PATCH_COVERAGE",
    "MultiLeadDataset",
    "MultiRegionSnapshotDownscalingDataset",
    "SnapshotRegionState",
    "downscaling_collate",
]
