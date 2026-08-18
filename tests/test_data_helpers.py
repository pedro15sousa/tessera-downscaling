"""Synthetic tests for :mod:`tessera_downscaling.data` (no /data needed)."""

import json

import numpy as np
import pandas as pd
import pytest
import torch

from tessera_downscaling.data.dataset import (
    MultiLeadDataset,
    MultiRegionSnapshotDownscalingDataset,
)
from tessera_downscaling.data.helpers import (
    MAX_LEAD_HOURS,
    build_context_grid,
    downscaling_collate,
    episodes_for_split,
    filter_stations_by_tessera_patches,
    filter_stations_by_vae_latents,
    filter_valid_indices_by_probe_active_from,
    resolve_drop_channel_indices,
    select_valid_targets,
    validate_target_variables,
)

# ---------------------------------------------------------------------------
# Target variables
# ---------------------------------------------------------------------------


def test_validate_target_variables_defaults_to_t2m_and_rejects_unknown():
    assert validate_target_variables(None) == ["t2m"]
    assert validate_target_variables(["wind", "t2m"]) == ["wind", "t2m"]
    with pytest.raises(ValueError, match="Unknown target variable"):
        validate_target_variables(["tmax"])


# ---------------------------------------------------------------------------
# Temporal split
# ---------------------------------------------------------------------------


def test_episodes_for_split_boundary_day_opens_the_next_split():
    # Boundary-date snapshots ("2020-12-31-HH") sort after the bare date
    # ("2020-12-31"), so they belong to the *next* split. This is the
    # convention behind every paper split (test starts 2021-12-31 00Z).
    hours = ["00", "06", "12", "18"]
    days = ["2020-12-30", "2020-12-31", "2021-01-01", "2021-12-31", "2022-01-01"]
    ts = [f"{d}-{h}" for d in days for h in hours]
    train_end, val_end = "2020-12-31", "2021-12-31"

    train = episodes_for_split(ts, "train", train_end, val_end)
    val = episodes_for_split(ts, "val", train_end, val_end)
    test = episodes_for_split(ts, "test", train_end, val_end)

    assert train == [t for t in ts if t[:10] == "2020-12-30"]
    assert val == [t for t in ts if t[:10] in ("2020-12-31", "2021-01-01")]
    assert test == [t for t in ts if t[:10] in ("2021-12-31", "2022-01-01")]
    assert len(train) + len(val) + len(test) == len(ts)
    with pytest.raises(ValueError, match="Unknown split"):
        episodes_for_split(ts, "holdout", train_end, val_end)


# ---------------------------------------------------------------------------
# Target selection
# ---------------------------------------------------------------------------


def test_select_valid_targets_masks_nan_and_sentinel(tmp_path):
    # ghcnh rows: 0..4; dataset stations map to ghcnh rows [4, 0, 2, 3].
    t2m = np.array([1.0, np.nan, 3.0, -999.0, 5.0], dtype=np.float32)
    wind = np.array([np.nan, 1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    path = tmp_path / "2021-01-01-06.npz"
    np.savez(path, t2m=t2m, wind=wind, obs_count=np.ones(5, dtype=np.int32))
    ghcnh_index_for_station = np.array([4, 0, 2, 3])

    valid, values = select_valid_targets(path, ghcnh_index_for_station, ["t2m"])
    assert valid.tolist() == [0, 1, 2]  # station 3 -> row 3 is the -999 sentinel
    assert values[0].tolist() == [5.0, 1.0, 3.0, -999.0]

    valid, values = select_valid_targets(path, ghcnh_index_for_station, ["t2m", "wind"])
    assert valid.tolist() == [0, 2]  # station 1 -> row 0 has NaN wind
    assert len(values) == 2


def test_probe_active_from_hides_probes_before_their_start():
    station_ids = np.array(["A", "B", "C"])
    valid = np.array([0, 1, 2])
    probes = {"B": "2020-06-01-00", "C": "9999-12-31"}
    early = filter_valid_indices_by_probe_active_from(
        valid, station_ids, "2020-01-01-00", probes
    )
    late = filter_valid_indices_by_probe_active_from(
        valid, station_ids, "2020-07-01-00", probes
    )
    assert early.tolist() == [0]
    assert late.tolist() == [0, 1]


# ---------------------------------------------------------------------------
# Collate
# ---------------------------------------------------------------------------


def _episode(
    n: int, *, with_mtpi: bool = True, latent_dim: int = 4, offset: int = 0
) -> dict:
    ep = {
        "context_grid": torch.zeros(3, 4, 5),
        "target_coords": torch.arange(2 * n, dtype=torch.float32).view(n, 2),
        "target_elev": torch.ones(n),
        "target_delta_elev": -torch.ones(n),
        "target_values": torch.full((n,), 7.0),
        "target_station_indices": torch.arange(n) + offset,
        "grid_lats": torch.arange(4.0),
        "grid_lons": torch.arange(5.0),
        "n_targets": n,
        "date": "2021-01-01-00",
        "target_tessera": torch.ones(n, latent_dim),
    }
    if with_mtpi:
        ep["target_mtpi"] = torch.full((n,), 2.0)
    return ep


def test_downscaling_collate_pads_and_masks():
    batch = downscaling_collate([_episode(2), _episode(0), _episode(3, offset=10)])
    assert batch["context_grid"].shape == (2, 3, 4, 5)  # zero-target episode dropped
    assert batch["target_coords"].shape == (2, 3, 2)
    assert batch["target_values"].shape == (2, 3)
    assert batch["target_tessera"].shape == (2, 3, 4)
    assert batch["target_mtpi"].shape == (2, 3)
    assert batch["target_mask"].tolist() == [[True, True, False], [True, True, True]]
    assert batch["target_station_indices"].tolist() == [[0, 1, -1], [10, 11, 12]]
    # Padding positions are zero for every per-station tensor.
    assert batch["target_values"][0, 2] == 0
    assert batch["target_tessera"][0, 2].abs().sum() == 0
    assert batch["grid_lats"].tolist() == [0.0, 1.0, 2.0, 3.0]


def test_downscaling_collate_mtpi_only_when_uniform_and_none_when_empty():
    batch = downscaling_collate([_episode(2), _episode(1, with_mtpi=False)])
    assert "target_mtpi" not in batch
    assert downscaling_collate([_episode(0), _episode(0)]) is None


# ---------------------------------------------------------------------------
# Station filters
# ---------------------------------------------------------------------------


def test_filter_stations_by_vae_latents_drops_nan_rows_and_zscores(tmp_path):
    latents = np.array(
        [[1.0, 10.0], [np.nan, 0.0], [3.0, 30.0], [5.0, 50.0]], dtype=np.float32
    )
    np.save(tmp_path / "lat.npy", latents)
    pd.DataFrame({"station_id": ["A", "B", "C", "D"]}).to_csv(
        tmp_path / "lat_stations.csv", index=False
    )
    # Dataset stations: E is absent from the latents CSV, B has a NaN row.
    stations = pd.DataFrame({"station_id": ["E", "A", "B", "C"]})
    spatial_indices = np.arange(4)

    res = filter_stations_by_vae_latents(
        stations,
        spatial_indices,
        tmp_path / "lat.npy",
        tmp_path / "lat_stations.csv",
        zscore=True,
    )
    assert res.kept_mask.tolist() == [False, True, False, True]
    assert res.latent_dim == 2
    valid = latents[[0, 2, 3]]
    expected = (latents[[0, 2]] - valid.mean(0)) / np.maximum(valid.std(0), 1e-6)
    np.testing.assert_allclose(res.latents, expected.astype(np.float32), rtol=1e-6)
    assert (tmp_path / "lat_global_stats.npz").exists()

    raw = filter_stations_by_vae_latents(
        stations,
        spatial_indices,
        tmp_path / "lat.npy",
        tmp_path / "lat_stations.csv",
        zscore=False,
    )
    np.testing.assert_array_equal(raw.latents, latents[[0, 2]])


def test_filter_stations_by_tessera_patches_coverage_threshold(tmp_path):
    h = w = 8
    patches = np.zeros((4, h, w, 3), dtype=np.float32)
    patches[0] = 1.0  # fully valid
    patches[1] = 1.0
    patches[1, h // 2, w // 2] = 0.0  # centre hole -> always rejected
    patches[2, :4, :4] = 1.0
    patches[2, h // 2, w // 2] = 1.0  # centre set but coverage 17/64 < 0.5
    patches[3, :, :5] = 1.0  # centre set, coverage 40/64 = 0.625
    np.save(tmp_path / "patches.npy", patches)
    pd.DataFrame({"station_id": ["P0", "P1", "P2", "P3"]}).to_csv(
        tmp_path / "patch_stations.csv", index=False
    )
    stations = pd.DataFrame({"station_id": ["P3", "P2", "P1", "P0", "PX"]})
    spatial_indices = np.arange(5)

    res = filter_stations_by_tessera_patches(
        stations,
        spatial_indices,
        tmp_path / "patches.npy",
        tmp_path / "patch_stations.csv",
        min_patch_coverage=0.5,
    )
    assert res.kept_mask.tolist() == [True, False, False, True, False]
    assert res.tessera_row_indices.tolist() == [3, 0]

    centre_only = filter_stations_by_tessera_patches(
        stations,
        spatial_indices,
        tmp_path / "patches.npy",
        tmp_path / "patch_stations.csv",
        min_patch_coverage=0.0,
    )
    assert centre_only.kept_mask.tolist() == [True, True, False, True, False]


# ---------------------------------------------------------------------------
# Context grid
# ---------------------------------------------------------------------------


def test_resolve_drop_channel_indices_strict_vs_lenient():
    names = ["a", "b", "c"]
    assert resolve_drop_channel_indices(None, names, strict=True) == []
    assert resolve_drop_channel_indices(["c", "a"], names, strict=True) == [0, 2]
    assert resolve_drop_channel_indices(["zzz"], names, strict=False) == []
    with pytest.raises(ValueError, match="not found"):
        resolve_drop_channel_indices(["zzz"], names, strict=True)


def test_build_context_grid_channel_layout(tmp_path):
    H, W = 3, 4
    era5 = np.arange(3 * H * W, dtype=np.float32).reshape(3, H, W)
    static = np.stack([np.full((H, W), 100.0), np.full((H, W), 200.0)]).astype(
        np.float32
    )
    lats = np.array([50.0, 49.0, 48.0], dtype=np.float32)
    lons = np.array([-2.0, -1.0, 0.0, 1.0], dtype=np.float32)
    era5_path = tmp_path / "2021-03-01-06.npy"
    np.save(era5_path, era5)

    grid = build_context_grid(
        era5_path,
        static,
        lats,
        lons,
        "2021-03-01",
        6,
        None,
        None,
    )
    assert grid.shape == (3 + 2 + 2 + 4, H, W)
    np.testing.assert_array_equal(grid[0:3], era5)
    np.testing.assert_array_equal(grid[3:5], static)
    np.testing.assert_array_equal(grid[5], np.repeat(lats[:, None], W, axis=1))
    np.testing.assert_array_equal(grid[6], np.repeat(lons[None, :], H, axis=0))
    doy = 60  # 1 March 2021
    np.testing.assert_allclose(grid[7], np.cos(2 * np.pi * doy / 365), rtol=1e-6)
    np.testing.assert_allclose(grid[8], np.sin(2 * np.pi * doy / 365), rtol=1e-6)
    np.testing.assert_allclose(grid[9], np.cos(2 * np.pi * 6 / 24), atol=1e-7)
    np.testing.assert_allclose(grid[10], np.sin(2 * np.pi * 6 / 24), atol=1e-7)

    # Normalisation covers dynamic + static + lat/lon only; lead channel is
    # appended last, unnormalised.
    mean = np.arange(7, dtype=np.float32)
    std = np.full(7, 2.0, dtype=np.float32)
    grid_n = build_context_grid(
        era5_path,
        static,
        lats,
        lons,
        "2021-03-01",
        6,
        mean,
        std,
        lead_hours=24,
    )
    assert grid_n.shape == (12, H, W)
    np.testing.assert_allclose(grid_n[:7], (grid[:7] - mean[:, None, None]) / 2.0)
    np.testing.assert_array_equal(grid_n[7:11], grid[7:11])
    np.testing.assert_allclose(grid_n[11], 24 / MAX_LEAD_HOURS)

    # Dropping a dynamic channel removes it from the grid AND from the stats.
    grid_d = build_context_grid(
        era5_path,
        static,
        lats,
        lons,
        "2021-03-01",
        6,
        mean,
        std,
        drop_dynamic_indices=[1],
    )
    assert grid_d.shape == (10, H, W)
    np.testing.assert_allclose(grid_d[1], (era5[2] - mean[2]) / 2.0)
    np.testing.assert_allclose(grid_d[2:4], (static - mean[3:5, None, None]) / 2.0)


# ---------------------------------------------------------------------------
# Dataset on a tiny synthetic multi_region_snapshot_v1 tree
# ---------------------------------------------------------------------------


def _write_synthetic_dataset(root, *, timestamps, n_dyn=3):
    """Two regions (eu: 3 stations, us: 2), 13 static channels, tiny grids."""
    stations = pd.DataFrame(
        {
            "station_id": ["E1", "E2", "E3", "U1", "U2"],
            "latitude": [50.0, 51.0, 52.0, 40.0, 41.0],
            "longitude": [0.0, 1.0, 2.0, -100.0, -99.0],
            "elevation": [10.0, 20.0, 30.0, 40.0, 50.0],
            "delta_elevation": [1.0, 2.0, 3.0, 4.0, 5.0],
            "region": ["eu", "eu", "eu", "us", "us"],
            "spatial_split": ["train", "train", "test", "train", "test"],
            "mtpi": [0.1, 0.2, 0.3, 0.4, 0.5],
        }
    )
    stations.to_csv(root / "stations.csv", index=False)
    np.save(root / "valid_station_indices.npy", np.arange(5))
    grids = {"eu": (3, 4), "us": (2, 3)}
    md = {
        "layout_version": "multi_region_snapshot_v1",
        "cadence": "6h",
        "era5_dynamic_channels": ["t2m", "u10", "precip"][:n_dyn],
        "n_dynamic_channels": n_dyn,
        "regions": {
            n: {"grid_shape": list(g), "n_static_channels": 13}
            for n, g in grids.items()
        },
        "valid_timestamps": timestamps,
        "temporal_split": {"train_end": "2020-12-31", "val_end": "2021-12-31"},
    }
    (root / "metadata.json").write_text(json.dumps(md))
    (root / "ghcnh_snapshot").mkdir()
    rng = np.random.default_rng(0)
    for ts in timestamps:
        t2m = rng.normal(size=5).astype(np.float32)
        wind = rng.uniform(0, 5, size=5).astype(np.float32)
        t2m[1] = np.nan  # E2 never observes t2m
        np.savez(
            root / "ghcnh_snapshot" / f"{ts}.npz",
            t2m=t2m,
            wind=wind,
            obs_count=np.ones(5, dtype=np.int32),
        )
    for name, (H, W) in grids.items():
        rd = root / "regions" / name
        (rd / "era5_snapshot").mkdir(parents=True)
        np.save(rd / "lats.npy", np.linspace(60, 40, H).astype(np.float32))
        np.save(rd / "lons.npy", np.linspace(-10, 10, W).astype(np.float32))
        np.save(
            rd / "static_fields.npy", rng.normal(size=(13, H, W)).astype(np.float32)
        )
        (rd / "region_metadata.json").write_text(json.dumps({"n_static_channels": 13}))
        for ts in timestamps:
            np.save(
                rd / "era5_snapshot" / f"{ts}.npy",
                rng.normal(size=(n_dyn, H, W)).astype(np.float32),
            )
    return stations


def test_multi_region_snapshot_dataset_end_to_end(tmp_path):
    ts = ["2020-01-01-00", "2020-01-01-06", "2021-06-01-12", "2022-03-01-18"]
    root = tmp_path / "ds"
    root.mkdir()
    _write_synthetic_dataset(root, timestamps=ts)
    # Precomputed vectors: U1 has a NaN row, so it is dropped when latents are on.
    lat = np.arange(10, dtype=np.float32).reshape(5, 2)
    lat[3] = np.nan
    np.save(tmp_path / "lat.npy", lat)
    pd.DataFrame({"station_id": ["E1", "E2", "E3", "U1", "U2"]}).to_csv(
        tmp_path / "lat_stations.csv", index=False
    )

    ds = MultiRegionSnapshotDownscalingDataset(
        root,
        region_specs={"eu": "train", "us": "all"},
        split="train",
        target_variables=["t2m"],
        include_static_fields=True,
        vae_latents_path=tmp_path / "lat.npy",
        vae_latents_station_csv=tmp_path / "lat_stations.csv",
    )
    # eu/train -> E1, E2; us/all -> U1, U2 minus NaN-latent U1.
    assert ds.station_ids.tolist() == ["E1", "E2", "U2"]
    assert ds.per_region["us"].flat_offset == 2
    assert ds.n_context_channels == 3 + 13 + 2 + 4
    assert ds.vae_latent_dim == 2
    assert len(ds) == 4  # 2 train timestamps x 2 regions
    assert ds.timestamps == ts[:2] + ts[:2]
    assert (root / "regions" / "eu" / "normalisation_stats.npz").exists()

    ep = ds[0]  # eu, first timestamp: E1 valid, E2 has NaN t2m
    assert ep["date"] == ts[0]
    assert ep["context_grid"].shape == (22, 3, 4)
    assert ep["target_station_indices"].tolist() == [0]
    assert ep["target_tessera"].shape == (1, 2)
    assert ep["target_mtpi"].tolist() == pytest.approx([0.1])
    ep_us = ds[2]
    assert ep_us["target_station_indices"].tolist() == [2]  # U2 at flat index 2
    assert ep_us["target_elev"].tolist() == [50.0]

    # Legacy form + drop channel + lead channel; wind target keeps E2.
    val = MultiRegionSnapshotDownscalingDataset(
        root,
        regions=["eu"],
        station_split="test",
        split="val",
        target_variables=["wind"],
        include_static_fields=False,
        drop_context_channels=["precip"],
        lead_hours=6,
    )
    assert val.station_ids.tolist() == ["E3"]
    assert val.n_context_channels == 2 + 2 + 4 + 1
    assert val.timestamps == ["2021-06-01-12"]
    ep = val[0]
    assert ep["context_grid"].shape == (9, 3, 4)
    assert float(ep["context_grid"][-1, 0, 0]) == pytest.approx(6 / MAX_LEAD_HOURS)
    assert "target_tessera" not in ep

    multi = MultiLeadDataset([val, val])
    assert len(multi) == 2
    assert multi[1]["date"] == "2021-06-01-12"
    assert multi.station_ids.tolist() == ["E3"]

    with pytest.raises(ValueError, match="per_region"):
        MultiRegionSnapshotDownscalingDataset(root, normalisation_policy="global")
