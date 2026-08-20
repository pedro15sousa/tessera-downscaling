"""Tests for the patch encoder (the VAE behind the station descriptors).

Two layers:

* Synthetic checks that run anywhere -- the model built from the paper's
  configuration has the expected shape and outputs, the shipped foundation-model
  configs build the geometry their data needs, the loss is finite and its KL
  ramp behaves, and the dataset's station filter, centre crop and normalisation
  do what the descriptors depend on.
* Checks against the real runs on the data root (skipped when it is absent):
  ``best.pt`` still loads into the ported model with ``strict=True``, and
  re-encoding the first stations reproduces the latents that run published --
  the proof that the migration did not move the descriptor space. This covers
  the paper's TESSERA run and both arms of the foundation-model benchmark.
"""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

from tessera_downscaling.patch_encoder.dataset import (
    TesseraPatchDataset,
    create_dataloaders,
    filter_elevation_sentinels,
    prepare_data,
)
from tessera_downscaling.patch_encoder.losses import VAELoss, linear_beta
from tessera_downscaling.patch_encoder.model import build_model
from tessera_downscaling.paths import data_root

# The configuration of the run whose latents the paper uses, as stored in
# <data root>/tessera_patch_encoder/outputs/vae/
#     p128_2017_crop64_lat16_grad0.5_auxon/config.yaml
REFERENCE_CONFIG = {
    "data": {"crop_size": 64},
    "model": {
        "in_channels": 128,
        "input_size": 64,
        "latent_dim": 16,
        "encoder_channels": [128, 256, 256, 512],
        "decoder_channels": [512, 256, 256, 128],
        "dropout": 0.1,
    },
    "auxiliary": {
        "enable": True,
        "targets": ["elevation", "latitude", "longitude"],
        "hidden_dim": 64,
        "weights": {"elevation": 1.0, "latitude": 0.5, "longitude": 0.5},
    },
    "loss": {
        "reconstruction": "mse",
        "gradient_weight": 0.5,
        "beta_end": 0.0005,
        "beta_warmup_steps": 5000,
    },
}

# The run that produced processed/vae_tessera_1B-M/
#     station_latents_1B-M_p128_2017_crop64_lat16_grad0.5_auxon.npy.
RUN_DIR = (
    data_root()
    / "tessera_patch_encoder"
    / "outputs"
    / "vae"
    / "p128_2017_crop64_lat16_grad0.5_auxon"
)
CACHE_FILE = (
    data_root()
    / "tessera_patch_encoder"
    / "outputs"
    / "dataset_cache"
    / "patch_embeddings_2017_p128"
    / "cache.npz"
)
PATCHES = (
    data_root()
    / "processed"
    / "tessera_station_patches"
    / "patch_embeddings_2017_p128.npy"
)
STATIONS = (
    data_root() / "processed" / "tessera_station_patches" / "station_list_filtered.csv"
)

# The run configs shipped with the repository.
CONFIG_DIR = Path(__file__).resolve().parents[1] / "scripts" / "patch_encoder"

# The foundation-model benchmark: the same recipe as the paper's run, trained on
# AlphaEarth (64-channel 10 m rasters, centre-cropped to 64 px) and OlmoEarth
# (768-channel 16x16 token grids, used whole) patches of the same stations.
FM_RUNS = {
    "alphaearth": {
        "config": "vae_alphaearth.yaml",
        "run_dir": data_root()
        / "tessera_patch_encoder"
        / "outputs"
        / "vae"
        / "alphaearth"
        / "2017"
        / "crop64_lat16_auxon",
        "patches": data_root()
        / "processed"
        / "alphaearth_station_patches"
        / "patch_embeddings_alphaearth_2017_p128.npy",
        "cache": data_root()
        / "tessera_patch_encoder"
        / "outputs"
        / "dataset_cache"
        / "patch_embeddings_alphaearth_2017_p128"
        / "cache.npz",
        "patch_shape": (64, 64, 64),  # (C, S, S) the encoder is fed
    },
    "olmoearth": {
        "config": "vae_olmoearth.yaml",
        "run_dir": data_root()
        / "tessera_patch_encoder"
        / "outputs"
        / "vae"
        / "olmoearth"
        / "2017"
        / "crop64_lat16_auxon",
        "patches": data_root()
        / "processed"
        / "olmoearth_station_patches"
        / "patch_embeddings_olmoearth_2017_g16.npy",
        "cache": data_root()
        / "tessera_patch_encoder"
        / "outputs"
        / "dataset_cache"
        / "patch_embeddings_olmoearth_2017_g16"
        / "cache.npz",
        "patch_shape": (768, 16, 16),
    },
}


def shipped_config(name: str, in_channels: int, input_size: int) -> dict:
    """Load a shipped run config and fill in what ``train_vae.py`` fills in.

    ``model.in_channels`` and ``model.input_size`` are not written by hand: the
    training script reads them off the patch file the run actually sees. Doing
    the same here keeps the test on the real config rather than a copy of it.
    """
    cfg = yaml.safe_load((CONFIG_DIR / name).read_text())
    cfg["model"]["in_channels"] = in_channels
    cfg["model"]["input_size"] = input_size
    return cfg


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


def test_reference_config_builds_the_documented_architecture():
    model = build_model(REFERENCE_CONFIG)

    # Four stride-2 stages 128 -> 256 -> 256 -> 512, each Conv -> BN -> ReLU.
    channels = [block.conv[0].out_channels for block in model.encoder.blocks]
    assert channels == [128, 256, 256, 512]
    assert all(block.conv[0].stride == (2, 2) for block in model.encoder.blocks)
    assert all(block.drop is not None for block in model.encoder.blocks)

    # 64 px through four halvings is 4 px, so the bottleneck flattens 512x4x4.
    assert model.encoder.final_spatial == 4
    assert tuple(model.encoder.fc_mu.weight.shape) == (16, 512 * 4 * 4)
    assert tuple(model.encoder.fc_logvar.weight.shape) == (16, 512 * 4 * 4)
    assert tuple(model.decoder.fc.weight.shape) == (512 * 4 * 4, 16)

    assert sorted(model.aux_heads) == ["elevation", "latitude", "longitude"]


def test_forward_returns_reconstruction_latent_and_auxiliary_predictions():
    model = build_model(REFERENCE_CONFIG).eval()
    x = torch.randn(2, 128, 64, 64)

    out = model(x)

    assert set(out) == {
        "x_recon",
        "mu",
        "logvar",
        "z",
        "aux_elevation",
        "aux_latitude",
        "aux_longitude",
    }
    assert out["x_recon"].shape == x.shape
    assert out["mu"].shape == out["logvar"].shape == out["z"].shape == (2, 16)
    for name in ("aux_elevation", "aux_latitude", "aux_longitude"):
        assert out[name].shape == (2,)
    assert torch.isfinite(out["x_recon"]).all()

    # In eval mode the posterior is not sampled: z is the mean, which is what
    # encode() returns and what the station descriptors are.
    torch.testing.assert_close(out["z"], out["mu"])
    torch.testing.assert_close(model.encode(x), out["mu"])


def test_encoder_rejects_patch_sizes_it_cannot_halve_four_times():
    cfg = copy.deepcopy(REFERENCE_CONFIG)
    cfg["model"]["input_size"] = 100
    with pytest.raises(ValueError, match="divisible"):
        build_model(cfg)


@pytest.mark.parametrize(("key", "value"), [("use_se", True), ("bottleneck", "gap")])
def test_build_model_rejects_the_ablation_branches_that_were_dropped(key, value):
    cfg = copy.deepcopy(REFERENCE_CONFIG)
    cfg["model"][key] = value
    with pytest.raises(ValueError, match=key):
        build_model(cfg)


def test_alphaearth_config_builds_a_64_channel_encoder():
    """AlphaEarth differs from TESSERA only in the channel count."""
    model = build_model(shipped_config("vae_alphaearth.yaml", 64, 64))

    assert model.encoder.blocks[0].conv[0].in_channels == 64
    assert [b.conv[0].out_channels for b in model.encoder.blocks] == [
        128,
        256,
        256,
        512,
    ]
    # Same four halvings and the same bottleneck as the paper's run.
    assert model.encoder.final_spatial == 4
    assert tuple(model.encoder.fc_mu.weight.shape) == (16, 512 * 4 * 4)

    out = model.eval()(torch.randn(2, 64, 64, 64))
    assert out["x_recon"].shape == (2, 64, 64, 64)
    assert out["mu"].shape == (2, 16)


def test_olmoearth_config_builds_a_three_stage_encoder_for_the_token_grid():
    """A 16x16 grid cannot be halved four times, so its config uses three stages."""
    model = build_model(shipped_config("vae_olmoearth.yaml", 768, 16))

    assert model.encoder.blocks[0].conv[0].in_channels == 768
    assert [b.conv[0].out_channels for b in model.encoder.blocks] == [256, 256, 512]
    # 16 px through three halvings is 2 px, so the bottleneck flattens 512x2x2.
    assert model.encoder.final_spatial == 2
    assert tuple(model.encoder.fc_mu.weight.shape) == (16, 512 * 2 * 2)
    assert tuple(model.decoder.fc.weight.shape) == (512 * 2 * 2, 16)

    out = model.eval()(torch.randn(2, 768, 16, 16))
    assert out["x_recon"].shape == (2, 768, 16, 16)
    assert out["mu"].shape == (2, 16)

    # Why three and not the default four: a fourth halving would leave a 1x1
    # feature map, so the flatten bottleneck would carry no spatial layout --
    # exactly what it exists to preserve.
    cfg = shipped_config("vae_olmoearth.yaml", 768, 16)
    cfg["model"]["encoder_channels"] = [128, 256, 256, 512]
    cfg["model"]["decoder_channels"] = [512, 256, 256, 128]
    assert build_model(cfg).encoder.final_spatial == 1


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------


def test_loss_is_finite_and_reports_every_component():
    torch.manual_seed(0)
    model = build_model(REFERENCE_CONFIG)
    criterion = VAELoss(
        REFERENCE_CONFIG["loss"], REFERENCE_CONFIG["auxiliary"]["weights"]
    )
    x = torch.randn(2, 128, 64, 64)
    targets = {
        # One missing elevation: the auxiliary loss must mask it, not poison
        # the total with a NaN.
        "elevation": torch.tensor([0.5, float("nan")]),
        "latitude": torch.tensor([-1.0, 0.25]),
        "longitude": torch.tensor([0.75, 1.5]),
    }

    total, log = criterion(model(x), x, targets)

    assert torch.isfinite(total)
    assert total.requires_grad
    assert {"loss/recon", "loss/grad", "loss/kl", "loss/beta", "loss/aux"} <= set(log)
    assert all(np.isfinite(value) for value in log.values())
    assert log["loss/grad"] > 0  # the gradient term is switched on (weight 0.5)
    assert "loss/aux_elevation" in log


def test_kl_weight_ramps_linearly_over_the_warmup():
    beta_end, warmup = 5e-4, 5000
    assert linear_beta(0, warmup, beta_end) == 0.0
    assert linear_beta(warmup // 2, warmup, beta_end) == pytest.approx(beta_end / 2)
    assert linear_beta(warmup, warmup, beta_end) == beta_end
    assert linear_beta(10 * warmup, warmup, beta_end) == beta_end

    criterion = VAELoss(REFERENCE_CONFIG["loss"])
    assert criterion.current_step == 0
    criterion.step()
    assert criterion.current_step == 1


def test_loss_rejects_a_schedule_that_was_dropped():
    cfg = dict(REFERENCE_CONFIG["loss"], beta_schedule="cyclical")
    with pytest.raises(ValueError, match="beta_schedule"):
        VAELoss(cfg)


# ---------------------------------------------------------------------------
# Dataset: station filter, centre crop, normalisation
# ---------------------------------------------------------------------------

STORED_SIZE = 16
CROP_SIZE = 8
N_CHANNELS = 4


@pytest.fixture
def synthetic_patches(tmp_path):
    """A miniature stand-in for the station patch file and its station CSV.

    Twenty stations of ``(16, 16, 4)``; station 3's patch is all zero (no
    TESSERA coverage), station 7's holds an extreme value (corrupted), and
    station 11's elevation is the GHCNh missing-data sentinel.
    """
    rng = np.random.RandomState(0)
    patches = rng.uniform(-2, 2, (20, STORED_SIZE, STORED_SIZE, N_CHANNELS))
    patches = patches.astype(np.float32)
    patches[3] = 0.0
    patches[7, 0, 0, 0] = 5000.0

    patches_path = tmp_path / "patch_embeddings_test_p16.npy"
    np.save(patches_path, patches)

    stations = pd.DataFrame(
        {
            "station_id": [f"S{i:03d}" for i in range(20)],
            "latitude": np.linspace(-60, 60, 20),
            "longitude": np.linspace(-170, 170, 20),
            "elevation": np.linspace(0, 1900, 20),
        }
    )
    stations.loc[11, "elevation"] = -999.9
    stations_path = tmp_path / "station_list_filtered.csv"
    stations.to_csv(stations_path, index=False)

    return patches, patches_path, stations, stations_path


def test_cache_flags_zero_and_outlier_patches_and_is_reused(
    synthetic_patches, tmp_path
):
    _, patches_path, _, _ = synthetic_patches
    cache_dir = tmp_path / "cache"

    cache = prepare_data(patches_path, cache_dir=cache_dir)

    assert cache["zero_indices"].tolist() == [3]
    assert cache["outlier_indices"].tolist() == [7]
    assert 3 not in cache["valid_indices"] and 7 not in cache["valid_indices"]
    assert len(cache["valid_indices"]) == 18
    assert cache["channel_mean"].shape == (N_CHANNELS,)
    assert (cache["channel_std"] > 0).all()

    cache_file = cache_dir / patches_path.stem / "cache.npz"
    assert cache_file.exists()
    reloaded = prepare_data(patches_path, cache_dir=cache_dir)
    assert np.array_equal(reloaded["valid_indices"], cache["valid_indices"])
    assert np.array_equal(reloaded["channel_mean"], cache["channel_mean"])


def test_elevation_sentinels_are_filtered_after_the_patch_scan(
    synthetic_patches, tmp_path
):
    _, patches_path, _, stations_path = synthetic_patches
    cache = prepare_data(patches_path, cache_dir=tmp_path / "cache")

    usable = filter_elevation_sentinels(cache["valid_indices"], stations_path)

    assert 11 not in usable
    assert usable.tolist() == [i for i in range(20) if i not in (3, 7, 11)]


def test_crop_is_centred_on_the_station_and_channels_come_first(
    synthetic_patches, tmp_path
):
    patches, patches_path, _, stations_path = synthetic_patches
    cache = prepare_data(patches_path, cache_dir=tmp_path / "cache")
    usable = filter_elevation_sentinels(cache["valid_indices"], stations_path)

    dataset = TesseraPatchDataset(
        patches_path=patches_path,
        stations_path=stations_path,
        valid_indices=usable,
        channel_mean=cache["channel_mean"],
        channel_std=cache["channel_std"],
        crop_size=CROP_SIZE,
    )

    assert dataset.stored_size == STORED_SIZE
    assert dataset.spatial_size == CROP_SIZE
    assert len(dataset) == len(usable)

    patch = dataset[0]["patch"]
    assert patch.shape == (N_CHANNELS, CROP_SIZE, CROP_SIZE)

    # The stored patch is station-centred, so the crop must be the middle
    # window -- offset (16 - 8) / 2 = 4 -- z-scored per channel.
    offset = (STORED_SIZE - CROP_SIZE) // 2
    expected = patches[
        usable[0], offset : offset + CROP_SIZE, offset : offset + CROP_SIZE
    ]
    expected = (expected - cache["channel_mean"]) / cache["channel_std"]
    torch.testing.assert_close(
        patch, torch.from_numpy(np.transpose(expected, (2, 0, 1)).copy())
    )


def test_full_patch_is_served_when_no_crop_is_requested(synthetic_patches, tmp_path):
    _, patches_path, _, stations_path = synthetic_patches
    cache = prepare_data(patches_path, cache_dir=tmp_path / "cache")

    dataset = TesseraPatchDataset(
        patches_path=patches_path,
        stations_path=stations_path,
        valid_indices=cache["valid_indices"],
        channel_mean=cache["channel_mean"],
        channel_std=cache["channel_std"],
        crop_size=None,
    )

    assert dataset.crop_size is None
    assert dataset[0]["patch"].shape == (N_CHANNELS, STORED_SIZE, STORED_SIZE)


def test_auxiliary_targets_are_z_scored_and_sentinels_are_masked(
    synthetic_patches, tmp_path
):
    _, patches_path, stations, stations_path = synthetic_patches
    cache = prepare_data(patches_path, cache_dir=tmp_path / "cache")
    # Keep station 11 (the elevation sentinel) to see it masked rather than
    # dropped; the dataloaders filter it out one step earlier.
    indices = np.array([0, 11, 19])

    dataset = TesseraPatchDataset(
        patches_path=patches_path,
        stations_path=stations_path,
        valid_indices=indices,
        channel_mean=cache["channel_mean"],
        channel_std=cache["channel_std"],
        aux_targets=["elevation", "latitude", "longitude"],
        crop_size=CROP_SIZE,
    )

    assert torch.isnan(dataset[1]["elevation"])
    assert torch.isfinite(dataset[0]["elevation"])
    stats = dataset.target_stats["latitude"]
    expected = (stations["latitude"][19] - stats["mean"]) / stats["std"]
    assert dataset[2]["latitude"].item() == pytest.approx(expected, abs=1e-5)


def test_dataloaders_split_the_usable_stations_deterministically(
    synthetic_patches, tmp_path
):
    _, patches_path, _, stations_path = synthetic_patches
    cfg = {
        "data": {
            "patches_path": str(patches_path),
            "stations_path": str(stations_path),
            "crop_size": CROP_SIZE,
        },
        "auxiliary": {"enable": True, "targets": ["elevation"]},
        "training": {
            "seed": 42,
            "val_split": 0.2,
            "batch_size": 2,
            "num_workers": 0,
            "pin_memory": False,
        },
    }

    train_loader, val_loader, dataset = create_dataloaders(
        cfg, cache_dir=tmp_path / "cache"
    )

    assert len(dataset) == 17  # 20 stations - zero - outlier - sentinel
    assert len(val_loader.dataset) == 3
    assert len(train_loader.dataset) == 14
    assert set(train_loader.dataset.indices).isdisjoint(val_loader.dataset.indices)

    same_split, _, _ = create_dataloaders(cfg, cache_dir=tmp_path / "cache")
    assert same_split.dataset.indices.tolist() == train_loader.dataset.indices.tolist()

    batch = next(iter(val_loader))
    assert batch["patch"].shape == (2, N_CHANNELS, CROP_SIZE, CROP_SIZE)
    assert "elevation" in batch


# ---------------------------------------------------------------------------
# The paper's run on the data root
# ---------------------------------------------------------------------------

data_root_only = pytest.mark.skipif(
    not data_root().exists(), reason=f"data root {data_root()} not present"
)


@data_root_only
def test_published_checkpoint_loads_into_the_ported_model_strictly():
    if not (RUN_DIR / "best.pt").exists():
        pytest.skip(f"checkpoint not present: {RUN_DIR / 'best.pt'}")

    ckpt = torch.load(RUN_DIR / "best.pt", map_location="cpu", weights_only=False)
    model = build_model(ckpt["config"])

    model.load_state_dict(ckpt["model"], strict=True)

    # The config travelled with the checkpoint, so it also describes the run.
    stored = yaml.safe_load((RUN_DIR / "config.yaml").read_text())
    assert stored["model"]["latent_dim"] == 16
    assert stored["data"]["crop_size"] == 64
    assert stored["loss"]["gradient_weight"] == 0.5
    assert stored["auxiliary"]["enable"] is True


@data_root_only
def test_re_encoding_reproduces_the_published_station_latents():
    """Encode the first stations again and compare with the published file.

    Everything that shapes a descriptor is exercised: the checkpoint's config,
    the cached normalisation statistics, the centre crop, and the posterior
    mean the encoder emits in eval mode.
    """
    published_path = RUN_DIR / "eval" / "station_latents.npy"
    for path in (published_path, CACHE_FILE, PATCHES, STATIONS):
        if not path.exists():
            pytest.skip(f"not present: {path}")

    ckpt = torch.load(RUN_DIR / "best.pt", map_location="cpu", weights_only=False)
    model = build_model(ckpt["config"])
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()

    published = np.load(published_path, mmap_mode="r")
    # NaN rows are stations the run could not encode; skip those.
    rows = [i for i in range(4) if np.isfinite(published[i]).all()]
    assert rows, "expected at least one encodable station among the first four"

    cache = np.load(CACHE_FILE)
    dataset = TesseraPatchDataset(
        patches_path=PATCHES,
        stations_path=STATIONS,
        valid_indices=np.array(rows),
        channel_mean=cache["channel_mean"],
        channel_std=cache["channel_std"],
        crop_size=ckpt["config"]["data"]["crop_size"],
    )
    x = torch.stack([dataset[i]["patch"] for i in range(len(rows))])
    assert x.shape == (len(rows), 128, 64, 64)

    with torch.no_grad():
        latents = model.encode(x).numpy()

    np.testing.assert_allclose(latents, np.asarray(published[rows]), atol=1e-5)


# ---------------------------------------------------------------------------
# The foundation-model benchmark runs on the data root
# ---------------------------------------------------------------------------


@data_root_only
@pytest.mark.parametrize("source", sorted(FM_RUNS))
def test_benchmark_checkpoint_loads_into_the_ported_model_strictly(source):
    """The AlphaEarth and OlmoEarth runs rebuild from their stored configs.

    Their geometries are the ones the migration had to keep configurable: a
    64-channel input for AlphaEarth, and a 768-channel 16x16 grid with a
    three-stage encoder for OlmoEarth.
    """
    run = FM_RUNS[source]
    if not (run["run_dir"] / "best.pt").exists():
        pytest.skip(f"checkpoint not present: {run['run_dir'] / 'best.pt'}")

    ckpt = torch.load(
        run["run_dir"] / "best.pt", map_location="cpu", weights_only=False
    )
    model = build_model(ckpt["config"])

    model.load_state_dict(ckpt["model"], strict=True)

    channels, size, _ = run["patch_shape"]
    assert ckpt["config"]["model"]["in_channels"] == channels
    assert ckpt["config"]["model"]["input_size"] == size
    assert ckpt["config"]["model"]["latent_dim"] == 16
    assert ckpt["config"]["auxiliary"]["enable"] is True
    # The recipe is the paper's, so only the surface embedding differs.
    assert ckpt["config"]["loss"]["gradient_weight"] == 0.5
    assert ckpt["config"]["loss"]["beta_end"] == 0.0005


@data_root_only
@pytest.mark.parametrize("source", sorted(FM_RUNS))
def test_re_encoding_reproduces_the_published_benchmark_latents(source):
    """Re-encode the first stations of each benchmark arm.

    Tolerance is looser than for the TESSERA run because these latents were
    published from a GPU evaluation (TF32 convolutions on a GH200) while the
    TESSERA one ran on CPU; the observed disagreement is ~3e-4 on latents whose
    entries reach 3.6.
    """
    run = FM_RUNS[source]
    published_path = run["run_dir"] / "eval" / "station_latents.npy"
    for path in (published_path, run["cache"], run["patches"], STATIONS):
        if not path.exists():
            pytest.skip(f"not present: {path}")

    ckpt = torch.load(
        run["run_dir"] / "best.pt", map_location="cpu", weights_only=False
    )
    model = build_model(ckpt["config"])
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()

    published = np.load(published_path, mmap_mode="r")
    # NaN rows are stations the run could not encode; skip those.
    rows = [i for i in range(4) if np.isfinite(published[i]).all()]
    assert rows, "expected at least one encodable station among the first four"

    cache = np.load(run["cache"])
    dataset = TesseraPatchDataset(
        patches_path=run["patches"],
        stations_path=STATIONS,
        valid_indices=np.array(rows),
        channel_mean=cache["channel_mean"],
        channel_std=cache["channel_std"],
        crop_size=ckpt["config"]["data"].get("crop_size"),
    )
    x = torch.stack([dataset[i]["patch"] for i in range(len(rows))])
    assert x.shape == (len(rows), *run["patch_shape"])

    with torch.no_grad():
        latents = model.encode(x).numpy()

    np.testing.assert_allclose(latents, np.asarray(published[rows]), atol=1e-3)
