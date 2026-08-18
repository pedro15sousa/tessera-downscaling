"""Checkpoint compatibility: real paper checkpoints must load with strict=True.

Rebuilds :class:`ConvCNPDownscaler` from the ``config`` dict stored inside
three trained checkpoints on the data root — the way ``tessera-evaluate``
does — loads their ``model_state_dict`` with ``strict=True`` and runs one
forward pass. This is the proof that pruning the model package preserved
the constructor semantics and the state-dict key layout.

Skipped when the data root is not present.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from tessera_downscaling.model.convcnp import BilinearInterp, ConvCNPDownscaler
from tessera_downscaling.paths import data_root, resolve

pytestmark = pytest.mark.skipif(
    not data_root().exists(), reason=f"data root {data_root()} not present"
)

# Three seed-42 runs from the paper: the Tessera arm for t2m (Gaussian) and wind
# (truncated normal), and the ERA5-only baseline for t2m.
_CHECKPOINTS = [
    "training_runs_snapshot_14y_eu_tessera_1B-M_2017/"
    "t2m_snap_vae_crop64_lat16_auxon_concat_mtpi_seed42/best_model.pt",
    "training_runs_snapshot_14y_eu_tessera_1B-M_2017/"
    "wind_truncnormal_snap_vae_crop64_lat16_auxon_concat_mtpi_seed42/best_model.pt",
    "training_runs_snapshot_14y_eu/t2m_snap_bilinear_baseline_mtpi_wd_seed42/best_model.pt",
]

# Context-grid channel budget of ``dataset_timestamp_global`` (snapshot layout):
# 20 ERA5 dynamic channels, 13 static fields (when ``include_static_fields``),
# lat + lon, and cos/sin of day-of-year and hour-of-day.
_N_DYNAMIC = 20
_N_STATIC = 13
_N_COORD_AND_TIME = 6


def _model_from_config(cfg: dict) -> ConvCNPDownscaler:
    """Mirror the config → constructor mapping used by ``tessera-evaluate``."""
    include_static = bool(cfg.get("include_static_fields", True))
    n_context_channels = (
        _N_DYNAMIC + (_N_STATIC if include_static else 0) + _N_COORD_AND_TIME
    )

    latents_path = cfg.get("vae_latents_path")
    uses_latents = latents_path is not None and str(latents_path) != "None"
    precomputed_dim = 0
    if uses_latents:
        latents_file = resolve(latents_path)
        if not latents_file.exists():
            pytest.skip(f"VAE latents not present: {latents_file}")
        precomputed_dim = int(np.load(latents_file, mmap_mode="r").shape[1])

    target_variables = cfg["target_variables"]
    likelihood = cfg.get("likelihood_per_variable") or dict.fromkeys(
        target_variables, "gaussian"
    )
    return ConvCNPDownscaler(
        n_context_channels=n_context_channels,
        cnn_hidden=cfg.get("cnn_hidden", 128),
        cnn_layers=cfg.get("cnn_layers", 7),
        cnn_kernel=cfg.get("cnn_kernel", 3),
        setconv_length_scale=cfg.get("setconv_length_scale", 0.5),
        interpolation=cfg.get("interpolation", "setconv"),
        mlp_hidden=cfg.get("mlp_hidden", 128),
        mlp_n_hidden=cfg.get("mlp_n_hidden", 3),
        n_elev_features=int(cfg.get("n_elev_features", 2)),
        include_elevation=cfg.get("include_elevation", True),
        target_variables=target_variables,
        likelihood_per_variable=likelihood,
        tessera_injection=cfg.get("tessera_injection", "concat"),
        tessera_features_precomputed=uses_latents,
        precomputed_tessera_dim=precomputed_dim,
    )


@pytest.mark.parametrize("rel_path", _CHECKPOINTS, ids=lambda p: Path(p).parent.name)
def test_paper_checkpoint_loads_strict_and_forwards(rel_path):
    ckpt_path = data_root() / rel_path
    if not ckpt_path.exists():
        pytest.skip(f"checkpoint not present: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    model = _model_from_config(cfg)

    # Every paper run used parameter-free bilinear interpolation.
    assert cfg["interpolation"] == "bilinear"
    assert isinstance(model.interp, BilinearInterp)

    # The key layout the pruned model produces must match the stored one exactly.
    state_dict = ckpt["model_state_dict"]
    assert set(model.state_dict()) == set(state_dict)
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    # One forward pass on random inputs of the right shapes.
    torch.manual_seed(0)
    B, N, n_lat, n_lon = 2, 5, 12, 16
    n_channels = state_dict["cnn.net.0.weight"].shape[1]
    grid_lats = torch.linspace(40.0, 51.0, n_lat)
    grid_lons = torch.linspace(-5.0, 10.0, n_lon)
    target_coords = torch.stack(
        [
            torch.empty(B, N).uniform_(40.5, 50.5),
            torch.empty(B, N).uniform_(-4.5, 9.5),
        ],
        dim=-1,
    )
    target_tessera = (
        torch.randn(B, N, model.precomputed_tessera_dim)
        if model.tessera_features_precomputed
        else None
    )
    with torch.no_grad():
        out = model(
            torch.randn(B, n_channels, n_lat, n_lon),
            grid_lats,
            grid_lons,
            target_coords,
            torch.rand(B, N) * 1500.0,  # elevation (m)
            torch.randn(B, N) * 200.0,  # Δelevation (m)
            torch.ones(B, N, dtype=torch.bool),
            target_tessera,
            torch.randn(B, N) * 50.0,  # mTPI
        )

    assert list(out) == cfg["target_variables"]
    for var, params in out.items():
        assert set(params) == {"mu", "log_var"}
        for tensor in params.values():
            assert tensor.shape == (B, N)
            assert torch.isfinite(tensor).all(), var
