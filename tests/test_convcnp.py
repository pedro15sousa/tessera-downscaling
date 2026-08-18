"""Tests for the ConvCNP downscaler with per-variable likelihood heads.

Covers construction + forward for the configurations the paper uses
(bilinear baseline, bilinear + concatenated 16-d TESSERA latent, Gaussian
and truncated-normal heads), the vanilla SetConv interpolator, argument
validation, and the state-dict key layout that checkpoints on disk rely on.
"""

import pytest
import torch

from tessera_downscaling.model.convcnp import (
    BilinearInterp,
    ConvCNPDownscaler,
    DecoderMLP,
    RBFSetConv,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _synthetic_batch(B=2, C=8, Nlat=6, Nlon=6, N=4):
    """Build a synthetic forward-pass input (no TESSERA, no mTPI)."""
    return dict(
        context_grid=torch.randn(B, C, Nlat, Nlon),
        grid_lats=torch.linspace(45.0, 60.0, Nlat),
        grid_lons=torch.linspace(0.0, 15.0, Nlon),
        target_coords=torch.stack(
            [
                torch.linspace(46.0, 59.0, N).repeat(B, 1),
                torch.linspace(1.0, 14.0, N).repeat(B, 1),
            ],
            dim=-1,
        ),
        target_elev=torch.randn(B, N) * 100.0,
        target_delta_elev=torch.randn(B, N) * 50.0,
        target_mask=torch.ones(B, N, dtype=torch.bool),
    )


def _small_model(**overrides):
    kwargs = dict(
        n_context_channels=8,
        cnn_hidden=12,
        cnn_layers=3,
        mlp_hidden=8,
        mlp_n_hidden=2,
        target_variables=["t2m"],
    )
    kwargs.update(overrides)
    return ConvCNPDownscaler(**kwargs)


# ---------------------------------------------------------------------------
# Construction + forward
# ---------------------------------------------------------------------------


def test_single_task_baseline_forward():
    """Single-task all-Gaussian (the default) builds and forwards."""
    torch.manual_seed(0)
    model = _small_model(likelihood_per_variable=None)  # default → all-Gaussian
    assert model.likelihood_per_variable == {"t2m": "gaussian"}
    assert isinstance(model.interp, BilinearInterp)  # bilinear is the default
    out = model(**_synthetic_batch())
    assert set(out.keys()) == {"t2m"}
    assert set(out["t2m"].keys()) == {"mu", "log_var"}
    assert out["t2m"]["mu"].shape == (2, 4)
    assert out["t2m"]["log_var"].shape == (2, 4)


def test_multi_task_mixed_distributions_forward():
    """t2m (Gaussian) + wind (truncated normal) with a concatenated 16-d
    precomputed TESSERA latent and mTPI — the paper's model family."""
    torch.manual_seed(0)
    model = _small_model(
        target_variables=["t2m", "wind"],
        likelihood_per_variable={"t2m": "gaussian", "wind": "truncated_normal"},
        n_elev_features=3,
        tessera_features_precomputed=True,
        precomputed_tessera_dim=16,
        tessera_injection="concat",
    )
    # Decoder input = 12 grid features + 3 topographic + 16 latent.
    assert model.mlp.net[0].in_features == 12 + 3 + 16

    B, N = 2, 4
    batch = _synthetic_batch(B=B, N=N)
    batch["target_tessera"] = torch.randn(B, N, 16)
    batch["target_mtpi"] = torch.randn(B, N) * 30.0

    out = model(**batch)
    assert list(out.keys()) == ["t2m", "wind"]
    for var in ("t2m", "wind"):
        assert set(out[var].keys()) == {"mu", "log_var"}
        assert out[var]["mu"].shape == (B, N)
        assert out[var]["log_var"].shape == (B, N)
    # Truncated-normal point predictions live on [0, ∞).
    assert (model.heads.heads["wind"].median(out["wind"]) >= 0).all()


def test_precomputed_latent_required_and_dim_checked():
    """A precomputed-latent model refuses a missing or mis-sized latent."""
    model = _small_model(tessera_features_precomputed=True, precomputed_tessera_dim=16)
    batch = _synthetic_batch()
    with pytest.raises(ValueError, match="target_tessera"):
        model(**batch)
    batch["target_tessera"] = torch.randn(2, 4, 8)
    with pytest.raises(ValueError, match="precomputed_tessera_dim"):
        model(**batch)


def test_tessera_injection_none_ignores_latent():
    """``tessera_injection='none'`` (cross-lead baselines) leaves the decoder
    input at grid + topography width and produces the baseline output."""
    model = _small_model(
        tessera_features_precomputed=True,
        precomputed_tessera_dim=16,
        tessera_injection="none",
    )
    assert model.mlp.net[0].in_features == 12 + 2
    batch = _synthetic_batch()
    batch["target_tessera"] = torch.randn(2, 4, 16)
    out = model(**batch)
    assert out["t2m"]["mu"].shape == (2, 4)


def test_mtpi_required_when_three_elev_features():
    model = _small_model(n_elev_features=3)
    with pytest.raises(ValueError, match="target_mtpi"):
        model(**_synthetic_batch())


@pytest.mark.parametrize("interpolation", ["bilinear", "setconv"])
def test_interpolation_variants_forward(interpolation):
    """Both interpolators build and forward; only SetConv has a parameter."""
    torch.manual_seed(0)
    model = _small_model(interpolation=interpolation, setconv_length_scale=0.7)
    expected = RBFSetConv if interpolation == "setconv" else BilinearInterp
    assert isinstance(model.interp, expected)
    out = model(**_synthetic_batch())
    assert out["t2m"]["mu"].shape == (2, 4)
    assert torch.isfinite(out["t2m"]["mu"]).all()
    interp_params = dict(model.interp.named_parameters())
    if interpolation == "setconv":
        assert set(interp_params) == {"log_scale"}
        assert interp_params["log_scale"].item() == pytest.approx(
            torch.tensor(0.7).log().item()
        )
    else:
        assert interp_params == {}


# ---------------------------------------------------------------------------
# Argument validation
# ---------------------------------------------------------------------------


def test_target_variables_required_and_non_empty():
    """target_variables must be a non-empty list."""
    with pytest.raises(ValueError, match="target_variables"):
        _small_model(target_variables=None)
    with pytest.raises(ValueError, match="target_variables"):
        _small_model(target_variables=[])


@pytest.mark.parametrize("bad", ["hypernet", "film"])
def test_removed_injection_modes_rejected(bad):
    """Only 'concat' and 'none' are supported."""
    with pytest.raises(ValueError, match="tessera_injection"):
        _small_model(tessera_injection=bad)


def test_unknown_interpolation_rejected():
    with pytest.raises(ValueError, match="interpolation"):
        _small_model(interpolation="cubic")


def test_precomputed_dim_required():
    with pytest.raises(ValueError, match="precomputed_tessera_dim"):
        _small_model(tessera_features_precomputed=True, precomputed_tessera_dim=0)


# ---------------------------------------------------------------------------
# Structure / state-dict layout
# ---------------------------------------------------------------------------


def test_decoder_mlp_emits_hidden_state():
    """DecoderMLP body emits (..., hidden_dim); the heads own the projections."""
    body = DecoderMLP(in_features=10, hidden_dim=16, n_hidden_layers=3)
    out = body(torch.randn(2, 5, 10))
    assert out.shape == (2, 5, 16)


@pytest.mark.parametrize("interpolation", ["bilinear", "setconv"])
def test_state_dict_layout(interpolation):
    """The state-dict key names checkpoints on disk use must not change.

    Prefixes: ``cnn.net.*``, ``interp.log_scale`` (SetConv only),
    ``mlp.net.*`` and ``heads.heads.<var>.linear.{weight,bias}``.
    """
    target_vars = ["t2m", "wind"]
    model = _small_model(
        target_variables=target_vars,
        likelihood_per_variable={"t2m": "gaussian", "wind": "truncated_normal"},
        interpolation=interpolation,
    )
    sd = model.state_dict()
    prefixes = {"cnn.", "interp.", "mlp.", "heads."}
    assert all(any(k.startswith(p) for p in prefixes) for k in sd), sorted(sd)

    # Grid CNN: cnn_layers=3 → input conv, one residual block, 1×1 output conv.
    assert sd["cnn.net.0.weight"].shape == (12, 8, 3, 3)
    assert {"cnn.net.2.conv1.weight", "cnn.net.2.conv2.weight"} <= set(sd)
    assert sd["cnn.net.3.weight"].shape == (12, 12, 1, 1)

    interp_keys = {k for k in sd if k.startswith("interp.")}
    assert interp_keys == (
        {"interp.log_scale"} if interpolation == "setconv" else set()
    )

    # Decoder body: mlp_n_hidden=2 → Linear at net.0 and net.2, nothing after.
    assert sd["mlp.net.0.weight"].shape == (8, 12 + 2)
    assert sd["mlp.net.2.weight"].shape == (8, 8)
    assert "mlp.net.4.weight" not in sd

    for var in target_vars:
        assert sd[f"heads.heads.{var}.linear.weight"].shape == (2, 8)
        assert sd[f"heads.heads.{var}.linear.bias"].shape == (2,)
