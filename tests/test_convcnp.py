"""Tests for the v4 ConvCNP downscaler with per-variable likelihood heads.

Two main areas are covered:

1. **Construction + forward.** The model must build cleanly for the
   single-task baseline, multi-task with mixed distributions (Gaussian +
   Weibull + Bernoulli-Gamma), and FiLM injection. Forward passes return
   the expected nested-dict structure with valid distribution-parameter
   constraints.

2. **The migration round-trip property.** The legacy code packed all
   variables' (μ, log_var) pairs row-wise into one
   ``Linear(H, 2V)`` projection at the end of the decoder body; the v4
   code splits that into per-variable ``Linear(H, n_params)`` heads. The
   migration script's correctness claim is that for any hidden state
   ``h``, applying ``Linear(H, 2V)`` then splitting into pairs is
   bit-for-bit identical to running the per-variable heads on the same
   ``h``. These tests verify that property at both the heads-only level
   and through a full end-to-end forward pass.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytest

from tessera_downscaling.model.convcnp import (
    ConvCNPDownscaler,
    DecoderMLP,
    FiLMDecoderMLP,
)
from tessera_downscaling.model.heads import (
    BernoulliGammaHead,
    GaussianHead,
    WeibullHead,
)


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------

def _synthetic_batch(B=2, C=8, Nlat=6, Nlon=6, N=4):
    """Build a synthetic forward-pass input (no TESSERA)."""
    return dict(
        context_grid=torch.randn(B, C, Nlat, Nlon),
        grid_lats=torch.linspace(45.0, 60.0, Nlat),
        grid_lons=torch.linspace(0.0, 15.0, Nlon),
        target_coords=torch.stack([
            torch.linspace(46.0, 59.0, N).repeat(B, 1),
            torch.linspace(1.0, 14.0, N).repeat(B, 1),
        ], dim=-1),
        target_elev=torch.randn(B, N) * 100.0,
        target_delta_elev=torch.randn(B, N) * 50.0,
        target_mask=torch.ones(B, N, dtype=torch.bool),
    )


class _FakeTesseraEncoder(nn.Module):
    """Minimal TesseraPatchEncoder stand-in for testing."""

    def __init__(self, dim=8, patch_size=4, embed_dim=128):
        super().__init__()
        self.output_dim = dim
        self.linear = nn.Linear(embed_dim * patch_size * patch_size, dim)

    def forward(self, x):
        return self.linear(x.reshape(x.shape[0], -1))


# ---------------------------------------------------------------------------
# Construction + forward
# ---------------------------------------------------------------------------

def test_single_task_baseline_forward():
    """Single-task all-Gaussian (the legacy default) builds and forwards."""
    torch.manual_seed(0)
    model = ConvCNPDownscaler(
        n_context_channels=8, cnn_hidden=12, cnn_layers=3,
        mlp_hidden=8, mlp_n_hidden=2,
        target_variables=["t2m"],
        likelihood_per_variable=None,   # default → all-Gaussian
    )
    assert model.likelihood_per_variable == {"t2m": "gaussian"}
    out = model(**_synthetic_batch())
    assert set(out.keys()) == {"t2m"}
    assert set(out["t2m"].keys()) == {"mu", "log_var"}
    assert out["t2m"]["mu"].shape == (2, 4)
    assert out["t2m"]["log_var"].shape == (2, 4)


def test_multi_task_mixed_distributions_forward():
    """Multi-task Gaussian+Weibull+B-G with FiLM injection forwards correctly."""
    torch.manual_seed(0)
    enc = _FakeTesseraEncoder(dim=8)
    model = ConvCNPDownscaler(
        n_context_channels=8, cnn_hidden=12, cnn_layers=3,
        mlp_hidden=8, mlp_n_hidden=2,
        target_variables=["t2m", "wind", "precip"],
        likelihood_per_variable={
            "t2m": "gaussian",
            "wind": "weibull",
            "precip": "bernoulli_gamma",
        },
        tessera_encoder=enc,
        tessera_injection="film",
    )
    B, N = 2, 4
    batch = _synthetic_batch(B=B, N=N)
    batch["target_tessera"] = torch.randn(B, N, 128, 4, 4)

    out = model(**batch)
    assert list(out.keys()) == ["t2m", "wind", "precip"]
    assert set(out["t2m"].keys()) == {"mu", "log_var"}
    assert set(out["wind"].keys()) == {"k", "lam"}
    assert set(out["precip"].keys()) == {"rho", "alpha", "beta"}

    # Distribution-parameter constraints.
    assert (out["wind"]["k"] > 0).all()
    assert (out["wind"]["lam"] > 0).all()
    assert (0 < out["precip"]["rho"]).all() and (out["precip"]["rho"] < 1).all()
    assert (out["precip"]["alpha"] > 0).all()
    assert (out["precip"]["beta"] > 0).all()


def test_hypernet_injection_rejected():
    """tessera_injection='hypernet' is no longer accepted."""
    with pytest.raises(ValueError, match="hypernet"):
        ConvCNPDownscaler(
            n_context_channels=8, cnn_hidden=12, cnn_layers=3,
            mlp_hidden=8, mlp_n_hidden=2,
            target_variables=["t2m"],
            tessera_injection="hypernet",
        )


def test_target_variables_required_and_non_empty():
    """target_variables must be a non-empty list."""
    with pytest.raises(ValueError, match="target_variables"):
        ConvCNPDownscaler(
            n_context_channels=8, cnn_hidden=12, cnn_layers=3,
            mlp_hidden=8, mlp_n_hidden=2,
            target_variables=None,
        )
    with pytest.raises(ValueError, match="target_variables"):
        ConvCNPDownscaler(
            n_context_channels=8, cnn_hidden=12, cnn_layers=3,
            mlp_hidden=8, mlp_n_hidden=2,
            target_variables=[],
        )


def test_decoder_mlp_emits_hidden_state_not_2v():
    """DecoderMLP body must emit (..., hidden_dim), not (..., 2V).

    The legacy body had a final ``Linear(hidden, 2V)``; v4 lifts that
    out into the heads. If this regresses, the body would output
    ``2 * len(target_variables)`` instead of ``hidden_dim``, which would
    silently break the head-dispatcher input shape.
    """
    body = DecoderMLP(in_features=10, hidden_dim=16, n_hidden_layers=3)
    x = torch.randn(2, 5, 10)
    out = body(x)
    assert out.shape == (2, 5, 16)


def test_film_decoder_mlp_emits_hidden_state_not_2v():
    """FiLMDecoderMLP body emits hidden state (no output_layer)."""
    body = FiLMDecoderMLP(
        in_features=10, hidden_dim=16, n_hidden_layers=3, tessera_dim=8,
    )
    x = torch.randn(2, 5, 10)
    t = torch.randn(2, 5, 8)
    out = body(x, tessera_features=t)
    assert out.shape == (2, 5, 16)
    # And without TESSERA features (baseline mode where FiLM decays to identity).
    out_baseline = body(x, tessera_features=None)
    assert out_baseline.shape == (2, 5, 16)


def test_film_decoder_mlp_no_output_layer_attribute():
    """The legacy `output_layer` attribute must be gone — its weights now
    live in the heads dispatcher."""
    body = FiLMDecoderMLP(
        in_features=10, hidden_dim=16, n_hidden_layers=3, tessera_dim=8,
    )
    assert not hasattr(body, "output_layer"), (
        "FiLMDecoderMLP still has an output_layer; the heads dispatcher "
        "should own the per-variable projections instead."
    )


# ---------------------------------------------------------------------------
# Migration round-trip property — the property the migration script depends on
#
# Two distinct levels of equality are at play here:
#
#   1. TENSOR level (bit-for-bit). The migration script's pre-write check
#      stacks the heads' per-variable Linear weights and asserts
#      ``torch.equal`` against the original legacy ``[2V, H]`` tensor.
#      That's a pure tensor copy operation — no float arithmetic — so it
#      really is bit-for-bit identical, and the migration script refuses
#      to write if it ever isn't. ``test_state_dict_layout_for_migration_target``
#      below covers the structural side of this.
#
#   2. PREDICTION level (~1e-7 absolute). Calling ``model.heads(h)`` (which
#      runs V × ``nn.Linear(H, 2)`` via ``F.linear`` → ``addmm``) is
#      numerically equivalent — but NOT bit-for-bit identical — to running
#      one ``nn.Linear(H, 2V)`` over the same hidden state. The GEMM
#      kernel dispatcher picks different code paths for output dim 2 vs
#      2V, and the fused-multiply-add accumulation rounds differently in
#      the last bit on some inputs. The resulting differences are at
#      ~1e-8 absolute (well below machine precision for our prediction
#      ranges) but ``torch.equal`` will detect them on some seeds.
#
# The tests below check level 2 with ``torch.allclose(atol=1e-6)``. Level 1
# is what the migration script actually enforces and is also tested
# directly in ``tests/test_migration.py``.
# ---------------------------------------------------------------------------

def _legacy_arithmetic(W_legacy, b_legacy, hidden, target_variables):
    """Apply legacy `Linear(H, 2V)` then split into pairs.

    This replicates what the legacy DecoderMLP / FiLMDecoderMLP
    output_layer + the legacy multitask split did:
      - Apply nn.Linear with weight [2V, H] and bias [2V]
      - var_i mean = output[..., 2i], var_i log_var = output[..., 2i+1]

    Uses ``F.linear`` (the function ``nn.Linear`` calls internally) so
    that this is bit-for-bit identical to what an actual
    ``nn.Linear(H, 2V)`` would produce — ``F.linear`` uses ``addmm``
    which fuses the matmul + bias add and would differ at the ULP
    level from a manual ``hidden @ W.T + b``.
    """
    flat = F.linear(hidden, W_legacy, b_legacy)
    out = {}
    for i, var in enumerate(target_variables):
        out[var] = {
            "mu":      flat[..., 2 * i],
            "log_var": flat[..., 2 * i + 1],
        }
    return out


@pytest.mark.parametrize("seed", [0, 1, 42, 2024])
def test_migration_roundtrip_at_heads_level(seed):
    """For any hidden state, heads(h) ≈ legacy single-Linear-then-split.

    Numerical equality at ~1e-6 absolute (see module-level comment for
    why this isn't bit-for-bit). The structural correctness — which
    rows of the legacy tensor map to which heads — is the real claim,
    and that is enforced bit-for-bit by the migration script's
    pre-write tensor-equality check.
    """
    torch.manual_seed(seed)
    target_vars = ["t2m", "wind"]
    model = ConvCNPDownscaler(
        n_context_channels=8, cnn_hidden=12, cnn_layers=3,
        mlp_hidden=8, mlp_n_hidden=2,
        target_variables=target_vars,
        likelihood_per_variable={var: "gaussian" for var in target_vars},
    )
    B, N, H = 3, 7, 8
    hidden = torch.randn(B, N, H)

    new_out = model.heads(hidden)

    # Reconstruct legacy weights by stacking the heads' per-variable Linears.
    W_legacy = torch.cat(
        [model.heads.heads[var].linear.weight for var in target_vars], dim=0,
    )  # shape [2V, H]
    b_legacy = torch.cat(
        [model.heads.heads[var].linear.bias for var in target_vars], dim=0,
    )  # shape [2V]

    legacy_out = _legacy_arithmetic(W_legacy, b_legacy, hidden, target_vars)

    for var in target_vars:
        for p in ("mu", "log_var"):
            torch.testing.assert_close(
                new_out[var][p], legacy_out[var][p],
                atol=1e-6, rtol=0,
                msg=f"Round-trip mismatch on {var}.{p} (seed={seed})",
            )


@pytest.mark.parametrize("seed", [0, 7, 42])
def test_migration_roundtrip_full_forward(seed):
    """End-to-end forward parity with the legacy body+single-Linear+split path.

    This is the integration version of the heads-only round-trip: the
    full ConvCNP forward (CNN → interp → MLP body → heads) produces the
    same output as the same body + a synthetic Linear(H, 2V) + split.
    """
    torch.manual_seed(seed)
    target_vars = ["t2m", "wind"]
    model = ConvCNPDownscaler(
        n_context_channels=8, cnn_hidden=12, cnn_layers=3,
        mlp_hidden=8, mlp_n_hidden=2,
        target_variables=target_vars,
        likelihood_per_variable={var: "gaussian" for var in target_vars},
    )
    model.eval()
    batch = _synthetic_batch()

    with torch.no_grad():
        full_out = model(**batch)

    # Recover hidden state by replicating the model's pre-heads forward.
    W_legacy = torch.cat(
        [model.heads.heads[var].linear.weight for var in target_vars], dim=0,
    )
    b_legacy = torch.cat(
        [model.heads.heads[var].linear.bias for var in target_vars], dim=0,
    )

    with torch.no_grad():
        grid_features = model.cnn(batch["context_grid"])
        interp_features = model.interp(
            grid_features,
            batch["grid_lats"], batch["grid_lons"],
            batch["target_coords"][:, :, 0], batch["target_coords"][:, :, 1],
        ).permute(0, 2, 1)
        elev = (batch["target_elev"] / 1000.0).unsqueeze(-1)
        delta = (batch["target_delta_elev"] / 1000.0).unsqueeze(-1)
        mlp_input = torch.cat([interp_features, elev, delta], dim=-1)
        body_hidden = model.mlp(mlp_input)
        legacy_pred = _legacy_arithmetic(
            W_legacy, b_legacy, body_hidden, target_vars,
        )

    for var in target_vars:
        for p in ("mu", "log_var"):
            torch.testing.assert_close(
                full_out[var][p], legacy_pred[var][p],
                atol=1e-6, rtol=0,
                msg=f"Full-forward mismatch on {var}.{p} (seed={seed})",
            )


def test_state_dict_layout_for_migration_target():
    """v4 model's state_dict has keys the migration script expects to write."""
    target_vars = ["t2m", "wind"]
    model = ConvCNPDownscaler(
        n_context_channels=8, cnn_hidden=12, cnn_layers=3,
        mlp_hidden=8, mlp_n_hidden=2,
        target_variables=target_vars,
        likelihood_per_variable={var: "gaussian" for var in target_vars},
    )
    sd = model.state_dict()
    for var in target_vars:
        assert f"heads.heads.{var}.linear.weight" in sd
        assert f"heads.heads.{var}.linear.bias" in sd
        assert sd[f"heads.heads.{var}.linear.weight"].shape == (2, 8)
        assert sd[f"heads.heads.{var}.linear.bias"].shape == (2,)
    # Body should NOT have a final 2V projection any more.
    assert "mlp.output_layer.weight" not in sd
    # The highest-indexed mlp.net.<N>.weight should output hidden_dim, not 2V.
    body_keys = sorted(
        (int(k.split(".")[2]), k)
        for k in sd
        if k.startswith("mlp.net.") and k.endswith(".weight")
    )
    last_idx, last_key = body_keys[-1]
    assert sd[last_key].shape == (8, 8) or sd[last_key].shape[0] == 8, (
        f"Body's final layer {last_key} has shape {sd[last_key].shape}, "
        f"expected first dim hidden_dim=8 (not 2V=4)"
    )