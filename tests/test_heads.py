"""Tests for per-variable likelihood heads.

The most important test in this file is :func:`test_gaussian_nll_matches_legacy`:
it pins ``GaussianHead.nll`` bit-for-bit to the explicit
``0.5 · (log σ² + (y−μ)²/σ² + log 2π)`` formula every checkpoint on disk was
trained with. If it ever drifts, existing checkpoints become silently
incompatible with the evaluator's NLL.

The other tests cover:
  - the raw ``log_var`` passing through ``forward`` unclamped;
  - the closed-form Gaussian CRPS against a scipy reference;
  - the truncated-normal head in the deeply-truncated regime (μ ≪ 0);
  - registry / construction / state-dict-layout sanity checks.
"""

import math

import numpy as np
import pytest
import torch

from tessera_downscaling.model.heads import (
    HEAD_REGISTRY,
    GaussianHead,
    LikelihoodHead,
    TruncatedNormalHead,
    build_head,
)

# ---------------------------------------------------------------------------
# GaussianHead — parity with the original gaussian_nll_loss
# ---------------------------------------------------------------------------


def _legacy_gaussian_nll_loss(pred_mean, pred_log_var, target, mask=None):
    """Verbatim reproduction of the original ``gaussian_nll_loss``.

    Kept here as a fixture so the parity test isn't tied to whether the
    original function is still importable.
    """
    pred_log_var = torch.clamp(pred_log_var, min=-10.0, max=10.0)
    nll = 0.5 * (
        pred_log_var
        + (target - pred_mean) ** 2 / torch.exp(pred_log_var)
        + math.log(2 * math.pi)
    )
    if mask is not None:
        return (nll * mask.float()).sum() / mask.float().sum().clamp(min=1.0)
    return nll.mean()


@pytest.mark.parametrize("seed", [0, 1, 42, 123, 2024])
def test_gaussian_nll_matches_legacy(seed):
    """GaussianHead.nll == original gaussian_nll_loss bit-for-bit.

    Tested unmasked and masked, across multiple seeds. Any drift here is
    a SEVERE bug — it changes the loss every checkpoint was trained on.
    """
    torch.manual_seed(seed)
    B, N, H = 4, 16, 32
    head = GaussianHead(hidden_dim=H)
    hidden = torch.randn(B, N, H)
    params = head(hidden)
    target = torch.randn(B, N)

    new = head.nll(params, target)
    legacy = _legacy_gaussian_nll_loss(params["mu"], params["log_var"], target)
    assert torch.equal(new, legacy), (
        f"Unmasked NLL mismatch (seed={seed}): new={new}, legacy={legacy}, "
        f"diff={(new - legacy).abs()}"
    )

    mask = torch.rand(B, N) > 0.3
    new_m = head.nll(params, target, mask=mask)
    legacy_m = _legacy_gaussian_nll_loss(
        params["mu"], params["log_var"], target, mask=mask
    )
    assert torch.equal(new_m, legacy_m), (
        f"Masked NLL mismatch (seed={seed}): new={new_m}, legacy={legacy_m}"
    )


def test_gaussian_log_var_clamp_applied_inside_nll_only():
    """Raw log_var passes through forward unchanged; clamp is in nll/cdf/sample."""
    torch.manual_seed(0)
    head = GaussianHead(hidden_dim=8)
    # Force the linear's log-var output channel to extreme values.
    with torch.no_grad():
        head.linear.weight.zero_()
        head.linear.bias[0] = 100.0  # mu
        head.linear.bias[1] = 50.0  # log_var (well outside [-10, 10])
    hidden = torch.zeros(2, 3, 8)
    params = head(hidden)
    # Raw log_var is preserved; only nll applies the clamp internally.
    assert torch.allclose(params["log_var"], torch.full_like(params["log_var"], 50.0))
    assert torch.allclose(params["mu"], torch.full_like(params["mu"], 100.0))


def test_gaussian_crps_matches_scipy():
    """Closed-form Gaussian CRPS matches a scipy-based reference."""
    from scipy import stats

    torch.manual_seed(0)
    head = GaussianHead(hidden_dim=4)
    mu = torch.tensor([0.0, 1.5, -2.0])
    log_var = torch.tensor([0.0, math.log(4.0), math.log(0.25)])  # σ = 1, 2, 0.5
    targets = torch.tensor([0.5, 0.0, -1.5])
    params = {"mu": mu, "log_var": log_var}
    crps_new = head.crps(params, targets).numpy()

    # Scipy reference: CRPS(N(μ, σ²), y) = σ * (z(2Φ(z) − 1) + 2φ(z) − 1/√π)
    sigma = np.exp(log_var.numpy() / 2)
    z = (targets.numpy() - mu.numpy()) / sigma
    crps_ref = sigma * (
        z * (2 * stats.norm.cdf(z) - 1) + 2 * stats.norm.pdf(z) - 1.0 / np.sqrt(np.pi)
    )
    np.testing.assert_allclose(crps_new, crps_ref, rtol=1e-6)


# ---------------------------------------------------------------------------
# TruncatedNormalHead — numerical stability in the deeply-truncated regime
#
# Regression tests for the μ ≪ 0 bug: the model legitimately predicts very
# negative μ for calm wind, where the old median/sample/CRPS code broke (the
# (1e-6, 1−1e-6) icdf clamp gave negative medians for μ/σ < −4.75, and the
# closed-form CRPS blew up to the hundreds). Ground truth is scipy.truncnorm.
# ---------------------------------------------------------------------------

# (μ, σ) spanning windy (μ/σ ≫ 0) through deeply-truncated (μ/σ ≪ 0).
_TN_PARAMS = [
    (8.0, 2.0),
    (4.0, 2.0),
    (1.0, 2.0),
    (0.0, 2.0),
    (-4.0, 2.0),
    (-12.0, 2.0),
    (-24.3, 2.0),
    (-50.0, 2.0),
]


def _tn_head_and_params():
    head = TruncatedNormalHead(hidden_dim=4)
    mus = torch.tensor([m for m, _ in _TN_PARAMS])
    sigmas = torch.tensor([s for _, s in _TN_PARAMS])
    params = {"mu": mus, "log_var": 2.0 * torch.log(sigmas)}
    return head, params, mus.numpy(), sigmas.numpy()


def test_truncnorm_median_matches_scipy():
    """Median matches scipy.truncnorm and is finite & ≥ 0 even for μ ≪ 0."""
    from scipy.stats import truncnorm

    head, params, mu, sigma = _tn_head_and_params()
    med = head.median(params).numpy()
    assert np.isfinite(med).all(), med
    assert (med >= 0).all(), f"truncated-normal median must be ≥ 0, got {med}"
    ref = truncnorm.median((0.0 - mu) / sigma, np.inf, loc=mu, scale=sigma)
    # float32 head params cap precision at ~1e-7; matches scipy to that level
    # across μ/σ ∈ [−25, 4] (the old code returned negatives below −4.75).
    np.testing.assert_allclose(med, ref, rtol=1e-5, atol=1e-5)


def test_truncnorm_samples_nonnegative_for_deep_truncation():
    """Inverse-CDF draws stay on the support [0, ∞) for μ ≪ 0."""
    torch.manual_seed(0)
    head, params, _, _ = _tn_head_and_params()
    samples = head.sample(params, n=500)
    assert torch.isfinite(samples).all()
    assert (samples >= 0).all(), "truncated-normal samples must be ≥ 0"


def test_truncnorm_crps_matches_montecarlo_and_is_finite():
    """Ensemble CRPS is finite and tracks a scipy Monte-Carlo reference even in
    the deeply-truncated regime where the closed form blew up (~474)."""
    from scipy.stats import truncnorm

    torch.manual_seed(0)
    head, params, mu, sigma = _tn_head_and_params()
    # Targets near each component's truncated mean (realistic wind values).
    a = (0.0 - mu) / sigma
    y = truncnorm.mean(a, np.inf, loc=mu, scale=sigma) + 0.5
    crps = head.crps(params, torch.tensor(y, dtype=torch.float32)).numpy()
    assert np.isfinite(crps).all()
    assert (crps < 50).all(), f"CRPS implausibly large (closed-form bug?): {crps}"

    rng = np.random.default_rng(0)
    ref = np.empty_like(crps)
    for i, (m, s, yi) in enumerate(zip(mu, sigma, y, strict=True)):
        X = truncnorm.rvs((0 - m) / s, np.inf, m, s, size=200000, random_state=rng)
        Xp = truncnorm.rvs((0 - m) / s, np.inf, m, s, size=200000, random_state=rng)
        ref[i] = np.abs(X - yi).mean() - 0.5 * np.abs(X - Xp).mean()
    # Ensemble estimator (n=200) — loose tolerance for its Monte-Carlo noise.
    # The point is "correct order of magnitude, not the closed form's ~474".
    np.testing.assert_allclose(crps, ref, rtol=0.15, atol=0.12)


def test_truncnorm_initialise_from_climatology():
    """Bias init gives (μ, σ) = (mean, std) at the first forward on zero input."""
    head = TruncatedNormalHead(hidden_dim=8)
    head.initialise_from_climatology(mean_target=4.0, std_target=2.5)
    with torch.no_grad():
        head.linear.weight.zero_()
    params = head(torch.zeros(1, 1, 8))
    assert params["mu"].item() == pytest.approx(4.0, abs=1e-6)
    assert params["log_var"].item() == pytest.approx(2.0 * math.log(2.5), abs=1e-6)
    with pytest.raises(ValueError):
        head.initialise_from_climatology(mean_target=0.0, std_target=1.0)


# ---------------------------------------------------------------------------
# Registry / API consistency
# ---------------------------------------------------------------------------


def test_registry_contents():
    """Only the two heads used by the paper are registered."""
    assert set(HEAD_REGISTRY) == {"gaussian", "truncated_normal"}


@pytest.mark.parametrize("name", list(HEAD_REGISTRY))
def test_build_head_returns_correct_type(name):
    head = build_head(name, hidden_dim=8)
    assert isinstance(head, LikelihoodHead)
    assert isinstance(head, HEAD_REGISTRY[name])


def test_build_head_rejects_unknown_distribution():
    with pytest.raises(ValueError):
        build_head("not_a_distribution", hidden_dim=8)


@pytest.mark.parametrize("name", list(HEAD_REGISTRY))
def test_param_names_match_forward_output(name):
    """A head's class-level ``param_names`` must match the dict keys it returns.

    The evaluator names the per-parameter arrays in ``test_predictions.npz``
    from ``param_names``.
    """
    head = build_head(name, hidden_dim=8)
    hidden = torch.randn(2, 3, 8)
    params = head(hidden)
    assert tuple(params.keys()) == head.param_names, (
        f"Head {name}: forward returned {list(params.keys())}, "
        f"class-level param_names is {head.param_names}"
    )


@pytest.mark.parametrize("name", list(HEAD_REGISTRY))
def test_state_dict_layout(name):
    """Each head's state_dict must contain ``linear.{weight,bias}`` only.

    Checkpoints on disk store ``heads.heads.<var>.linear.{weight,bias}``;
    a head adding submodules would break ``load_state_dict(strict=True)``.
    """
    head = build_head(name, hidden_dim=8)
    keys = set(head.state_dict())
    assert keys == {"linear.weight", "linear.bias"}, (
        f"Head {name}: unexpected state_dict keys {keys}"
    )


@pytest.mark.parametrize("name", list(HEAD_REGISTRY))
def test_head_methods_are_differentiable(name):
    """nll() flows gradients to the head's Linear parameters."""
    head = build_head(name, hidden_dim=8)
    hidden = torch.randn(2, 3, 8, requires_grad=False)
    params = head(hidden)
    target = torch.randn(2, 3)
    loss = head.nll(params, target)
    loss.backward()
    assert head.linear.weight.grad is not None
    assert torch.isfinite(head.linear.weight.grad).all()
