"""Tests for per-variable likelihood heads.

The most important test in this file is :func:`test_gaussian_nll_matches_legacy`:
it proves that ``GaussianHead.nll`` is bit-for-bit identical to the legacy
``gaussian_nll_loss`` from the pre-v4 ``convcnp.py``. This is the property
the migration script depends on — if it ever drifts, every legacy
checkpoint becomes silently incompatible with v4 inference.

The other tests cover:
  - ``param_names`` matching the actual dict keys produced by ``forward``;
  - analytical mean/median formulas for Weibull;
  - the Bernoulli-Gamma hurdle NLL on hand-computed fixtures;
  - the climatology-bias init helper;
  - registry / construction sanity checks.
"""
import math

import numpy as np
import pytest
import torch

from tessera_downscaling.model.heads import (
    HEAD_REGISTRY,
    BernoulliGammaHead,
    GaussianHead,
    LikelihoodHead,
    TruncatedNormalHead,
    WeibullHead,
    build_head,
)


# ---------------------------------------------------------------------------
# GaussianHead — parity with legacy gaussian_nll_loss
# ---------------------------------------------------------------------------

def _legacy_gaussian_nll_loss(pred_mean, pred_log_var, target, mask=None):
    """Verbatim reproduction of legacy convcnp.py's gaussian_nll_loss.

    Kept here as a fixture so the parity test isn't tied to whether the
    legacy function is still importable.
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
    """GaussianHead.nll == legacy gaussian_nll_loss bit-for-bit.

    Tested unmasked and masked, across multiple seeds. Any drift here is
    a SEVERE bug — it breaks the migration script's correctness claim.
    """
    torch.manual_seed(seed)
    B, N, H = 4, 16, 32
    head = GaussianHead(hidden_dim=H)
    hidden = torch.randn(B, N, H)
    params = head(hidden)
    target = torch.randn(B, N)

    new = head.nll(params, target)
    legacy = _legacy_gaussian_nll_loss(
        params["mu"], params["log_var"], target,
    )
    assert torch.equal(new, legacy), (
        f"Unmasked NLL mismatch (seed={seed}): new={new}, legacy={legacy}, "
        f"diff={(new - legacy).abs()}"
    )

    mask = torch.rand(B, N) > 0.3
    new_m = head.nll(params, target, mask=mask)
    legacy_m = _legacy_gaussian_nll_loss(
        params["mu"], params["log_var"], target, mask=mask,
    )
    assert torch.equal(new_m, legacy_m), (
        f"Masked NLL mismatch (seed={seed}): new={new_m}, legacy={legacy_m}"
    )


def test_gaussian_log_var_clamp_applied_inside_nll_only():
    """Raw log_var passes through forward unchanged; clamp is in nll/cdf/sample.

    This is the property that lets the migration round-trip work: a
    legacy state-dict's final-projection rows are *exactly* what the
    new heads emit as raw log_var, without any softplus or clamp
    rewriting them at parameter-emission time.
    """
    torch.manual_seed(0)
    head = GaussianHead(hidden_dim=8)
    # Force the linear's log-var output channel to extreme values.
    with torch.no_grad():
        head.linear.weight.zero_()
        head.linear.bias[0] = 100.0   # mu
        head.linear.bias[1] = 50.0    # log_var (well outside [-10, 10])
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
        z * (2 * stats.norm.cdf(z) - 1)
        + 2 * stats.norm.pdf(z)
        - 1.0 / np.sqrt(np.pi)
    )
    np.testing.assert_allclose(crps_new, crps_ref, rtol=1e-6)


# ---------------------------------------------------------------------------
# WeibullHead — analytical mean/median formulas
# ---------------------------------------------------------------------------

def test_weibull_mean_formula():
    """Mean of Weibull(k, λ) is λ · Γ(1 + 1/k)."""
    head = WeibullHead(hidden_dim=4)
    params = {
        "k":   torch.tensor([1.0, 2.0, 5.0]),
        "lam": torch.tensor([1.0, 3.0, 2.5]),
    }
    expected = torch.tensor([
        1.0 * math.gamma(1 + 1 / 1.0),
        3.0 * math.gamma(1 + 1 / 2.0),
        2.5 * math.gamma(1 + 1 / 5.0),
    ])
    actual = head.mean(params)
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)


def test_weibull_median_formula():
    """Median of Weibull(k, λ) is λ · (ln 2)^(1/k)."""
    head = WeibullHead(hidden_dim=4)
    params = {
        "k":   torch.tensor([1.0, 2.0, 5.0]),
        "lam": torch.tensor([1.0, 3.0, 2.5]),
    }
    expected = torch.tensor([
        1.0 * math.log(2) ** (1 / 1.0),
        3.0 * math.log(2) ** (1 / 2.0),
        2.5 * math.log(2) ** (1 / 5.0),
    ])
    actual = head.median(params)
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)


def test_weibull_forward_constraints():
    """Forward output for Weibull always has positive k and λ."""
    torch.manual_seed(0)
    head = WeibullHead(hidden_dim=8)
    # Push raw outputs to extreme negatives — softplus + floor must still
    # produce strictly positive parameters.
    hidden = torch.full((4, 5, 8), -100.0)
    params = head(hidden)
    assert (params["k"] > 0).all()
    assert (params["lam"] > 0).all()


def test_weibull_nll_handles_zero_target():
    """y == 0 is undefined for Weibull; defensive clamp keeps NLL finite."""
    torch.manual_seed(0)
    head = WeibullHead(hidden_dim=4)
    hidden = torch.randn(2, 3, 4)
    params = head(hidden)
    target = torch.zeros(2, 3)
    nll = head.nll(params, target)
    assert torch.isfinite(nll), "Weibull NLL must stay finite at y=0"


# ---------------------------------------------------------------------------
# TruncatedNormalHead — numerical stability in the deeply-truncated regime
#
# Regression tests for the μ ≪ 0 bug: the model legitimately predicts very
# negative μ for calm wind, where the old median/sample/CRPS code broke (the
# (1e-6, 1−1e-6) icdf clamp gave negative medians for μ/σ < −4.75, and the
# closed-form CRPS blew up to the hundreds). Ground truth is scipy.truncnorm.
# ---------------------------------------------------------------------------

# (μ, σ) spanning windy (μ/σ ≫ 0) through deeply-truncated (μ/σ ≪ 0).
_TN_PARAMS = [(8.0, 2.0), (4.0, 2.0), (1.0, 2.0), (0.0, 2.0),
              (-4.0, 2.0), (-12.0, 2.0), (-24.3, 2.0), (-50.0, 2.0)]


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
    for i, (m, s, yi) in enumerate(zip(mu, sigma, y)):
        X = truncnorm.rvs((0 - m) / s, np.inf, m, s, size=200000, random_state=rng)
        Xp = truncnorm.rvs((0 - m) / s, np.inf, m, s, size=200000, random_state=rng)
        ref[i] = np.abs(X - yi).mean() - 0.5 * np.abs(X - Xp).mean()
    # Ensemble estimator (n=200) — loose tolerance for its Monte-Carlo noise.
    # The point is "correct order of magnitude, not the closed form's ~474".
    np.testing.assert_allclose(crps, ref, rtol=0.15, atol=0.12)


# ---------------------------------------------------------------------------
# BernoulliGammaHead
# ---------------------------------------------------------------------------

def test_bernoulli_gamma_nll_dry_branch():
    """For y == 0 the NLL reduces to −log(1 − ρ)."""
    head = BernoulliGammaHead(hidden_dim=4)
    # Pin parameters to known values.
    rho_value = 0.2
    params = {
        "rho":   torch.full((1, 1), rho_value),
        "alpha": torch.full((1, 1), 1.5),
        "beta":  torch.full((1, 1), 0.5),
    }
    target = torch.zeros(1, 1)
    nll = head.nll(params, target)
    expected = -math.log(1 - rho_value)
    torch.testing.assert_close(
        nll, torch.tensor(expected), rtol=1e-6, atol=1e-6,
    )


def test_bernoulli_gamma_nll_wet_branch():
    """For y > 0 the NLL is −[log ρ + log Gamma(y; α, β)]."""
    head = BernoulliGammaHead(hidden_dim=4)
    rho_value, alpha_value, beta_value, y_value = 0.7, 2.0, 0.5, 3.0
    params = {
        "rho":   torch.full((1, 1), rho_value),
        "alpha": torch.full((1, 1), alpha_value),
        "beta":  torch.full((1, 1), beta_value),
    }
    target = torch.full((1, 1), y_value)
    nll = head.nll(params, target)

    # Reference: Gamma(α, β) log-density at y, plus log ρ, both negated.
    gamma_dist = torch.distributions.Gamma(
        concentration=torch.tensor(alpha_value),
        rate=torch.tensor(beta_value),
    )
    ref = -(math.log(rho_value) + gamma_dist.log_prob(torch.tensor(y_value)).item())
    torch.testing.assert_close(
        nll, torch.tensor(ref), rtol=1e-6, atol=1e-6,
    )


def test_bernoulli_gamma_initialise_rho_bias_from_climatology():
    """ρ-bias init sets bias[0] = logit(p_wet)."""
    head = BernoulliGammaHead(hidden_dim=8)
    head.initialise_rho_bias_from_climatology(0.1)
    assert head.linear.bias[0].item() == pytest.approx(math.log(0.1 / 0.9), abs=1e-6)

    # ρ at zero input should equal p_wet (bias-only contribution through sigmoid).
    zero_hidden = torch.zeros(1, 1, 8)
    # Set the rho-channel weights to zero so only the bias matters.
    with torch.no_grad():
        head.linear.weight[0].zero_()
    params = head(zero_hidden)
    assert params["rho"].item() == pytest.approx(0.1, abs=1e-5)


def test_bernoulli_gamma_init_rejects_invalid_p_wet():
    """p_wet must be in (0, 1) — boundary values would produce ±inf logit."""
    head = BernoulliGammaHead(hidden_dim=4)
    with pytest.raises(ValueError):
        head.initialise_rho_bias_from_climatology(0.0)
    with pytest.raises(ValueError):
        head.initialise_rho_bias_from_climatology(1.0)
    with pytest.raises(ValueError):
        head.initialise_rho_bias_from_climatology(-0.1)


def test_bernoulli_gamma_forward_constraints():
    """B-G forward emits ρ ∈ (0, 1), α > 0, β > 0."""
    torch.manual_seed(0)
    head = BernoulliGammaHead(hidden_dim=8)
    hidden = torch.randn(4, 6, 8) * 100  # extreme inputs
    params = head(hidden)
    assert (params["rho"] > 0).all() and (params["rho"] < 1).all()
    assert (params["alpha"] > 0).all()
    assert (params["beta"] > 0).all()


# ---------------------------------------------------------------------------
# Registry / API consistency
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", list(HEAD_REGISTRY))
def test_build_head_returns_correct_type(name):
    head = build_head(name, hidden_dim=8)
    assert isinstance(head, LikelihoodHead)
    # Most registry values are the head class itself; the non-negative
    # generative variant is a functools.partial preset, so only type-check
    # entries that are actually classes.
    factory = HEAD_REGISTRY[name]
    if isinstance(factory, type):
        assert isinstance(head, factory)


def test_build_head_rejects_unknown_distribution():
    with pytest.raises(ValueError):
        build_head("not_a_distribution", hidden_dim=8)


@pytest.mark.parametrize("name", list(HEAD_REGISTRY))
def test_param_names_match_forward_output(name):
    """A head's class-level ``param_names`` must match the dict keys it returns.

    This is what the heads dispatcher and the migration logic both rely
    on for naming the per-parameter arrays in test_predictions.npz and
    in the migrated state-dict.
    """
    head = build_head(name, hidden_dim=8)
    hidden = torch.randn(2, 3, 8)
    params = head(hidden)
    assert tuple(params.keys()) == head.param_names, (
        f"Head {name}: forward returned {list(params.keys())}, "
        f"class-level param_names is {head.param_names}"
    )


# Parametric heads share the single-Linear state-dict layout the migration
# script depends on. The implicit generative heads do not (they own a
# generator MLP) — and they are brand-new with no legacy checkpoint to
# migrate, so every ``generative*`` variant is excluded from this contract.
_PARAMETRIC_HEADS = [n for n in HEAD_REGISTRY if not n.startswith("generative")]


@pytest.mark.parametrize("name", _PARAMETRIC_HEADS)
def test_state_dict_layout_for_migration(name):
    """Each parametric head's state_dict must contain ``linear.{weight,bias}`` only.

    The migration script relies on this exact layout (it writes
    ``heads.heads.<var>.linear.weight``). A new head adding additional
    submodules would require a migration-script update too — which is exactly
    why the generative head (which has no legacy checkpoint) is excluded here.
    """
    head = build_head(name, hidden_dim=8)
    keys = set(head.state_dict())
    assert keys == {"linear.weight", "linear.bias"}, (
        f"Head {name}: unexpected state_dict keys {keys}"
    )


@pytest.mark.parametrize("name", _PARAMETRIC_HEADS)
def test_head_methods_are_differentiable(name):
    """nll() flows gradients to the parametric head's Linear parameters."""
    head = build_head(name, hidden_dim=8)
    hidden = torch.randn(2, 3, 8, requires_grad=False)
    params = head(hidden)
    # Targets must be valid for each distribution: positive for
    # Weibull, non-negative for B-G, anything for Gaussian.
    if name == "weibull":
        target = torch.rand(2, 3) + 0.1
    elif name == "bernoulli_gamma":
        target = torch.where(
            torch.rand(2, 3) > 0.5, torch.rand(2, 3) + 0.1, torch.zeros(2, 3),
        )
    else:
        target = torch.randn(2, 3)
    loss = head.nll(params, target)
    loss.backward()
    assert head.linear.weight.grad is not None
    assert torch.isfinite(head.linear.weight.grad).all()


# ---------------------------------------------------------------------------
# GenerativeHead — implicit model trained by CRPS minimisation
# ---------------------------------------------------------------------------

def test_generative_forward_returns_hidden_passthrough():
    """forward() returns the hidden state under the 'hidden' key, unchanged.

    The hidden state *is* the per-point distribution descriptor (the
    generative head's analogue of (mu, log_var)); sampling happens later.
    """
    from tessera_downscaling.model.heads import GenerativeHead
    head = GenerativeHead(hidden_dim=8)
    hidden = torch.randn(2, 3, 8)
    params = head(hidden)
    assert tuple(params.keys()) == head.param_names == ("hidden",)
    assert torch.equal(params["hidden"], hidden)


def test_generative_nll_raises():
    """An implicit head has no tractable density; nll() must refuse."""
    from tessera_downscaling.model.heads import GenerativeHead
    head = GenerativeHead(hidden_dim=8)
    params = head(torch.randn(2, 3, 8))
    assert head.has_density is False
    with pytest.raises(NotImplementedError):
        head.nll(params, torch.randn(2, 3))


def test_generative_nonneg_samples_are_nonnegative():
    """The non-negative variant (softplus output) never emits samples < 0.

    For wind/precip the generator must respect the variable's support; without
    the constraint it can place mass below 0, costing CRPS sharpness vs the
    parametric heads (TruncNormal/Weibull/Gamma) that build the support in.
    """
    from tessera_downscaling.model.heads import GenerativeHead, build_head
    torch.manual_seed(0)
    head = build_head("generative_nonneg", hidden_dim=8)
    assert isinstance(head, GenerativeHead) and head.nonneg is True
    hidden = torch.randn(4, 6, 8) * 3.0   # spread out so an unconstrained head would go negative
    samples = head.sample({"hidden": hidden}, n=2000)
    assert (samples >= 0).all(), "generative_nonneg must keep samples >= 0"
    # Sanity: the plain variant is unconstrained and does dip below 0.
    plain = GenerativeHead(hidden_dim=8)
    assert (plain.sample({"hidden": hidden}, n=2000) < 0).any()


def test_generative_crps_shape_and_gradient():
    """crps() returns per-observation values and flows gradients to G.

    The reparameterisation (z ~ randn, differentiable G) is what makes the
    ensemble CRPS trainable — this asserts the gradient path is intact.
    """
    from tessera_downscaling.model.heads import GenerativeHead
    torch.manual_seed(0)
    head = GenerativeHead(hidden_dim=8, n_train_samples=16)
    head.train()
    hidden = torch.randn(2, 3, 8)
    params = head(hidden)
    target = torch.randn(2, 3)
    crps = head.crps(params, target)
    assert crps.shape == target.shape
    crps.mean().backward()
    first_linear = head.generator[0]
    assert first_linear.weight.grad is not None
    assert torch.isfinite(first_linear.weight.grad).all()
    assert first_linear.weight.grad.abs().sum() > 0


def test_generative_crps_fair_estimator_matches_brute_force():
    """The O(M log M) sorted spread term equals the explicit fair pairwise sum.

    Guards the order-statistic identity used in crps() against the definition
        CRPS = mean_m |X_m - y| - 1/(2 M(M-1)) sum_{i!=j} |X_i - X_j|.
    A regression here would mean a biased spread term (the head would learn to
    under-disperse).
    """
    from tessera_downscaling.model.heads import GenerativeHead
    torch.manual_seed(0)
    head = GenerativeHead(hidden_dim=4, n_eval_samples=64)
    head.eval()  # use n_eval_samples
    hidden = torch.randn(5, 4)
    target = torch.randn(5)
    # Fix the noise so head.crps and the brute-force reference see the same
    # ensemble: draw samples once, reimplement both terms by hand.
    n = head.n_eval_samples
    torch.manual_seed(123)
    samples = head._generate(hidden, n)            # (n, 5)
    abs_err = (samples - target.unsqueeze(0)).abs().mean(dim=0)
    diffs = (samples.unsqueeze(0) - samples.unsqueeze(1)).abs()  # (n, n, 5)
    spread = diffs.sum(dim=(0, 1)) / (n * (n - 1))   # 1/(n(n-1)) sum_{i,j} (i=j adds 0)
    ref = abs_err - 0.5 * spread

    torch.manual_seed(123)
    got = head.crps({"hidden": hidden}, target)
    torch.testing.assert_close(got, ref, rtol=1e-5, atol=1e-6)

# NOTE: the end-to-end "recover a skewed distribution by CRPS training" test
# was intentionally removed — it ran a 400-step Adam loop, which is real
# training and must not execute on an HPC login node (PyTorch fans out across
# all cores and destabilises the shared node). The CRPS gradient path is
# already covered cheaply by test_generative_crps_shape_and_gradient and the
# fair-estimator correctness by test_generative_crps_fair_estimator_matches_
# brute_force. Verify end-to-end recovery as part of a real training run on a
# compute node, not in the unit suite.