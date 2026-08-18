"""Per-variable likelihood heads for the multi-task ConvCNP.

Each ``LikelihoodHead`` projects a shared ``(batch, n_targets, hidden_dim)``
hidden state to its own predictive-distribution parameters and provides
the per-variable NLL, mean/median, CDF (for PIT calibration), CRPS, and
sampling. The model owns a ``LikelihoodHeadDict`` (one head per target
variable) and dispatches per variable, so distribution-specific logic is
encapsulated in the head class rather than spread across the loss
function and the evaluator.

Distribution choices and prior work
-----------------------------------

  - **Gaussian head** (default, ``t2m``).
    Heteroscedastic with ``log_var`` parameterisation, *deliberately*
    matching the legacy ``gaussian_nll_loss`` in ``convcnp.py``: raw
    network output is treated as ``log σ²``, with the same ``[-10, 10]``
    clamp applied inside the NLL. This makes the new path bit-for-bit
    equivalent to the legacy code on the same input + weights, which is
    what the migration script's round-trip check relies on. We do *not*
    use Vaughan's softplus → σ scheme even though it is the convCNPClimate
    reference, because doing so would silently invalidate every legacy
    checkpoint we want to reload.

  - **Weibull head** (``wind`` / ``wind_mean``).
    WMO-recommended for surface wind speed; physically motivated as the
    magnitude of two-component Gaussian wind components. Two parameters
    ``(k, λ)`` (shape, scale), both positive via softplus + 0.01 floor.
    ``mean = λ · Γ(1 + 1/k)``, ``median = λ · (ln 2)^(1/k)``; for skewed
    distributions these differ and the metric system reports both
    (``*_mae_at_median`` minimises expected MAE; ``*_rmse_at_mean``
    minimises expected MSE).

  - **Truncated Normal head** (alternative for ``wind``).
    Left-truncated Gaussian at 0, support [0, ∞). Storage-compatible with
    the Gaussian head (same ``(μ, log_var)`` parameterisation) so the two
    can be diagnostically compared without re-running. Field-standard
    probabilistic head for short-term wind in NWP post-processing
    (non-homogeneous Gaussian regression / EMOS; Thorarinsdottir &
    Gneiting 2010). Differs from plain Gaussian only when σ is comparable
    to μ — i.e. in calm regimes near the truncation point.

  - **Bernoulli-Gamma hurdle head** (``precip``).
    The standard precipitation hurdle from Cannon (2008) and Vaughan et
    al. (2022). Three parameters ``(ρ, α, β)`` — rain probability
    (sigmoid + clamp), Gamma shape/rate (softplus + floor). NLL gates
    the Gamma term by ``r = 1{y > 0}`` so the dry-day branch reduces to
    ``log(1 - ρ)``. Matches convCNPClimate's ``loss_functions.py``
    rather than the slightly imprecise paper Eq. 10.

Numerical guards (lifted with adaptation from convCNPClimate's
``final_layers.py``):

  - Softplus floor of ``0.01`` prevents positive parameters from
    collapsing toward zero (which would diverge the Gamma density at
    small ``y``).
  - Hard clamps on parameter values prevent gradient blow-up at the
    optimisation boundary: ``ρ ∈ [1e-5, 1−1e-5]``, ``α/β ∈ [1e-5, 1e5]``,
    ``k/λ ∈ [1e-5, 1e3]``.
  - ``log_var`` is clamped to ``[-10, 10]`` *inside* the Gaussian NLL
    (matching the legacy ``gaussian_nll_loss`` exactly), not at parameter
    emission time. The raw value is preserved in the parameters dict so
    a downstream consumer who wants to inspect "what the network actually
    output" can do so.
  - NaN target handling is the responsibility of the caller (mask in the
    loss layer); heads operate on dense tensors with an optional mask.
"""
from __future__ import annotations

import functools
import math
from abc import ABC, abstractmethod
from typing import Callable, Final

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Numerical guards
# ---------------------------------------------------------------------------

# Floor added to softplus output so positive parameters never reach zero.
# Vaughan's code uses 0.01; we keep the same value for direct parity.
_POSITIVE_FLOOR: Final[float] = 0.01

# Clamp ranges. Keep one place to adjust them so the training loss and
# the evaluator don't drift apart.
_RHO_MIN: Final[float] = 1e-5
_RHO_MAX: Final[float] = 1.0 - 1e-5
_POSITIVE_MIN: Final[float] = 1e-5
_POSITIVE_MAX: Final[float] = 1e5
_SCALE_MAX: Final[float] = 1e3   # tighter clamp for k / λ which shouldn't
                                 # realistically exceed O(100).

# Legacy log_var clamp: matches ``gaussian_nll_loss`` exactly. This is the
# *training-time* numerical-stability clamp, not a parameter-emission
# constraint — the raw log_var is left in the params dict for inspection,
# and the clamp is applied where it matters (NLL, σ derivation for CDF,
# CRPS, sampling).
_LOG_VAR_MIN: Final[float] = -10.0
_LOG_VAR_MAX: Final[float] = 10.0


def _softplus_floor(x: torch.Tensor) -> torch.Tensor:
    """Numerically robust positive activation with a floor."""
    return _POSITIVE_FLOOR + F.softplus(x)


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class LikelihoodHead(nn.Module, ABC):
    """Per-variable likelihood head.

    Each subclass owns a single ``Linear`` projecting the shared hidden
    state to its distribution's parameters (raw, unconstrained). The
    :meth:`forward` method returns a dict of named, constrained
    distribution parameters; the loss / evaluation methods consume that
    dict.

    Subclasses declare ``param_names`` and ``n_params`` at class level.
    Both are used by sanity checks and serialisation; they should reflect
    the exact dict keys returned by :meth:`forward`.
    """

    n_params: int
    param_names: tuple[str, ...]

    # Whether this head defines a tractable likelihood (closed-form density).
    # Parametric heads do; the implicit GenerativeHead does not, and is
    # trained by scoring-rule minimisation instead. The training loop refuses
    # ``loss_function='nll'`` for a head with ``has_density = False`` (the
    # head's ``nll`` raises), and the evaluator skips the NLL metric for it.
    has_density: bool = True

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.linear = nn.Linear(hidden_dim, self.n_params)

    # ---- The interface every head implements ------------------------------

    @abstractmethod
    def forward(self, hidden: torch.Tensor) -> dict[str, torch.Tensor]:
        """Project hidden state to constrained distribution parameters.

        Args:
            hidden: tensor of shape ``(..., hidden_dim)``.

        Returns:
            dict mapping parameter name (matching ``self.param_names``)
            to a tensor of the same leading shape.
        """

    @abstractmethod
    def nll(
        self,
        params: dict[str, torch.Tensor],
        target: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Mean negative log-likelihood under this head.

        ``mask`` is an optional bool tensor of the same shape as
        ``target``; entries where ``mask`` is False are ignored. If
        ``mask`` is None, every entry contributes.
        """

    @abstractmethod
    def mean(self, params: dict[str, torch.Tensor]) -> torch.Tensor:
        """Predictive mean E[Y | params]."""

    @abstractmethod
    def median(self, params: dict[str, torch.Tensor]) -> torch.Tensor:
        """Predictive median Median(Y | params).

        Preferred over ``mean`` for skewed distributions (Weibull, Gamma)
        when reporting MAE — the median minimises expected MAE, the mean
        minimises expected MSE.
        """

    @abstractmethod
    def cdf(
        self,
        params: dict[str, torch.Tensor],
        target: torch.Tensor,
    ) -> torch.Tensor:
        """CDF F(target | params). The unconditional CDF.

        For mixed-discrete-continuous heads (B-G hurdle) this includes
        the point mass at zero. For pure continuous distributions
        (Gaussian, Weibull) this is the only CDF there is.

        Used for forecast comparison (e.g. Brier-style proper scoring)
        but NOT for PIT calibration tests on hurdle distributions —
        see :meth:`pit_cdf` and :meth:`pit_mask` for that.
        """

    def pit_cdf(
        self,
        params: dict[str, torch.Tensor],
        target: torch.Tensor,
    ) -> torch.Tensor:
        """CDF used for PIT calibration tests.

        For continuous distributions this is identical to :meth:`cdf`.
        For hurdle distributions (Bernoulli-Gamma) this is the
        *conditional* CDF given the continuous branch (e.g. F_Gamma)
        rather than the mixed CDF — including the point mass at zero
        in a uniform-PIT test biases the χ² statistic by the (1−ρ)
        atom. Vaughan et al. (2022, Fig. 10) compute PIT on wet days
        using the conditional Gamma CDF for exactly this reason.
        """
        return self.cdf(params, target)

    def pit_mask(
        self,
        target: torch.Tensor,
    ) -> torch.Tensor | None:
        """Boolean mask selecting observations to include in the PIT test.

        Returns ``None`` to indicate "all observations". The B-G head
        overrides this to mask out dry days (where the conditional
        Gamma CDF is undefined/uninformative).
        """
        return None

    @abstractmethod
    def crps(
        self,
        params: dict[str, torch.Tensor],
        target: torch.Tensor,
    ) -> torch.Tensor:
        """Continuous Ranked Probability Score per observation.

        Closed-form where available; ensemble approximation where not.
        Always returns a tensor of the same shape as ``target``.
        """

    def sample(
        self,
        params: dict[str, torch.Tensor],
        n: int = 1,
    ) -> torch.Tensor:
        """Draw ``n`` samples per (..., target) entry from the predictive
        distribution.

        Returns shape ``(n, *target_shape)``. Default implementation
        builds a ``torch.distributions`` object via :meth:`_distribution`
        if the subclass provides one; otherwise subclass must override.
        """
        dist = self._distribution(params)
        if dist is None:
            raise NotImplementedError(
                f"{type(self).__name__} must override sample() — no "
                "torch.distributions analogue available."
            )
        return dist.sample((n,))

    def _distribution(
        self, params: dict[str, torch.Tensor]
    ) -> torch.distributions.Distribution | None:
        """Optional helper: return a torch.distributions object.

        Subclasses that have a clean torch.distributions analogue can
        return one here and reuse :meth:`sample` and the default
        log_prob in :meth:`nll`. Subclasses with custom likelihoods
        (e.g. Bernoulli-Gamma) return None.
        """
        return None


# ---------------------------------------------------------------------------
# Helper: ensemble-CRPS for distributions without a closed-form expression.
# Used by Weibull and Bernoulli-Gamma.
# ---------------------------------------------------------------------------

def _ensemble_crps(
    head: "LikelihoodHead",
    params: dict[str, torch.Tensor],
    target: torch.Tensor,
    n_samples: int = 200,
) -> torch.Tensor:
    """Unbiased ensemble estimator of CRPS.

        CRPS(F, y) ≈ E |X − y| − ½ E |X − X'|

    where X, X' are i.i.d. from the predictive distribution. The
    second-term expectation is approximated via a single random
    permutation (cheap, unbiased, slightly noisier than a full N²
    pairing but adequate for our N = 200 default).
    """
    samples = head.sample(params, n=n_samples)        # (n, *target_shape)
    term1 = (samples - target.unsqueeze(0)).abs().mean(dim=0)
    perm = torch.randperm(n_samples, device=samples.device)
    term2 = 0.5 * (samples - samples[perm]).abs().mean(dim=0)
    return term1 - term2


# ---------------------------------------------------------------------------
# Gaussian (log_var parameterisation — matches legacy gaussian_nll_loss)
# ---------------------------------------------------------------------------

class GaussianHead(LikelihoodHead):
    """Heteroscedastic Gaussian, ``params = {"mu": ..., "log_var": ...}``.

    ``μ`` is unconstrained; ``log_var`` is the *raw* network output (also
    unconstrained, per the legacy parameterisation). The
    numerical-stability clamp ``log_var ∈ [-10, 10]`` is applied inside
    consumer methods (``nll``, ``cdf``, ``crps``, ``sample``), exactly
    matching ``gaussian_nll_loss`` from the legacy ``convcnp.py``. This
    is the property that lets the migration script reload a legacy
    checkpoint into the new code path with bit-for-bit identical
    predictions.
    """

    n_params = 2
    param_names = ("mu", "log_var")

    def forward(self, hidden: torch.Tensor) -> dict[str, torch.Tensor]:
        raw = self.linear(hidden)
        mu = raw[..., 0]
        log_var = raw[..., 1]   # unconstrained — matches legacy
        return {"mu": mu, "log_var": log_var}

    def _sigma(self, params: dict[str, torch.Tensor]) -> torch.Tensor:
        """σ = exp(log_var/2) with the legacy stability clamp applied."""
        log_var_clamped = params["log_var"].clamp(_LOG_VAR_MIN, _LOG_VAR_MAX)
        return torch.exp(0.5 * log_var_clamped)

    def _distribution(self, params):
        return torch.distributions.Normal(
            loc=params["mu"], scale=self._sigma(params)
        )

    def nll(self, params, target, mask=None):
        # Reproduces legacy gaussian_nll_loss exactly (bit-for-bit), via
        # the explicit formula
        #   0.5 * (log_var + (y - μ)² / exp(log_var) + log(2π))
        # — NOT via torch.distributions.Normal(μ, σ).log_prob, which
        # computes the density through a different sequence of float
        # operations (((y-μ)/σ)² vs (y-μ)²/exp(log_var)) and differs by
        # ~1 ULP in some configurations. Bit-for-bit parity is the
        # property the migration script's correctness claim depends on.
        log_var_clamped = params["log_var"].clamp(_LOG_VAR_MIN, _LOG_VAR_MAX)
        nll = 0.5 * (
            log_var_clamped
            + (target - params["mu"]) ** 2 / torch.exp(log_var_clamped)
            + math.log(2 * math.pi)
        )
        if mask is not None:
            mask_f = mask.to(nll.dtype)
            return (nll * mask_f).sum() / mask_f.sum().clamp(min=1.0)
        return nll.mean()

    def mean(self, params):
        return params["mu"]

    def median(self, params):
        return params["mu"]   # Gaussian: mean == median.

    def cdf(self, params, target):
        return self._distribution(params).cdf(target)

    def crps(self, params, target):
        # Closed-form CRPS for Gaussian:
        #   CRPS(N(μ, σ²), y) = σ · [ z·(2Φ(z) − 1) + 2φ(z) − 1/√π ]
        # where z = (y − μ) / σ.
        mu = params["mu"]
        sigma = self._sigma(params)
        z = (target - mu) / sigma
        normal = torch.distributions.Normal(
            loc=torch.zeros_like(z), scale=torch.ones_like(z)
        )
        cdf_z = normal.cdf(z)
        pdf_z = torch.exp(normal.log_prob(z))
        return sigma * (z * (2 * cdf_z - 1) + 2 * pdf_z - 1.0 / math.sqrt(math.pi))


# ---------------------------------------------------------------------------
# Weibull
# ---------------------------------------------------------------------------

class WeibullHead(LikelihoodHead):
    """Two-parameter Weibull(k, λ), ``params = {"k": ..., "lam": ...}``.

    PDF: ``f(y; k, λ) = (k/λ)·(y/λ)^(k−1) · exp(−(y/λ)^k)`` for ``y > 0``.
    Mean: ``λ · Γ(1 + 1/k)``. Median: ``λ · (ln 2)^(1/k)``.

    Both parameters positive via softplus + 0.01 floor. Note PyTorch's
    ``torch.distributions.Weibull`` parameterises as
    ``(scale, concentration) = (λ, k)`` — scale first.
    """

    n_params = 2
    param_names = ("k", "lam")

    def forward(self, hidden):
        raw = self.linear(hidden)
        k = _softplus_floor(raw[..., 0]).clamp(_POSITIVE_MIN, _SCALE_MAX)
        lam = _softplus_floor(raw[..., 1]).clamp(_POSITIVE_MIN, _SCALE_MAX)
        return {"k": k, "lam": lam}

    def _distribution(self, params):
        return torch.distributions.Weibull(
            scale=params["lam"], concentration=params["k"]
        )

    def nll(self, params, target, mask=None):
        # Weibull is undefined at y == 0. GHCNh wind reports of exactly
        # 0.0 m/s do exist (anemometer thresholds, calm conditions); the
        # caller is expected to either include these in the mask
        # (clamping them to a small ε) or exclude them entirely. We
        # apply a defensive clamp here for robustness.
        # Mask form matches legacy gaussian_nll_loss for bit-for-bit
        # consistency with the existing per-variable NLL aggregation.
        target_safe = target.clamp(min=_POSITIVE_MIN)
        log_prob = self._distribution(params).log_prob(target_safe)
        nll = -log_prob
        if mask is not None:
            mask_f = mask.to(nll.dtype)
            return (nll * mask_f).sum() / mask_f.sum().clamp(min=1.0)
        return nll.mean()

    def mean(self, params):
        # E[Weibull(k, λ)] = λ · Γ(1 + 1/k)
        k = params["k"]
        lam = params["lam"]
        return lam * torch.exp(torch.lgamma(1.0 + 1.0 / k))

    def median(self, params):
        # Median(Weibull(k, λ)) = λ · (ln 2)^(1/k)
        k = params["k"]
        lam = params["lam"]
        return lam * torch.pow(
            torch.tensor(math.log(2.0), device=k.device, dtype=k.dtype),
            1.0 / k,
        )

    def cdf(self, params, target):
        return self._distribution(params).cdf(target.clamp(min=0.0))

    def crps(self, params, target):
        # No published closed-form for Weibull CRPS (unlike Gaussian,
        # Gamma, GEV which appear in scoringRules). Use the unbiased
        # ensemble estimator.
        return _ensemble_crps(self, params, target, n_samples=200)


# ---------------------------------------------------------------------------
# Left-truncated Normal at 0 (support [0, ∞))
# ---------------------------------------------------------------------------

class TruncatedNormalHead(LikelihoodHead):
    """Left-truncated Gaussian at 0, ``params = {"mu": ..., "log_var": ...}``.

    Parameterisation deliberately matches :class:`GaussianHead` so the raw
    parameter tensors are storage-compatible — a TruncNormal checkpoint can
    be loaded into a Gaussian head (and vice versa) for diagnostic
    comparisons, only the consumer methods differ. The truncation point is
    fixed at zero because this head exists specifically for non-negative
    wind speed; making it a configurable parameter would add API surface
    that no caller uses.

    Density on [0, ∞):
        f(y; μ, σ) = (1/σ) · φ((y − μ)/σ) / Z,    Z = 1 − Φ(−μ/σ) = Φ(μ/σ).

    Closed-form moments (α := −μ/σ):
        mean   = μ + σ · φ(α) / Z
        median = μ + σ · Φ⁻¹(0.5 + 0.5·Φ(α))
        CDF(y) = (Φ((y − μ)/σ) − Φ(α)) / Z         for y ≥ 0; 0 otherwise.

    Justification: TruncNormal is the field-standard probabilistic head for
    short-term wind forecasting in NWP post-processing (non-homogeneous
    Gaussian regression / EMOS; Thorarinsdottir & Gneiting 2010). Versus
    plain Gaussian, the gain shows up where σ is comparable to μ — i.e.
    near calm. For typical wind regimes (μ ≫ σ) the two distributions are
    near-indistinguishable; the literature reports ~1–3 % NLL / CRPS gain.

    Numerical strategy: ``log Z`` is computed via ``torch.special.log_ndtr``
    (log of the standard-normal CDF), which is stable in both tails. Quantiles
    (``median`` / ``sample``) go through :meth:`_quantile`, which evaluates the
    inverse CDF on the *lower-tail* probability ``Φ(μ/σ)·(1−p)`` in float64 and
    floors the result at 0 — this stays correct for μ ≪ 0, where the earlier
    "``Φ⁻¹(0.5 + 0.5·Φ(−μ/σ))`` then clamp to (1e-6, 1−1e-6)" form saturated and
    returned *negative* medians for μ/σ < −4.75. CRPS uses the unbiased ensemble
    estimator (like the Weibull / Bernoulli-Gamma heads): the closed form for the
    lower-truncated normal suffers catastrophic cancellation once μ/σ ≲ −4 (the
    ``1/Z`` amplification meets ``2Φ(y_z)−2 → 0``), which the model routinely hits
    on calm wind, so the analytic version is unsafe here despite being correct on
    paper.
    """

    n_params = 2
    param_names = ("mu", "log_var")

    def forward(self, hidden):
        raw = self.linear(hidden)
        mu = raw[..., 0]
        log_var = raw[..., 1]   # storage-compatible with GaussianHead.
        return {"mu": mu, "log_var": log_var}

    def _sigma(self, params):
        """σ = exp(log_var/2) with the legacy stability clamp applied. The
        clamp is intentionally the same one ``GaussianHead`` uses — keeps
        the two heads on the same gradient surface near initialisation."""
        log_var_clamped = params["log_var"].clamp(_LOG_VAR_MIN, _LOG_VAR_MAX)
        return torch.exp(0.5 * log_var_clamped)

    def initialise_from_climatology(
        self,
        mean_target: float,
        std_target: float,
    ) -> None:
        """Set the ``Linear`` biases so initial (μ, σ) ≈ (mean, std) of
        the positive-target climatology.

        For wind this is ``targets[targets > 0].mean()`` and ``.std()`` over
        the train split. The truncation pulls the *implied* mean of the
        predictive distribution slightly upward from μ, but the correction
        is small when μ ≫ σ (typical wind), and the network absorbs the
        residual within a few epochs — exactly the pattern documented for
        Weibull init.

        Idempotent; call again with different statistics to re-initialise.

        Args:
            mean_target: empirical mean of positive target values.
            std_target:  empirical std of positive target values.
        """
        if mean_target <= 0 or std_target <= 0:
            raise ValueError(
                f"mean_target and std_target must be > 0; "
                f"got mean={mean_target}, std={std_target}"
            )
        with torch.no_grad():
            # Linear is initialised with bias = 0; setting it directly
            # gives (μ, log_var) = (mean, 2·log(std)) at the first forward.
            self.linear.bias[0] = float(mean_target)
            self.linear.bias[1] = 2.0 * math.log(float(std_target))

    def nll(self, params, target, mask=None):
        # log f_trunc(y) = log φ((y − μ)/σ) − log σ − log Z
        # for y ≥ 0; the truncated normal has zero density at y < 0.
        # Mirror the Weibull defensive clamp for targets at exactly zero
        # (which exist in GHCNh — anemometer threshold reports).
        target_safe = target.clamp(min=_POSITIVE_MIN)
        mu = params["mu"]
        sigma = self._sigma(params)
        z = (target_safe - mu) / sigma
        log_phi = -0.5 * z * z - 0.5 * math.log(2.0 * math.pi)
        log_sigma = torch.log(sigma)
        log_Z = torch.special.log_ndtr(mu / sigma)   # = log Φ(μ/σ) = log(1 − Φ(α))
        nll = -(log_phi - log_sigma - log_Z)
        if mask is not None:
            mask_f = mask.to(nll.dtype)
            return (nll * mask_f).sum() / mask_f.sum().clamp(min=1.0)
        return nll.mean()

    def mean(self, params):
        # E[Y | Y > 0] = μ + σ · φ(α) / Z
        # Work in log-domain for the ratio to avoid catastrophic loss in
        # the calm regime where Z is small.
        mu = params["mu"]
        sigma = self._sigma(params)
        alpha = -mu / sigma
        log_phi_alpha = -0.5 * alpha * alpha - 0.5 * math.log(2.0 * math.pi)
        log_Z = torch.special.log_ndtr(mu / sigma)
        return mu + sigma * torch.exp(log_phi_alpha - log_Z)

    def _quantile(self, params, p):
        """``p``-quantile of the truncated normal on [0, ∞), stable for all
        μ/σ — including the deeply-truncated regime (μ ≪ 0) the model enters
        for calm wind.

        Solves ``F(x) = p``:
            x = μ − σ · Φ⁻¹( Φ(μ/σ) · (1 − p) )
        Using the *lower-tail* probability ``Φ(μ/σ)`` (the truncation
        normaliser Z) keeps the argument to ``Φ⁻¹`` small rather than near 1.
        The old form solved ``Φ⁻¹(0.5 + 0.5·Φ(−μ/σ))``, whose argument → 1 for
        μ ≪ 0 and saturated the ``(1e-6, 1−1e-6)`` icdf clamp — producing
        *negative* medians (down to −15 on real data) for μ/σ < −4.75, the
        exact value of ``Φ⁻¹(1−1e-6)``. Computed in float64 and floored at 0
        (the support's lower bound) to cover the extreme underflow tail.
        Verified against ``scipy.stats.truncnorm`` across μ/σ ∈ [−25, 4].

        ``p`` may be a scalar or a tensor broadcastable against the parameter
        shape (used by :meth:`sample` with per-draw uniforms).
        """
        mu = params["mu"].double()
        sigma = self._sigma(params).double()
        p = torch.as_tensor(p, dtype=mu.dtype, device=mu.device)
        Z = torch.exp(torch.special.log_ndtr(mu / sigma))     # Φ(μ/σ)
        q = (Z * (1.0 - p)).clamp(min=1e-300, max=1.0 - 1e-15)
        # torch.special.ndtri (probit) is tail-accurate for tiny q, unlike
        # Normal.icdf, which routes through erfinv(2q−1) and returns −inf once
        # 2q−1 rounds to exactly −1 (q ≲ 1e-16) — the very regime μ ≪ 0 hits.
        x = mu - sigma * torch.special.ndtri(q)
        return x.clamp(min=0.0).to(params["mu"].dtype)

    def median(self, params):
        # F_trunc(m) = 0.5; see _quantile for the numerically stable form (the
        # previous closed form returned negative medians for μ ≪ 0).
        return self._quantile(params, 0.5)

    def cdf(self, params, target):
        # F(y) = (Φ((y − μ)/σ) − Φ(α)) / (1 − Φ(α))  for y ≥ 0;
        # F(y) = 0 elsewhere.
        mu = params["mu"]
        sigma = self._sigma(params)
        std_normal = torch.distributions.Normal(
            loc=torch.zeros_like(mu), scale=torch.ones_like(mu)
        )
        y_nonneg = target.clamp(min=0.0)
        z = (y_nonneg - mu) / sigma
        alpha = -mu / sigma
        Phi_z = std_normal.cdf(z)
        Phi_alpha = std_normal.cdf(alpha)
        Z = (1.0 - Phi_alpha).clamp(min=1e-12)
        cdf = (Phi_z - Phi_alpha) / Z
        return torch.where(target >= 0, cdf, torch.zeros_like(cdf))

    def crps(self, params, target):
        # Unbiased ensemble estimator (same approach as the Weibull and
        # Bernoulli-Gamma heads), using the numerically stable :meth:`sample`.
        #
        # The closed form for the lower-truncated normal (Thorarinsdottir &
        # Gneiting 2010; scoringRules::crps_tnorm) is exact on paper but
        # catastrophically unstable in the deeply-truncated regime: the
        # ``c = 1/Φ(μ/σ)`` amplification meets the cancellation
        # ``2Φ((y−μ)/σ) − 2 → 0`` once μ/σ ≲ −4, blowing the per-observation
        # CRPS up by orders of magnitude (in float32 — the eval dtype — values
        # of ~9 at μ/σ=−4 and ~376 at μ/σ=−6 vs a true ~0.8, which inflated the
        # reported wind CRPS to the hundreds). The model legitimately predicts
        # μ ≪ 0 for calm wind, so that regime is common, not pathological — the
        # ensemble avoids the cancellation entirely. Validated against
        # ``scipy.stats.truncnorm`` Monte-Carlo CRPS across μ/σ ∈ [−12, 4].
        return _ensemble_crps(self, params, target, n_samples=200)

    def sample(self, params, n: int = 1) -> torch.Tensor:
        # Inverse-CDF sampling through the stable :meth:`_quantile` (lower-tail
        # form), so draws stay ≥ 0 even for μ ≪ 0. The earlier direct form
        # ``μ + σ·Φ⁻¹(Φ(α) + u·(1−Φ(α)))`` saturated the (1e-6, 1−1e-6) clamp
        # for μ ≪ 0 and emitted large negative samples there.
        mu = params["mu"]
        u = torch.rand((n, *mu.shape), device=mu.device, dtype=mu.dtype)
        return self._quantile(params, u)


# ---------------------------------------------------------------------------
# Bernoulli-Gamma hurdle
# ---------------------------------------------------------------------------

class BernoulliGammaHead(LikelihoodHead):
    """Hurdle Bernoulli-Gamma for precipitation.

    ``params = {"rho": ..., "alpha": ..., "beta": ...}`` where:
      - ``ρ`` is ``P(rain > 0) ∈ (0, 1)`` via sigmoid + clamp.
      - ``α, β`` are Gamma shape/rate, both positive via softplus + floor.

    PDF (hurdle):
      ``P(Y = 0) = 1 − ρ``,  ``P(Y = y > 0) = ρ · Gamma(y; α, β)``.

    NLL (Cannon 2008; Vaughan et al. 2022):
      ``−[ r · (log ρ + log Gamma(y; α, β)) + (1 − r) · log(1 − ρ) ]``
    where ``r = 1{y > 0}``. The Gamma term is gated by ``r`` so it
    never sees ``y = 0`` (which would be undefined for ``α < 1``).
    """

    n_params = 3
    param_names = ("rho", "alpha", "beta")

    def forward(self, hidden):
        raw = self.linear(hidden)
        rho = torch.sigmoid(raw[..., 0]).clamp(_RHO_MIN, _RHO_MAX)
        alpha = _softplus_floor(raw[..., 1]).clamp(_POSITIVE_MIN, _POSITIVE_MAX)
        beta = _softplus_floor(raw[..., 2]).clamp(_POSITIVE_MIN, _POSITIVE_MAX)
        return {"rho": rho, "alpha": alpha, "beta": beta}

    def initialise_rho_bias_from_climatology(self, p_wet: float) -> None:
        """Set the ``ρ``-channel bias to ``logit(p_wet)``.

        Initialising ρ at the regional climatological wet-day frequency
        (rather than at 0.5 from the default Linear init) prevents the
        early training loss from being dominated by the dry-day branch
        ``log(1 − ρ)`` in dry regions, where the model otherwise spends
        many epochs slowly drifting ρ toward its true climatological
        value while learning nothing about α, β.

        Call this after head construction, before the first optimiser
        step. Idempotent — call again with a different ``p_wet`` to
        re-initialise.

        Args:
            p_wet: Climatological probability of a wet observation
                (``0 < p_wet < 1``). Computed by the training script as
                ``(targets > 0).mean()`` over the train split.
        """
        if not (0.0 < p_wet < 1.0):
            raise ValueError(
                f"p_wet must be in (0, 1); got {p_wet}"
            )
        # ρ-channel is index 0 in self.linear's output.
        with torch.no_grad():
            self.linear.bias[0] = math.log(p_wet / (1.0 - p_wet))

    def _gamma(self, params):
        # PyTorch Gamma(concentration, rate) = Gamma(α, β).
        return torch.distributions.Gamma(
            concentration=params["alpha"], rate=params["beta"]
        )

    def nll(self, params, target, mask=None):
        rho = params["rho"]
        # r = 1 if rain observed, 0 otherwise. Replace y=0 with a small
        # dummy so log_prob doesn't blow up; the dry-day contribution
        # gates this with r=0 anyway. Adapted from convCNPClimate's
        # `make_r_mask` utility.
        r = (target > 0).to(target.dtype)
        target_safe = torch.where(
            target > 0,
            target,
            torch.full_like(target, _POSITIVE_FLOOR),
        )
        log_gamma = self._gamma(params).log_prob(target_safe)
        # Stable log(1 - ρ) — ρ is already clamped < 1, but use log1p
        # of -ρ for stability when ρ is near 1.
        log_one_minus_rho = torch.log1p(-rho)
        loss_per_entry = -(
            r * (torch.log(rho) + log_gamma)
            + (1.0 - r) * log_one_minus_rho
        )
        if mask is not None:
            mask_f = mask.to(loss_per_entry.dtype)
            return (loss_per_entry * mask_f).sum() / mask_f.sum().clamp(min=1.0)
        return loss_per_entry.mean()

    def mean(self, params):
        # E[Y] = ρ · α/β  (mass at zero contributes 0; conditional mean
        # of the Gamma is α/β).
        return params["rho"] * params["alpha"] / params["beta"]

    def median(self, params):
        # The median is 0 if ρ < 0.5; otherwise it's the (1 − 0.5/ρ)
        # quantile of the Gamma. We use the conditional Gamma median
        # only when ρ ≥ 0.5, returning 0 elsewhere — matching the
        # wet/dry decision rule in Vaughan et al. (ρ ≥ 0.5 → wet).
        rho = params["rho"]
        alpha = params["alpha"]
        beta = params["beta"]
        # icdf isn't implemented for Gamma in older PyTorch; the
        # Wilson-Hilferty approximation is accurate to <1% for α > 1
        # (and the typical α range here):
        #   median(Gamma) ≈ α · (1 − 1/(9α))³ / β.
        wilson_hilferty = alpha * torch.pow(1.0 - 1.0 / (9.0 * alpha), 3) / beta
        return torch.where(
            rho >= 0.5, wilson_hilferty, torch.zeros_like(wilson_hilferty)
        )

    def cdf(self, params, target):
        # Mixed CDF: F(y) = (1 − ρ) · 1{y ≥ 0} + ρ · F_Gamma(y; α, β) · 1{y > 0}.
        # Used for forecast scoring but NOT directly for PIT — see
        # pit_cdf / pit_mask for the wet-day Vaughan-style PIT.
        rho = params["rho"]
        gamma_cdf = self._gamma(params).cdf(target.clamp(min=_POSITIVE_FLOOR))
        # For y == 0: F(0) = (1 − ρ). For y > 0: F(y) = (1 − ρ) + ρ · F_Gamma(y).
        return torch.where(
            target > 0,
            (1.0 - rho) + rho * gamma_cdf,
            (1.0 - rho),
        )

    def pit_cdf(self, params, target):
        # Conditional Gamma CDF for wet observations: F_Gamma(y; α, β).
        # Combined with pit_mask (wet obs only), this gives the PIT
        # test Vaughan et al. (2022) report — uniform under a correctly
        # calibrated Gamma fit, regardless of the (1−ρ) classification.
        return self._gamma(params).cdf(target.clamp(min=_POSITIVE_FLOOR))

    def pit_mask(self, target):
        # Restrict PIT to wet observations. Dry days contribute the
        # (1−ρ) point mass at 0, which would pile up at PIT=0 and
        # collapse the χ² test.
        return target > 0

    def crps(self, params, target):
        # No clean closed form; ensemble approximation.
        return _ensemble_crps(self, params, target, n_samples=200)

    def sample(self, params, n: int = 1) -> torch.Tensor:
        rho = params["rho"]
        # Bernoulli sample for wet/dry, then Gamma sample for wet entries.
        wet = torch.bernoulli(rho.expand(n, *rho.shape))
        gamma_samples = self._gamma(params).sample((n,))
        return wet * gamma_samples


# ---------------------------------------------------------------------------
# Generative (implicit) head — trained by CRPS minimisation
# ---------------------------------------------------------------------------

class GenerativeHead(LikelihoodHead):
    """Implicit generative likelihood head, trained by scoring-rule minimisation.

    Unlike the parametric heads, this head does *not* emit fixed-form
    distribution parameters. The predictive distribution at a query point is
    defined *implicitly*: a small generator MLP ``G`` maps the concatenation
    of that point's hidden state and an i.i.d. noise vector ``z ~ N(0, I)`` to
    a single scalar sample. Drawing many ``z``'s gives an ensemble of draws
    from the (arbitrary-shaped) predictive distribution — skew, heavy tails
    and multimodality that the Gaussian / TruncNormal / Weibull / B-G families
    cannot represent. This is the approach of Pacchiardi & Dutta (2024) and
    the recent ML-ensemble weather models (GenCast, AIFS-CRPS): an implicit
    generator trained to minimise a strictly proper scoring rule, with **no
    adversarial critic / discriminator** (Gneiting & Raftery 2007).

    Interface mapping. Every head implements ``forward(hidden) -> params``,
    where ``params`` is the *noise-free* description of the per-point
    distribution that the sampling / scoring methods consume. For a Gaussian
    head that is ``(mu, log_var)``; here the generator ``G`` is a *shared*
    trained module, so the only per-point quantity that pins down a point's
    distribution is its **hidden state**. Hence ``params = {"hidden": hidden}``.
    Holding it fixed and drawing ``M`` independent ``z``'s yields the ensemble
    the way ``mu + sigma * z`` does for the Gaussian — see the discussion in
    the module-level design notes.

    ``G`` is separate from, and downstream of, the shared ``DecoderMLP`` body
    (which ingests weather features, elevation, TESSERA latent and the gridded
    interpolation). It replaces the per-variable ``Linear(hidden_dim, n_params)``
    that the parametric heads own — the shared body is untouched.

    Training objective. Reparameterised by construction (``z`` is drawn with
    ``torch.randn`` and ``G`` is differentiable), so the ensemble CRPS is
    differentiable end-to-end. :meth:`crps` returns the per-observation CRPS,
    which the training loop masks-and-means exactly like ``head.nll`` for the
    other heads. **There is no tractable density**, so :meth:`nll` raises and
    ``has_density = False``; this head must be trained with
    ``loss_function='crps'``.

    Fair (unbiased) CRPS estimator. The empirical CRPS is

        CRPS(F, y) = E|X - y| - 1/2 E|X - X'|,   X, X' ~ F i.i.d.

    The spread term is estimated with the **unbiased** ("fair") estimator
    ``(1/(M(M-1))) sum_{i != j} |X_i - X_j|``, *not* the biased ``1/M^2`` form
    that includes the ``i = j`` zeros. The biased form underestimates the
    ensemble spread by a factor ``(M-1)/M`` and trains the generator to
    *under-disperse* (collapse its variance) — the well-documented fair-CRPS
    issue (Zamo & Naveau 2018; Leutbecher 2019; Lang et al. 2024). The spread
    is computed in ``O(M log M)`` via the order-statistic identity
    ``sum_{i,j} |x_i - x_j| = 2 sum_i (2i - M - 1) x_(i)`` (sorted ascending),
    which is exact, differentiable through the sort, and avoids materialising
    the ``O(M^2)`` pairwise matrix.
    """

    # No fixed-form parameters: the conditioning hidden state *is* the
    # per-point "parameter" (see class docstring).
    param_names = ("hidden",)
    n_params = 0          # unused — the generator replaces the base Linear.
    has_density = False   # implicit model: no closed-form likelihood.

    def __init__(
        self,
        hidden_dim: int,
        z_dim: int = 8,
        generator_width: int = 128,
        generator_depth: int = 2,
        n_train_samples: int = 16,
        n_eval_samples: int = 100,
        nonneg: bool = False,
    ):
        # Skip LikelihoodHead.__init__ deliberately: it builds
        # ``self.linear = Linear(hidden_dim, n_params)``, which a generative
        # head has no use for. We build the generator MLP instead.
        nn.Module.__init__(self)
        if n_train_samples < 2 or n_eval_samples < 2:
            raise ValueError(
                "GenerativeHead needs >= 2 samples for the fair CRPS spread "
                f"term; got n_train_samples={n_train_samples}, "
                f"n_eval_samples={n_eval_samples}."
            )
        self.hidden_dim = hidden_dim
        self.z_dim = z_dim
        self.n_train_samples = n_train_samples
        self.n_eval_samples = n_eval_samples
        # When True, every generated sample is passed through softplus so the
        # predictive support is (0, inf) — for non-negative targets (wind,
        # precip). Without it the generator can place mass below 0, which costs
        # CRPS sharpness vs the parametric heads that build in the support
        # (TruncatedNormal/Weibull/Gamma). t2m (real-valued) leaves this False.
        self.nonneg = nonneg

        layers: list[nn.Module] = []
        in_dim = hidden_dim + z_dim
        for _ in range(generator_depth):
            layers += [nn.Linear(in_dim, generator_width), nn.SiLU()]
            in_dim = generator_width
        layers += [nn.Linear(in_dim, 1)]
        self.generator = nn.Sequential(*layers)

    def _n_samples(self) -> int:
        """Sample count: small (cheap, gradient-noisy but fine for SGD) while
        training, larger (low-variance metric) at eval, switched by
        ``self.training`` exactly like dropout/BN."""
        return self.n_train_samples if self.training else self.n_eval_samples

    def _generate(self, hidden: torch.Tensor, n: int) -> torch.Tensor:
        """Draw ``n`` reparameterised samples per query point.

        Args:
            hidden: ``(..., hidden_dim)`` conditioning state.
            n: number of i.i.d. noise draws per point.

        Returns:
            ``(n, ...)`` samples — leading sample axis, then the conditioning
            shape with ``hidden_dim`` collapsed. Differentiable w.r.t. the
            generator weights (noise enters only through ``torch.randn``).
        """
        z = torch.randn(
            n, *hidden.shape[:-1], self.z_dim,
            device=hidden.device, dtype=hidden.dtype,
        )
        h = hidden.unsqueeze(0).expand(n, *hidden.shape)
        x = torch.cat([h, z], dim=-1)
        out = self.generator(x).squeeze(-1)
        if self.nonneg:
            # Constrain samples to (0, inf) for non-negative variables. Softplus
            # (not ReLU) keeps the map smooth so gradients flow everywhere.
            out = F.softplus(out)
        return out

    def forward(self, hidden: torch.Tensor) -> dict[str, torch.Tensor]:
        # The conditioning state is the (noise-free) per-point distribution
        # descriptor; sampling happens later in crps/sample/mean/... .
        return {"hidden": hidden}

    def nll(self, params, target, mask=None):
        raise NotImplementedError(
            "GenerativeHead is an implicit model with no tractable density, "
            "so NLL is undefined. Train any variable using this head with "
            "loss_function='crps' (scoring-rule minimisation); the evaluator "
            "skips the NLL metric for it (has_density=False)."
        )

    def crps(self, params, target):
        # Fair (unbiased) empirical CRPS, per observation, differentiable.
        n = self._n_samples()
        samples = self._generate(params["hidden"], n)        # (n, *target_shape)
        abs_err = (samples - target.unsqueeze(0)).abs().mean(dim=0)   # E|X - y|

        # Spread term 1/2 E|X - X'| via the order-statistic identity
        #   sum_{i,j} |x_i - x_j| = 2 sum_i (2i - n - 1) x_(i)
        # with the unbiased 1/(n(n-1)) normaliser (excludes i=j zeros).
        sorted_s, _ = torch.sort(samples, dim=0)
        idx = torch.arange(1, n + 1, device=samples.device, dtype=samples.dtype)
        weights = (2.0 * idx - n - 1.0).view(n, *([1] * (sorted_s.dim() - 1)))
        spread = (weights * sorted_s).sum(dim=0) / (n * (n - 1))   # = 1/2 E|X-X'|
        return abs_err - spread

    def sample(self, params, n: int = 1) -> torch.Tensor:
        return self._generate(params["hidden"], n)

    def mean(self, params):
        return self._generate(params["hidden"], self.n_eval_samples).mean(dim=0)

    def median(self, params):
        return self._generate(
            params["hidden"], self.n_eval_samples
        ).median(dim=0).values

    def cdf(self, params, target):
        # Empirical CDF of the ensemble: F(y) ~ (1/M) sum_m 1{X_m <= y}.
        samples = self._generate(params["hidden"], self.n_eval_samples)
        return (samples <= target.unsqueeze(0)).to(target.dtype).mean(dim=0)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

# Keys used in the YAML / CLI config to select a head per variable. Values are
# *factories* taking ``hidden_dim`` — usually the head class itself, but the
# non-negative generative variant is a preset (softplus output) of the same
# class, so it is a ``functools.partial`` rather than a distinct class.
HEAD_REGISTRY: dict[str, Callable[..., LikelihoodHead]] = {
    "gaussian":          GaussianHead,
    "truncated_normal":  TruncatedNormalHead,
    "weibull":           WeibullHead,
    "bernoulli_gamma":   BernoulliGammaHead,
    "generative":        GenerativeHead,
    # Non-negative variant for wind / precip: generator output passed through
    # softplus so samples stay >= 0 (the support those variables live on).
    "generative_nonneg": functools.partial(GenerativeHead, nonneg=True),
}


def build_head(name: str, hidden_dim: int) -> LikelihoodHead:
    """Construct a head from its registry key."""
    if name not in HEAD_REGISTRY:
        raise ValueError(
            f"Unknown likelihood {name!r}. "
            f"Choose from {list(HEAD_REGISTRY)}."
        )
    return HEAD_REGISTRY[name](hidden_dim=hidden_dim)