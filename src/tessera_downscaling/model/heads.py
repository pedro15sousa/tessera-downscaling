"""Per-variable likelihood heads for the ConvCNP downscaler.

Each ``LikelihoodHead`` projects the shared decoder hidden state
``(batch, n_targets, hidden_dim)`` to its own predictive-distribution
parameters through a single ``nn.Linear`` and provides the per-variable
NLL, mean / median, CDF (for PIT calibration), CRPS and sampling. The model
owns a ``LikelihoodHeadDict`` (one head per target variable, see
:mod:`.heads_dispatch`), so distribution-specific logic lives on the head
class rather than in the loss function or the evaluator.

Two heads are provided:

  - **Gaussian** (``"gaussian"``; used for 2 m temperature).
    Heteroscedastic ``(μ, log σ²)``. The raw network output is treated as
    ``log σ²`` and clamped to ``[-10, 10]`` inside the consumer methods
    (NLL, σ derivation for CDF / CRPS / sampling), not at emission time, so
    the parameters dict holds exactly what the network produced. NLL is
    written out as the explicit ``0.5 · (log σ² + (y−μ)²/σ² + log 2π)``
    formula; every existing checkpoint was trained on that arithmetic and it
    is kept verbatim.

  - **Truncated Normal** (``"truncated_normal"``; used for 10 m wind speed).
    Gaussian left-truncated at 0, support ``[0, ∞)``, same ``(μ, log σ²)``
    parameterisation as the Gaussian head. Field-standard head for
    short-term wind in NWP post-processing (EMOS; Thorarinsdottir & Gneiting
    2010). Its quantile / sampling / CRPS paths are written for the deeply
    truncated regime ``μ ≪ 0`` that the model routinely enters on calm wind
    — see the class docstring before changing any of it.

CRPS: closed form for the Gaussian; for the truncated normal the 200-sample
fair ensemble estimator :func:`_ensemble_crps` (the paper's wind CRPS).

NaN-target handling is the caller's responsibility (mask in the loss
layer); heads operate on dense tensors with an optional boolean mask.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Final

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Numerical guards
# ---------------------------------------------------------------------------

# log_var clamp applied inside the consumer methods (NLL, σ derivation for
# CDF, CRPS, sampling) — never at parameter-emission time, so the raw
# network output survives in the params dict.
_LOG_VAR_MIN: Final[float] = -10.0
_LOG_VAR_MAX: Final[float] = 10.0

# Floor applied to targets inside the truncated-normal NLL so an observation
# of exactly 0 (which exists in GHCNh wind: anemometer-threshold reports)
# sits strictly inside the support.
_POSITIVE_MIN: Final[float] = 1e-5


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------


class LikelihoodHead(nn.Module, ABC):
    """Per-variable likelihood head.

    Each subclass owns a single ``Linear`` projecting the shared hidden
    state to its distribution's parameters (raw, unconstrained). The
    :meth:`forward` method returns a dict of named distribution
    parameters; the loss / evaluation methods consume that dict.

    Subclasses declare ``param_names`` and ``n_params`` at class level;
    they must reflect the exact dict keys returned by :meth:`forward`.
    """

    n_params: int
    param_names: tuple[str, ...]

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.linear = nn.Linear(hidden_dim, self.n_params)

    # ---- The interface every head implements ------------------------------

    @abstractmethod
    def forward(self, hidden: torch.Tensor) -> dict[str, torch.Tensor]:
        """Project hidden state to distribution parameters.

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

        Preferred over ``mean`` for skewed distributions when reporting
        MAE — the median minimises expected MAE, the mean expected MSE.
        """

    @abstractmethod
    def cdf(
        self,
        params: dict[str, torch.Tensor],
        target: torch.Tensor,
    ) -> torch.Tensor:
        """CDF F(target | params)."""

    def pit_cdf(
        self,
        params: dict[str, torch.Tensor],
        target: torch.Tensor,
    ) -> torch.Tensor:
        """CDF used for PIT calibration tests.

        Identical to :meth:`cdf` for the continuous heads here. A head with
        a point mass would override it with the conditional CDF of its
        continuous branch.
        """
        return self.cdf(params, target)

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
        """Draw ``n`` samples per (..., target) entry from the predictive distribution.

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
        """Return a ``torch.distributions`` analogue of this head, if it has one.

        Subclasses with a clean torch.distributions analogue return one
        here and reuse the default :meth:`sample`; others return None.
        """
        return None


# ---------------------------------------------------------------------------
# Helper: fair ensemble CRPS for distributions without a usable closed form.
# ---------------------------------------------------------------------------


def _ensemble_crps(
    head: LikelihoodHead,
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
    samples = head.sample(params, n=n_samples)  # (n, *target_shape)
    term1 = (samples - target.unsqueeze(0)).abs().mean(dim=0)
    perm = torch.randperm(n_samples, device=samples.device)
    term2 = 0.5 * (samples - samples[perm]).abs().mean(dim=0)
    return term1 - term2


# ---------------------------------------------------------------------------
# Gaussian (log_var parameterisation)
# ---------------------------------------------------------------------------


class GaussianHead(LikelihoodHead):
    """Heteroscedastic Gaussian, ``params = {"mu": ..., "log_var": ...}``.

    ``μ`` is unconstrained; ``log_var`` is the *raw* network output (also
    unconstrained). The numerical-stability clamp ``log_var ∈ [-10, 10]``
    is applied inside the consumer methods (``nll``, ``cdf``, ``crps``,
    ``sample``), not at emission time. Every trained checkpoint depends
    on this exact arithmetic — keep it as is.
    """

    n_params = 2
    param_names = ("mu", "log_var")

    def forward(self, hidden: torch.Tensor) -> dict[str, torch.Tensor]:
        raw = self.linear(hidden)
        mu = raw[..., 0]
        log_var = raw[..., 1]  # unconstrained
        return {"mu": mu, "log_var": log_var}

    def _sigma(self, params: dict[str, torch.Tensor]) -> torch.Tensor:
        """σ = exp(log_var/2) with the stability clamp applied."""
        log_var_clamped = params["log_var"].clamp(_LOG_VAR_MIN, _LOG_VAR_MAX)
        return torch.exp(0.5 * log_var_clamped)

    def _distribution(self, params):
        return torch.distributions.Normal(loc=params["mu"], scale=self._sigma(params))

    def nll(self, params, target, mask=None):
        # Explicit formula
        #   0.5 * (log_var + (y - μ)² / exp(log_var) + log(2π))
        # — NOT torch.distributions.Normal(μ, σ).log_prob, which computes
        # the density through a different sequence of float operations
        # (((y-μ)/σ)² vs (y-μ)²/exp(log_var)) and differs at the ULP level.
        # This is the arithmetic every checkpoint was trained on.
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
        return params["mu"]  # Gaussian: mean == median.

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
    estimator: the closed form for the lower-truncated normal suffers
    catastrophic cancellation once μ/σ ≲ −4 (the ``1/Z`` amplification meets
    ``2Φ(y_z)−2 → 0``), which the model routinely hits on calm wind, so the
    analytic version is unsafe here despite being correct on paper.
    """

    n_params = 2
    param_names = ("mu", "log_var")

    def forward(self, hidden):
        raw = self.linear(hidden)
        mu = raw[..., 0]
        log_var = raw[..., 1]  # storage-compatible with GaussianHead.
        return {"mu": mu, "log_var": log_var}

    def _sigma(self, params):
        """σ = exp(log_var/2) with the stability clamp applied.

        The clamp is intentionally the same one ``GaussianHead`` uses — keeps
        the two heads on the same gradient surface near initialisation.
        """
        log_var_clamped = params["log_var"].clamp(_LOG_VAR_MIN, _LOG_VAR_MAX)
        return torch.exp(0.5 * log_var_clamped)

    def initialise_from_climatology(
        self,
        mean_target: float,
        std_target: float,
    ) -> None:
        """Set the ``Linear`` biases so initial (μ, σ) ≈ (mean, std) of the target.

        Uses the positive-target climatology: for wind this is
        ``targets[targets > 0].mean()`` and ``.std()`` over the train split.
        The truncation pulls the *implied* mean of the
        predictive distribution slightly upward from μ, but the correction
        is small when μ ≫ σ (typical wind), and the network absorbs the
        residual within a few epochs.

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
        # Defensive clamp for targets at exactly zero (which exist in GHCNh
        # — anemometer threshold reports).
        target_safe = target.clamp(min=_POSITIVE_MIN)
        mu = params["mu"]
        sigma = self._sigma(params)
        z = (target_safe - mu) / sigma
        log_phi = -0.5 * z * z - 0.5 * math.log(2.0 * math.pi)
        log_sigma = torch.log(sigma)
        log_Z = torch.special.log_ndtr(mu / sigma)  # = log Φ(μ/σ) = log(1 − Φ(α))
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
        """``p``-quantile of the truncated normal on [0, ∞), stable for all μ/σ.

        Including the deeply-truncated regime (μ ≪ 0) the model enters for
        calm wind. Solves ``F(x) = p``:
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
        Z = torch.exp(torch.special.log_ndtr(mu / sigma))  # Φ(μ/σ)
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
        # Unbiased ensemble estimator using the numerically stable
        # :meth:`sample`.
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
# Registry
# ---------------------------------------------------------------------------

# Keys used by ``--likelihood var=<key>`` and stored in ``config.json`` under
# ``likelihood_per_variable``.
HEAD_REGISTRY: dict[str, type[LikelihoodHead]] = {
    "gaussian": GaussianHead,
    "truncated_normal": TruncatedNormalHead,
}


def build_head(name: str, hidden_dim: int) -> LikelihoodHead:
    """Construct a head from its registry key."""
    if name not in HEAD_REGISTRY:
        raise ValueError(
            f"Unknown likelihood {name!r}. Choose from {list(HEAD_REGISTRY)}."
        )
    return HEAD_REGISTRY[name](hidden_dim=hidden_dim)
