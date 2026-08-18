"""Multi-variable likelihood-head dispatcher.

A thin ``nn.ModuleDict`` wrapper that holds one ``LikelihoodHead`` per
target variable and provides batched dispatch for training and
evaluation. Lives alongside ``heads.py``; pulled out here so the model
file (``convcnp.py``) has a single small import and the dispatch logic
isn't duplicated between train and eval.

The dispatcher consumes a hidden state of shape
``(batch, n_stations, hidden_dim)`` produced by the shared MLP body
(MLP run in ``body_only=True`` mode). The ``n_stations`` axis is the
per-batch target station axis — NOT a variable axis. For each target
variable, the dispatcher applies the variable's head (which owns its
own ``Linear(hidden_dim, n_params)``) to the *same* hidden state. Each
head produces a dict of named, constrained distribution parameters of
shape ``(batch, n_stations)``. Returns
``{var_name: {param_name: tensor}}``.

This contrasts with the alternative single-output design (one
``Linear(hidden_dim, sum_of_n_params)`` and slice the output) that
gives identical parameter count but loses encapsulation: the
distribution-specific constraints (softplus floor, sigmoid, clamps),
NLL formulas, mean/median/CDF/CRPS/sample methods all live on the
head class rather than scattered across the loss and evaluator.

Variable iteration order follows ``target_variables``. Order matters
for downstream consumers that pack/unpack tensors by index (e.g.
``targets[:, :, vi]`` in the multi-task training loop).

Backwards compatibility: the dispatcher is only constructed when the
model is in per-variable-heads mode. The legacy flat-output pathway
in ``convcnp.py`` does not use this module at all.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .heads import LikelihoodHead, build_head


class LikelihoodHeadDict(nn.Module):
    """ModuleDict wrapper providing per-variable dispatch.

    Args:
        likelihood_per_variable: mapping ``{var_name: dist_name}`` where
            ``dist_name`` is a key in ``HEAD_REGISTRY``
            (``gaussian``, ``weibull``, ``bernoulli_gamma``).
        target_variables: ordered list of variable names. Determines
            iteration order; used by callers (training loop) to pack
            and unpack per-variable tensors by index.
        hidden_dim: dimension of the shared MLP body output. Each head
            owns a ``Linear(hidden_dim, n_params)`` of its own.

    Raises:
        ValueError: when the keys of ``likelihood_per_variable`` do
            not match ``target_variables`` exactly (extra or missing
            entries). The training pipeline assumes a 1:1
            correspondence; failing fast at construction time means
            mismatches surface before any training cost is incurred.
    """

    def __init__(
        self,
        likelihood_per_variable: dict[str, str],
        target_variables: list[str],
        hidden_dim: int,
    ):
        super().__init__()
        # Strict 1:1 check. Set arithmetic gives the clearest error.
        spec_vars = set(likelihood_per_variable)
        target_vars = set(target_variables)
        missing = target_vars - spec_vars
        extra = spec_vars - target_vars
        if missing or extra:
            raise ValueError(
                "likelihood_per_variable does not match target_variables. "
                f"target_variables={sorted(target_vars)}, "
                f"likelihood_per_variable keys={sorted(spec_vars)}. "
                f"Missing={sorted(missing)}, extra={sorted(extra)}. "
                "Each target variable must have exactly one likelihood entry "
                "and vice versa. Either add the missing entries to "
                "--likelihood, drop the extra ones, or omit --likelihood "
                "entirely to use the legacy Gaussian-everywhere path."
            )

        self.target_variables = list(target_variables)
        self.likelihood_per_variable = dict(likelihood_per_variable)
        self.heads = nn.ModuleDict({
            var: build_head(likelihood_per_variable[var], hidden_dim)
            for var in target_variables
        })

    def forward(
        self, hidden: torch.Tensor
    ) -> dict[str, dict[str, torch.Tensor]]:
        """Dispatch the hidden state through each variable's head.

        Args:
            hidden: tensor of shape ``(batch, n_stations, hidden_dim)``,
                produced by the shared MLP body in ``body_only=True``
                mode. The ``n_stations`` axis is the per-batch target
                station axis (NOT a variable axis — each variable head
                sees the same hidden state and projects to its own
                distribution-specific parameters via its own
                ``Linear(hidden, n_params)``).

        Returns:
            ``{var_name: {param_name: tensor of shape (batch, n_stations)}}``.
            The leading dims are preserved; only the trailing
            ``hidden_dim`` axis is collapsed into the head's
            parameter-named dict.
        """
        out: dict[str, dict[str, torch.Tensor]] = {}
        for var in self.target_variables:
            head: LikelihoodHead = self.heads[var]
            # head.forward applies its Linear(hidden, n_params) along the
            # last axis and returns a dict of named parameters, each of
            # shape (batch, n_stations).
            out[var] = head(hidden)
        return out

    # ------------------------------------------------------------------
    # Aggregation helpers — used by train.py and evaluate.py so they
    # don't have to iterate the dict themselves.
    # ------------------------------------------------------------------

    def total_nll(
        self,
        params_per_variable: dict[str, dict[str, torch.Tensor]],
        targets_per_variable: dict[str, torch.Tensor],
        masks_per_variable: dict[str, torch.Tensor] | None = None,
        weights: dict[str, float] | None = None,
    ) -> torch.Tensor:
        """Summed weighted NLL across variables.

        Args:
            params_per_variable: output of :meth:`forward`. Each
                head's parameters have shape ``(batch, n_stations)``.
            targets_per_variable: ``{var: tensor of shape (batch, n_stations)}``.
            masks_per_variable: optional ``{var: bool tensor of shape
                (batch, n_stations)}``; entries where mask is False are
                dropped from the loss.
            weights: optional ``{var: float}`` for multi-task
                weighting. Defaults to all ones.

        Returns:
            scalar tensor — the weighted sum of per-variable NLLs.
        """
        weights = weights or {var: 1.0 for var in self.target_variables}
        masks = masks_per_variable or {}
        total = torch.zeros((), device=next(self.parameters()).device)
        for var in self.target_variables:
            head: LikelihoodHead = self.heads[var]
            var_nll = head.nll(
                params_per_variable[var],
                targets_per_variable[var],
                mask=masks.get(var),
            )
            total = total + weights.get(var, 1.0) * var_nll
        return total

    def predictive_means(
        self,
        params_per_variable: dict[str, dict[str, torch.Tensor]],
    ) -> dict[str, torch.Tensor]:
        """Per-variable predictive means.

        For Gaussian this is μ; for Weibull it's λ·Γ(1+1/k); for
        Bernoulli-Gamma it's ρ·α/β. Use this when you want the same
        point estimate that legacy code assumed (mean prediction).
        """
        return {
            var: self.heads[var].mean(params_per_variable[var])
            for var in self.target_variables
        }

    def predictive_medians(
        self,
        params_per_variable: dict[str, dict[str, torch.Tensor]],
    ) -> dict[str, torch.Tensor]:
        """Per-variable predictive medians.

        Preferred over means for skewed distributions (Weibull, Gamma)
        when reporting MAE — the median minimises expected MAE while
        the mean minimises expected MSE.
        """
        return {
            var: self.heads[var].median(params_per_variable[var])
            for var in self.target_variables
        }