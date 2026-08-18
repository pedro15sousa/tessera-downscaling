"""Multi-variable likelihood-head dispatcher.

``LikelihoodHeadDict`` holds one :class:`~.heads.LikelihoodHead` per target
variable (an ``nn.ModuleDict`` under ``self.heads``, hence the state-dict
prefix ``heads.heads.<var>.``) and applies each of them to the *same*
decoder hidden state ``(batch, n_stations, hidden_dim)``. The ``n_stations``
axis is the per-batch target-station axis, not a variable axis. Each head
returns a dict of named distribution parameters of shape
``(batch, n_stations)``; the dispatcher returns
``{var_name: {param_name: tensor}}``.

Variable iteration order follows ``target_variables``. Order matters for
callers that pack / unpack per-variable tensors by index (e.g.
``targets[:, :, vi]`` in the training loop). Per-variable losses and
metrics are computed by the caller through ``model.heads.heads[var]``.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .heads import LikelihoodHead, build_head


class LikelihoodHeadDict(nn.Module):
    """ModuleDict wrapper providing per-variable dispatch.

    Args:
        likelihood_per_variable: mapping ``{var_name: dist_name}`` where
            ``dist_name`` is a key in
            :data:`~tessera_downscaling.model.heads.HEAD_REGISTRY`.
        target_variables: ordered list of variable names. Determines
            iteration order.
        hidden_dim: dimension of the shared MLP body output. Each head
            owns a ``Linear(hidden_dim, n_params)`` of its own.

    Raises:
        ValueError: when the keys of ``likelihood_per_variable`` do
            not match ``target_variables`` exactly (extra or missing
            entries). Failing fast at construction time means mismatches
            surface before any training cost is incurred.

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
                "entirely to use Gaussian heads throughout."
            )

        self.target_variables = list(target_variables)
        self.likelihood_per_variable = dict(likelihood_per_variable)
        self.heads = nn.ModuleDict(
            {
                var: build_head(likelihood_per_variable[var], hidden_dim)
                for var in target_variables
            }
        )

    def forward(self, hidden: torch.Tensor) -> dict[str, dict[str, torch.Tensor]]:
        """Dispatch the hidden state through each variable's head.

        Args:
            hidden: tensor of shape ``(batch, n_stations, hidden_dim)``,
                produced by the shared MLP body.

        Returns:
            ``{var_name: {param_name: tensor of shape (batch, n_stations)}}``.

        """
        out: dict[str, dict[str, torch.Tensor]] = {}
        for var in self.target_variables:
            head: LikelihoodHead = self.heads[var]
            out[var] = head(hidden)
        return out
