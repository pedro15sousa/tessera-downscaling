"""Training objective of the patch-encoder VAE.

    L = L_recon + w_grad * L_grad + beta(t) * KL + sum_i lambda_i * L_aux_i

* ``L_recon`` -- per-element MSE (or L1) between the reconstruction and the
  z-scored input patch.
* ``L_grad`` -- the same distance between the *finite differences* of the two,
  along both axes. TESSERA embeddings carry sharp field/road/coast boundaries
  that a plain MSE blurs away; the paper's run weights this term 0.5.
* ``KL`` -- against the standard normal prior, ramped in linearly over
  ``beta_warmup_steps`` optimiser steps to a small ``beta_end`` (5e-4). A
  full-strength KL from step 0 collapses the posterior onto the prior.
* ``L_aux`` -- masked MSE of each auxiliary head against its (already z-scored)
  target. Stations whose target is missing carry NaN and drop out of the mean,
  which is how elevation sentinels are excluded.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def reconstruction_loss(
    x_recon: torch.Tensor, x: torch.Tensor, mode: str = "mse"
) -> torch.Tensor:
    """Per-element reconstruction loss, averaged over batch and pixels."""
    if mode == "mse":
        return F.mse_loss(x_recon, x)
    if mode == "l1":
        return F.l1_loss(x_recon, x)
    raise ValueError(f"Unknown reconstruction loss: {mode!r} (expected 'mse' or 'l1')")


def _spatial_gradients(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Finite differences of ``(B, C, H, W)`` along the width and height axes."""
    dx = x[:, :, :, 1:] - x[:, :, :, :-1]
    dy = x[:, :, 1:, :] - x[:, :, :-1, :]
    return dx, dy


def gradient_loss(
    x_recon: torch.Tensor, x: torch.Tensor, mode: str = "mse"
) -> torch.Tensor:
    """Distance between the spatial gradients of reconstruction and target."""
    dx_r, dy_r = _spatial_gradients(x_recon)
    dx_t, dy_t = _spatial_gradients(x)
    if mode == "mse":
        return F.mse_loss(dx_r, dx_t) + F.mse_loss(dy_r, dy_t)
    if mode == "l1":
        return F.l1_loss(dx_r, dx_t) + F.l1_loss(dy_r, dy_t)
    raise ValueError(f"Unknown gradient loss mode: {mode!r} (expected 'mse' or 'l1')")


def kl_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """``KL(q(z|x) || N(0, I))``, summed over latent dims and averaged over batch.

    ``logvar`` is clamped to ``[-10, 10]`` before exponentiation, the same
    clamp the sampler applies; the bounds are far outside any useful posterior
    width (exp(+-10) is 2.2e4 / 4.5e-5) and only guard against overflow.
    """
    logvar = torch.clamp(logvar, min=-10, max=10)
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1).mean()


def auxiliary_loss(
    predictions: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
    weights: dict[str, float],
) -> tuple[torch.Tensor, dict[str, float]]:
    """Weighted sum of per-target MSEs, ignoring non-finite target entries.

    Args:
        predictions: ``{target name: (B,) prediction}``.
        targets: ``{target name: (B,) target}``; NaN entries are masked out,
            and a target absent from the dict is skipped entirely.
        weights: ``{target name: weight}``, defaulting to 1.0.

    Returns:
        The weighted total and an unweighted ``{name: float}`` breakdown for
        logging.
    """
    total = torch.tensor(0.0, device=next(iter(predictions.values())).device)
    breakdown: dict[str, float] = {}

    for name, pred in predictions.items():
        if name not in targets:
            continue
        target = targets[name]
        valid = torch.isfinite(target)
        if not valid.any():
            continue
        loss = F.mse_loss(pred[valid], target[valid])
        breakdown[name] = loss.item()
        total = total + weights.get(name, 1.0) * loss

    return total, breakdown


def linear_beta(step: int, warmup_steps: int, beta_end: float) -> float:
    """KL weight at ``step``: a linear ramp from 0 to ``beta_end``, then flat."""
    if step >= warmup_steps:
        return beta_end
    return beta_end * (step / warmup_steps)


class VAELoss:
    """The full objective, holding the optimiser-step counter that drives beta.

    Call it once per batch and :meth:`step` once per optimiser step::

        criterion = VAELoss(cfg["loss"], cfg["auxiliary"]["weights"])
        total, log = criterion(model(x), x, targets)
        total.backward(); optimizer.step(); criterion.step()

    Args:
        loss_cfg: The ``loss`` block of the run config (``reconstruction``,
            ``gradient_weight``, ``beta_end``, ``beta_warmup_steps``).
        aux_weights: ``{target name: weight}`` for the auxiliary heads.
    """

    def __init__(
        self, loss_cfg: dict, aux_weights: dict[str, float] | None = None
    ) -> None:
        schedule = loss_cfg.get("beta_schedule", "linear")
        if schedule != "linear":
            raise ValueError(
                f"loss.beta_schedule={schedule!r} is not supported: only the "
                "linear KL ramp was kept when the patch encoder moved here."
            )
        self.loss_cfg = loss_cfg
        self.recon_mode = loss_cfg.get("reconstruction", "mse")
        self.grad_weight = loss_cfg.get("gradient_weight", 0.0)
        self.beta_end = loss_cfg["beta_end"]
        self.beta_warmup_steps = loss_cfg.get("beta_warmup_steps", 1000)
        self.aux_weights = aux_weights or {}
        self.current_step = 0

    def __call__(
        self,
        model_output: dict[str, torch.Tensor],
        x: torch.Tensor,
        targets: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Return the scalar loss and a flat dict of components for logging."""
        l_recon = reconstruction_loss(model_output["x_recon"], x, self.recon_mode)

        if self.grad_weight > 0:
            l_grad = gradient_loss(model_output["x_recon"], x, self.recon_mode)
        else:
            l_grad = torch.tensor(0.0, device=x.device)

        l_kl = kl_divergence(model_output["mu"], model_output["logvar"])
        beta = linear_beta(self.current_step, self.beta_warmup_steps, self.beta_end)

        aux_preds = {
            k.removeprefix("aux_"): v
            for k, v in model_output.items()
            if k.startswith("aux_")
        }
        if aux_preds and targets:
            l_aux, aux_breakdown = auxiliary_loss(aux_preds, targets, self.aux_weights)
        else:
            l_aux = torch.tensor(0.0, device=x.device)
            aux_breakdown = {}

        total = l_recon + self.grad_weight * l_grad + beta * l_kl + l_aux

        log = {
            "loss/total": total.item(),
            "loss/recon": l_recon.item(),
            "loss/grad": l_grad.item(),
            "loss/kl": l_kl.item(),
            "loss/beta": beta,
            "loss/beta_kl": (beta * l_kl).item(),
            "loss/aux": l_aux.item(),
        }
        for name, value in aux_breakdown.items():
            log[f"loss/aux_{name}"] = value

        return total, log

    def step(self) -> None:
        """Advance the beta schedule by one optimiser step."""
        self.current_step += 1
