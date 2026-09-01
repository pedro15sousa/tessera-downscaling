#!/usr/bin/env python3
"""Train the patch-encoder VAE on the per-station TESSERA patches.

Reads a run config (``vae.yaml`` next to this script by default, overridable
key by key on the command line), builds the train/validation loaders and fits
the VAE with AdamW and a cosine learning-rate schedule. Every epoch it reports
the validation loss and its components, the number of active latent dimensions
and the auxiliary heads' R^2, and writes into the run directory:

    config.yaml               the resolved config, with input_size and
                              in_channels filled in from the data
    best.pt                   best-validation checkpoint
    checkpoint_epoch<N>.pt    every 20 epochs, to resume from
    last.pt                   final epoch; its presence marks a run complete
    history.json              per-epoch metrics

Checkpoints carry the config that produced them, so ``eval_vae.py`` and
``encode_dense_grid.py`` rebuild the model without being told the settings.

Usage (the paper's run: 2017 patches, 64 px crop, 16-d latent, the settings
of ``scripts/patch_encoder/vae.yaml``; about 200 epochs on one GPU):

    uv run python scripts/patch_encoder/train_vae.py \\
        --outdir tessera_patch_encoder/outputs/vae/p128_2017_crop64_lat16_grad0.5_auxon

The same script trains the foundation-model arms of the benchmark, which differ
only in their data and the resulting geometry -- pass ``--config
scripts/patch_encoder/vae_alphaearth.yaml`` or ``vae_olmoearth.yaml``, or use
``slurm/submit_fm_sweep.sh`` for the whole 16-run sweep.

Relative paths (``--outdir``, ``--patches-path``, ``--stations-path``,
``--cache-dir``, ``--resume``) are interpreted under the data root.
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from tessera_downscaling.patch_encoder.dataset import create_dataloaders
from tessera_downscaling.patch_encoder.losses import VAELoss
from tessera_downscaling.patch_encoder.model import build_model
from tessera_downscaling.paths import patch_encoder_dir, resolve

logger = logging.getLogger("train_vae")

DEFAULT_CONFIG = Path(__file__).with_name("vae.yaml")
EPOCH_HEADER = (
    f"{'epoch':>6} {'train':>8} {'val':>8} {'recon':>8} {'kl':>7} "
    f"{'beta':>7} {'active':>6} {'elev_R2':>8} {'lat_R2':>7} {'lon_R2':>7}"
)


def default_runs_dir() -> Path:
    """``<data root>/tessera_patch_encoder/outputs/vae`` -- where runs are kept."""
    return patch_encoder_dir("outputs", "vae")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the patch-encoder VAE.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--device", default="auto", help="cuda, cpu, or auto")
    parser.add_argument(
        "--outdir",
        default=None,
        help="Run directory. Default: a name derived from the config under "
        "<data root>/tessera_patch_encoder/outputs/vae/.",
    )
    parser.add_argument(
        "--resume", default=None, help="Checkpoint to continue training from."
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="Root of the dataset cache tree (default: "
        "<data root>/tessera_patch_encoder/outputs/dataset_cache).",
    )
    parser.add_argument(
        "--rebuild-cache",
        action="store_true",
        help="Rescan the patch file instead of reusing its cache.",
    )

    overrides = parser.add_argument_group("config overrides")
    overrides.add_argument("--patches-path", default=None)
    overrides.add_argument("--stations-path", default=None)
    overrides.add_argument(
        "--crop-size",
        type=int,
        default=None,
        help="Centred crop taken out of each stored patch (e.g. 64 of 128).",
    )
    overrides.add_argument(
        "--aux",
        choices=["on", "off"],
        default=None,
        help="Toggle the elevation/latitude/longitude heads.",
    )
    overrides.add_argument("--epochs", type=int, default=None)
    overrides.add_argument("--reconstruction", choices=["mse", "l1"], default=None)
    overrides.add_argument("--gradient-weight", type=float, default=None)
    overrides.add_argument("--beta-end", type=float, default=None)
    overrides.add_argument("--latent-dim", type=int, default=None)
    overrides.add_argument(
        "--suffix", default=None, help="Extra tag for the generated run name."
    )
    return parser


def apply_overrides(cfg: dict, args: argparse.Namespace) -> dict:
    """Apply the command-line overrides on top of the loaded config."""
    if args.patches_path is not None:
        cfg["data"]["patches_path"] = args.patches_path
    if args.stations_path is not None:
        cfg["data"]["stations_path"] = args.stations_path
    if args.crop_size is not None:
        cfg["data"]["crop_size"] = args.crop_size
    if args.aux is not None:
        cfg.setdefault("auxiliary", {})["enable"] = args.aux == "on"
    if args.epochs is not None:
        cfg["training"]["epochs"] = args.epochs
    if args.reconstruction is not None:
        cfg["loss"]["reconstruction"] = args.reconstruction
    if args.gradient_weight is not None:
        cfg["loss"]["gradient_weight"] = args.gradient_weight
    if args.beta_end is not None:
        cfg["loss"]["beta_end"] = args.beta_end
    if args.latent_dim is not None:
        cfg["model"]["latent_dim"] = args.latent_dim
    return cfg


def make_run_name(cfg: dict, suffix: str | None = None) -> str:
    """Fallback run-directory name, summarising the settings that vary."""
    model_cfg, loss_cfg = cfg["model"], cfg["loss"]
    parts = []
    if cfg["data"].get("crop_size"):
        parts.append(f"crop{cfg['data']['crop_size']}")
    parts += [f"lat{model_cfg['latent_dim']}", f"beta{loss_cfg['beta_end']}"]
    if loss_cfg.get("reconstruction", "mse") != "mse":
        parts.append(loss_cfg["reconstruction"])
    if loss_cfg.get("gradient_weight", 0.0) > 0:
        parts.append(f"grad{loss_cfg['gradient_weight']}")
    parts.append("auxon" if cfg.get("auxiliary", {}).get("enable") else "auxoff")
    if suffix:
        parts.append(suffix)
    parts.append(datetime.now().strftime("%Y%m%d_%H%M%S"))
    return "_".join(parts)


def select_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def describe_diverged_batch(
    log: dict[str, float], patches: torch.Tensor, out: dict[str, torch.Tensor]
) -> str:
    """Summarise a batch whose loss went non-finite, for the raised error."""
    lines = [f"    {k}: {v}" for k, v in log.items()]
    for name, tensor in [
        ("input", patches),
        *((k, out[k]) for k in ("mu", "logvar", "x_recon")),
    ]:
        lines.append(
            f"    {name}: min={tensor.min().item():.4f} "
            f"max={tensor.max().item():.4f} nan={torch.isnan(tensor).any().item()}"
        )
    return "\n".join(lines)


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: VAELoss,
    device: torch.device,
    grad_clip: float,
) -> tuple[float, dict[str, float]]:
    """One pass over the training split; returns mean loss and mean components."""
    model.train()
    total_loss = 0.0
    totals: dict[str, float] = {}
    n_batches = 0

    for batch_idx, batch in enumerate(loader):
        patches = batch["patch"].to(device, non_blocking=True)
        targets = {
            k: v.to(device, non_blocking=True) for k, v in batch.items() if k != "patch"
        }
        if not torch.isfinite(patches).all():
            logger.warning(f"non-finite values in input patch at batch {batch_idx}")
            patches = torch.nan_to_num(patches, nan=0.0, posinf=10.0, neginf=-10.0)

        out = model(patches)
        loss, log = criterion(out, patches, targets)
        if not torch.isfinite(loss):
            raise RuntimeError(
                f"non-finite loss at batch {batch_idx}:\n"
                + describe_diverged_batch(log, patches, out)
            )

        optimizer.zero_grad()
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        criterion.step()

        total_loss += loss.item()
        for key, value in log.items():
            totals[key] = totals.get(key, 0.0) + value
        n_batches += 1

    return total_loss / n_batches, {k: v / n_batches for k, v in totals.items()}


@torch.no_grad()
def validate(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: VAELoss,
    device: torch.device,
) -> tuple[float, dict[str, float]]:
    """Validation loss plus latent-health and auxiliary-R^2 diagnostics."""
    model.eval()
    total_loss = 0.0
    totals: dict[str, float] = {}
    n_batches = 0

    all_mu: list[torch.Tensor] = []
    preds: dict[str, list[torch.Tensor]] = {name: [] for name in model.aux_heads}
    truths: dict[str, list[torch.Tensor]] = {name: [] for name in model.aux_heads}

    for batch in loader:
        patches = batch["patch"].to(device, non_blocking=True)
        targets = {
            k: v.to(device, non_blocking=True) for k, v in batch.items() if k != "patch"
        }
        out = model(patches)
        loss, log = criterion(out, patches, targets)
        if not torch.isfinite(loss):
            logger.warning("non-finite validation loss, skipping batch")
            continue

        total_loss += loss.item()
        for key, value in log.items():
            totals[key] = totals.get(key, 0.0) + value
        n_batches += 1

        all_mu.append(out["mu"].cpu())
        for name in model.aux_heads:
            if f"aux_{name}" in out:
                preds[name].append(out[f"aux_{name}"].cpu())
            if name in targets:
                truths[name].append(targets[name].cpu())

    if n_batches == 0:
        return float("inf"), {}

    logs = {k: v / n_batches for k, v in totals.items()}

    for name in model.aux_heads:
        if not (preds[name] and truths[name]):
            continue
        y_pred = torch.cat(preds[name]).numpy()
        y_true = torch.cat(truths[name]).numpy()
        valid = np.isfinite(y_true) & np.isfinite(y_pred)
        if valid.sum() > 10:
            ss_res = ((y_true[valid] - y_pred[valid]) ** 2).sum()
            ss_tot = ((y_true[valid] - y_true[valid].mean()) ** 2).sum()
            logs[f"aux_r2/{name}"] = 1 - ss_res / (ss_tot + 1e-8)

    mu = torch.cat(all_mu).numpy()
    logs["latent/mean_norm"] = float(np.linalg.norm(mu, axis=1).mean())
    logs["latent/std_across_batch"] = float(mu.std(axis=0).mean())
    logs["latent/active_dims"] = int((mu.std(axis=0) > 0.1).sum())

    return total_loss / n_batches, logs


def format_epoch(epoch: int, train_loss: float, val_loss: float, log: dict) -> str:
    """One line of the per-epoch table (see :data:`EPOCH_HEADER`)."""
    return (
        f"{epoch + 1:>6} {train_loss:>8.4f} {val_loss:>8.4f} "
        f"{log['loss/recon']:>8.4f} {log['loss/kl']:>7.2f} {log['loss/beta']:>7.4f} "
        f"{log['latent/active_dims']:>6} "
        f"{log.get('aux_r2/elevation', float('nan')):>8.3f} "
        f"{log.get('aux_r2/latitude', float('nan')):>7.3f} "
        f"{log.get('aux_r2/longitude', float('nan')):>7.3f}"
    )


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    args = build_parser().parse_args()
    with open(args.config) as f:
        cfg = apply_overrides(yaml.safe_load(f), args)

    outdir = (
        resolve(args.outdir)
        if args.outdir
        else default_runs_dir() / make_run_name(cfg, args.suffix)
    )
    outdir.mkdir(parents=True, exist_ok=True)
    device = select_device(args.device)
    logger.info(f"Run directory: {outdir}")
    logger.info(f"Device: {device}")

    torch.manual_seed(cfg["training"]["seed"])
    np.random.seed(cfg["training"]["seed"])

    train_loader, val_loader, dataset = create_dataloaders(
        cfg, cache_dir=args.cache_dir, rebuild_cache=args.rebuild_cache
    )
    # The convolution arithmetic depends on the patch the dataset actually
    # emits, so record its size and channel count before building the model;
    # the saved config then reproduces the run exactly.
    cfg["model"]["input_size"] = dataset.spatial_size
    cfg["model"]["in_channels"] = int(dataset.mmap.shape[-1])
    with open(outdir / "config.yaml", "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)

    model = build_model(cfg).to(device)
    logger.info(
        f"Model: {sum(p.numel() for p in model.parameters()):,} params "
        f"(encoder {sum(p.numel() for p in model.encoder.parameters()):,}, "
        f"decoder {sum(p.numel() for p in model.decoder.parameters()):,}, "
        f"aux {sum(p.numel() for p in model.aux_heads.parameters()):,}), "
        f"latent_dim={cfg['model']['latent_dim']}"
    )

    criterion = VAELoss(cfg["loss"], cfg.get("auxiliary", {}).get("weights", {}))
    train_cfg = cfg["training"]
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=train_cfg["lr"], weight_decay=train_cfg["weight_decay"]
    )
    scheduler = None
    if train_cfg["scheduler"] == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=train_cfg["epochs"]
        )

    def snapshot(path: Path, epoch: int, val_loss: float) -> None:
        torch.save(
            {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict() if scheduler else None,
                "loss_step": criterion.current_step,
                "val_loss": val_loss,
                "config": cfg,
            },
            path,
        )

    start_epoch = 0
    history: list[dict] = []
    best_val_loss = float("inf")
    if args.resume:
        ckpt = torch.load(resolve(args.resume), map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        if scheduler and ckpt.get("scheduler"):
            scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        criterion.current_step = ckpt.get("loss_step", 0)

        # history.json is rewritten every epoch, so starting from an empty list
        # would discard everything trained before the resume. Keep the epochs
        # the checkpoint covers -- a timeout can leave history.json ahead of the
        # last periodic checkpoint -- and restore the best loss so far, else the
        # first resumed epoch would overwrite a better best.pt.
        history_path = outdir / "history.json"
        if history_path.exists():
            with open(history_path) as f:
                history = [r for r in json.load(f) if r["epoch"] < start_epoch]
            if history:
                best_val_loss = min(r["val_loss"] for r in history)
        logger.info(
            f"Resumed at epoch {start_epoch} ({len(history)} prior epochs kept, "
            f"best val {best_val_loss:.4f})"
        )

    logger.info(
        f"Training {train_cfg['epochs']} epochs, batch {train_cfg['batch_size']}, "
        f"lr {train_cfg['lr']:g}, beta -> {cfg['loss']['beta_end']:g} over "
        f"{cfg['loss'].get('beta_warmup_steps', 1000)} steps"
    )
    logger.info(EPOCH_HEADER)

    val_loss = float("inf")
    for epoch in range(start_epoch, train_cfg["epochs"]):
        train_loss, train_log = train_one_epoch(
            model, train_loader, optimizer, criterion, device, train_cfg["grad_clip"]
        )
        val_loss, val_log = validate(model, val_loader, criterion, device)
        if scheduler:
            scheduler.step()

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "lr": optimizer.param_groups[0]["lr"],
                **{f"train/{k}": v for k, v in train_log.items()},
                **{f"val/{k}": v for k, v in val_log.items()},
            }
        )
        logger.info(format_epoch(epoch, train_loss, val_loss, val_log))

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            snapshot(outdir / "best.pt", epoch, val_loss)
        if (epoch + 1) % 20 == 0:
            snapshot(outdir / f"checkpoint_epoch{epoch + 1}.pt", epoch, val_loss)
        # Rewritten every epoch so a crashed run keeps its metrics.
        with open(outdir / "history.json", "w") as f:
            json.dump(history, f, indent=2, default=str)

    snapshot(outdir / "last.pt", train_cfg["epochs"] - 1, val_loss)
    logger.info(
        f"Done -- best validation loss {best_val_loss:.4f}, outputs in {outdir}"
    )


if __name__ == "__main__":
    main()
