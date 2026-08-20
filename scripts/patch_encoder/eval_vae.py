#!/usr/bin/env python3
"""Encode every station patch with a trained VAE and score the run.

Rebuilds the model from the config stored in the checkpoint, runs the encoder
over all usable stations and writes into ``<run_dir>/eval/``:

    station_latents.npy         (n_stations, latent_dim) float32, row-aligned
                                with the station CSV; NaN rows are the stations
                                the encoder cannot describe (no TESSERA
                                coverage, corrupted patch, elevation sentinel)
    latents.npz                 the same latents without the NaN padding, plus
                                log-variances, their global row indices and
                                per-dimension statistics
    reconstruction_metrics.npz  per-channel reconstruction MSE over a random
                                subset, and a handful of patch/reconstruction
                                pairs to look at
    probe_metrics.json          5-fold cross-validated ridge regression from
    probe_predictions.npz       the frozen latents to station elevation

``station_latents.npy`` is the artefact the downscaler consumes: copy it to
``processed/vae_tessera_1B-M/`` (recording the run in ``provenance.txt``) and
point ``tessera-train --vae-latents-path`` at it. Its row alignment with
``station_list_filtered.csv`` is the contract that lets the downscaler look a
station's descriptor up by ``station_id``.

Usage:

    uv run python scripts/patch_encoder/eval_vae.py \\
        tessera_patch_encoder/outputs/vae/p128_2017_crop64_lat16_grad0.5_auxon

Relative paths are interpreted under the data root. The patch file and station
CSV default to the ones the run trained on; when that recorded path no longer
exists (runs trained on the HPC store its absolute paths) the file of the same
name under ``processed/tessera_station_patches/`` is used instead.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from tessera_downscaling.patch_encoder.dataset import (
    ELEV_SENTINEL_HIGH,
    ELEV_SENTINEL_LOW,
    STATION_PATCH_DIR,
    TesseraPatchDataset,
    filter_elevation_sentinels,
    prepare_data,
)
from tessera_downscaling.patch_encoder.model import build_model
from tessera_downscaling.paths import processed_dir, resolve

logger = logging.getLogger("eval_vae")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained patch-encoder run."
    )
    parser.add_argument("run_dir", help="Run directory written by train_vae.py.")
    parser.add_argument(
        "--checkpoint", default="best.pt", help="Checkpoint inside the run directory."
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--patches-path", default=None, help="Override the run's patch file."
    )
    parser.add_argument(
        "--stations-path", default=None, help="Override the run's station CSV."
    )
    parser.add_argument("--cache-dir", default=None, help="Root of the cache tree.")
    parser.add_argument(
        "--n-recon-eval",
        type=int,
        default=500,
        help="Patches used for the per-channel reconstruction-MSE distribution.",
    )
    parser.add_argument(
        "--n-recon-samples",
        type=int,
        default=12,
        help="Patch/reconstruction pairs saved for qualitative inspection.",
    )
    return parser


def locate_data_file(recorded: str, override: str | None) -> Path:
    """Resolve a data path recorded in a checkpoint config.

    ``--patches-path`` / ``--stations-path`` win. Otherwise the recorded path
    is used if it exists; runs trained on the HPC recorded absolute paths that
    the data root does not reproduce, so as a last resort we take the file of
    the same name under ``processed/tessera_station_patches/``.
    """
    if override is not None:
        return resolve(override)
    path = resolve(recorded)
    if path.exists():
        return path
    fallback = processed_dir(STATION_PATCH_DIR, path.name)
    if fallback.exists():
        logger.warning(f"{path} does not exist; using {fallback}")
        return fallback
    raise SystemExit(
        f"Neither {path} (recorded in the checkpoint) nor {fallback} exists; "
        f"pass --patches-path / --stations-path explicitly."
    )


@torch.no_grad()
def encode_all(
    model: torch.nn.Module,
    dataset: TesseraPatchDataset,
    device: torch.device,
    batch_size: int,
    num_workers: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Posterior mean and log-variance of every patch in ``dataset``."""
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    mus, logvars = [], []
    n_done = 0
    for batch in loader:
        x = batch["patch"].to(device, non_blocking=True)
        mu, logvar = model.encoder(x)
        mus.append(mu.cpu().numpy())
        logvars.append(logvar.cpu().numpy())
        n_done += x.size(0)
        if n_done % (batch_size * 20) == 0 or n_done == len(dataset):
            logger.info(f"  encoded {n_done}/{len(dataset)}")
    return np.concatenate(mus), np.concatenate(logvars)


@torch.no_grad()
def reconstruction_metrics(
    model: torch.nn.Module,
    dataset: TesseraPatchDataset,
    device: torch.device,
    n_eval: int,
    n_samples: int,
) -> dict[str, np.ndarray]:
    """Per-channel reconstruction MSE plus a few patch/reconstruction pairs."""
    rng = np.random.RandomState(0)

    per_channel_mse = []
    for idx in rng.choice(len(dataset), min(n_eval, len(dataset)), replace=False):
        x = dataset[int(idx)]["patch"].unsqueeze(0).to(device)
        recon = model(x)["x_recon"]
        per_channel_mse.append((recon - x).pow(2).mean(dim=(0, 2, 3)).cpu().numpy())

    sample_idx = rng.choice(len(dataset), n_samples, replace=False)
    originals, reconstructions = [], []
    for idx in sample_idx:
        x = dataset[int(idx)]["patch"]
        recon = model(x.unsqueeze(0).to(device))["x_recon"].squeeze(0).cpu()
        originals.append(x.numpy())
        reconstructions.append(recon.numpy())

    return {
        "per_channel_mse": np.array(per_channel_mse),
        "sample_indices": sample_idx,
        "sample_originals": np.array(originals),
        "sample_reconstructions": np.array(reconstructions),
    }


def probe_elevation(latents: np.ndarray, dataset: TesseraPatchDataset) -> dict:
    """Cross-validated ridge regression from the frozen latents to elevation.

    The headline diagnostic of the encoder: how much of the terrain the latent
    still carries after compression. Elevation is *also* an auxiliary training
    target when ``auxiliary.enable`` is set, so read the number as a check that
    the latent is well-formed rather than as an independent probe.
    """
    elevation = dataset.metadata["elevation"].to_numpy().astype(np.float32)
    valid = (elevation > ELEV_SENTINEL_LOW) & (elevation <= ELEV_SENTINEL_HIGH)
    x = StandardScaler().fit_transform(latents[valid])
    y = elevation[valid]

    y_pred = cross_val_predict(
        Ridge(alpha=1.0), x, y, cv=KFold(5, shuffle=True, random_state=42)
    )
    return {
        "r2": float(r2_score(y, y_pred)),
        "mae": float(mean_absolute_error(y, y_pred)),
        "rmse": float(np.sqrt(((y - y_pred) ** 2).mean())),
        "n": int(len(y)),
        "y_true": y,
        "y_pred": y_pred,
    }


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    args = build_parser().parse_args()

    run_dir = resolve(args.run_dir)
    ckpt_path = run_dir / args.checkpoint
    if not ckpt_path.exists():
        raise SystemExit(f"Checkpoint not found: {ckpt_path}")
    eval_dir = run_dir / "eval"
    eval_dir.mkdir(exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Run {run_dir}, checkpoint {args.checkpoint}, device {device}")

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    model = build_model(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    logger.info(
        f"Loaded epoch {ckpt['epoch'] + 1}, val_loss={ckpt['val_loss']:.4f}, "
        f"latent_dim={cfg['model']['latent_dim']}"
    )

    patches_path = locate_data_file(cfg["data"]["patches_path"], args.patches_path)
    stations_path = locate_data_file(cfg["data"]["stations_path"], args.stations_path)

    # Same two-stage station filter and same cached normalisation statistics as
    # training: latents are only comparable across runs if both match.
    cache = prepare_data(patches_path, cache_dir=args.cache_dir)
    usable_idx = filter_elevation_sentinels(cache["valid_indices"], stations_path)
    aux_cfg = cfg.get("auxiliary", {})
    dataset = TesseraPatchDataset(
        patches_path=patches_path,
        stations_path=stations_path,
        valid_indices=usable_idx,
        channel_mean=cache["channel_mean"],
        channel_std=cache["channel_std"],
        aux_targets=aux_cfg.get("targets", []) if aux_cfg.get("enable", False) else [],
        crop_size=cfg["data"].get("crop_size"),
    )
    logger.info(f"Encoding {len(dataset)} usable patches of {patches_path}")

    latents, logvars = encode_all(
        model, dataset, device, args.batch_size, args.num_workers
    )
    active_dims = int((latents.std(0) > 0.1).sum())
    logger.info(f"Latents {latents.shape}, active dims (std>0.1) {active_dims}")
    np.savez(
        eval_dir / "latents.npz",
        Z=latents,
        LV=logvars,
        global_indices=usable_idx,
        active_dims=active_dims,
        per_dim_std=latents.std(0),
        per_dim_mean=latents.mean(0),
    )

    n_stations = len(pd.read_csv(stations_path))
    station_latents = np.full((n_stations, latents.shape[1]), np.nan, dtype=np.float32)
    station_latents[usable_idx] = latents
    np.save(eval_dir / "station_latents.npy", station_latents)
    logger.info(
        f"Wrote {eval_dir / 'station_latents.npy'}: {n_stations} rows, "
        f"{int((~np.isnan(station_latents).any(1)).sum())} valid"
    )

    recon = reconstruction_metrics(
        model, dataset, device, args.n_recon_eval, args.n_recon_samples
    )
    np.savez(eval_dir / "reconstruction_metrics.npz", **recon)
    logger.info(
        f"Mean per-channel reconstruction MSE {recon['per_channel_mse'].mean():.4f}"
    )

    probe = probe_elevation(latents, dataset)
    np.savez(
        eval_dir / "probe_predictions.npz",
        y_true=probe.pop("y_true"),
        y_pred=probe.pop("y_pred"),
    )
    with open(eval_dir / "probe_metrics.json", "w") as f:
        json.dump(probe, f, indent=2)
    logger.info(
        f"Latent -> elevation (5-fold CV): R2={probe['r2']:.4f}, "
        f"MAE={probe['mae']:.1f} m, RMSE={probe['rmse']:.1f} m"
    )
    logger.info(f"Done -- artefacts in {eval_dir}")


if __name__ == "__main__":
    main()
