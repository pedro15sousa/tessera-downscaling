#!/usr/bin/env python3
"""Encode a dense grid of TESSERA patches with a trained VAE.

The map figures need a descriptor at every point of a regular 0.05 degree grid,
not just at stations. ``scripts/maps/extract_dense_grid_patches.py`` writes the
patches for such a grid; this script turns them into latents, the dense-grid
counterpart of ``eval_vae.py`` (no station list, no probes).

Input patches must be ``(N, 64, 64, 128)`` -- the grid extractor already cuts
them to the final 64 px window, so nothing is cropped here and the checkpoint
must be one trained at ``input_size = 64``.

The one thing that must not go wrong is normalisation: the grid is z-scored
with the per-channel statistics of the run's *training* extraction, never with
statistics recomputed from the grid, or the latents will not live in the same
space as the station latents the downscaler was fitted on. Those statistics are
read from the dataset cache of the patch file recorded in the checkpoint;
``--cache`` overrides the lookup.

Grid points whose patch is all zero or holds extreme values (ocean, missing
tiles) are written as NaN rows and flagged in ``valid_mask``, matching the
station pipeline's convention.

Usage:

    uv run python scripts/patch_encoder/encode_dense_grid.py \\
        --patches processed/tessera_dense_grid/norway_0.05deg_2024/patch_embeddings.npy \\
        --run-dir tessera_patch_encoder/outputs/vae/lat16_beta0.0005_grad0.5_e200 \\
        --coords  processed/tessera_dense_grid/norway_0.05deg_2024/grid_points.csv \\
        --out     processed/dense/norway/norway_0.05deg_2024.npz

Relative paths are interpreted under the data root.
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from tessera_downscaling.patch_encoder.dataset import (
    OUTLIER_THRESHOLD,
    default_cache_dir,
)
from tessera_downscaling.patch_encoder.model import build_model
from tessera_downscaling.paths import resolve

logger = logging.getLogger("encode_dense_grid")

DENSE_PATCH_SHAPE = (64, 64, 128)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Encode dense-grid TESSERA patches.")
    parser.add_argument(
        "--patches", required=True, help="Dense-grid patches .npy, (N, 64, 64, 128)."
    )
    parser.add_argument(
        "--run-dir", required=True, help="Run directory of the encoder."
    )
    parser.add_argument("--out", required=True, help="Output .npz path.")
    parser.add_argument("--checkpoint", default="best.pt")
    parser.add_argument(
        "--cache",
        default=None,
        help="cache.npz holding the training channel_mean/channel_std. Default: "
        "the cache of the patch file recorded in the checkpoint.",
    )
    parser.add_argument(
        "--coords",
        default=None,
        help="Per-patch coordinates (.csv or .npy) copied through for gridding.",
    )
    parser.add_argument("--batch-size", type=int, default=256)
    return parser


def resolve_cache_path(cfg: dict, cache_dir: Path) -> tuple[Path, bool]:
    """Locate the channel statistics that trained this checkpoint.

    Caches are namespaced by the patch filename, so the statistics live in
    ``<cache_dir>/<patch stem>/cache.npz``. Runs from before that namespacing
    (the v1 encoder behind the dense maps) kept theirs at the top level;
    falling back to it is flagged, because silently normalising with another
    extraction's statistics produces latents that look fine and are wrong.

    Returns the path and whether it is that legacy fallback.
    """
    stem = Path(str(cfg.get("data", {}).get("patches_path", ""))).stem
    namespaced = cache_dir / stem / "cache.npz"
    if namespaced.exists():
        return namespaced, False
    legacy = cache_dir / "cache.npz"
    if stem and legacy.exists():
        return legacy, True
    return namespaced, False


def load_channel_stats(cache_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read ``channel_mean``/``channel_std`` from a dataset cache."""
    if not cache_path.exists():
        available = sorted(str(p) for p in cache_path.parent.parent.glob("*/cache.npz"))
        raise SystemExit(
            f"Channel statistics not found at {cache_path}. These are computed "
            f"during training and must be reused, never recomputed from the "
            f"grid. Available caches:\n    " + ("\n    ".join(available) or "(none)")
        )
    npz = np.load(cache_path)
    if "channel_mean" not in npz.files or "channel_std" not in npz.files:
        raise SystemExit(f"{cache_path} has no channel_mean/channel_std: {npz.files}")
    return (
        npz["channel_mean"].astype(np.float32),
        np.maximum(npz["channel_std"].astype(np.float32), 1e-6),
    )


def load_coords(coords_path: Path, n_patches: int) -> dict[str, np.ndarray]:
    """Load per-patch coordinates to copy into the output verbatim."""
    if coords_path.suffix == ".npy":
        extra = {"coords": np.load(coords_path)}
    elif coords_path.suffix in (".csv", ".tsv"):
        sep = "\t" if coords_path.suffix == ".tsv" else ","
        frame = pd.read_csv(coords_path, sep=sep)
        extra = {
            "coords": frame.to_records(index=False),
            "coord_columns": np.array(list(frame.columns)),
        }
    else:
        logger.warning(f"unrecognised --coords extension, skipping: {coords_path}")
        return {}
    if len(extra["coords"]) != n_patches:
        logger.warning(f"coords length {len(extra['coords'])} != {n_patches} patches")
    return extra


@torch.no_grad()
def encode_grid(
    model: torch.nn.Module,
    patches: np.ndarray,
    ch_mean: np.ndarray,
    ch_std: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Stream the grid through the encoder; returns latents and a validity mask."""
    n_patches = patches.shape[0]
    latents = np.full((n_patches, model.latent_dim), np.nan, dtype=np.float32)
    valid_mask = np.zeros(n_patches, dtype=bool)
    started = time.time()

    for start in range(0, n_patches, batch_size):
        end = min(start + batch_size, n_patches)
        chunk = np.asarray(patches[start:end], dtype=np.float32)

        # Per-patch validity, mirroring the scan in patch_encoder.dataset.
        max_abs = np.abs(chunk).reshape(chunk.shape[0], -1).max(axis=1)
        ok = np.isfinite(max_abs) & (max_abs > 0) & (max_abs <= OUTLIER_THRESHOLD)
        if not ok.any():
            continue

        good = (chunk[ok] - ch_mean) / ch_std  # broadcast over the channel axis
        good = np.transpose(good, (0, 3, 1, 2))  # -> (b, 128, 64, 64)
        x = torch.from_numpy(np.ascontiguousarray(good)).to(device)

        idx = np.arange(start, end)[ok]
        latents[idx] = model.encode(x).cpu().numpy()
        valid_mask[idx] = True

        if start % (batch_size * 20) == 0 or end == n_patches:
            rate = end / max(time.time() - started, 1e-9)
            logger.info(f"  {end}/{n_patches} ({rate:.0f} patch/s)")

    return latents, valid_mask


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    args = build_parser().parse_args()

    run_dir = resolve(args.run_dir)
    out_path = resolve(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_path = run_dir / args.checkpoint
    if not ckpt_path.exists():
        raise SystemExit(f"Checkpoint not found: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    if cfg["model"].get("input_size", 64) != DENSE_PATCH_SHAPE[0]:
        raise SystemExit(
            f"This run encodes {cfg['model']['input_size']} px patches, but dense "
            f"grids are extracted at {DENSE_PATCH_SHAPE[0]} px. Use a run trained "
            f"with that input size."
        )
    model = build_model(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    logger.info(
        f"Run {run_dir} (epoch {ckpt['epoch'] + 1}, val_loss={ckpt['val_loss']:.4f}, "
        f"latent_dim={model.latent_dim}), device {device}"
    )

    if args.cache:
        cache_path, is_legacy = resolve(args.cache), False
    else:
        cache_path, is_legacy = resolve_cache_path(cfg, default_cache_dir())
    ch_mean, ch_std = load_channel_stats(cache_path)
    logger.info(f"Trained on {cfg.get('data', {}).get('patches_path', '?')}")
    logger.info(f"Channel statistics from {cache_path}")
    if is_legacy:
        logger.warning(
            "using the pre-namespacing top-level cache.npz; check that it belongs "
            "to this run's extraction, or the latents will not match its station "
            "latents (pass --cache to be explicit)"
        )

    patches = np.load(resolve(args.patches), mmap_mode="r")
    if patches.ndim != 4 or patches.shape[1:] != DENSE_PATCH_SHAPE:
        raise SystemExit(
            f"Expected patches of shape (N, 64, 64, 128), got {patches.shape}. "
            f"Reshape or concatenate the grid first."
        )
    logger.info(f"Patches {patches.shape} from {args.patches}")

    latents, valid_mask = encode_grid(
        model, patches, ch_mean, ch_std, device, args.batch_size
    )
    n_valid = int(valid_mask.sum())
    logger.info(
        f"Encoded {n_valid}/{len(valid_mask)} patches "
        f"({len(valid_mask) - n_valid} zero/outlier -> NaN); active latent dims "
        f"(std>0.1) {int((np.nanstd(latents, axis=0) > 0.1).sum())}/{model.latent_dim}"
    )

    extra = load_coords(resolve(args.coords), len(valid_mask)) if args.coords else {}
    np.savez(
        out_path,
        Z=latents,
        valid_mask=valid_mask,
        latent_dim=model.latent_dim,
        run_name=run_dir.name,
        checkpoint=args.checkpoint,
        checkpoint_epoch=ckpt["epoch"] + 1,
        patches_path=str(args.patches),
        channel_stats_source=str(cache_path),
        **extra,
    )
    # A bare .npy of the latents alongside, for consumers that want just those.
    np.save(out_path.with_suffix(".Z.npy"), latents)
    logger.info(f"Wrote {out_path} and {out_path.with_suffix('.Z.npy')}")


if __name__ == "__main__":
    main()
