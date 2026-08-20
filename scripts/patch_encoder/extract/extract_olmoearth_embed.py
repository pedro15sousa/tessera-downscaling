#!/usr/bin/env python3
"""Stage 2 of the OlmoEarth arm: encode the stored imagery into embeddings.

Runs the OlmoEarth encoder (a direct forward pass through the
``olmoearth-pretrain`` package -- no rslearn) over the monthly Sentinel-2
composites written by ``extract_olmoearth_imagery.py``, and stores one
token-grid embedding patch per station-year:

    (N, S/p, S/p, D) float32     e.g. S = 64, p = 4, v1 Base -> (N, 16, 16, 768)

Unlike TESSERA and AlphaEarth patches, these are *token* grids rather than 10 m
pixel rasters: one token covers p x p pixels (40 m at the defaults), so a 16x16
grid spans the same 640 m window as the 64 px crop used for the other two
sources. The VAE config for this source therefore uses a three-stage encoder and
no runtime crop (``scripts/patch_encoder/vae_olmoearth.yaml``).

Per batch the pipeline is:

    uint16 DNs -> float32 -> Normalizer(Strategy.COMPUTED) min-max scaling ->
    MaskedOlmoEarthSample with a per-month mask (ONLINE_ENCODER where the month
    has imagery, MISSING where it does not) and (day=15, month, year)
    timestamps -> encoder(..., fast_pass=True, patch_size=p) ->
    tokens (B, S/p, S/p, T, band sets, D) -> mean over the band-set axis, then
    the mean over the *valid* months only.

The normaliser's output is cast back to float32 explicitly: it promotes to
float64, which the convolutions reject.

Stations with no valid month keep an all-zero row, the same "invalid row"
convention as the TESSERA and AlphaEarth patch files. Resume is stateless: a
station is recomputed exactly when its output row is all zero while it has at
least one valid month (a transformer never outputs exactly zero, so finished
rows are never redone).

Optional dependency (not in the default environment): ``olmoearth-pretrain``.
It is imported inside the functions that need it, so ``--help`` works without
it.

Usage (relative paths are interpreted under the data root):

    # Smoke test on CPU with the Nano model.
    uv run python scripts/patch_encoder/extract/extract_olmoearth_embed.py \\
        --years 2024 --model NANO --device cpu --limit 10

    # Full run on one GPU (Base, ViT patch size 4).
    uv run python scripts/patch_encoder/extract/extract_olmoearth_embed.py \\
        --years 2017 2024

Output files, per year, in ``--output-dir``:

    patch_embeddings_olmoearth_<year>_g<S/p>.npy   (N, S/p, S/p, D) float32
    embed_metadata.json                            model, pooling, parameters
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np

from tessera_downscaling.paths import processed_dir, resolve

logger = logging.getLogger("extract_olmoearth_embed")

N_MONTHS = 12
N_BANDSETS = 3  # sentinel2_l2a band-set groups (10 m / 20 m / 60 m)
# Embedding width per OlmoEarth v1 size, used to catch a mismatched resume.
EMBED_DIM = {
    "OLMOEARTH_V1_NANO": 128,
    "OLMOEARTH_V1_TINY": 192,
    "OLMOEARTH_V1_BASE": 768,
    "OLMOEARTH_V1_LARGE": 1024,
}


def encode_year(
    year: int,
    imagery_dir: Path,
    output_dir: Path,
    model,
    normalizer,
    spec,
    model_name: str,
    patch_size: int,
    in_patch: int,
    batch_size: int,
    device: str,
    limit: int | None,
) -> dict:
    """Encode one year of stored imagery into its token-grid embedding file."""
    import torch
    from olmoearth_pretrain.datatypes import MaskedOlmoEarthSample, MaskValue

    cube_path = imagery_dir / f"s2_{year}_p{in_patch}.npy"
    months_path = imagery_dir / f"s2_{year}_p{in_patch}_months.npy"
    if not cube_path.exists():
        raise SystemExit(
            f"{cube_path} not found -- run extract_olmoearth_imagery.py first"
        )
    cube = np.load(str(cube_path), mmap_mode="r")  # (N, S, S, 12, 12)
    valid = np.load(str(months_path), mmap_mode="r")  # (N, 12)
    n_stations, side = cube.shape[0], cube.shape[1]
    grid = side // patch_size

    out_path = output_dir / f"patch_embeddings_olmoearth_{year}_g{grid}.npy"
    out = None
    if out_path.exists():
        out = np.lib.format.open_memmap(str(out_path), mode="r+")
        want = EMBED_DIM.get(model_name)
        if want is not None and out.shape[-1] != want:
            raise SystemExit(
                f"{out_path} has embedding dim {out.shape[-1]} but {model_name} "
                f"produces {want} -- delete it or use a different output dir."
            )
        logger.info(f"Opening existing {out_path} (stateless resume)")

    idx_all = np.arange(n_stations if limit is None else min(limit, n_stations))
    has_data = valid[idx_all].any(axis=1)
    if out is not None:
        done = out[idx_all].reshape(len(idx_all), -1).any(axis=1)
        todo = idx_all[has_data & ~done]
    else:
        todo = idx_all[has_data]
    n_no_data = int((~has_data).sum())
    logger.info(
        f"Year {year}: {len(todo)} stations to encode ({n_no_data} without "
        f"imagery, {len(idx_all) - len(todo) - n_no_data} already done)"
    )

    started = time.time()
    n_done = 0
    for batch_start in range(0, len(todo), batch_size):
        ids = todo[batch_start : batch_start + batch_size]
        n_batch = len(ids)
        images = np.asarray(cube[ids], dtype=np.float32)  # (B, S, S, 12, 12)
        images = normalizer.normalize(spec, images).astype(np.float32)
        month_mask = np.asarray(valid[ids], dtype=np.float32)  # (B, 12)

        mask = np.where(
            month_mask[:, None, None, :, None] > 0,
            float(MaskValue.ONLINE_ENCODER.value),
            float(MaskValue.MISSING.value),
        )
        mask = np.broadcast_to(mask, (n_batch, side, side, N_MONTHS, N_BANDSETS)).copy()
        timestamps = np.stack(
            [
                np.stack(
                    [
                        np.full(N_MONTHS, 15),
                        np.arange(N_MONTHS),
                        np.full(N_MONTHS, year),
                    ],
                    axis=-1,
                )
            ]
            * n_batch
        )

        sample = MaskedOlmoEarthSample(
            sentinel2_l2a=torch.from_numpy(images).to(device),
            sentinel2_l2a_mask=torch.from_numpy(mask).float().to(device),
            timestamps=torch.from_numpy(timestamps).long().to(device),
        )
        with torch.no_grad():
            result = model.encoder(sample, fast_pass=True, patch_size=patch_size)
        feats = result["tokens_and_masks"].sentinel2_l2a  # (B, g, g, T, 3, D)
        feats = feats.float().mean(dim=4)  # over the band sets
        weights = torch.from_numpy(month_mask).to(feats.device)  # (B, T)
        weights = weights / weights.sum(dim=1, keepdim=True).clamp(min=1)
        feats = (feats * weights[:, None, None, :, None]).sum(dim=3)  # valid months
        feats = feats.cpu().numpy().astype(np.float32)

        if out is None:
            dim = feats.shape[-1]
            logger.info(
                f"Pre-allocating ({n_stations}, {grid}, {grid}, {dim}) float32 "
                f"({n_stations * grid * grid * dim * 4 / 1e9:.1f} GB) at {out_path}"
            )
            out = np.lib.format.open_memmap(
                str(out_path),
                mode="w+",
                dtype=np.float32,
                shape=(n_stations, grid, grid, dim),
            )
        out[ids] = feats
        n_done += n_batch
        if (batch_start // batch_size) % 10 == 0:
            out.flush()
            rate = n_done / max(time.time() - started, 1)
            logger.info(
                f"  {n_done}/{len(todo)} ({rate:.1f} st/s, ETA "
                f"{(len(todo) - n_done) / max(rate, 1e-9) / 3600:.1f} h)"
            )
    if out is not None:
        out.flush()
    logger.info(f"Year {year}: done ({n_done} encoded)")
    return {
        "file": out_path.name,
        "shape": list(out.shape) if out is not None else None,
        "n_no_imagery": n_no_data,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Encode the fetched Sentinel-2 imagery with OlmoEarth."
    )
    parser.add_argument(
        "--imagery-dir",
        default=str(processed_dir("olmoearth_imagery")),
        help="Directory written by extract_olmoearth_imagery.py.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(processed_dir("olmoearth_station_patches")),
        help="Where the token-grid patch file and metadata are written.",
    )
    parser.add_argument("--years", type=int, nargs="+", default=[2017, 2024])
    parser.add_argument(
        "--model", default="BASE", help="NANO | TINY | BASE | LARGE (OlmoEarth v1)."
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        default=4,
        help="ViT patch size in pixels; a 64 px input with p=4 gives a 16x16 "
        "token grid at 40 m per token.",
    )
    parser.add_argument(
        "--in-patch",
        type=int,
        default=64,
        help="Input imagery patch size (must match stage 1).",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--device", default=None, help="cuda | cpu (auto-detected if unset)."
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only the first N stations (smoke test).",
    )
    return parser


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    args = build_parser().parse_args()

    import torch
    from olmoearth_pretrain.data.constants import Modality
    from olmoearth_pretrain.data.normalize import Normalizer, Strategy
    from olmoearth_pretrain.model_loader import ModelID, load_model_from_id

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model_id = getattr(ModelID, f"OLMOEARTH_V1_{args.model.upper()}")
    logger.info(f"Loading {model_id.name} on {device}")
    model = load_model_from_id(model_id)
    model.eval().to(device)
    normalizer = Normalizer(Strategy.COMPUTED)
    spec = Modality.SENTINEL2_L2A

    output_dir = resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    imagery_dir = resolve(args.imagery_dir)

    meta_path = output_dir / "embed_metadata.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    meta.setdefault("files", {})
    meta.update(
        {
            "model": model_id.name,
            "license": "OlmoEarth Artifact License (Ai2) -- attribution required",
            "vit_patch_size": args.patch_size,
            "input": f"{args.in_patch}px @10 m S2 L2A monthly composites "
            f"(T<=12), COMPUTED normalization",
            "pooling": "mean over band-set dim, weighted mean over valid months",
            "layout": "(N, S/p, S/p, D) float32, north-up, station at grid "
            "centre; zero rows = no imagery",
        }
    )

    for year in args.years:
        logger.info(f"=== Year {year} ===")
        info = encode_year(
            year,
            imagery_dir,
            output_dir,
            model,
            normalizer,
            spec,
            model_id.name,
            args.patch_size,
            args.in_patch,
            args.batch_size,
            device,
            args.limit,
        )
        meta["files"].setdefault(info["file"], {}).update(
            {k: v for k, v in info.items() if k != "file"}
        )
    meta_path.write_text(json.dumps(meta, indent=2))
    logger.info(f"Metadata saved to {meta_path}")


if __name__ == "__main__":
    main()
