#!/usr/bin/env python3
"""Shuffle VAE-latent rows to break the (station, latent) mapping.

Produces a sibling ``.npy`` for the shuffled-latent control runs.
The shuffle preserves NaN-row positions exactly (rows with any NaN in
the unshuffled file stay NaN in the same positions), so the dataset
loader's NaN-based station filter yields the identical station set on
the shuffled run as on the unshuffled run. Only the (station_id, latent
vector) mapping is broken: each station now receives a latent that
originally belonged to some other (non-NaN-row) station.

This is a one-shot preprocessing step. Run once per latent file you
want to shuffle; point the shuffled-control experiment YAML entries at
the output path via ``--vae-latents-path``.

Usage
-----
    python scripts/shuffle_latents.py \\
        --input  .tmp_output/processed/station_latents_lat16_grad0.5.npy \\
        --output .tmp_output/processed/station_latents_lat16_grad0.5_shuffle_seed0.npy \\
        --seed   0
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Row-shuffle a VAE latents .npy for the shuffled-latent control."
    )
    parser.add_argument("--input", type=Path, required=True,
                        help="Path to unshuffled latents .npy.")
    parser.add_argument("--output", type=Path, required=True,
                        help="Path to write shuffled latents .npy. "
                             "Must be different from --input.")
    parser.add_argument("--seed", type=int, required=True,
                        help="Random seed for the permutation. The shuffled "
                             "file is fully deterministic from this seed.")
    args = parser.parse_args()

    if not args.input.exists():
        raise SystemExit(f"Input file not found: {args.input}")
    if args.input.resolve() == args.output.resolve():
        raise SystemExit("--input and --output must be different paths.")

    latents = np.load(args.input)
    if latents.ndim != 2:
        raise SystemExit(
            f"Expected 2-D (n_stations, latent_dim) array; got shape {latents.shape}."
        )
    n_total, latent_dim = latents.shape
    print(f"Loaded {args.input}: shape={latents.shape}, dtype={latents.dtype}")

    # Rows with any NaN are filtered out by the dataset loader (see
    # filter_stations_by_vae_latents in helpers.py — it checks
    # valid_mask_global[row] before keeping a station). Preserving NaN
    # positions therefore preserves the filtered station set.
    row_is_valid = ~np.isnan(latents).any(axis=1)
    n_valid = int(row_is_valid.sum())
    print(f"Valid (non-NaN) rows: {n_valid}/{n_total}")
    if n_valid < 2:
        raise SystemExit("Need at least 2 valid rows to shuffle.")

    rng = np.random.default_rng(args.seed)
    valid_row_indices = np.where(row_is_valid)[0]
    perm_of_valid = rng.permutation(n_valid)

    # new_to_old[i] = which row of the original goes to position i of
    # the output. Identity for invalid rows (NaN stays NaN at the same
    # position); shuffled-within-valid for valid rows.
    new_to_old = np.arange(n_total)
    new_to_old[valid_row_indices] = valid_row_indices[perm_of_valid]

    shuffled = latents[new_to_old]

    # Invariant: NaN positions must be unchanged.
    new_valid = ~np.isnan(shuffled).any(axis=1)
    assert np.array_equal(row_is_valid, new_valid), (
        "NaN positions changed after shuffle — invariant violated."
    )

    # Invariant: every valid row must be a permutation of the originals.
    n_fixed_points = int(
        (new_to_old[valid_row_indices] == valid_row_indices).sum()
    )
    print(f"Permutation fixed points (rows where shuffle is a no-op): "
          f"{n_fixed_points}/{n_valid}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, shuffled)
    perm_path = args.output.with_name(args.output.stem + ".perm.npy")
    np.save(perm_path, new_to_old)

    print(f"Wrote shuffled latents: {args.output}")
    print(f"Wrote row mapping (new→old): {perm_path}")
    print(f"Seed: {args.seed}")
    print(f"Spot check — valid row {valid_row_indices[0]} now contains "
          f"the latent originally at row {new_to_old[valid_row_indices[0]]}.")


if __name__ == "__main__":
    main()