"""CLI: extract TESSERA patches at every point of a regular dense grid.

Used to build the inputs for high-resolution downscaling map comparisons
(e.g. "what does TESSERA's contribution look like across Iberia at 0.05°
resolution"), where we need a TESSERA patch centred at every grid point.

Output goes to ``.tmp_output/processed/tessera_dense_grid/<region>_<resolution>deg_<year>/``
with a memory-mapped ``patch_embeddings.npy`` plus a sibling ``grid_points.csv``
and metadata. The output format mirrors the existing
``processed/tessera_global/patch_embeddings_<year>.npy`` so downstream
consumers (VAE encoder, ConvCNP inference) can use the same code path.

Usage:
    # Predefined region (matches the analysis notebook's REGIONS dict)
    uv run --group core python projects/tessera_downscaling/scripts/maps/extract_dense_grid_patches.py \\
        --region iberia \\
        --resolution 0.05 \\
        --year 2024 \\
        --output-dir .tmp_output/processed/tessera_dense_grid

    # Arbitrary bbox
    uv run --group core python projects/tessera_downscaling/scripts/maps/extract_dense_grid_patches.py \\
        --bbox 36.0 43.5 -10.0 3.0 \\
        --region-name iberia \\
        --resolution 0.05 \\
        --output-dir .tmp_output/processed/tessera_dense_grid

The script is **resumable**: if it crashes or is killed, re-running the
same command picks up at the next unfinished sub-bbox.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Add the project src to the path. Mirrors the convention used by
# scripts/extract_tessera.py.
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[3]
sys.path.insert(0, str(REPO_ROOT / "projects" / "tessera_downscaling" / "src"))

from tessera_downscaling.data.dense_grid_patches import (  # noqa: E402
    compute_grid_points,
    extract_dense_grid_patches,
)


# Pre-defined regions. Matches the REGIONS dict in
# notebooks/single_folder_analysis.ipynb (cell 32).
PREDEFINED_REGIONS: dict[str, dict[str, float]] = {
    "alps":         {"lat_min": 45.5, "lat_max": 48.0, "lon_min": 6.0,   "lon_max": 16.0},
    "norway":       {"lat_min": 58.0, "lat_max": 71.0, "lon_min": 4.0,   "lon_max": 16.0},
    "iberia":       {"lat_min": 36.0, "lat_max": 43.5, "lon_min": -10.0, "lon_max": 3.0},
    "british_isles":{"lat_min": 50.0, "lat_max": 59.0, "lon_min": -11.0, "lon_max": 2.0},
    "north_eu_plain":{"lat_min": 50.0, "lat_max": 55.0, "lon_min": 5.0,  "lon_max": 25.0},
    "med_coast":    {"lat_min": 36.0, "lat_max": 44.0, "lon_min": -5.0,  "lon_max": 20.0},
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    region_group = parser.add_mutually_exclusive_group(required=True)
    region_group.add_argument(
        "--region", choices=list(PREDEFINED_REGIONS.keys()),
        help="One of the predefined regions matching the analysis notebook.",
    )
    region_group.add_argument(
        "--bbox", nargs=4, type=float, metavar=("LAT_MIN", "LAT_MAX", "LON_MIN", "LON_MAX"),
        help="Custom bounding box. Use --region-name to label the output dir.",
    )
    parser.add_argument(
        "--region-name", type=str, default=None,
        help="Output subfolder name when using --bbox. Required with --bbox.",
    )
    parser.add_argument(
        "--resolution", type=float, default=0.05,
        help="Grid spacing in degrees (default: 0.05° ≈ 5km).",
    )
    parser.add_argument(
        "--year", type=int, default=2024,
        help="TESSERA embedding year (default: 2024).",
    )
    parser.add_argument(
        "--patch-size", type=int, default=64,
        help="Patch side length in pixels at 10m resolution (default: 64).",
    )
    parser.add_argument(
        "--sub-bbox-size", type=float, default=0.3,
        help="Sub-bbox side length for batched mosaic fetching, degrees "
             "(default: 0.3 → ~5 GB peak mosaic at mid-latitudes).",
    )
    parser.add_argument(
        "--output-dir", type=Path, required=True,
        help="Parent directory under which the run's output subfolder is "
             "created (e.g. .tmp_output/processed/tessera_dense_grid).",
    )
    parser.add_argument(
        "--embeddings-cache-dir", type=Path, default=None,
        help="Where geotessera caches downloaded TESSERA tiles. "
             "Defaults to {output_dir}/{run_name}/_tile_cache. Tiles are "
             "preserved across runs to avoid re-downloads.",
    )
    parser.add_argument(
        "--no-resume", action="store_true",
        help="Start fresh, ignoring any existing progress.json.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Resolve bbox + region name
    if args.region is not None:
        r = PREDEFINED_REGIONS[args.region]
        lat_min, lat_max = r["lat_min"], r["lat_max"]
        lon_min, lon_max = r["lon_min"], r["lon_max"]
        region_name = args.region
    else:
        lat_min, lat_max, lon_min, lon_max = args.bbox
        if args.region_name is None:
            parser.error("--region-name is required when --bbox is given.")
        region_name = args.region_name

    # Per-run output directory.
    run_name = f"{region_name}_{args.resolution}deg_{args.year}"
    run_dir = args.output_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    # Build the grid.
    grid_points = compute_grid_points(
        lat_min=lat_min, lat_max=lat_max,
        lon_min=lon_min, lon_max=lon_max,
        resolution_deg=args.resolution,
    )
    print(
        f"Region: {region_name}\n"
        f"  bbox: lat ∈ [{lat_min}, {lat_max}], lon ∈ [{lon_min}, {lon_max}]\n"
        f"  resolution: {args.resolution}°\n"
        f"  grid points: {len(grid_points)}\n"
        f"  output: {run_dir}\n"
    )

    extract_dense_grid_patches(
        grid_points=grid_points,
        bbox=(lon_min, lat_min, lon_max, lat_max),
        output_dir=run_dir,
        year=args.year,
        patch_size=args.patch_size,
        sub_bbox_size_deg=args.sub_bbox_size,
        embeddings_cache_dir=args.embeddings_cache_dir,
        resume=not args.no_resume,
    )


if __name__ == "__main__":
    main()
