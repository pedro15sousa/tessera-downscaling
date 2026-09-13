"""Stage 1 of the Aurora-context pipeline: generate Aurora forecasts, cropped
to the regions of interest.

This script runs the pretrained Aurora 0.25 deg model from ERA5 initial
conditions and, for each harvested forecast frame, crops Aurora's global output
to the regions of interest *before* writing, so we never persist the ~286 MB
global frame (a full trainval run would be ~15 TB). Each region's crop is
written in the SAME on-disk layout as the ERA5 WeatherBench2 staging, under a
per-region subtree:

    <output_root>/lead{L}h/<region>/processed/era5_wb2_quarter_<var>/data/<valid_ts>.nc

The crop uses ``compute_grid_crop_indices`` from
``tessera_downscaling.preprocessing.helpers`` -- the same function the dataset
preprocessor uses, including the longitude roll that makes Europe's
0-deg-crossing box contiguous -- and is asserted bit-for-bit against each
region's reference grid in ``dataset_timestamp_global``, so the Aurora context
aligns exactly with the ERA5 dataset. Region bboxes come from the global
dataset's ``metadata.json``. Because the per-region crops are already regional,
the companion Stage-2 script (``scripts/preprocessing/preprocess_aurora.py``)
consumes them directly (no second crop).

Design decisions (see session notes):
  * Model: Aurora 0.25 deg *Pretrained*. The fine-tuned 0.25 checkpoint is
    matched to IFS HRES T0 inputs; we initialise from ERA5, so the pretrained
    model is the in-distribution, defensible choice. It also keeps the
    experiment a single distribution shift (reanalysis -> forecast).
  * We keep Aurora's full native field set (4 surface + 5 atmospheric vars at
    all 13 pressure levels = 69 dynamic fields per frame), fp32, leaving any
    channel/level subsetting to Stage 2 -- only the *spatial* extent is cropped
    here.
  * Only ``rollout``'s predictions are used, so no source patch to the upstream
    ``aurora`` package is required.

Lead times are harvested from a SINGLE rollout per init:
    6 h  -> step 1,   24 h -> step 4,   72 h -> step 12
so adding the 24 h lead costs no extra rollouts. Per-init we only roll out as
many steps as the longest lead that init actually feeds.

Inputs: 13-level ERA5 staging (``scripts/data/download_era5_wb2.py --levels aurora``)
and the ERA5 static file (z / lsm / slt); see ``scripts/aurora/submit_aurora_forecasts.sh``.
Needs the ``aurora`` extra (``uv sync --extra aurora``).

Usage (dry run first -- no model load, just accounting):
    uv run python scripts/aurora/generate_aurora_forecasts.py \
        --global-metadata <data root>/dataset_timestamp_global/metadata.json \
        --era5-staging-root <data root>/ingest/aurora_inputs \
        --static-file <data root>/ingest/processed/era5_static/era5_static_0p25_all.nc \
        --output-root <data root>/ingest/aurora \
        --dry-run

Real run (GPU node):        ... --model pretrained
Plumbing smoke test (GPU):  ... --model small --limit 3
"""

from __future__ import annotations

import argparse
from pathlib import Path

from _common import (
    _build_batch,
    _load_model,
    _open_static,
    _write_frame,
    add_common_arguments,
    build_schedule,
    dry_run_report,
    harvest_done,
    load_times,
    region_n_cells,
    resolve_device,
    resolve_region_bboxes,
    resolve_region_crops,
)


def run(args) -> None:
    output_root = Path(args.output_root)
    era5_staging_root = Path(args.era5_staging_root)
    global_dataset_dir = Path(args.global_metadata).parent

    test_times = load_times(
        args.global_metadata, args.split, args.start_date, args.end_date
    )
    inits, per_init = build_schedule(test_times, args.leads)
    dtype_bytes = 2 if args.dtype == "float16" else 4

    # Region bboxes (from the global metadata) + cell counts.
    region_bboxes = resolve_region_bboxes(args.global_metadata, args.regions)
    region_names = list(region_bboxes)
    region_cells = {r: region_n_cells(bb) for r, bb in region_bboxes.items()}

    if args.limit:
        inits = inits[: args.limit]

    if args.dry_run:
        dry_run_report(
            test_times,
            inits,
            per_init,
            args.leads,
            era5_staging_root,
            output_root,
            dtype_bytes,
            region_bboxes,
            region_cells,
            static_file=args.static_file,
        )
        return

    # Heavy imports only on the real path (keeps --dry-run torch/aurora-free).
    import torch
    from aurora import rollout
    from tqdm import tqdm

    device = resolve_device(args.device, args.allow_cpu)
    static_ds = _open_static(Path(args.static_file))
    model = _load_model(args.model, device)

    # Crop indices are resolved once, from Aurora's actual output grid (the first
    # prediction), then reused. This mirrors Stage 2 and asserts europe/east_asia
    # against the dataset reference grid before any write happens.
    region_crops = None

    for t0 in tqdm(inits, desc=f"Aurora rollouts ({args.model})"):
        harvests = per_init[t0]
        todo = [
            (lead, step, valid)
            for (lead, step, valid) in harvests
            if not harvest_done(output_root, lead, valid, region_names)
        ]
        if not todo:
            continue
        max_step = max(step for (_l, step, _v) in todo)
        want = {step: (lead, valid) for (lead, step, valid) in todo}

        batch = _build_batch(era5_staging_root, t0, static_ds, device)
        with torch.inference_mode():
            for i, pred in enumerate(rollout(model, batch, steps=max_step), start=1):
                if i in want:
                    lead, valid = want[i]
                    pred = pred.to("cpu")
                    if region_crops is None:
                        print(
                            f"Resolving region crops on Aurora's "
                            f"{len(pred.metadata.lat)}x{len(pred.metadata.lon)} output grid:"
                        )
                        region_crops = resolve_region_crops(
                            region_bboxes,
                            pred.metadata.lat.cpu().numpy(),
                            pred.metadata.lon.cpu().numpy(),
                            global_dataset_dir,
                            logger=print,
                        )
                    _write_frame(
                        output_root, lead, valid, pred, args.dtype, region_crops
                    )


def parse_args():
    p = argparse.ArgumentParser(
        description="Generate global Aurora forecasts in ERA5-staging layout (Stage 1)."
    )
    add_common_arguments(p)
    p.add_argument(
        "--output-root",
        required=True,
        help="Aurora staging root; writes lead{L}h/<region>/processed/... under it",
    )
    p.add_argument("--dtype", choices=["float32", "float16"], default="float32")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
