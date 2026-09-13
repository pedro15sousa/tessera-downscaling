"""Extract Aurora's internal latents from the same rollouts that produce the
forecast context, cropped to the regions of interest.

Per init T0 one rollout is run (as in ``generate_aurora_forecasts.py``) and two
things are captured with forward hooks, without touching the ``aurora``
package:

* the **backbone output** at the harvested steps (default steps 1/4/12 =
  leads 6/24/72 h) -- the complete input of Aurora's decoder, ``(1, 4*180*360,
  2*embed_dim)`` tokens in ``(level, row, col)`` order, level 0 = surface;
* the **encoder output** at step 1 -- Aurora's embedding of the two ERA5
  analysis frames (T0-6h, T0), the lead-0 counterpart, ``(1, 4*180*360,
  embed_dim)``.

Each is cropped per region on the token grid (exactly, see
``tessera_downscaling.preprocessing.aurora_latent``), widened by a margin of
tokens, and written level-major as ``(levels*D, nh, nw)`` ``.npy`` files:

    <output_root>/lead{L}h/<region>/latent_backbone/<valid_ts>.npy
    <output_root>/lead0h/<region>/latent_encoder/<init_ts>.npy
    + latent_meta.json, latent_lats.npy, latent_lons.npy per directory

Storage precision is decided ONCE for a whole run from a calibration pass
(``--calibrate N``: N inits in fp32, per-channel range and fp16 round-trip
error, written to ``<output_root>/latent_calibration.json``), so shards that
run in parallel agree; ``--dtype auto`` (default) reads that file, an explicit
``--dtype`` overrides it (smoke tests). Every file is checked finite and
in-range before the atomic write, and re-runs skip files already present.

Optional checks: ``--verify-decoder`` also hooks the decoder's input and asserts
it IS the captured backbone output; ``--verify-physical-every N`` writes the
decoded fields of every N-th 6-hourly init through the generator's own writer
under ``--verify-root`` so ``verify_aurora_latents.py`` can quantify drift
against the original forecasts.

Inputs: 13-level ERA5 staging (``scripts/data/download_era5_wb2.py --levels
aurora``), the ERA5 static file and the global dataset's metadata (bboxes,
valid timestamps) + region grids (alignment check). Needs the ``aurora`` extra.

Usage (dry run first -- no model load, just accounting):
    uv run python scripts/aurora/extract_aurora_latents.py \
        --global-metadata <data root>/datasets/dataset_timestamp_global/metadata.json \
        --era5-staging-root <staging root> \
        --static-file <data root>/ingest/processed/era5_static/era5_static_0p25_all.nc \
        --regions europe east_asia --split all --dry-run
Smoke (GPU):      ... --model small --limit 3 --dtype float16 --verify-decoder
Calibration:      ... --model pretrained --calibrate 50
Full run (shard): ... --model pretrained --start-date 2012-01-01 --end-date 2012-12-31-18
"""

from __future__ import annotations

import argparse
import datetime as dt
import importlib.metadata
import json
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd
from _common import (
    TIME_DELTA_HOURS,
    _build_batch,
    _load_model,
    _open_static,
    _write_frame,
    add_common_arguments,
    build_schedule,
    lead_to_step,
    load_times,
    missing_input_frames,
    required_input_frames,
    resolve_device,
    resolve_region_bboxes,
    resolve_region_crops,
)

from tessera_downscaling.paths import ingest_dir
from tessera_downscaling.preprocessing import aurora_latent as al

KIND_BACKBONE = "backbone"
KIND_ENCODER = "encoder"
KIND_SUBDIR = {KIND_BACKBONE: "latent_backbone", KIND_ENCODER: "latent_encoder"}
CALIBRATION_NAME = "latent_calibration.json"
DEFAULT_REGIONS = ["europe", "east_asia"]
# (encoder embed_dim, latent levels) per model kind -- accounting only; the
# real run derives every dimension from the captured tensors.
MODEL_DIMS = {"pretrained": (512, 4), "small": (256, 4)}
# Every N-th 6-hourly init (counted from a fixed epoch, so shards agree) gets
# its decoded fields written for the drift check.
PHYSICAL_EPOCH = pd.Timestamp("2000-01-01")


# --------------------------------------------------------------------------- #
# Paths and resume markers
# --------------------------------------------------------------------------- #


def ts_stem(t: pd.Timestamp) -> str:
    return f"{t.year:04d}-{t.month:02d}-{t.day:02d}-{t.hour:02d}"


def latent_dir(output_root: Path, kind: str, lead_hours: int, region: str) -> Path:
    return output_root / f"lead{lead_hours}h" / region / KIND_SUBDIR[kind]


def latent_path(
    output_root: Path, kind: str, lead_hours: int, region: str, t: pd.Timestamp
) -> Path:
    return latent_dir(output_root, kind, lead_hours, region) / f"{ts_stem(t)}.npy"


def backbone_done(
    output_root: Path, lead_hours: int, valid: pd.Timestamp, regions: list[str]
) -> bool:
    """Done only once EVERY region's file exists (a half-written init is redone)."""
    return all(
        latent_path(output_root, KIND_BACKBONE, lead_hours, r, valid).exists()
        for r in regions
    )


def encoder_done(output_root: Path, t0: pd.Timestamp, regions: list[str]) -> bool:
    return all(
        latent_path(output_root, KIND_ENCODER, 0, r, t0).exists() for r in regions
    )


def physical_subset(t0: pd.Timestamp, every: int) -> bool:
    if every <= 0:
        return False
    slot = int((t0 - PHYSICAL_EPOCH) / pd.Timedelta(hours=TIME_DELTA_HOURS))
    return slot % every == 0


def plan_init(args, output_root: Path, t0, harvests, regions):
    """``(todo, need_encoder)`` for one init after the resume check."""
    todo = [
        (lead, step, valid)
        for (lead, step, valid) in harvests
        if not backbone_done(output_root, lead, valid, regions)
    ]
    need_encoder = (not args.no_encoder) and not encoder_done(output_root, t0, regions)
    return todo, need_encoder


# --------------------------------------------------------------------------- #
# Storage decision
# --------------------------------------------------------------------------- #


def load_storage(args, output_root: Path) -> dict[str, dict]:
    """``{kind: {"dtype", "offset", "calibration"}}`` for this run."""
    if args.dtype != "auto":
        s = {"dtype": args.dtype, "offset": None, "calibration": None}
        return {KIND_BACKBONE: dict(s), KIND_ENCODER: dict(s)}
    cal_file = Path(args.calibration_file or (output_root / CALIBRATION_NAME))
    if not cal_file.exists():
        raise SystemExit(
            f"--dtype auto needs the calibration file {cal_file}, which does not "
            "exist. Run once with --calibrate N (N inits, e.g. 50) to create it, "
            "or pass --dtype float16|float32 explicitly (smoke tests only)."
        )
    cal = json.loads(cal_file.read_text())
    return {k: cal["storage"][k] for k in (KIND_BACKBONE, KIND_ENCODER)}


# --------------------------------------------------------------------------- #
# Dry run
# --------------------------------------------------------------------------- #


def bytes_per_file(pc: al.PatchCrop, n_channels: int, dtype: str) -> int:
    return n_channels * pc.nh * pc.nw * (4 if dtype == "float32" else 2)


def dry_run(args, output_root: Path, valid_times, inits, per_init, crops, storage):
    regions = list(crops)
    enc_dim, levels = MODEL_DIMS[args.model]
    bb_dim = 2 * enc_dim
    n_bb_todo = 0
    n_enc_todo = 0
    n_rollouts = 0
    total_steps = 0
    for t0 in inits:
        todo, need_enc = plan_init(args, output_root, t0, per_init[t0], regions)
        if not todo and not need_enc:
            continue
        n_rollouts += 1
        total_steps += max([s for (_l, s, _v) in todo], default=1)
        n_bb_todo += len(todo)
        n_enc_todo += int(need_enc)
    in_frames = required_input_frames(inits)
    print("=" * 70)
    print("Aurora latent extraction -- DRY RUN")
    print("=" * 70)
    print(
        f"Model                 : {args.model}  (encoder D={enc_dim}, backbone D={bb_dim}, {levels} levels)"
    )
    print(
        f"Leads (hours)         : {args.leads}  ->  steps {[lead_to_step(h) for h in args.leads]}"
    )
    print(
        f"Valid-times ({args.split:8s}): {len(valid_times)}  ({valid_times[0]} .. {valid_times[-1]})"
    )
    print(f"Init times (union)    : {len(inits)}  ({inits[0]} .. {inits[-1]})")
    print(f"Rollouts to run       : {n_rollouts}  (after skipping completed)")
    print(f"Total rollout steps   : {total_steps}")
    print(f"Backbone files to do  : {n_bb_todo} (lead, valid) x {len(regions)} regions")
    print(
        f"Encoder files to do   : {n_enc_todo} inits x {len(regions)} regions"
        + ("  [disabled]" if args.no_encoder else "")
    )
    print(
        f"Input frames needed   : {len(in_frames)}  ({in_frames[0]} .. {in_frames[-1]})"
    )
    print("Regions (token crops, margin included):")
    gb = 0.0
    for name, pc in crops.items():
        b_bb = bytes_per_file(pc, levels * bb_dim, storage[KIND_BACKBONE]["dtype"])
        b_enc = bytes_per_file(pc, levels * enc_dim, storage[KIND_ENCODER]["dtype"])
        rs, cs = al.interior_patch_slices(pc)
        print(
            f"  {name:14s} {pc.nh}x{pc.nw} tokens (interior {rs.stop - rs.start}x"
            f"{cs.stop - cs.start} at [{rs.start}, {cs.start}]), "
            f"backbone {b_bb / 1e6:.1f} MB, encoder {b_enc / 1e6:.1f} MB per file"
        )
        gb += (n_bb_todo * b_bb + (0 if args.no_encoder else n_enc_todo * b_enc)) / 1e9
    print(
        f"Storage               : backbone {storage[KIND_BACKBONE]['dtype']}, "
        f"encoder {storage[KIND_ENCODER]['dtype']}; ~{gb:,.0f} GB to write"
    )
    missing = missing_input_frames(Path(args.era5_staging_root), inits)
    if missing:
        print(
            f"  *** MISSING INPUTS  : {len(missing)} (var, frame) pairs absent under "
            f"{args.era5_staging_root}; e.g. {[(v, str(f)) for v, f in missing[:3]]}"
        )
    else:
        print(
            f"  input check         : all {len(in_frames) * 9} required ERA5 files present  [ok]"
        )
    static_ok = Path(args.static_file).exists()
    print(
        f"Static file           : {'present [ok]' if static_ok else '*** MISSING ***'}"
    )
    print("=" * 70)
    go = (not missing) and static_ok
    print(
        "PRE-FLIGHT:",
        "READY TO RUN" if go else "*** RESOLVE ISSUES ABOVE BEFORE RUNNING ***",
    )
    print("=" * 70)


def disable_tf32(torch) -> None:
    """Keep fp32 matmuls/convolutions at full precision on CUDA (reproducibility
    across GPUs); a no-op on other backends. torch >= 2.9 exposes this through
    ``fp32_precision``; older builds through the ``allow_tf32`` flags."""
    import warnings

    try:
        torch.backends.cuda.matmul.fp32_precision = "ieee"
        torch.backends.cudnn.conv.fp32_precision = "ieee"
        return
    except (AttributeError, RuntimeError):
        pass
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
    except Exception:  # noqa: BLE001
        pass


# --------------------------------------------------------------------------- #
# Provenance
# --------------------------------------------------------------------------- #


def git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip()
    except Exception:  # noqa: BLE001
        return None


def model_info(args, device, model) -> dict:
    import torch

    try:
        version = importlib.metadata.version("microsoft-aurora")
    except importlib.metadata.PackageNotFoundError:
        version = None
    return {
        "name": args.model,
        "aurora_class": type(model).__name__,
        "aurora_version": version,
        "checkpoint": getattr(model, "default_checkpoint_name", None),
        "torch_version": torch.__version__,
        "device": str(device),
        "tf32_disabled": True,
    }


def provenance(args) -> dict:
    return {
        "created": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "git_commit": git_commit(),
        "cli": " ".join(sys.argv),
        "margin_patches": args.margin_patches,
    }


# --------------------------------------------------------------------------- #
# The run
# --------------------------------------------------------------------------- #


def run(args) -> None:
    output_root = Path(args.output_root)
    era5_staging_root = Path(args.era5_staging_root)
    global_dataset_dir = Path(args.global_metadata).parent

    valid_times = load_times(
        args.global_metadata, args.split, args.start_date, args.end_date
    )
    inits, per_init = build_schedule(valid_times, args.leads)
    if args.limit:
        inits = inits[: args.limit]
    region_bboxes = resolve_region_bboxes(args.global_metadata, args.regions)
    regions = list(region_bboxes)

    # Token crops, validated against the dataset reference grids where present.
    crops: dict[str, al.PatchCrop] = {}
    for name, bbox in region_bboxes.items():
        pc = al.patch_crop_from_bbox(name, bbox, margin_patches=args.margin_patches)
        ref_dir = global_dataset_dir / "regions" / name
        if (ref_dir / "lats.npy").exists() and (ref_dir / "lons.npy").exists():
            import numpy as np

            al.validate_patch_crop(
                pc, np.load(ref_dir / "lats.npy"), np.load(ref_dir / "lons.npy")
            )
            checked = "interior matches dataset grid"
        else:
            checked = "NEW region (no reference grid to check against)"
        print(
            f"  region {name:14s}: {pc.nh}x{pc.nw} tokens, margin {pc.margin}, "
            f"patch_roll={pc.patch_roll}  [{checked}]"
        )
        crops[name] = pc

    if args.calibrate:
        storage = {
            KIND_BACKBONE: {"dtype": "float32", "offset": None, "calibration": None},
            KIND_ENCODER: {"dtype": "float32", "offset": None, "calibration": None},
        }
    else:
        storage = load_storage(args, output_root)

    if args.dry_run:
        dry_run(args, output_root, valid_times, inits, per_init, crops, storage)
        return

    # Heavy imports only on the real path.
    import numpy as np
    import torch
    from aurora import rollout
    from tqdm import tqdm

    device = resolve_device(args.device, args.allow_cpu)
    disable_tf32(torch)
    static_ds = _open_static(Path(args.static_file))
    model = _load_model(args.model, device)
    latent_levels = int(model.encoder.latent_levels)
    if int(model.encoder.patch_size) != al.PATCH_SIZE:
        raise SystemExit(
            f"Model patch size {model.encoder.patch_size} != {al.PATCH_SIZE}; "
            "the token-grid crop assumes the 0.25 deg / patch-4 configuration."
        )
    info = model_info(args, device, model)
    prov = provenance(args)

    bb_cap = al.LatentCapture(model.backbone)
    enc_cap = (
        None
        if args.no_encoder and not args.calibrate
        else al.LatentCapture(model.encoder)
    )
    dec_cap = al.LatentCapture(model.decoder, pre=True) if args.verify_decoder else None

    if args.calibrate:
        _calibrate(
            args,
            output_root,
            inits,
            per_init,
            crops,
            model,
            static_ds,
            device,
            era5_staging_root,
            latent_levels,
            bb_cap,
            enc_cap,
            info,
            prov,
            rollout,
            tqdm,
            np,
            torch,
        )
        return

    region_crops = None  # cell-level crops for the physical writer, resolved once
    ensured_dirs: set[Path] = set()
    dims: dict[str, int] = {}  # kind -> D, learned from the first capture
    n_files = 0
    n_rollouts = 0
    n_steps = 0
    t_start = time.time()

    def write_kind(kind: str, x, lead: int, step: int, t: pd.Timestamp) -> None:
        nonlocal n_files
        grid = al.tokens_to_grid(x, latent_levels)
        d = grid.shape[-1]
        if kind in dims and dims[kind] != d:
            raise RuntimeError(f"{kind} width changed from {dims[kind]} to {d}")
        dims[kind] = d
        for name, pc in crops.items():
            directory = latent_dir(output_root, kind, lead, name)
            if directory not in ensured_dirs:
                meta = al.latent_meta(
                    pc,
                    kind=kind,
                    lead_hours=lead,
                    rollout_step=step,
                    embed_dim=d,
                    latent_levels=latent_levels,
                    storage=storage[kind],
                    model=info,
                    extra=prov,
                )
                al.write_sidecars(directory, meta, pc)
                ensured_dirs.add(directory)
            crop = al.crop_latent_tokens(grid, pc)
            al.write_latent(
                latent_path(output_root, kind, lead, name, t), crop, storage[kind]
            )
            n_files += 1

    for t0 in tqdm(inits, desc=f"Aurora latents ({args.model})"):
        todo, need_enc = plan_init(args, output_root, t0, per_init[t0], regions)
        if not todo and not need_enc:
            continue
        max_step = max([s for (_l, s, _v) in todo], default=1)
        want = {step: (lead, valid) for (lead, step, valid) in todo}
        bb_cap.reset(keep_steps=set(want))
        if enc_cap is not None:
            enc_cap.reset(keep_steps={1} if need_enc else set())
        if dec_cap is not None:
            dec_cap.reset(keep_steps=set(want))
        write_physical = physical_subset(t0, args.verify_physical_every)

        batch = _build_batch(era5_staging_root, t0, static_ds, device)
        with torch.inference_mode():
            for i, pred in enumerate(rollout(model, batch, steps=max_step), start=1):
                if i == 1 and need_enc:
                    write_kind(KIND_ENCODER, enc_cap.pop(1), 0, 1, t0)
                if i in want:
                    lead, valid = want[i]
                    x = bb_cap.pop(i)
                    if dec_cap is not None:
                        dec_in = dec_cap.pop(i)
                        if not torch.equal(x, dec_in):
                            raise RuntimeError(
                                f"step {i}: decoder input differs from the captured "
                                "backbone output -- hook placement is wrong"
                            )
                    write_kind(KIND_BACKBONE, x, lead, i, valid)
                    if write_physical:
                        pred_cpu = pred.to("cpu")
                        if region_crops is None:
                            region_crops = resolve_region_crops(
                                region_bboxes,
                                pred_cpu.metadata.lat.cpu().numpy(),
                                pred_cpu.metadata.lon.cpu().numpy(),
                                global_dataset_dir,
                                logger=print,
                            )
                        _write_frame(
                            Path(args.verify_root),
                            lead,
                            valid,
                            pred_cpu,
                            "float32",
                            region_crops,
                        )
        if bb_cap.calls != max_step:
            raise RuntimeError(
                f"backbone was called {bb_cap.calls} times for a {max_step}-step "
                "rollout -- hook accounting is wrong"
            )
        n_rollouts += 1
        n_steps += max_step

    elapsed = time.time() - t_start
    print(
        f"Done: {n_rollouts} rollouts, {n_steps} steps, {n_files} latent files in "
        f"{elapsed / 3600:.2f} h"
        + (f" ({elapsed / n_steps:.2f} s/step)" if n_steps else "")
    )
    if dec_cap is not None and n_steps:
        print(
            "verify-decoder: decoder input == backbone output at every harvested step [ok]"
        )


def _calibrate(
    args,
    output_root,
    inits,
    per_init,
    crops,
    model,
    static_ds,
    device,
    era5_staging_root,
    latent_levels,
    bb_cap,
    enc_cap,
    info,
    prov,
    rollout,
    tqdm,
    np,
    torch,
):
    """Run the first N inits in fp32, accumulate per-channel statistics of the
    region crops for each kind, decide the storage encoding and write the
    calibration file. Nothing else is written."""
    inits = inits[: args.calibrate]
    max_step = max(lead_to_step(h) for h in args.leads)
    keep = {lead_to_step(h) for h in args.leads}
    acc: dict[str, al.CalibrationAccumulator] = {}
    dims: dict[str, int] = {}
    n_steps = 0
    t_start = time.time()

    def feed(kind: str, x) -> None:
        grid = al.tokens_to_grid(x, latent_levels)
        dims[kind] = grid.shape[-1]
        for pc in crops.values():
            crop = al.crop_latent_tokens(grid, pc)
            if kind not in acc:
                acc[kind] = al.CalibrationAccumulator(crop.shape[0])
            acc[kind].update(crop)

    for t0 in tqdm(inits, desc=f"Aurora calibration ({args.model})"):
        bb_cap.reset(keep_steps=keep)
        enc_cap.reset(keep_steps={1})
        batch = _build_batch(era5_staging_root, t0, static_ds, device)
        with torch.inference_mode():
            for i, _pred in enumerate(rollout(model, batch, steps=max_step), start=1):
                if i == 1:
                    feed(KIND_ENCODER, enc_cap.pop(1))
                if i in keep:
                    feed(KIND_BACKBONE, bb_cap.pop(i))
        n_steps += max_step
    elapsed = time.time() - t_start

    result = {
        "n_inits": len(inits),
        "inits": [ts_stem(t) for t in inits],
        "leads": list(args.leads),
        "regions": list(crops),
        "margin_patches": args.margin_patches,
        "latent_levels": latent_levels,
        "dims": dims,
        "seconds_per_step": elapsed / max(n_steps, 1),
        "model": info,
        "provenance": prov,
        "storage": {},
    }
    for kind, a in acc.items():
        summary = a.summary()
        decision = al.decide_storage(summary)
        result["storage"][kind] = decision
        result["storage"][kind]["channel_mean"] = [float(v) for v in summary["mean"]]
        result["storage"][kind]["channel_std"] = [float(v) for v in summary["std"]]
        c = decision["calibration"]
        print(
            f"{kind:8s}: D={dims[kind]}, abs-max {c['absmax_max']:.4g}, fp16 rel err "
            f"{c['rel_err_float16_max']:.2e} (centred {c['rel_err_centred_float16_max']:.2e})"
            f" -> {decision['dtype']}"
        )
    output_root.mkdir(parents=True, exist_ok=True)
    cal_file = Path(args.calibration_file or (output_root / CALIBRATION_NAME))
    tmp = cal_file.with_name(cal_file.name + ".tmp")
    tmp.write_text(json.dumps(result, indent=2))
    tmp.replace(cal_file)
    print(
        f"Calibration written to {cal_file} ({len(inits)} inits, {n_steps} steps, "
        f"{result['seconds_per_step']:.2f} s/step)"
    )


def parse_args():
    p = argparse.ArgumentParser(
        description="Extract Aurora backbone/encoder latents, cropped to regions."
    )
    add_common_arguments(p)
    p.set_defaults(regions=DEFAULT_REGIONS, split="all")
    p.add_argument(
        "--output-root",
        default=str(ingest_dir("aurora")),
        help="Latent root; writes lead{L}h/<region>/latent_backbone/ and "
        "lead0h/<region>/latent_encoder/ under it (default: <data root>/ingest/aurora)",
    )
    p.add_argument(
        "--margin-patches",
        type=int,
        default=4,
        help="Tokens (1 deg each) of context kept around every region bbox",
    )
    p.add_argument(
        "--no-encoder",
        action="store_true",
        help="Do not store the step-1 encoder output (the lead-0 latent)",
    )
    p.add_argument(
        "--dtype",
        choices=["auto", "float16", "float32"],
        default="auto",
        help="Storage precision: 'auto' reads the calibration file (see --calibrate)",
    )
    p.add_argument(
        "--calibrate",
        type=int,
        default=0,
        metavar="N",
        help="Calibration mode: run N inits in fp32, decide the storage encoding, "
        "write the calibration file and exit (no latents written)",
    )
    p.add_argument(
        "--calibration-file",
        default=None,
        help=f"Calibration JSON (default: <output-root>/{CALIBRATION_NAME})",
    )
    p.add_argument(
        "--verify-decoder",
        action="store_true",
        help="Also hook the decoder input and assert it equals the captured "
        "backbone output (smoke tests)",
    )
    p.add_argument(
        "--verify-physical-every",
        type=int,
        default=200,
        metavar="N",
        help="Write the decoded fields of every N-th 6-hourly init under "
        "--verify-root for the drift check (0 = off)",
    )
    p.add_argument(
        "--verify-root",
        default=str(ingest_dir("aurora_verify")),
        help="Where the physical verification subset goes "
        "(default: <data root>/ingest/aurora_verify)",
    )
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
