#!/usr/bin/env python3
"""Backfill the precip drop into the config EMBEDDED in cross-lead checkpoints.

evaluate.py reads its config from the checkpoint, not config.json:

    ckpt = torch.load(args.checkpoint, ...)
    config = ckpt.get("config", {})          # <- this is what eval uses
    drop_context_channels = config.get("drop_context_channels", None)

train.py embeds `vars(args)` as that "config" when it saves best_model.pt. Runs
trained before the fix had `args.drop_context_channels = None` (the precip drop
was force-applied to shared_kwargs, not args), so the embedded config has no
drop -> eval keeps all 20 ERA5 channels for eval_lead0h and builds a 40-channel
model against the 39-channel checkpoint. (Patching config.json does nothing,
because eval never reads it.)

This rewrites `ckpt["config"]["drop_context_channels"]` to include
`total_precipitation_sum`, idempotently, and re-saves best_model.pt (keeping
every other key — model/optimizer state, epoch, val_loss — untouched). A
one-time best_model.pt.orig backup is written per run. Existing checkpoints then
evaluate correctly with no retraining; new runs (explicit --drop-context-channels)
already embed the drop and don't need this.

Requires torch -> run with the training venv, e.g.:
    ${REPO_ROOT}/.venv/bin/python patch_cross_lead_checkpoints.py [OUTPUT_BASE]
                                  [--regions europe east_asia] [--dry-run] [--no-backup]
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

DEFAULT_BASE = (
    "/projects/u6do/pmms2/end-to-end-forecasting/projects/tessera_downscaling/"
    ".tmp_output/training_runs_snapshot_14y_cross_lead"
)
DROP_NAME = "total_precipitation_sum"


def ensure_drop(cfg: dict) -> tuple[dict, bool]:
    """Return (cfg, changed): ensure DROP_NAME is in cfg['drop_context_channels']
    as a list. Pure / torch-free so the mutation logic is unit-testable.
    Idempotent. Also normalises a bare-string value to a list (a string would be
    iterated character-by-character by evaluate.py and silently drop nothing).
    """
    original = cfg.get("drop_context_channels")
    drops = original or []
    if isinstance(drops, str):
        drops = [drops]
    drops = list(drops)
    if DROP_NAME not in drops:
        drops.append(DROP_NAME)
    if drops == original:
        return cfg, False
    cfg = dict(cfg)
    cfg["drop_context_channels"] = drops
    return cfg, True


def patch_one(pt_path: Path, dry_run: bool, backup: bool) -> str:
    import torch  # local import: only needed at runtime, on the cluster venv
    run = pt_path.parent.name
    try:
        ckpt = torch.load(pt_path, map_location="cpu", weights_only=False)
    except Exception as e:  # noqa: BLE001
        return f"ERROR  {run}: torch.load failed ({e})"
    if "config" not in ckpt or not isinstance(ckpt["config"], dict):
        return f"ERROR  {run}: checkpoint has no 'config' dict"

    new_cfg, changed = ensure_drop(ckpt["config"])
    if not changed:
        return f"ok     {run}: already has drop={new_cfg['drop_context_channels']}"
    if dry_run:
        return f"WOULD  {run}: set drop={new_cfg['drop_context_channels']}"

    if backup:
        bak = pt_path.with_suffix(".pt.orig")
        if not bak.exists():
            bak.write_bytes(pt_path.read_bytes())
    ckpt["config"] = new_cfg
    torch.save(ckpt, pt_path)
    return f"PATCH  {run}: set drop={new_cfg['drop_context_channels']}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("output_base", nargs="?", default=DEFAULT_BASE,
                    help=f"Cross-lead training-runs root (default: {DEFAULT_BASE})")
    ap.add_argument("--regions", nargs="+", default=["europe", "east_asia"],
                    help="Region subfolders to scan (default: europe east_asia).")
    ap.add_argument("--dry-run", action="store_true",
                    help="Report what would change without writing.")
    ap.add_argument("--no-backup", dest="backup", action="store_false", default=True,
                    help="Skip writing best_model.pt.orig backups.")
    args = ap.parse_args()

    base = Path(args.output_base)
    if not base.is_dir():
        print(f"output_base not found: {base}", file=sys.stderr)
        return 2

    ckpts: list[Path] = []
    for region in args.regions:
        ckpts.extend(sorted((base / region).glob("*_seed*/best_model.pt")))
    if not ckpts:
        print(f"No best_model.pt under {base}/<region>/*_seed*/ "
              f"(regions={args.regions}).", file=sys.stderr)
        return 1

    print(f"{'[dry-run] ' if args.dry_run else ''}Scanning {len(ckpts)} "
          f"checkpoints under {base}\n")
    n = 0
    for c in ckpts:
        status = patch_one(c, args.dry_run, args.backup)
        print(" ", status)
        if status.startswith(("PATCH", "WOULD")):
            n += 1
    verb = "would patch" if args.dry_run else "patched"
    print(f"\n{verb} {n} / {len(ckpts)} checkpoints "
          f"({len(ckpts) - n} already correct or errored).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())