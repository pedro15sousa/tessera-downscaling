#!/usr/bin/env python3
"""Delete the training_runs directories of shortlisted configs so the
existing per-region submit.sh scripts re-queue only those, not the
whole 140-config matrix.

Reads the flat shortlist file produced by build_shortlist.py
(<shortlist>.flat.txt — one '<folder> <config_name>' per line) and,
for each entry, removes the directory:

    <repo_root>/projects/tessera_downscaling/.tmp_output/training_runs_<folder_suffix>/<config_name>_seed<seed>/

across all configured seeds. The matching folder_suffix is derived from
the experiment-folder name (stripping the 'snapshot_' prefix, since the
training_runs dirs use that convention — see existing reeval_all.sh
behaviour). For example:

    snapshot_14y_eu  ->  training_runs_snapshot_14y_eu/

Writes a manifest of every action taken to a timestamped log file. The
script is dry-run by default; pass --apply to actually delete.

Usage:
    # Preview what would be deleted (default).
    .venv/bin/python projects/tessera_downscaling/scripts/prepare_rerun.py \
        --shortlist projects/tessera_downscaling/scripts/experiments/shortlist_post_fix.json.flat.txt

    # Actually delete.
    .venv/bin/python projects/tessera_downscaling/scripts/prepare_rerun.py \
        --shortlist projects/tessera_downscaling/scripts/experiments/shortlist_post_fix.json.flat.txt \
        --apply

After running, kick off the per-region submit.sh scripts as normal —
they'll see the missing test_summary.json files for the deleted runs
and re-queue them. Everything else is preserved.

Also handles the simple-baseline subdirs: those don't have a
best_model.pt and aren't affected by retraining the trained ConvCNPs.
For consistency, if the shortlist mentions a simple-baseline folder
(name ends in _era5_interp_baseline / _persistence_baseline), this
script deletes it too so FORCE_BASELINES=1 doesn't have to be set.
"""
from __future__ import annotations
import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

DEFAULT_SEEDS = [42, 123, 456]
TRAINING_RUNS_PREFIX = "training_runs_"


def is_simple_baseline_name(name: str) -> bool:
    return name.endswith(("_era5_interp_baseline", "_persistence_baseline"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--shortlist", type=Path, required=True,
        help="Path to the .flat.txt file from build_shortlist.py "
             "(one '<folder> <name>' per line).",
    )
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=DEFAULT_SEEDS,
        help="Seeds whose run dirs should be deleted. Default 42 123 456.",
    )
    parser.add_argument(
        "--repo-root", type=Path, default=None,
        help="Repo root. Default: walk up from this script's location.",
    )
    parser.add_argument(
        "--apply", action="store_true",
        help="Actually delete. Without this flag the script is dry-run.",
    )
    parser.add_argument(
        "--manifest-dir", type=Path, default=None,
        help="Where to write the deletion manifest. Default: "
             "<repo_root>/logs/prepare_rerun/.",
    )
    args = parser.parse_args()

    if args.repo_root is None:
        # scripts/prepare_rerun.py -> tessera_downscaling -> projects -> repo_root
        repo_root = Path(__file__).resolve().parents[3]
    else:
        repo_root = args.repo_root.resolve()
    if not (repo_root / "projects" / "tessera_downscaling").is_dir():
        print(f"ERROR: {repo_root} doesn't look like the repo root.", file=sys.stderr)
        sys.exit(1)

    base_output = repo_root / "projects" / "tessera_downscaling" / ".tmp_output"
    if not base_output.is_dir():
        print(f"ERROR: {base_output} not found.", file=sys.stderr)
        sys.exit(1)

    if not args.shortlist.exists():
        print(f"ERROR: shortlist file not found: {args.shortlist}", file=sys.stderr)
        sys.exit(1)

    shortlist_entries: list[tuple[str, str]] = []
    for line in args.shortlist.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) != 2:
            print(f"  WARNING: skipping malformed line: {line!r}")
            continue
        shortlist_entries.append((parts[0], parts[1]))

    print(f"Loaded {len(shortlist_entries)} shortlist entries from "
          f"{args.shortlist}")

    manifest_dir = args.manifest_dir or (repo_root / "logs" / "prepare_rerun")
    manifest_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    manifest_path = manifest_dir / f"manifest_{timestamp}.json"

    actions: list[dict] = []
    n_planned = 0
    n_done = 0
    n_missing = 0

    for folder, name in shortlist_entries:
        # folder example: "snapshot_14y_eu"
        # training_runs dir: ".tmp_output/training_runs_snapshot_14y_eu/"
        training_root = base_output / f"{TRAINING_RUNS_PREFIX}{folder}"
        if not training_root.is_dir():
            print(f"  (no training_runs dir for folder={folder}; skipping)")
            actions.append({
                "folder": folder, "name": name, "training_root": str(training_root),
                "status": "training_root_missing",
            })
            continue

        seeds_to_use = args.seeds
        if is_simple_baseline_name(name):
            # Simple baselines also get one dir per seed in the
            # current layout. Same loop applies.
            pass

        for seed in seeds_to_use:
            run_dir = training_root / f"{name}_seed{seed}"
            entry = {
                "folder": folder,
                "name": name,
                "seed": seed,
                "run_dir": str(run_dir),
            }
            if not run_dir.is_dir():
                entry["status"] = "missing"
                n_missing += 1
                actions.append(entry)
                continue
            n_planned += 1
            entry["status"] = "would_delete" if not args.apply else "deleted"
            if args.apply:
                shutil.rmtree(run_dir)
                n_done += 1
            actions.append(entry)

    summary = {
        "timestamp": timestamp,
        "shortlist_path": str(args.shortlist),
        "apply": args.apply,
        "n_entries": len(shortlist_entries),
        "n_seeds_per_entry": len(args.seeds),
        "n_planned_deletions": n_planned,
        "n_actually_deleted": n_done,
        "n_missing_run_dirs": n_missing,
        "actions": actions,
    }
    with open(manifest_path, "w") as f:
        json.dump(summary, f, indent=2)

    mode = "APPLIED" if args.apply else "DRY RUN"
    print(f"\n[{mode}] {n_planned} run dirs targeted")
    if args.apply:
        print(f"          {n_done} deleted")
    print(f"          {n_missing} were already absent (still recorded in manifest)")
    print(f"\nManifest written to: {manifest_path}")
    if not args.apply:
        print("\nRe-run with --apply to actually delete.")


if __name__ == "__main__":
    main()