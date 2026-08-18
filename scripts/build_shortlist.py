#!/usr/bin/env python3
"""Build the per-(region, variable) shortlist of configs to retrain.

Reads existing test_summary.json files from training_runs_*/ and uses
the notebook helpers' shortlist_experiments() to pick top-K TESSERA
variants per region × variable. Always includes every no-TESSERA
bilinear baseline (`*_bilinear_baseline_wd`) regardless of where it
ranks, because the baselines drive the headline-comparison rows.

Output: a JSON file with this shape:

    {
      "snapshot_14y_eu": {
        "t2m": {
          "top_tessera": [
            {"name": "...", "label": "...", "mae_mean": ..., "mae_std": ..., "rank": 1},
            ...
          ],
          "baselines": [
            {"name": "t2m_snap_bilinear_baseline_wd", ...}
          ]
        },
        "wind": {...}
      },
      ...
    }

Optionally, a flat list view is also written to <out>.flat.txt: one
"<folder> <config_name>" per line. This is what prepare_rerun.py reads
to decide which run directories to delete.

Usage (from repo root):
    .venv/bin/python projects/tessera_downscaling/scripts/build_shortlist.py \
        --top-k 15 \
        --out projects/tessera_downscaling/scripts/experiments/shortlist_post_fix.json
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

# Add the notebooks/ dir to sys.path so we can import _helpers without
# pip-installing it. The helper module is intentionally not packaged —
# it lives next to the analysis notebooks.
SCRIPT_DIR = Path(__file__).resolve().parent
NOTEBOOKS_DIR = SCRIPT_DIR.parent / "notebooks"
sys.path.insert(0, str(NOTEBOOKS_DIR))

from _helpers import (  # noqa: E402
    build_experiment_defs,
    find_repo_root,
    list_folders,
    load_folder_results,
    shortlist_experiments,
    BASELINE_NAMES,
    is_tessera,
)

# Variables we care about for the snapshot 14y / 6y experiments. Daily
# runs (tmax / wind_mean) aren't part of the current paper; if we ever
# want shortlists for them, add them here.
SNAPSHOT_VARIABLES = ["t2m", "wind"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--top-k", type=int, default=15,
        help="How many top TESSERA configs to shortlist per (region × variable). "
             "Default 15. Top-K is computed by the notebook's "
             "shortlist_experiments() composite ranking (mae_mean rank + "
             "0.3 × mae_std rank).",
    )
    parser.add_argument(
        "--folders", type=str, nargs="+", default=None,
        help="Restrict to specific experiment folders. Default = all "
             "folders under scripts/experiments/.",
    )
    parser.add_argument(
        "--variables", type=str, nargs="+", default=SNAPSHOT_VARIABLES,
        help="Variables to build shortlists for. Default = t2m wind.",
    )
    parser.add_argument(
        "--out", type=Path, required=True,
        help="JSON output path. A companion <out>.flat.txt is also "
             "written, listing one '<folder> <name>' per line for "
             "consumption by prepare_rerun.py.",
    )
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=[42, 123, 456],
        help="Which seeds to load when reading run results.",
    )
    args = parser.parse_args()

    repo_root = find_repo_root()
    folders = args.folders or list_folders(repo_root)
    if not folders:
        print(f"No experiment folders found under "
              f"{repo_root}/projects/tessera_downscaling/scripts/experiments/")
        sys.exit(1)

    print(f"Loading results from {len(folders)} folder(s): {folders}")
    output: dict = {}
    flat_lines: list[str] = []

    for folder in folders:
        print(f"\n=== {folder} ===")
        df = load_folder_results(
            folder, seeds=args.seeds, repo_root=repo_root,
        )
        if df.empty:
            print(f"  (no results found in {folder} — skipping)")
            continue

        output[folder] = {}
        for variable in args.variables:
            mae_col = f"{variable}_mae"
            if mae_col not in df.columns or not df[mae_col].notna().any():
                print(f"  {variable}: no results — skipping")
                continue

            # Top-K TESSERA-only shortlist via the notebook helper.
            # tessera_only=True drops bilinear baselines; we add them
            # separately below so they're always retrained.
            top_names = shortlist_experiments(
                df, variable, top_n=args.top_k,
                tessera_only=True, print_table=False,
            )
            top_records = []
            for rank, name in enumerate(top_names, start=1):
                edata = df[(df["experiment"] == name) & df[mae_col].notna()]
                if edata.empty:
                    continue
                top_records.append({
                    "name": name,
                    "label": edata["label"].iloc[0],
                    "mae_mean": float(edata[mae_col].mean()),
                    "mae_std": float(edata[mae_col].std()) if len(edata) > 1 else 0.0,
                    "n_seeds": int(len(edata)),
                    "rank": rank,
                })
                flat_lines.append(f"{folder} {name}")

            # All baselines for this variable, regardless of rank.
            baselines = BASELINE_NAMES.get(variable, [])
            baseline_records = []
            for bname in baselines:
                edata = df[(df["experiment"] == bname) & df[mae_col].notna()]
                if edata.empty:
                    # Baseline didn't run / didn't produce metrics.
                    # Record None so the file documents the absence.
                    baseline_records.append({
                        "name": bname,
                        "label": None,
                        "mae_mean": None,
                        "mae_std": None,
                        "n_seeds": 0,
                    })
                    flat_lines.append(f"{folder} {bname}")
                    continue
                baseline_records.append({
                    "name": bname,
                    "label": edata["label"].iloc[0],
                    "mae_mean": float(edata[mae_col].mean()),
                    "mae_std": float(edata[mae_col].std()) if len(edata) > 1 else 0.0,
                    "n_seeds": int(len(edata)),
                })
                flat_lines.append(f"{folder} {bname}")

            output[folder][variable] = {
                "top_tessera": top_records,
                "baselines": baseline_records,
            }
            n_top = len(top_records)
            n_base = len(baseline_records)
            print(f"  {variable}: top-{n_top} TESSERA + {n_base} baselines")

    # Deduplicate flat_lines (same baseline can be in t2m and wind cells
    # — only need to delete its run dir once).
    flat_lines = sorted(set(flat_lines))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(output, f, indent=2)
    flat_path = Path(str(args.out) + ".flat.txt")
    with open(flat_path, "w") as f:
        f.write("\n".join(flat_lines) + "\n")

    print(f"\nWrote: {args.out}")
    print(f"Wrote: {flat_path}  ({len(flat_lines)} unique runs)")


if __name__ == "__main__":
    main()