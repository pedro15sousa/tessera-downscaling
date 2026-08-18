# Snapshot-experiment analysis notebooks

Two notebooks, one shared helpers module. Both replace the old
`convcnp_experiment_analysis.ipynb` which mixed daily and snapshot
results in a single hardcoded `EXPERIMENT_DEFS` dict.

## Files

```
notebooks/
├── README.md                          (this file)
├── _helpers.py                        (shared loader, tables, shortlists)
├── cross_folder_analysis.ipynb        (compare across experiment folders)
└── single_folder_analysis.ipynb       (detailed analysis of one folder)
```

## When to use which

**`cross_folder_analysis.ipynb`** — overview / cross-cutting reads.
Loads results from every folder under `scripts/experiments/`. For each
folder it prints a `print_summary` table per target variable, picks a
top-N TESSERA shortlist per (folder, variable), then assembles a
cross-folder pivot showing baseline + best variant side-by-side.
Useful for: "how does the wind baseline degrade as we move from EU to
Asia?", "is the lat16+concat winner consistent across all 5 regions?".

**`single_folder_analysis.ipynb`** — deep-dive one folder at a time.
Set `ACTIVE_FOLDER` near the top, run from there, and the rest of the
notebook (12 sections: summary, per-seed detail, shortlist, variant
comparison, per-seed stability, training curves, seasonal, calibration,
prediction distribution, per-station maps, per-station baseline-vs-VAE
comparison, regional breakdown) operates on that one folder's results.
Switch folder by changing the constant and re-running.

## How they share data

Both notebooks import `_helpers` from the same directory. The helpers
provide:

- `find_repo_root`, `experiments_dir`, `output_dir_for_folder` — path
  discovery (walk up from cwd until `projects/tessera_downscaling/`
  is found).
- `list_folders` — every folder under `scripts/experiments/` with a
  valid `experiments.yaml`.
- `load_experiments_yaml(folder)` — parse one folder's YAML.
- `build_experiment_defs(folders)` — replacement for the old
  hardcoded `EXPERIMENT_DEFS` dict, keyed by experiment name.
- `load_folder_results(folder, seeds)` — load runs for one folder.
- `load_all_results(folders, seeds)` — concatenate runs across many
  folders. Each row carries `source_folder` for downstream filtering.
- `print_summary(df, variable, title_suffix)` — the per-variable
  table from the original cell-7, lifted out as a function.
- `shortlist_experiments(df, variable, top_n)` — composite-rank
  shortlist (MAE-mean + MAE-stability), as in the original cell-9.
- `get_baseline_and_shortlist(shortlist, variable)` — pair the
  shortlist with the canonical baseline name for that variable.
- `cross_folder_table(df, variable)` — pivot to (experiment ×
  source_folder) MAE table. Cross-folder-only.
- `baselines_across_folders(df, variable)` — the baseline-only
  cross-folder pivot.

Each helper is type-annotated and has a one-paragraph docstring; if
you're unsure what something does, the source is < 500 lines.

## Detailed analysis sections (single-folder notebook)

| Section | Content | Cell |
|---|---|---|
| 0 | Configuration — set ACTIVE_FOLDER | 3 |
| 1 | Load results | 5 |
| 2 | Summary tables (one per target variable) | 7 |
| 3 | Per-seed detail | 9 |
| 4 | Shortlist + baseline resolution | 11 |
| 5 | Variant comparison (bar chart) | 13 |
| 6 | Per-seed stability (scatter MAE/RMSE) | 15 |
| 7 | Training curves (train + val MAE/NLL) | 17–19 |
| 8 | Seasonal error patterns | 21 |
| 9 | Uncertainty calibration | 23 |
| 10 | Per-station error analysis | 25–27 |
| 11 | Per-station baseline-vs-VAE comparison | 29–30 |
| 12 | Regional breakdown by lat/lon bbox + terrain | 32–34 |

The `REGIONS` dict in section 12 is hardcoded for European stations
(Alps, Norway, Iberian Peninsula, etc.). When you switch
`ACTIVE_FOLDER` to a non-EU folder, redefine `REGIONS` in cell 32 to
match the new geography (e.g. for `snapshot_14y_us`, "Rocky Mountains",
"Florida coast", "Great Plains"). The `regional_comparison_with_density`
function itself is region-agnostic — only the dict needs editing.

## Migration from the old notebook

Verbatim ports of these original cells live in single-folder:

| Original cell | New location | Notes |
|---|---|---|
| 1, 2 | new cell 1 | path discovery now in `_helpers.find_repo_root` |
| 4 (EXPERIMENT_DEFS, 1088 lines) | gone | replaced by `_helpers.build_experiment_defs(folders)` reading YAMLs |
| 5 (load_all_results) | new cell 5 | replaced by `_helpers.load_folder_results(ACTIVE_FOLDER)` |
| 7 (print_summary) | new cell 7 | function lifted to `_helpers.print_summary` |
| 8 (per-seed detail) | new cell 9 | identical, with `None`-safe formatting |
| 9 (shortlist_experiments) | new cell 11 | function lifted to `_helpers.shortlist_experiments` |
| 10 (get_baseline_and_shortlist) | new cell 11 | function lifted to `_helpers.get_baseline_and_shortlist` |
| 12 (plot_comparison) | new cell 13 | identical |
| 14 (per-seed scatter) | new cell 15 | identical |
| 16–18 (training curves) | new cells 17–19 | identical |
| 20 (seasonal) | new cell 21 | identical |
| 22 (calibration) | new cell 23 | identical |
| 24 (prediction distribution) | dropped | the old cell relied on test_predictions.npz files which aren't always written; can re-add later |
| 25 (tail/calibration) | dropped | same reason as 24 |
| 27–31 (per-station + regional) | new cells 25–34 | identical functions, with hardcoded driver lines stripped and replaced by ACTIVE_FOLDER-respecting drivers |

## Running

```bash
cd projects/tessera_downscaling/notebooks
jupyter notebook cross_folder_analysis.ipynb
# or
jupyter notebook single_folder_analysis.ipynb
```

Both have been validated end-to-end against a mock dataset — every
cell executes without error, gracefully skipping variables that have
no data in the active folder.
