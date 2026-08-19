# Analysis notebooks

The notebooks in this directory read the stored run artefacts under
`$TESSERA_DATA_ROOT` (default `/data/weather-downscaling`; see
`tessera_downscaling.paths` and `DATA.md`) and hold the analyses behind the
paper's tables and figures. Their saved outputs are the record of the numbers
printed in the paper; `scripts/paper/make_paper_figures.py` and
`scripts/paper/make_paper_tables.py` re-derive the delivered figures/tables
from the same inputs, with numeric cross-checks against the paper.

Run them from this directory (`uv run jupyter lab`); every notebook resolves
its inputs through `tessera_downscaling.paths`, so no path editing is needed
when the data root is set.

## Files

```
notebooks/
├── README.md                              (this file)
├── _helpers.py                            (shared loader, summary tables, shortlists,
│                                           run classification)
├── cross_folder_analysis.ipynb            Table 1, shuffled/summary-stats control table,
│                                           Fig 2, train-station table
├── analyze_cross_lead.ipynb               Fig 6 + Fig 10 (Aurora leads), calibration table
├── data_efficiency_temporal_rollout.ipynb Fig 7 (simulated Norway deployment)
├── residual_structure_analysis.ipynb      Fig 5 + Fig 8 (RF residual probe), §4 alignment
├── extra_descriptors_stratified.ipynb     Table 6 / App. C (hand-crafted descriptor arm)
├── tessera_1bm_variant_shortlist.ipynb    model-selection provenance (VAE variant choice)
├── uncertainty_estimation.ipynb           calibration / resolution study (appendix material)
├── stratified_performance.ipynb           qualitative stratified-skill claims
├── cross_lead_analysis_outputs/           written by analyze_cross_lead (all_results_tidy.csv
│                                           is a cross-check input of make_paper_figures)
├── norway_analysis_outputs/               written by data_efficiency_temporal_rollout and
│                                           scripts/analysis/norway_descriptor_spaces.py
├── descriptor_analysis_outputs/           written by scripts/analysis/residual_probe_spaces.py
└── uncertainty_analysis_outputs/          written by uncertainty_estimation
```

Figure/table numbers refer to the preprint; the AMS draft renumbers
(Fig 5→3, Fig 6→8, Fig 7→9, Figs 3–4→4–7, Figs 11–12→F1–F2, Table 4→C1,
Table 6→B1).

## What each notebook backs

| Notebook | Paper item | Runs / inputs it reads |
|---|---|---|
| `cross_folder_analysis.ipynb` | **Table 1** (per-region MAE/RMSE/CRPS; cell "+mTPI summary"), **Table 4** shuffled-latents + summary-stats control (cell "Shuffled-latent control"), **Fig 2** CRPS uplift (cell "CRPS uplift"), train-station generalisation table (last cell). Earlier sections are the generic per-folder summaries / shortlists. | `training_runs_snapshot_14y_{region}/` (baseline, persistence, ERA5-interp, `*_extradesc_*`), `training_runs_snapshot_14y_{region}_tessera_1B-M_2017/` (TESSERA arm, `*_stats16_*`), `…_shuffled/`, `<run>/eval_train_stations/` from `scripts/reeval_train_stations.sh` |
| `analyze_cross_lead.ipynb` | **Fig 6** Aurora-lead RMSE uplift, **Fig 10** relative CRPS decay vs lead, calibration table (§9d) | `training_runs_snapshot_14y_cross_lead{,_tessera_1B-M_2017}/{europe,east_asia}/*/eval_lead{0,6,24,72}h/`, `*_era5_interp{,_lapse}_baseline_matched_seed42`; writes `cross_lead_analysis_outputs/` |
| `data_efficiency_temporal_rollout.ipynb` | **Fig 7** simulated Norway deployment (the per-station CRPS-from-head-parameters logic lives only here) | `training_runs_snapshot_14y_eu_temporal_rollout_norway_lat16_mtpi/` (baseline + ERA5-interp), `…_norway_tessera_1B-M_2017/` (TESSERA), `test_station_errors.npz`; `scripts/experiments/snapshot_14y_eu_temporal_rollout_norway_lat16_mtpi/rollout_schedule.json` |
| `residual_structure_analysis.ipynb` | **Fig 5** (§3c) and **Fig 8** (§3g) descriptor-space RF probes of the ERA5-interp residual; §4 residual-alignment (imports `scripts/maps/residual_alignment_fulltest.py`) | per-run `test_predictions.npz` / `test_station_errors.npz`, `dataset_timestamp_global/stations.csv` + `regions/*/{static_fields,lats,lons}.npy`, `processed/extra_descriptors.npy` (+ `_names.json`) |
| `extra_descriptors_stratified.ipynb` | **Table 6 / App. C** stratified tables for the hand-crafted-descriptor arm | `training_runs_snapshot_14y_eu/*_extradesc_*` (+ baseline/TESSERA arms), `processed/station_extra_descriptors.csv` |
| `tessera_1bm_variant_shortlist.ipynb` | provenance of the canonical latent choice (1B-M 2017, `crop64_lat16_auxon`) | the VAE-variant runs in `training_runs_snapshot_14y_{region}_tessera_1B-M_{2017,2024}/` |
| `uncertainty_estimation.ipynb` | calibration / resolution appendix material (σ reliability, spread–skill, terrain-conditional coverage) | per-run `test_predictions.npz` of the +mTPI headline runs; writes `uncertainty_analysis_outputs/` |
| `stratified_performance.ipynb` | qualitative "where does TESSERA help" claims (elevation, |Δelev|, density, latitude, season strata). It auto-selects the best TESSERA run per folder — the paper's run stems are the +mTPI ones. | `test_station_errors.npz` of the per-region folders, `dataset_timestamp_global/stations.csv` |

The dense-map figures (Figs 1, 3, 4, 9) are not notebook products; they come
from `scripts/maps/` (see `scripts/maps/run_region_maps.sh` and the provenance
note in `scripts/maps/generate_maps.py`). Figs 11–12 come from
`scripts/analysis/norway_descriptor_spaces.py`.

## `_helpers.py`

Imported by the notebooks as `import _helpers` (the notebook directory is on
`sys.path` when Jupyter runs from here). It provides:

- `find_repo_root`, `experiments_dir`, `output_dir_for_folder` — repository
  root, `scripts/experiments/`, and `training_runs_dir(folder)` under the data
  root.
- `list_folders`, `load_experiments_yaml(folder)` — the experiment definitions.
- `load_folder_results(folder, seeds)`, `load_all_results(folders, seeds)` —
  one row per (run, seed) from `test_summary.json`, with head-aware point
  metrics (`mae_at_median` / `rmse_at_mean` for the truncated-normal wind head).
- `print_summary(df, variable, ...)` — the per-(variable, distribution) table.
- `shortlist_experiments`, `baselines_for` — shortlisting and baseline names.
- `RunCategory`, `classify_run`, `_augment_df_with_classification`,
  `print_centralised_analysis` — config-driven classification of runs
  (baseline / TESSERA / shuffled control / …) and the paired tables built on it.
