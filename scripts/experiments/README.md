# Experiment folders

Every folder here is one output directory of the paper: `<folder>/` drives
`<data root>/training_runs_<folder>/` (data root = `$TESSERA_DATA_ROOT`,
default `/data/weather-downscaling`; see `DATA.md`). Each folder holds

* `experiments.yaml` -- the list of configurations (the single source of truth;
  `notebooks/_helpers.py` and `scripts/paper/*` read the same files), and
* `submit.sh` -- the Slurm submitter, a few lines on top of `_lib.sh`.

All trained runs share the dataset `dataset_timestamp_global`
(`multi_region_snapshot_v1` layout, per-region normalisation stats), the
hyperparameters in `_lib.sh` (batch 1, 100 epochs, patience 10, lr 2.5e-5
with 5 % warm-up, CNN 128x7, MLP 128x3) and seeds 42/123/456. Every run --
no-model references included -- is filtered to the TESSERA-valid station set
(`processed/tessera_global/patch_embeddings_2024.npy`, centre pixel non-zero
and >= 50 % coverage; filter only, no patches are loaded).

## Folders

| Folder | Entries | What |
|---|---|---|
| `snapshot_14y_{eu,us,east_asia,australia,southern_africa}` | 12 | latents-independent arms of one region: persistence and ERA5-interp references (t2m also with lapse-rate correction), the no-TESSERA ConvCNP (+mTPI, static fields on), the same + 17 hand-crafted descriptors, and the v1 TESSERA arm |
| `snapshot_14y_<region>_tessera_1B-M_2017` | 20 (eu, east_asia) / 12 (others) | the paper's TESSERA arm (`*_vae_crop64_lat16_auxon_concat_mtpi`), the VAE-variant model-selection sweep, the +descriptors and summary-statistics controls |
| `snapshot_14y_<region>_tessera_1B-M_2017_shuffled` | 2 | shuffled-latent control (Table 4 / App. C) |
| `snapshot_14y_cross_lead` | 7 | lead-conditioned models trained on the ERA5 + Aurora {6, 24, 72} h mix for europe and east_asia, evaluated per lead into `eval_lead{0,6,24,72}h/`; plus the per-lead interp references on the matched station set |
| `snapshot_14y_cross_lead_tessera_1B-M_2017` | 2 | the paper's TESSERA arm of the cross-lead experiment |
| `snapshot_14y_eu_temporal_rollout_norway_lat16_mtpi` | 4 architectures x 10 sweep points + 4 references | Norway station-network rollout (Fig 7), baseline and v1 TESSERA arms |
| `snapshot_14y_eu_temporal_rollout_norway_tessera_1B-M_2017` | 2 x 10 | the paper's TESSERA arm of the rollout (same schedule files) |

The regional folders are kept entry-for-entry identical across the five
regions; the `_tessera_1B-M_2017` folders differ only in how much of the
variant sweep was run in that region.

## Running

```bash
bash scripts/experiments/snapshot_14y_eu/submit.sh              # sbatch one job per (entry, seed)
DRY_RUN=1 bash scripts/experiments/snapshot_14y_eu/submit.sh    # print the commands only
LOCAL=1   bash scripts/experiments/snapshot_14y_eu/submit.sh    # run sequentially in this shell
```

Runs with a `test_summary.json` are skipped, so re-running a submitter only
dispatches what is missing. Knobs (env): `TESSERA_DATA_ROOT`, `OUTPUT_ROOT`,
`SEEDS="42 123"`, `TIME`, `PARTITION`, `LOG_DIR`. Trained entries are one GPU
job (`uv run tessera-train ... && uv run tessera-evaluate ...`); reference
entries (`baseline_kind` / `simple_baselines`) are one CPU job running
`uv run tessera-baselines`.

## YAML grammar

Flat folders (everything except the two rollout folders) are a list:

```yaml
- name: t2m_snap_vae_crop64_lat16_auxon_concat_mtpi
  label: "t2m VAE crop64-lat16 auxon concat (elev + mTPI, no static, wd=1e-4)"
  colour: "#1f77b4"                 # plotting only
  target_variables: [t2m]           # t2m | wind
  extra_args: "--interpolation bilinear --tessera-injection concat ..."
  baseline_kind: era5_interp        # only on no-model references
```

`extra_args` is the tail of the `tessera-train` command line. `${VAR}`
references are expanded from the environment exported by `_lib.sh` /
`submit.sh` (`VAE_LATENTS_PATH`, `VAE_LATENTS_CSV`, `VAE_LATENTS_DIR`,
`TESSERA_VARIANT`, `EMBED_YEAR`, `EXTRA_DESCRIPTORS_PATH`,
`SUMMARY_STATS_PATH`), so no absolute path lives in a yaml.

The rollout folders use three top-level keys: `sweep_points`
(`label`/`description`), `architectures` (`name`/`label`/`family`/
`target_variables`/`extra_args`) and `simple_baselines`
(`name`/`baseline`/`target_variables`[/`extra_args`]). Their sidecars are
built once and committed: `probe_station_ids.json` by `pick_probe_set.py`,
`rollout_schedule.json` by `build_rollout_schedule.py`; `submit.sh` derives
`probe_active_from_<sweep>.json`, `train_end_overrides.json` and
`region_specs_{train,test}.json` from them on every run.

## Outputs

`<data root>/training_runs_<folder>/<name>_seed<S>/` (cross-lead:
`<region>/<name>_seed<S>/eval_lead<L>h/`; rollout: `<arch>_<sweep>_seed<S>/`)
with `best_model.pt`, `config.json`, `test_summary.json`, `test_results.json`,
`test_predictions.npz`, `test_station_errors.npz`, `training_curves.npz`,
`training_summary.json`. `DATA.md` lists the per-file contents.
