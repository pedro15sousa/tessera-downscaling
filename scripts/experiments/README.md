# Snapshot-cadence experiments

Each subdirectory here is self-contained: one YAML (the experiment
list) + one bash script (the Slurm submitter). The YAML is the single
source of truth for what experiments belong to each output directory;
the notebook's `EXPERIMENT_DEFS` will be migrated to load from the
same YAMLs in a follow-up.

## Layout

```
experiments/
├── README.md
├── snapshot_6y_eu/
│   ├── experiments.yaml
│   └── submit.sh
├── snapshot_14y_eu/
│   ├── experiments.yaml
│   └── submit.sh
├── snapshot_14y_us/
│   ├── experiments.yaml
│   └── submit.sh
├── snapshot_14y_east_asia/
│   ├── experiments.yaml
│   └── submit.sh
├── snapshot_14y_australia/
│   ├── experiments.yaml
│   └── submit.sh
├── snapshot_14y_southern_africa/
│   ├── experiments.yaml
│   └── submit.sh
└── snapshot_global_14y/
    ├── experiments.yaml
    └── submit.sh
```

Each folder's name matches its output directory
(`.tmp_output/training_runs_<folder_name>/`), so `snapshot_14y_us/`
drives `training_runs_snapshot_14y_us/`, and so on.

## What each script does

| Folder | Dataset | Region(s) | Normalisation | Configs | Jobs |
|---|---|---|---|---|---|
| snapshot_6y_eu | dataset_timestamp (flat, 2017-23) | Europe | built-in | 28 | 84 |
| snapshot_14y_eu | dataset_timestamp_global | europe | per_region | 28 | 84 |
| snapshot_14y_us | dataset_timestamp_global | us | per_region | 28 | 84 |
| snapshot_14y_east_asia | dataset_timestamp_global | east_asia | per_region | 28 | 84 |
| snapshot_14y_australia | dataset_timestamp_global | australia | per_region | 28 | 84 |
| snapshot_14y_southern_africa | dataset_timestamp_global | southern_africa | per_region | 28 | 84 |
| snapshot_global_14y | dataset_timestamp_global | multi-region + transfer | global | 24 | 72 |

Grand total at 14y cadence: 6 × 84 + 72 = 576 jobs. Add the 84 for
6y-EU (being re-run for clean comparison) = 660 jobs.

## Running

```bash
# From anywhere (the script resolves its YAML relative to itself):
bash projects/tessera_downscaling/scripts/experiments/snapshot_14y_us/submit.sh
```

Dry run (prints sbatch commands without submitting):

```bash
DRY_RUN=1 bash projects/tessera_downscaling/scripts/experiments/snapshot_14y_us/submit.sh
```

## YAML schema

Each entry in an `experiments.yaml` is a dict with these fields:

```yaml
- name: wind_snap_vae_lat16_concat_with_elev_no_static_wd_drop0
  label: "Wind snap VAE-lat16 concat (with elev, no static, wd=1e-4)"
  colour: "#1f77b4"           # used by the notebook's plotting
  target_variables: [wind]    # list; multi-task = [t2m, wind]
  extra_args: "--interpolation bilinear --tessera-injection concat ..."
  # Only used by snapshot_global_14y/experiments.yaml:
  region_specs_train: {europe: train, us: all}
  region_specs_test: {europe: test}
```

`extra_args` is shell-formatted as it would appear on the `train.py`
command line. The submit script expands `${VAE_LATENTS_PATH}`,
`${VAE_LATENTS_PATH_LAT64}`, and `${VAE_LATENTS_CSV}` from the
environment (set at the top of each `submit.sh`) so the YAML doesn't
hardcode absolute paths.

`region_specs_train` / `region_specs_test` are only used by the
`snapshot_global_14y` YAML — for single-region experiments, the
script's `--train-regions` flag handles region selection.

## Editing the 28-config matrix

The 6 single-region YAMLs (5 × 14y + 1 × 6y EU) share an identical
28-config matrix — they differ only in which dataset the submit
script points at. If you add, remove, or modify an entry, **apply
the same change to all 6 YAMLs**. Each YAML's header comment names
the other 5 siblings to make the required edits obvious.

A diff check to catch drift:

```bash
cd projects/tessera_downscaling/scripts/experiments
for f in snapshot_6y_eu snapshot_14y_eu snapshot_14y_us \
         snapshot_14y_east_asia snapshot_14y_australia \
         snapshot_14y_southern_africa; do
    python3 -c "
import yaml
exps = yaml.safe_load(open('$f/experiments.yaml'))
for e in exps:
    print(e['name'], e['target_variables'], e['extra_args'])
" > /tmp/$f.txt
done
md5sum /tmp/snapshot_*.txt  # all 6 should be byte-identical
```

The `snapshot_global_14y` YAML is structurally different (uses
`region_specs_train`/`region_specs_test`) and is not part of this
6-way sync.

## How a submit.sh runs

1. Loads hyperparameters (seeds, epochs, patience, LR schedule, etc.)
   at the top.
2. Exports `VAE_LATENTS_PATH{,_LAT64}`, `VAE_LATENTS_CSV` so the
   Python-in-bash YAML loader can expand the shell refs.
3. Runs a preflight check: dataset exists, `metadata.json` has the
   expected `layout_version`, per-region stats file exists (for
   single-region scripts), or global stats file exists (for
   `snapshot_global_14y`).
4. Iterates `(config, seed)`. For each pair:
   - If `<run_dir>/test_summary.json` exists already, skip (lets you
     re-run the script to submit any new configs without re-running
     the existing ones).
   - Builds an `sbatch --wrap "cd <repo> && python train.py ... && python evaluate.py ..."`.
   - Submits (or prints, under `DRY_RUN=1`).

## Expected runtime

At 14y cadence, each job runs for up to the Slurm `--time=24:00:00`
cap with `--patience 10`. Most jobs finish well within the budget.
The full suite (660 jobs) completes in ~2-3 days wall-clock given
typical GPU availability on Isambard.

## Renames and preservation

The old `training_runs_snapshot_global/` directory (from the 6y
multi-region experiments) has been renamed to
`training_runs_snapshot_global_6y/` to preserve those results
alongside the new 14y ones.
