# Tessera downscaling — standalone-repo migration plan

*Prepared 2026-08-17 against `cambridge-mlg/end-to-end-forecasting@68e54b0` (branch `pedro15sousa/tessera-downscaling`), the preprint "Earth observation embeddings are effective sub-grid descriptors for probabilistic weather downscaling" (39 pp.) and the AMS journal draft (46 pp.). Data/model root: `/data/weather-downscaling`. Updated 2026-08-18 after scoping decisions (see "Status").*

## Status (2026-08-18)

Scoping decisions taken:
- **Wind-energy project stays out**: `scripts/preprocessing/wind_energy/` remains in the monorepo untouched, to be migrated to its own repo later. `capacity_factor` and the two extra ERA5 variables it needed (`surface_pressure`, `boundary_layer_height`) do not belong to this repo.
- **Data pipeline is end-to-end**: the `dataprocessing` ingestion scripts are vendored (§2.6). `era5_static_0p25_all.nc` (13 ERA5 invariant fields, 12 MB; one-off CDS/cfgrib download 2026-03-18) is **treated as given** — it ships with the data root and is documented in `DATA.md`; no downloader is written for it.
- **Vanilla SetConv stays**: `RBFSetConv`, `setconv_length_scale` and `--interpolation setconv` are **KEEP** for future re-runs; only the embedding-conditioned / stream / attention SetConv variants are cut. **Default `--interpolation` is `bilinear`** (what every paper run used), so a bare invocation reproduces the paper.
- **Mechanics**: the monorepo checkout is left exactly as is; the new repository is built locally at `/home/pmms2/tessera-downscaling` and pushed to a new remote when ready.

Done (all in the new repo `/home/pmms2/tessera-downscaling`; the monorepo checkout is unchanged). The history was rebuilt once on 2026-08-18 to include the data package in the import commit, so the SHAs below supersede earlier ones:

| commit | content |
|---|---|
| `0e6ea1c` | verbatim import of `projects/tessera_downscaling` (601 files) minus `wind_energy/`, the git-ignored `maps/outputs`, caches, the 18 MB `convcnp_experiment_analysis.ipynb` |
| `de1856b` | vendored `dataprocessing` ingestion scripts + `download_{era5,ghcnh}.sh` (working-tree versions) |
| `3ea484f` | `pyproject.toml` (uv, cu126 torch index, pinned deps), README, HISTORY, `uv.lock`; fresh env → 74/74 tests |
| `1b8dfd1` `6814c1b` `1d9fcab` | this plan (+ directives; VAE aux heads verified) |
| `dbfe546` | `tessera_downscaling.paths` — the single data-root resolver with Isambard-prefix remapping |
| `19ed42c` | gitignore fix (`/data/` anchored) |
| `de5cfff` | model package pruned 2,578 → 1,115 lines; vanilla SetConv kept, default bilinear; bit-identical outputs on three real checkpoints (`tests/test_checkpoint_compat.py`); params 984,322 / 997,250 |
| `aa9d456` | data package pruned 2,283 → 812 lines; synthetic tests; real-dataset smoke test |
| `0137c7a` | figures / maps / analysis scripts and notebooks reorganised on the data root; 7 dead notebooks deleted; 21 figures regenerate with 0 drift warnings |
| `40697d8` | `scripts/paper/make_paper_tables.py` + `paper/tables/`: Table 1 144/144, Table 4 120/120, Table 6 120/120 reproduced; AMS hand-crafted row typos 2.73→2.72, 1.30→1.29 |
| `90ae969` | data pipeline under `scripts/data` + `scripts/preprocessing` + `scripts/aurora`; `io_utils.py`, `preprocessing/helpers.py` in the package; 19 experiment folders on a shared `_lib.sh`, regional YAMLs pruned to 12 entries; `DATA.md`, Makefile, pre-commit, CI |
| `07791df` | `tessera-train` / `tessera-evaluate` / `tessera-baselines` console entry points (train 2088→1228, evaluate 1615→915, baselines 1136→1017 lines); tolerant config reads; `tests/test_evaluate_config_compat.py` loads every checkpoint family on disk |
| `b181570` | `ruff format` + lint policy (E, W, F, I, B, UP), tree lint-clean |
| `89412d9` | notebook outputs stripped (hook), hooks scoped, LICENSE, README |

Verification on the final tree: `uv run pytest` 65 passed; `make_paper_figures.py` 21/21 PDFs, 23 `[ok]`, 0 `[warn]`; `make_paper_tables.py` identical to the committed tables; `pre-commit run --all-files` clean; no legacy path literal left in code (`paths.py` holds the two Isambard prefixes for remapping). Repository size 27 MB (was 83 MB imported, 194 MB in the monorepo).

Still to do:
1. ~~Data rescue~~ — **complete 2026-08-20**: all 133 migration jobs done, zero failures; `dataset_timestamp_global` has 19,032 snapshots in every region + GHCNh, `metadata.json`, `valid_station_indices.npy`; the three Aurora-lead `ghcnh_snapshot` symlinks are relinked (relative, resolving); the patch-encoder source tree is archived at `tessera_patch_encoder/repo/`. Phase 5.4 passed: `tessera-evaluate` on the stored Southern Africa Tessera t2m seed-42 checkpoint reproduces all 23 stored metrics to ≤4e-4 relative (float32 reduction noise; identical at paper precision). Phase 5.5 passed: `tessera-train` one epoch on `southern_africa` (2,008 steps, canonical Tessera flags) → `best_model.pt`, `training_curves.npz`, `training_summary.json`; 0 non-finite-gradient skips, val loss 2.53.
2b. **AlphaEarth / OlmoEarth benchmark arms restored** (author's request, 2026-08-20, commit `6ad2bf1`): `scripts/patch_encoder/extract/{extract_alphaearth,extract_olmoearth_imagery,extract_olmoearth_embed}.py`, `vae_{alphaearth,olmoearth}.yaml`, Slurm extract + 16-run sweep drivers; both FM checkpoints strict-load and reproduce their latents to ~3e-4; `fm` extra in pyproject (the OlmoEarth encoder package pins torch<2.8 and lives in its own venv); JEPA left out. The data (AlphaEarth 2×152 GB, OlmoEarth imagery 2×43 GB + embeddings) is now KEEP on the mount — §4's drop list is superseded for those entries.
2. ~~Fold the VAE patch encoder in~~ — done 2026-08-20: `src/tessera_downscaling/patch_encoder/` + `scripts/patch_encoder/` (JEPA / AlphaEarth / OlmoEarth dropped); the paper checkpoint loads strict=True and its published latents reproduce to 1.8e-6; 81 tests green.
3. Push to a new remote (personal GitHub account), enable the two GitHub workflows. **The monorepo copy is kept** (author's decision 2026-08-20) — Phase 6 is dropped; `projects/tessera_downscaling` in `end-to-end-forecasting` remains as the historical tree (and still hosts the untracked `wind_energy/` code until its own migration).
4. Manuscript items §0.1–0.4 (bilinear vs SetConv wording; dense-map provenance; VAE aux heads; the two AMS Table 1 typos).

Answer to "is anything still missing from the mount?" — two things, both Isambard-only:
- **`dataset_timestamp_global`** (the 6-hourly ERA5 crops + GHCNh snapshots + `metadata.json` for all five regions, 2010–2023; ~176 GB). On `/data` it is a 7.6 MB skeleton (verified 2026-08-18: all five `era5_snapshot/` empty, no `ghcnh_snapshot/`, no `metadata.json`; the three `dataset_timestamp_aurora_lead*h/ghcnh_snapshot` symlinks dangle into it; it was never in `_migration/jobs.tsv`). It is *not* needed to regenerate the paper figures/tables from the stored run outputs, but it is needed for anything else: retraining, re-evaluating a checkpoint, the Aurora-lead evaluations, simple baselines, regenerating the dense maps. It **is rebuildable** from what is on the mount (`_staging/processed/era5_wb2_quarter_*` 2 TB + `ghcnh/data` 153 GB + `era5_static`) via `preprocess_timestamp_global.py` — hours of compute and a risk of tiny differences in station sets / normalisation stats — so pulling the original 176 GB while Isambard is up is the better option.
- **The VAE patch-encoder code** (`~/tessera-patch-encoder` on Isambard, uncommitted): its outputs (checkpoints, latents) are on the mount, the code is not.
- Everything else the paper needs is on the mount (models, latents, descriptors, Aurora-lead datasets, dense grids, DEMs, figure inputs, raw ERA5/GHCNh, Tessera 2017/2024 patches, VAE checkpoints).

Isambard rescue (2026-08-18): eight jobs appended to `/data/weather-downscaling/_migration/jobs.tsv` — `dsg_top`, `dsg_ghcnh`, `dsg_{europe,us,east_asia,southern_africa,australia}` (→ `dataset_timestamp_global/`) and `pe_repo` (→ `tessera_patch_encoder/repo/`, excludes venv/outputs/data) — and four workers started; they poll every 300 s until the clifton cert on this machine is valid (it still showed 2026-08-11 → 08-12 at 19:46 UTC), then transfer. `bash /data/weather-downscaling/_migration/status.sh` shows progress.


---

## 0. Executive summary

**The split is clean.** `projects/tessera_downscaling` has **zero Python imports** from `core`, `dataprocessing`, `downscaling`, `auroraft`, `obsmodel`, `neuralprocesses` or `wandb`. The only monorepo coupling is at the *data-production* level: seven small scripts in `projects/dataprocessing/scripts/{era5,ghcnh,gee}` + `src/dataprocessing/utils.py` (~1,450 lines total) produce the raw ERA5 / GHCNh / mTPI / hand-crafted-descriptor inputs. Vendor them and the new repo is self-contained.

**Roughly half of the code is dead for the paper.** Of ~7,600 package lines, ~2,600–3,800 can go (daily-aggregate datasets, end-to-end patch encoder, FiLM/attention/embedding-conditioned kernels, Weibull/Bernoulli-Gamma/generative heads, station allowlists, global normalisation). `train.py`/`evaluate.py` lose ~25 %/20 %. Of 35 experiment folders, 19 survive (and their YAMLs shrink from ~2,100 entries to ~120). Of 17 notebooks, 5 are load-bearing, 4 are optional provenance, 7 are dead ends (~23 MB, incl. an 18 MB notebook). The `wind_energy/` tree (24 files, ~7.7k lines) is a separate project and leaves entirely.

**Five things must be resolved before or during the migration — they affect the *manuscript*, not just the code:**

1. **The trained models use parameter-free bilinear interpolation, not the learned Gaussian-kernel SetConv the paper describes.** Every paper run has `interpolation: 'bilinear'` in its `config.json` (verified on `/data`), and `BilinearInterp` is `F.grid_sample` with no parameters ([convcnp.py:177](src/tessera_downscaling/model/convcnp.py#L177)). Preprint Eq. (1) + App. A.4 and AMS Table A5 ("Kernel length scale learned as log ℓ, initialised ℓ = 0.5°") describe `RBFSetConv`, which no paper run used. Fix the text (or re-run with SetConv). This also touches the parameter counts quoted (9.84×10⁵ / 9.97×10⁵ — small effect, but re-derive).
2. **The dense-map figures come from a different model generation than Table 1.** Figs 3/4/9 (preprint) = Figs 4–7 + App. E (AMS) were generated from `training_runs_snapshot_14y_eu/{t2m,wind_truncnormal}_snap_vae_lat16_concat_with_elev_no_static_wd` — the *v1* runs (v1 latents from VAE `lat16_beta0.0005_grad0.5_e200` on Tessera v1 **2024** p64 patches, no mTPI) — with 2024 dense-grid patches encoded by that same v1 VAE ([regions.py:36-45](scripts/maps/regions.py#L36-L45); `processed/dense/*.npz` metadata `run_name='lat16_beta0.0005_grad0.5_e200'`). Table 1 / Fig 2 use 1B-M **2017** latents. The paper says 2017 1B-M is used for "all training, validation, and test snapshots". Either regenerate the maps with the 1B-M 2017 models (needs 2017 v2 dense patches for Iberia/Norway + the `dataset_timestamp_global` ERA5 snapshots for the four dates) or disclose the difference.
3. **The canonical VAE latent is `crop64_lat16_grad0.5_auxon`** — **verified 2026-08-19 in `tessera-patch-encoder` (`src/vae/model.py`, `src/vae/losses.py`, the checkpoint's `config.yaml`)**: `auxiliary.enable: true` adds three MLP heads (hidden 64) on the latent `z` that regress **elevation (weight 1.0), latitude (0.5) and longitude (0.5)** with masked, normalised-target MSE, so the training loss is `L = L_recon + 0.5·L_grad + β·KL + Σ λ_i·L_aux_i`. App. A.3 / Table A5 describe only the first three terms. This needs an explicit sentence in the paper, and it bears on the interpretation of the descriptor-space comparisons (Figs 5/11/12), where the Tessera latent is contrasted with the geographic and elevation + mTPI spaces it was partly supervised to encode. (The `_auxoff` sibling latents exist on `/data` and in the `_tessera_1B-M_2017` folders' sweep for an ablation.)
4. **Table 1, Table 4 (shuffled) and Table 6 / App. C have no script** — they exist only as cells 14–18 of `cross_folder_analysis.ipynb` and `extra_descriptors_stratified.ipynb`, hard-coded to the Isambard path. Port them to `make_paper_tables.py` and confirm they reproduce the PDF numbers.
5. **`dataset_timestamp_global` — the actual training input for every paper run — was never migrated.** `/data/weather-downscaling/dataset_timestamp_global/` is a 7.6 MB skeleton (`stations.csv` + per-region `lats/lons/static_fields.npy`); all five `era5_snapshot/` dirs are empty and `ghcnh_snapshot/` is absent (the three `dataset_timestamp_aurora_lead*h/ghcnh_snapshot` symlinks dangle into it). It is ~176 GB on Isambard. Rescue it while Isambard access lasts (`clifton auth`, then one job in the existing `_migration` harness); otherwise rebuild from `_staging/processed` (2 TB ERA5 + 152 GB GHCNh must therefore stay). Without it nothing can be retrained and the Aurora-lead evaluation datasets are unusable.

Also not in this repo but part of the paper: the **VAE patch encoder** (§2.2 / App. A.3) lives in the separate, still-uncommitted `tessera-patch-encoder` repo on Isambard. Rescue `src/{vae,common}`, `scripts/vae/*`, `configs/vae.yaml` and fold them into the new repo as `patch_encoder/`.

Recommended git strategy: **fresh repository**, single verbatim "import" commit of the keep-set, then reviewable prune/refactor commits. The project's history is 8 commits by one author dominated by a bulk dump (`b4408ac`) and an 18 MB notebook blob — nothing worth `git subtree split`ting (and `git filter-repo` isn't installed).

---

## 1. What the papers actually need

Both drafts contain the same experiment set; the AMS draft promotes the hand-crafted-descriptor arm into Table 1 / Fig 2 and re-numbers. Every item below maps to code that exists, except where flagged.

| Paper item (preprint → AMS) | Producer | Runs / inputs on `/data` |
|---|---|---|
| Fig 1 region overview | `scripts/maps/plot_region_overview.py` → `make_paper_figures.fig01` (copies `overview/region_overview.pdf`) | `paper_figure_outputs/maps_outputs/overview/`, `processed/overview_cache/` |
| Table 1 per-region MAE/RMSE/CRPS | **no script** — `cross_folder_analysis.ipynb` cell 15 (`CELLS_MTPI`) | `training_runs_snapshot_14y_{region}/{t2m,wind_truncnormal}_snap_bilinear_baseline_mtpi_wd_seed*` (baseline), `…_bilinear_baseline_mtpi_extradesc_wd` (AMS hand-crafted row), `*_era5_interp*`, `*_persistence_baseline`; `training_runs_snapshot_14y_{region}_tessera_1B-M_2017/{t2m,wind_truncnormal}_snap_vae_crop64_lat16_auxon_concat_mtpi_seed*` |
| Fig 2 (+ext) CRPS uplift | `make_paper_figures.fig02/fig02ext` (= `cross_folder_analysis` cell 17) | same + `…/eval_train_stations/` (from `reeval_train_stations.sh`) |
| Figs 3–4 → 4–7 dense maps | `scripts/maps/{fetch_dem,extract_dense_grid_patches,generate_maps}.py` → `make_paper_figures.fig03/fig04` | `paper_figure_outputs/maps_outputs/{iberia,norway}/*_dem.npz`, `processed/dense/`, `processed/dem_cache/` — **not regenerable today** (see §0.5, §0.2) |
| Fig 5 (+ Fig 8) → Fig 3 RF residual probe | `residual_structure_analysis.ipynb` §3c/§3g → `make_paper_figures.fig05/fig08` | `dataset_timestamp_global/stations.csv` + `regions/{us,europe}/{static_fields,lats,lons}.npy` (present), per-run `test_predictions.npz`, `processed/extra_descriptors.npy`. NB `scripts/descriptor_analysis/residual_probe_spaces.py` is a *different* probe (different station set / RF params) — its numbers ≠ the paper's |
| Fig 6 → Fig 8 Aurora RMSE uplift; Fig 10 (preprint App. F) CRPS decay | `analyze_cross_lead.ipynb` → `make_paper_figures.fig06/fig10` | `training_runs_snapshot_14y_cross_lead{,_tessera_1B-M_2017}/{europe,east_asia}/*/eval_lead{0,6,24,72}h/test_summary.json`; `*_era5_interp{,_lapse}_baseline_matched_seed42`; cross-check `notebooks/cross_lead_analysis_outputs/all_results_tidy.csv` |
| Fig 7 → Fig 9 Norway rollout | `data_efficiency_temporal_rollout.ipynb` → `make_paper_figures.fig07` | `training_runs_snapshot_14y_eu_temporal_rollout_norway_lat16_mtpi` (baseline+interp) + `…_tessera_1B-M_2017` (Tessera), `test_station_errors.npz`; `scripts/experiments/snapshot_14y_eu_temporal_rollout_norway_lat16_mtpi/rollout_schedule.json` |
| Fig 9 (preprint) error-alignment scatter; App. D/E texture + random-perturbation control | `scripts/maps/{station_eval,summary_table,residual_alignment_fulltest}.py` → `make_paper_figures.fig09` | `maps_outputs/*/*_stations.npz`, `*_summary.csv`; per-run `test_predictions.npz` |
| Figs 11–12 → F1–F2 reachability / AUC | `scripts/norway_rollout_descriptors/norway_descriptor_spaces.py` → `make_paper_figures.fig11/fig12(+ext)` | `processed/vae_tessera_1B-M/station_latents_1B-M_p128_2017_crop64_lat16_grad0.5_auxon.npy`, `processed/tessera_global/station_list_filtered.csv`, europe static/lats/lons, `extra_descriptors.npy`, `rollout_schedule.json` |
| Table 4 → C1 shuffled + summary-stats controls | **no script** — `cross_folder_analysis` cell 16; latents from `scripts/shuffle_latents.py`, `scripts/preprocessing/build_summary_stats_latents.py` | `…_tessera_1B-M_2017_shuffled/`, `…_tessera_1B-M_2017/*_stats16_crop64_concat_mtpi` |
| Table 6 → B1 extended descriptor | **no script** — `extra_descriptors_stratified.ipynb`; features from `build_extra_descriptors.py` ← `dataprocessing/scripts/gee/fetch_station_extra_descriptors.py` | `*_extradesc_*` runs in both folder families |
| Tables 2/3 → A1/A2 predictors, boxes | `preprocess_timestamp_global.py` `REGIONS` (authority) = `plot_region_overview.py:65-71` | — |
| Table A5 (AMS) VAE + ConvCNP settings | `tessera-patch-encoder` (Isambard) + `train.py` | see §0.1, §0.3 |
| App. E (preprint) US-vs-EU orography displacement diagnostics | **producer not located** anywhere (repo, notebooks, Isambard home) | flag |
| Simple baselines (persistence, ERA5-interp, fitted lapse) | `scripts/baselines/evaluate_simple_baselines.py` (called directly by each regional `submit.sh`) | `*_era5_interp_lapse_baseline*`, `*_persistence_baseline*` |
| Model-selection provenance (why crop64/lat16/auxon/2017) | `tessera_1bm_variant_shortlist.ipynb` | the 6 other VAE variants in `_tessera_1B-M_2017` folders, the 5 `_tessera_1B-M_2024` folders |

**Canonical paper configuration** (from the YAMLs + `config.json`):

```
# ConvCNP + Tessera
--interpolation bilinear --tessera-injection concat
--vae-latents-path processed/vae_tessera_1B-M/station_latents_1B-M_p128_2017_crop64_lat16_grad0.5_auxon.npy
--vae-latents-station-csv processed/tessera_global/station_list_filtered.csv
--no-static-fields --use-mtpi --weight-decay 1e-4 [--likelihood wind=truncated_normal]   # t2m → gaussian
# ConvCNP no-Tessera baseline
--interpolation bilinear --use-mtpi --weight-decay 1e-4   # static ERA5 fields ON
# every run additionally: --tessera-path processed/tessera_global/patch_embeddings_2024.npy (station-validity filter only)
#   --min-tessera-patch-coverage 0.5, per-region normalisation, seeds 42/123/456, cnn 128×7×k3, mlp 128×3
```

---

## 2. Keep / cut manifest

Verdicts: **KEEP** · **CUT** · **?** (decision needed; my recommendation in brackets).

### 2.1 Package `src/tessera_downscaling/` (7,635 lines incl. tests)

| Module | Keep | Cut (est. lines) | Notes |
|---|---|---|---|
| `data/dataset.py` (2,283) | `MultiRegionSnapshotDownscalingDataset`, `SnapshotRegionState`, `MultiLeadDataset`, `_load_optional_station_mtpi`, `DEFAULT_MIN_TESSERA_PATCH_COVERAGE`; kwargs `region_specs`/`regions`+`station_split` (`train`/`test`/`all`), `probe_active_from`, `train_end_override`, `drop_context_channels(_strict)`, `lead_hours`, `vae_latents_*`, `include_static_fields`, `min_patch_coverage` | `DailyDownscalingDataset` (274), `RegionState`+`MultiRegionDownscalingDataset` (536), raw-patch serving `load_tessera_patches`/`_patches_mmap` (~80), `train_station_allowlist`+`_apply_post_filter_mask` (~90), `normalisation_policy="global"` (~20) ≈ **1,000** | **?** `SnapshotDownscalingDataset` flat `snapshot_v1` (556) — only `snapshot_6y_eu` + legacy `dataset_timestamp/` + `evaluate_simple_baselines.py` import it [cut with them] |
| `data/helpers.py` (759) | everything else | `slice_tessera_patches`, point-embedding branch, `keep_mmap_alive`, dead `include_mtpi` param; shrink `SUPPORTED_TARGET_VARIABLES` to `{"t2m","wind"}` (drops `capacity_factor` from `eac2df2`, `tmax`, `wind_mean`, `precip`) ≈ **50** | |
| `data/vae_latents.py` (153) | all | — | load path for VAE latents, shuffled latents, summary-stats, extra descriptors |
| `data/dense_grid_patches.py` (405) | all | unused `Iterable` import | map inference; uses `geotessera` download path; year default 2024 (see §0.2) |
| `data/tessera.py` (326) | — | `extract_point_embeddings` (36) at minimum | **?** whole module [cut] — the paper's patches were produced by `scripts/extract_tessera_patches_local.py` (v2 mount, bit-identical to geotessera mosaic). Keep only if you want a mount-free geotessera path; if cut, also cut `scripts/extract_tessera.py` + `tests/test_tessera.py` |
| `model/convcnp.py` (1,280) | `ResidualBlock`, `GridCNN`, `BilinearInterp`, **`RBFSetConv` (vanilla SetConv, kept for future re-runs; default `--interpolation` → `bilinear`)**; state-dict prefixes `cnn.`, `interp.` (setconv), `mlp.`, `heads.heads.<var>.`, `DecoderMLP`, `ConvCNPDownscaler` (trimmed), `tessera_injection ∈ {concat, none}` (`none` used by cross-lead baselines) | `EmbeddingConditionedSetConv`, `EmbeddingStreamSetConv`, `TargetEmbeddingAttention` + design-note block (~300, only `snapshot_14y_east_asia` sweep + `embedding_mechanism_analysis.ipynb`), `FiLMDecoderMLP` (107), `tessera_encoder` arg, `precomputed_proj_*`, `decoder_kernel`, `use_target_embed_stream`, `target_embed_attention`, `detach_attn_embed` ≈ **530** | **?** `precomputed_drop_prob` (10) [cut]. `setconv_length_scale` stays with `RBFSetConv`; fix the module docstring so it describes both interpolators |
| `model/heads.py` (1,006) | `LikelihoodHead`, `_ensemble_crps` (the 200-sample fair CRPS), `GaussianHead`, `TruncatedNormalHead` (incl. `initialise_from_climatology`, `_quantile` stability path — do not simplify), `HEAD_REGISTRY`, `build_head` | `WeibullHead` (65; also broken — `train.py:1644` calls a method it lacks), `BernoulliGammaHead` (142), `GenerativeHead` (173), `_softplus_floor`, `has_density`, unused clamps ≈ **410** | |
| `model/heads_dispatch.py` (190) | `LikelihoodHeadDict` + `forward` | `total_nll`, `predictive_means`, `predictive_medians` (0 callers) ≈ **62** | |
| `model/tessera_encoder.py` (102) | — | whole module (`--tessera-method` appears in no YAML/submit) + ~100 downstream lines | |
| `tests/test_heads.py` (526) | Gaussian (legacy-parity + scipy CRPS), TruncNormal μ≪0 suite, registry tests | Weibull / B-G / Generative blocks ≈ **230** | |
| `tests/test_convcnp.py` (366) | fixtures, baseline forward, hypernet-rejection, target-var, decoder-hidden tests | `_FakeTesseraEncoder`, FiLM tests; rewrite the mixed-distribution test as gaussian+TN+concat-latent ≈ **60** | **?** migration round-trip tests (~210) — only justify `migrate_legacy_checkpoints.py` [cut both] |
| `tests/test_lapse_rate_baseline.py` (133) | all | — | loads `scripts/baselines/evaluate_simple_baselines.py` **by path** — preserve that location or make baselines importable |
| `tests/test_tessera.py` (104) | — | with `data/tessera.py` | |

### 2.2 Entry points `scripts/train.py` (2,088) and `scripts/evaluate.py` (1,615)

Keep flags: `--dataset-dir --output-dir --tessera-path --tessera-station-csv --min-tessera-patch-coverage --tessera-injection{concat,none} --vae-latents-path --vae-latents-station-csv --vae-latents-drop-prob(?) --extra-descriptors-path/-station-csv --train-regions --region-specs-{train,val,test}-file --target-variables{t2m,wind} --likelihood --use-mtpi --no-static-fields --normalisation-policy(pin per_region) --lr-warmup-pct --cnn-{hidden,layers,kernel} --interpolation --mlp-{hidden,n-hidden} --epochs --lr --weight-decay --drop-context-channels --lead-datasets --lead-hours --grad-clip-norm --batch-size --num-workers --patience --seed --probe-active-from-file --train-end-override --station-split --filter-vae-latents-path/-station-csv --checkpoint`.

Cut flags/paths: `--tessera-method/-output-dim/-chunk-size(never read)/-drop-prob`, `--decoder-kernel`, `--use-target-embed-stream`, `--target-embed-attention`, `--detach-attn-embed` (asymmetric train/eval anyway), `--vae-latents-proj-dim/-proj-mlp`, `--vae-latents-no-zscore`, `--val-regions`, inline `--region-specs-*`, `--loss-function crps` (CRPS *loss*; CRPS *metric* stays), `--region-balanced-sampling`, `--train-station-allowlist-file`, `--test-regions`, target names `tmax/wind_mean/precip`, `film`; code: `build_output_dir_name` + relative `.tmp_output` default ([train.py:1026](scripts/train.py#L1026), [evaluate.py:199](scripts/evaluate.py#L199)), region-balanced sampler, daily dataset branch, Bernoulli-Gamma & Weibull climatology inits, Kendall multi-task weights, the two per-batch NaN-diagnostic blocks (self-labelled temporary, [train.py:1748](scripts/train.py#L1748)), hypernet rejection, `GenerativeHead` inline diagnostics, Weibull/B-G/Generative metric blocks. ≈ 500 + 325 lines.

Keep: the non-finite-gradient skip guard, TN climatology init, warm-up, checkpointing, `test_station_errors.npz` per-station/per-subset aggregation (Fig 7/9 read it), `subset_per_station` labelling (`probe`/`always_on`/…), `setconv.→interp.` state-dict key shim.

**Constraint:** `evaluate.py` reads config from `ckpt["config"]`, so every removed flag must survive as `config.get(key, default)` if the existing checkpoints on `/data` are to remain evaluable. Do the removals as "stop parsing, keep tolerant read". Also fix the root cause of `patch_cross_lead_configs.py`: set `args.drop_context_channels` before `vars(args)` is embedded in the checkpoint.

### 2.3 Experiment folders (`scripts/experiments/`, 35 → 19)

| Verdict | Folders | Notes |
|---|---|---|
| **KEEP, prune YAML** | `snapshot_14y_{eu,us,east_asia,australia,southern_africa}` (122/251/319/227/256 entries) | keep only: `{t2m,wind_truncnormal}_snap_bilinear_baseline_mtpi_wd`, `…_bilinear_baseline_mtpi_extradesc_wd`, `{t2m,wind}_snap_era5_interp_baseline`, `t2m_snap_era5_interp_lapse_baseline`, `*_persistence_baseline`, and the v1 `{t2m,wind_truncnormal}_snap_vae_lat16_concat_with_elev_mtpi_no_static_wd` (fig02 train-station counts; rollout arch) → ~14–16 entries each. Delete `experiments_old.yaml`, `experiments.yaml.bak` |
| **KEEP** | `snapshot_14y_{region}_tessera_1B-M_2017` ×5 (12–20 entries: VAE variant sweep + extradesc + stats16) | canonical Tessera arm; the 6 non-canonical VAE variants back the model-selection provenance |
| **KEEP** | `snapshot_14y_{region}_tessera_1B-M_2017_shuffled` ×5 | Table 4/C1 |
| **KEEP (4 of 6 entries)** | `snapshot_14y_cross_lead` | drop the 2 `_film_` entries; keeps `*_era5_interp*_matched` refs |
| **KEEP** | `snapshot_14y_cross_lead_tessera_1B-M_2017` | |
| **KEEP** | `snapshot_14y_eu_temporal_rollout_norway_lat16_mtpi`, `…_norway_tessera_1B-M_2017` (+ `rollout_schedule.json`, `probe_station_ids.json`, `train_end_overrides.json`, `probe_active_from_r*.json`, `region_specs_*.json`) | Fig 7 + Figs 11/12 |
| **KEEP helpers** | `build_rollout_schedule.py`, `pick_probe_set.py` | regenerate the Norway schedule |
| **CUT** | `snapshot_14y_{region}_tessera_1B-M_2024` ×5 | **?** — runs exist on `/data`; keep artefacts, drop configs unless a "which year" sentence needs them [cut] |
| **CUT** | `snapshot_6y_eu` (177, flat layout, none of the canonical arms), `snapshot_14y_aurora_zeroshot` (no runs on `/data`; superseded by joint cross-lead), `snapshot_14y_eu_temporal_rollout_norway` (v1 proj8, superseded), `…_rollout_alps`, `…_temporal_efficiency{,_norway,_alps}`, `…_station_count_efficiency`, `…_placement_norway{,_stationcount}` (never submitted; 95 files each) | plus `build_station_count_schedule.py`, `rank_probes_by_coverage.py`; rewrite `experiments/README.md` (describes a deleted `snapshot_global_14y` design) |

Every kept `submit.sh` hard-codes `REPO_ROOT=/projects/u6do/…` and `BASE_DIR=…/.tmp_output`, uses `sbatch --wrap`, and calls `evaluate_simple_baselines.py` directly for `baseline_kind` entries — rewrite the header block against the new data-root env (§3.3).

### 2.4 Scripts

| Area | KEEP | CUT / ? |
|---|---|---|
| `scripts/paper_figures/` (untracked!) | `make_paper_figures.py` (1,722 lines, self-contained, regenerates all 21 PDFs with built-in numeric cross-checks) + **new** `make_paper_tables.py` | — |
| `scripts/maps/` | `regions.py`, `generate_maps.py`, `fetch_dem.py`, `extract_dense_grid_patches.py`, `plot_region_overview.py`, `station_eval.py`, `summary_table.py`, `residual_alignment_fulltest.py` (imported by `residual_structure_analysis.ipynb`), `run_region_maps.sh` (rewrite: drop cut steps) | `replot_paper_maps.py` (superseded by fig03/04), `analyze_maps.py`, `dem_plots.py`, `station_error_maps.py` (exploratory, ESRI tiles); **?** `select_dates.py` [keep as provenance, 99 lines], `interpret_plots.py` [cut; `summary_table.py` computes the same Spearman] |
| `scripts/baselines/` | `evaluate_simple_baselines.py` (1,136; persistence / ERA5-interp / fitted lapse; imported by tests + `residual_probe_spaces.py`) | — |
| `scripts/descriptor_analysis/` | **?** `residual_probe_spaces.py` [keep, labelled "independent robustness probe — not the Fig 5 source"] | — |
| `scripts/norway_rollout_descriptors/` | `norway_descriptor_spaces.py` | `norway_ood_latent_analysis.py` (v1, proj8) |
| `scripts/aurora/` | `generate_aurora_forecasts.py` (own `.venv-aurora`, `microsoft-aurora==1.8.0`; drop `UK_BBOX`, dedupe `compute_grid_crop_indices`), `submit_aurora_forecasts.sh` (rewrite paths) | **?** `migrate_global_staging_to_regions.py` [keep as provenance] |
| `scripts/preprocessing/` | `helpers.py`, `timestamp/preprocess_timestamp_global.py` (+submit), `aurora_timestamp/{preprocess_aurora,validate_aurora_datasets}.py` (+submit), `backfill_station_mtpi.py`, `build_extra_descriptors.py`, `build_summary_stats_latents.py`, `concat_station_vectors.py` | `daily/*` (2 files) + root `preprocess_daily.py`; **`wind_energy/`** (24 files, untracked, separate project — move to its own repo); **?** `timestamp/preprocess_timestamp.py` single-region precursor [cut] |
| top-level `scripts/` | `reeval_train_stations.sh`, `reeval_truncated_normal.sh` (paper wind numbers depend on it), `shuffle_latents.py`, `extract_tessera_patches_local.py` (production p128 extractor; make mount/landmask dirs configurable, remove `/home/pmms2/…`), **?** `shortlist_tessera_tiles.py` [keep, provisioning] | `run_experiments.sh`, `submit_daily_experiments.sh`, `submit_snapshot_experiments.sh`, `submit_snapshot_global.sh`, `submit_snapshot_jepa.sh`, `prepare_rerun.py`, `build_shortlist.py` (imports `notebooks/_helpers` via `sys.path`), `migrate_legacy_checkpoints.py`, `patch_cross_lead_configs.py` (one-shot; fix root cause instead), `extract_tessera.py`, `preprocess_daily.py` |

### 2.5 Notebooks (`notebooks/`, 17 → 5 + 4 optional)

| Verdict | Notebook | Why |
|---|---|---|
| **KEEP** | `cross_folder_analysis.ipynb` | Table 1 (cell 15), Table 4 (16), Fig 2 (17), train-station table (18); hard-codes HPC `BASE_DIR` (cell 14) |
| **KEEP** | `analyze_cross_lead.ipynb` | Figs 6/10, calibration table (cell 33); writes `cross_lead_analysis_outputs/all_results_tidy.csv` (cross-check input) |
| **KEEP** | `data_efficiency_temporal_rollout.ipynb` | Fig 7; the per-station CRPS-from-head-params logic lives only here |
| **KEEP** | `residual_structure_analysis.ipynb` | Figs 5/8, §4 alignment (imports `residual_alignment_fulltest`) |
| **KEEP** | `extra_descriptors_stratified.ipynb` | Table 6 / App. C stratified tables |
| **?** | `tessera_1bm_variant_shortlist.ipynb` [keep — model-selection provenance], `uncertainty_estimation.ipynb` [keep only if the calibration appendix survives review], `stratified_performance.ipynb` [keep; source of qualitative prose claims, but it auto-selects "best" runs — pin the paper's run stems], `single_folder_analysis.ipynb` [keep as dev tool] | |
| **CUT** | `convcnp_experiment_analysis.ipynb` (18 MB, replaced per `notebooks/README.md`), `analyze_aurora_zeroshot.ipynb` (+26 tracked outputs), `architecture_meta_analysis.ipynb`, `embedding_mechanism_analysis.ipynb` (5/9 sections are "(scaffold)"), `latent_interpretability.ipynb` (v1 latents), `normalisation_analysis.ipynb` (settled), `stratified_performance_latent_clusters.ipynb` (v1) | ≈ 23 MB |
| **KEEP** | `_helpers.py` (140 KB, imports nothing from the package; hard-codes `.tmp_output` at `output_dir_for_folder`), `README.md` (update) | |
| outputs dirs | keep `cross_lead_analysis_outputs/`, `norway_analysis_outputs/`; **?** `descriptor_analysis_outputs/`, `uncertainty_analysis_outputs/`; cut `aurora_zeroshot_analysis_outputs/` | |
| deleted-unstaged | `data_efficiency_station_count{,_old}.ipynb`, `data_efficiency_temporal.ipynb` show as `D` in `git status` — confirm intentional (they belong to cut experiments) | |

### 2.6 To vendor from the monorepo (the only real coupling, ~1,450 lines)

| Source | Lines | Role | Where it goes |
|---|---|---|---|
| `projects/dataprocessing/src/dataprocessing/utils.py` | 169 | `parallel_foreach`, `atomic`, `compute_file_name` (`YYYY-MM-DD-HH.nc` — the on-disk convention every preprocessor assumes), `write_and_flush_dataset` | `src/tessera_downscaling/io_utils.py` |
| `…/scripts/era5/weatherbench2.py` | 86 | WB2 zarr → `era5_wb2_quarter_{var}/data/*.nc`, levels 500/700/850 | `scripts/data/download_era5_wb2.py` |
| `…/scripts/era5/weatherbench2_aurora_levels.py` | 84 | 13-level variant for Aurora ICs | `scripts/data/download_era5_wb2_aurora_levels.py` |
| `…/scripts/era5/arco.py` (**uncommitted**, +141) | ~336 | ARCO-ERA5 fallback made drop-in compatible with the WB2 layout; real CLI | `scripts/data/download_era5_arco.py` — commit it |
| `…/scripts/ghcnh/ghcnh.py` | 215 | GHCNh PSV → 6-hourly NetCDF; its `ROOT` is *already* the tessera staging dir | `scripts/data/download_ghcnh.py` |
| `…/scripts/gee/fetch_station_mtpi.py` | 216 | ALOS mTPI via Earth Engine → `station_mtpi.csv` (`--use-mtpi` in every run) | `scripts/data/fetch_station_mtpi.py` (+ `earthengine-api` dep) |
| `…/scripts/gee/fetch_station_extra_descriptors.py` | 428 | 17 Bakketun features → `station_extra_descriptors.csv` | `scripts/data/fetch_station_extra_descriptors.py` |
| root `download_era5.sh`, `download_ghcnh.sh` | — | Slurm wrappers for the two downloaders (mtime = tessera work) | `scripts/data/slurm/` |
| `…/scripts/era5/cds.py` | 220 | CDS API helper (`ecmwf.datastores`), not used by tessera today | base for the missing `download_era5_static.py` |
| **uncommitted** `dataprocessing/utils.py` diff | 4 | adds `surface_pressure`, `boundary_layer_height` — **wind-energy inputs, not paper**; the `weatherbench2.py` diff retargets to `/data/weather-downscaling/_staging/processed` | carry as-is; note provenance |

Not needed: all of `core/` (20k lines), `projects/downscaling`, `auroraft`, `obsmodel`, `dataprocessing/scripts/{gmt,regridding,accumulate,…}`, `cds.py`.

### 2.7 Also required, but not in this repo: the VAE patch encoder

`tessera-patch-encoder` on Isambard (`~/tessera-patch-encoder`, uncommitted as of 2026-08-10). Rescue and fold in: `src/vae/{model,losses}.py`, `src/common/{dataset,blocks}.py`, `scripts/vae/{train,eval,encode_dense_grid,prebuild_cache,submit_dates.sh}`, `configs/vae.yaml`, `README.md`. Drop `src/jepa/*`, `scripts/jepa/*`, `scripts/extract/{extract_alphaearth,extract_olmoearth_*}.py`, `configs/vae_{alphaearth,olmoearth}.yaml` (Tessera-v2-paper benchmark work). Checkpoints are already on `/data/weather-downscaling/tessera_patch_encoder/outputs/vae/` — keep `p128_2017_crop64_lat16_grad0.5_auxon/` (+ the 7 sibling variants named in `processed/vae_tessera_1B-M/provenance.txt`), and the v1 `lat16_beta0.0005_grad0.5_e200/` that produced the dense-map latents; drop `jepa/`, `alphaearth/`, `olmoearth/`. Note `encode_dense_grid.py` hard-codes `(N,64,64,128)`.

### 2.8 Repo-root strays

`station_list.csv` (NOAA master, twin on `/data`; CWD-relative input of `extract_tessera_patches_local.py` — don't commit), `tessera_tiles_shortlist.{csv,txt}` (regenerable output), `global_0.1_degree_tiff_all/` (93 landmask tiffs; `/data/…/_cache/geotessera/` has 96 — delete), `runs/` (wind-energy smoke tests — delete). Also `.gitignore`'s bare `outputs/` is what swallowed `scripts/maps/outputs/` (94 MB of figure inputs that were never in git; only backup is `/data/…/paper_figure_outputs/maps_outputs/`).

---

## 3. The migration, step by step

### Phase 0 — Rescue and freeze (before touching any code)

1. **Isambard, while it lasts.** Run `clifton auth` (interactive), then:
   - append one `rsync` job to `/data/weather-downscaling/_migration/jobs.tsv` for `…/projects/tessera_downscaling/.tmp_output/dataset_timestamp_global/` → `/data/weather-downscaling/dataset_timestamp_global/` (~176 GB; the harness self-resumes across cert expiries). Verify afterwards with `du -sb` (not `stat`) and by loading one `era5_snapshot/*.npy` (`(20,161,257) float32` for europe) and one `ghcnh_snapshot/*.npz` (`t2m`, `wind`, `obs_count`).
   - `git add -A && git commit` inside `~/tessera-patch-encoder`, then rsync the repo (minus `venv/`, `outputs/`, `data/`) to `/data/weather-downscaling/tessera_patch_encoder/repo/`.
   - copy the Isambard checkout's `.gitignore` addition (`.tmp_output`) — irrelevant post-migration, just don't lose local edits.
2. **Commit the uncommitted monorepo work** on the current branch so nothing is lost when the tree is deleted later: `projects/dataprocessing/scripts/era5/{arco,weatherbench2}.py`, `src/dataprocessing/utils.py`, and `git rm` the three deleted notebooks. Push.
3. **Snapshot the untracked/ignored keep-set** somewhere durable (`/data/weather-downscaling/_migration/repo_snapshot_2026-08-17.tar` — `paper/`, `scripts/paper_figures/`, `scripts/extract_tessera_patches_local.py`, `scripts/maps/outputs/`, `MIGRATION_PLAN.md`).
4. Repoint the dangling `dataset_timestamp_aurora_lead{6,24,72}h/ghcnh_snapshot` symlinks to `../dataset_timestamp_global/ghcnh_snapshot` (relative) once step 1 lands.

### Phase 1 — New repository skeleton  ✅ done

Fresh `git init` (recommended name: `tessera-downscaling`; keep the *package* name `tessera_downscaling` so checkpoints, `config.json` keys and notebooks keep working). Files written from scratch:

- `pyproject.toml`: single uv project (no workspace). `requires-python >=3.12`; `[tool.uv.index] pytorch-cu126 explicit` + `torch = {index=…}` (carry over — this is how CUDA wheels resolve); dependencies materialised (the current project pyproject declares **none**): `torch==2.9.0`, `numpy==2.3.4`, `pandas`, `scipy`, `scikit-learn`, `matplotlib==3.10.7`, `cartopy==0.25.0`, `xarray==2025.9.0`, `netcdf4`, `h5netcdf`, `h5py`, `zarr==2.18.7`, `pyyaml`, `tqdm`, `rasterio`, `geotessera>=0.7`, `gcsfs`; optional groups `aurora = ["microsoft-aurora==1.8.0"]`, `gee = ["earthengine-api"]`, `dev = [pytest, pytest-cov, ruff==0.12.2, pre-commit, nbstripout]`. Drop `wandb`, `neuralprocesses`, `flask`, `frozendict`, azure. `[project.scripts]` entry points: `tessera-train`, `tessera-evaluate`, `tessera-baselines`, `tessera-figures`. `[tool.ruff]` copied verbatim (line-length 88, `select=["ALL"]`, the 19 ignores, `tests/`+`scripts/` per-file ignores; drop the auroraft block). `[tool.pytest.ini_options] testpaths=["tests"]`.
- `.python-version` (3.13), `.pre-commit-config.yaml` (same hooks; raise `check-added-large-files` or exclude `paper/figures/` and add `nbstripout`), `.gitignore` (**never** a bare `outputs/`; ignore `data/`, `runs/`, `.venv`, `wandb/`), `.github/workflows/{ci,pre-commit}.yml` (trimmed from the monorepo's `CI.yml`/`pre-commit.yml`), `Makefile` (`setup-env`, `test`, `format`, `figures`), **commit `uv.lock`**.
- `README.md` (start from `src/tessera_downscaling.egg-info/PKG-INFO`), `LICENSE` (monorepo is MIT — check with co-authors), `CITATION.cff`, `HISTORY.md` (`git log --format=… -- projects/tessera_downscaling` + a pointer to `cambridge-mlg/end-to-end-forecasting@68e54b0`), `DATA.md` (§4 below).

Target layout:

```
tessera-downscaling/
├── src/tessera_downscaling/
│   ├── paths.py                 # NEW: DATA_ROOT resolver + HPC→local remap (one place)
│   ├── data/{dataset,helpers,vae_latents,dense_grid_patches}.py
│   ├── model/{convcnp,heads,heads_dispatch}.py
│   ├── train.py, evaluate.py    # moved in from scripts/ → importable, no sys.path hacks
│   ├── baselines.py             # evaluate_simple_baselines.py (tests + residual probe import it)
│   ├── io_utils.py              # vendored dataprocessing/utils.py
│   └── patch_encoder/           # VAE from tessera-patch-encoder (model, losses, dataset, blocks)
├── scripts/
│   ├── data/          download_era5_{wb2,wb2_aurora_levels,arco}.py, download_ghcnh.py,
│   │                  fetch_station_{mtpi,extra_descriptors}.py, backfill_station_mtpi.py,
│   │                  build_extra_descriptors.py, build_summary_stats_latents.py,
│   │                  concat_station_vectors.py, shuffle_latents.py,
│   │                  extract_tessera_patches_local.py, shortlist_tessera_tiles.py, slurm/
│   ├── preprocessing/ preprocess_timestamp_global.py, helpers.py, preprocess_aurora.py,
│   │                  validate_aurora_datasets.py, slurm/
│   ├── aurora/        generate_aurora_forecasts.py, submit_aurora_forecasts.sh
│   ├── patch_encoder/ train_vae.py, eval_vae.py, encode_stations.py, encode_dense_grid.py
│   ├── experiments/   <19 folders>, build_rollout_schedule.py, pick_probe_set.py, README.md
│   ├── reeval_train_stations.sh, reeval_truncated_normal.sh
│   ├── maps/          regions.py, fetch_dem.py, extract_dense_grid_patches.py, generate_maps.py,
│   │                  station_eval.py, summary_table.py, residual_alignment_fulltest.py,
│   │                  plot_region_overview.py, select_dates.py, run_region_maps.sh
│   ├── analysis/      norway_descriptor_spaces.py, residual_probe_spaces.py
│   └── paper/         make_paper_figures.py, make_paper_tables.py (NEW), figure_region_overview.tex
├── paper/figures/     the 21 delivered PDFs (tracked; drop preview/ PNGs)
├── notebooks/         5 keep (+4 optional), _helpers.py, README.md, *_analysis_outputs/
└── tests/             test_convcnp.py, test_heads.py, test_lapse_rate_baseline.py (+ smoke tests)
```

### Phase 2 — Import commit (verbatim, reviewable)  ✅ done

Copy the keep-set **file by file** (an explicit `rsync --files-from` manifest, not a directory copy) from the monorepo checkout — including the untracked `paper/`, `scripts/paper_figures/`, `scripts/extract_tessera_patches_local.py` and the vendored `dataprocessing` files — into the new layout, and commit as `Import from cambridge-mlg/end-to-end-forecasting@68e54b0 (projects/tessera_downscaling)`. No edits in this commit, so every later prune/refactor is a clean diff. Keep the 5 paper notebooks' outputs in this commit (they are the record of the numbers as computed on Isambard); strip afterwards.

### Phase 3 — Decouple paths and packaging  ✅ done

1. `src/tessera_downscaling/paths.py`: `DATA_ROOT = Path(os.environ.get("TESSERA_DS_DATA_ROOT", "/data/weather-downscaling"))`, plus `MOUNT_ROOT` (`/tessera/v2/…`), `LANDMASK_DIR`, and `remap(path)` that rewrites the two Isambard prefixes (`/projects/u6do/…/.tmp_output`, `/lus/lfs1aip2/…/.tmp_output`) recorded inside `config.json`/checkpoints to `DATA_ROOT`. Replace: `train.py:1026`, `evaluate.py:199`, `make_paper_figures.py:45,886` (`DATA`, `HPC_TMP`, and `MAPS_OUT` → `DATA_ROOT/paper_figure_outputs/maps_outputs`), `notebooks/_helpers.py:50-59`, `scripts/maps/regions.py:24-27` (+ the other six `/lus/…` `REPO=` lines), `norway_descriptor_spaces.py:88`, `residual_probe_spaces.py:88`, `extract_tessera_patches_local.py:77-81`, `evaluate_simple_baselines.py`, every kept `submit.sh` header, both `reeval_*.sh`, `submit_aurora_forecasts.sh`, `submit_preprocess_*.sh`. (Full inventory: 31 files with `/projects/u6do/…`, 7 with `/lus/…`, ~73 with `.tmp_output`.)
2. Make the package installable (`uv sync`; `hatchling` build backend) and delete the ~30 `sys.path.insert` bootstraps; move `train.py`/`evaluate.py`/`evaluate_simple_baselines.py` into the package with console entry points; update `tests/test_lapse_rate_baseline.py` to import instead of loading by path.
3. Slurm-agnostic runner: factor the per-entry loop of `submit.sh` into `scripts/experiments/run_folder.py --folder … --seeds … [--sbatch|--local|--dry-run]` reading `experiments.yaml`; keep the YAML grammar unchanged so `notebooks/_helpers.py` still parses it.
4. `notebooks`: replace the hard-coded `BASE_DIR`/`RESULTS_ROOT` cells with `from tessera_downscaling.paths import DATA_ROOT`; `nbstripout` them; rely on `*_analysis_outputs/` for stored numbers.

### Phase 4 — Prune (one commit per area, tests green after each)  ✅ done

Order: (a) `wind_energy/`, daily/, dead scripts, dead experiment folders, dead notebooks; (b) `tessera_encoder.py` + `--tessera-method` plumbing + raw-patch serving in datasets; (c) heads (Weibull/B-G/Generative) + their train/eval blocks + tests; (d) convcnp mechanisms (embedding kernels, attention, stream, FiLM, projection); (e) datasets (daily, allowlist, global-norm; flat `snapshot_v1` if agreed); (f) `data/tessera.py` if agreed; (g) YAML pruning of the five regional folders; (h) `SUPPORTED_TARGET_VARIABLES`, `--target-variables` choices, docstrings (the `train.py` module docstring is entirely about the dead `--tessera-method` path). Keep `config.get(...)` tolerant reads in `evaluate.py` throughout.

### Phase 5 — Verify  ✅ all done (4–5 on the rescued dataset, 2026-08-20)

1. `uv run pytest` (4 → 3 files) green.
2. `tessera-figures --out /tmp/figs` regenerates all 21 PDFs from `/data` with **zero `[warn]` drift lines** (the script cross-checks Fig 2/5/6/8/11/12 numbers against stored expectations and `all_results_tidy.csv`).
3. New `make_paper_tables.py` reproduces Table 1 (2 d.p.), Table 4/C1 and Table 6/B1 (3 d.p.) from `test_summary.json` — diff against the PDF.
4. Evaluate one existing checkpoint per family with the pruned `evaluate.py` (`_tessera_1B-M_2017` t2m + wind, a rollout run with `region_specs_test.json`, a cross-lead run with `--lead-hours 24`) and diff `test_summary.json` against the on-disk originals — proves the tolerant config reads.
5. Smoke-train 1 epoch on `southern_africa` (smallest grid, 81×81) once `dataset_timestamp_global` is back; run `tessera-baselines --lapse-rate-mode fitted` on it.
6. `ruff`/`pre-commit` clean; CI green on GitHub.

### Phase 6 — Retire the monorepo copy  ❌ dropped (monorepo copy is kept)

After Phase 5 passes: PR on `end-to-end-forecasting` that `git rm -r projects/tessera_downscaling`, removes it from `pyproject.toml` (workspace members/sources, pytest/mypy/ruff paths — 7 places) and `Makefile:9`, moves/deletes `download_era5.sh`/`download_ghcnh.sh`, and leaves a one-paragraph `projects/tessera_downscaling/README.md` pointer to the new repo. Delete the untracked strays (`runs/`, `global_0.1_degree_tiff_all/`, `station_list.csv`, `tessera_tiles_shortlist.*`) from the local checkout.

---

## 4. Data plan for `/data/weather-downscaling` (9.26 TB today)

Document this as `DATA.md` in the new repo; the run directories carry **no** copy of `experiments.yaml`, so the mount is uninterpretable without the repo's YAMLs.

**Keep — reproduces every paper number (~115 GB + 264 GB + the rescued 176 GB):**
`training_runs_snapshot_14y_{eu,us,east_asia,southern_africa,australia}` (baseline subsets, 4.0 GB) · `…_tessera_1B-M_2017` ×5 (7.4 GB) · `…_2017_shuffled` ×5 (0.9 GB) · `…_cross_lead` (6.9 GB) + `…_cross_lead_tessera_1B-M_2017` (1.0 GB) · `…_eu_temporal_rollout_norway_lat16_mtpi` (24.2 GB) + `…_norway_tessera_1B-M_2017` (11.8 GB) · `dataset_timestamp_aurora_lead{6,24,72}h` (88 GB each) · **`dataset_timestamp_global` (rescue, ~176 GB)** · `processed/vae_tessera_1B-M/` (92 MB) · `processed/{extra_descriptors.npy,_names.json,_global_stats.npz,station_extra_descriptors.csv,station_mtpi.csv,station_summary_stats_1B-M_p128_2017_crop64_dim16*}` · `processed/tessera_global/station_list_filtered.csv` (row-alignment key) · `processed/dense/`, `processed/dem_cache/` (5.1 GB), `processed/overview_cache/` · `paper_figure_outputs/maps_outputs/` (97 MB) · `tessera_patch_encoder/outputs/vae/{p128_2017_*,lat16_beta0.0005_grad0.5_e200}` · `isambard_home/slurm_accounting/` (the 9,600 GPU-h figure) · `_staging/processed/era5_static/`.

**Keep only while rebuilding is possible (~2.5 TB):** `_staging/processed/era5_wb2_quarter_{12 paper vars}` (2.0 TB), `_staging/processed/ghcnh/data` (153 GB), `processed/tessera_station_patches/patch_embeddings_2017_p128.npy` (326 GB — only to re-encode latents), `processed/tessera_dense_grid/{norway,iberia}_0.05deg_2024/{patch_embeddings.npy,grid_points.csv}` (only to regenerate `processed/dense/`; and only if the maps stay v1), `dataset_timestamp/` (29 GB legacy EU flat — format reference until `_global` is back).

**Drop (~7.5 TB):** `_staging/raw/ghcnh/` (1.69 TB, 544k PSV, regenerable), `_staging/aurora/` (1.89 TB raw forecasts — europe/east_asia already distilled; 4 regions never used), `_staging/processed/ghcnh/data.legacy` (138 GB), `era5_wb2_quarter_{100m_u,100m_v,surface_pressure,boundary_layer_height}` (223 GB, wind-energy), `processed/alphaearth_station_patches` (326 GB), `processed/olmoearth_*` (153 GB), `processed/tessera_station_patches/patch_embeddings_2024_p128.npy` (326 GB), `processed/tessera_dense_grid/*/_tile_cache/` (>1 TB download cache), `training_runs_snapshot_14y_*_tessera_1B-M_2024` ×5 (5 GB; unless kept for provenance), the 11 empty `training_runs_*` dirs, `wind_energy/` (3.2 GB — belongs to the other project), `dataset_timestamp_aurora_lead6h_test`, `tessera_patch_encoder/outputs/{jepa,vae/lat32*,vae/lat64*}` (the `vae/{alphaearth,olmoearth}` runs are KEEP since 2026-08-20), `isambard_home/claude_projects/` (archive elsewhere), `_migration/{chunks,state}` (keep `jobs.tsv`).

**Two traps before deleting anything under `processed/`:**
- `processed/tessera_global` is a **symlink** to `tessera_global_v1_p64_2024_outdated/`; every run's `--tessera-path` points at its 81.5 GB `patch_embeddings_2024.npy` purely as the **station-validity filter** (centre-pixel non-zero ∧ coverage ≥ 0.5). Deleting it silently changes the station set. First freeze the mask (compare with `processed/overview_cache/patch_valid_2024.npz`, then write `processed/station_masks/tessera_v1_2024_patch_valid.npz`), add a `--tessera-valid-mask` option to the dataset, re-verify one `test_summary.json`, *then* drop the `.npy`.
- `dataset_timestamp_aurora_lead*h/ghcnh_snapshot` are dangling symlinks until Phase 0.4.

---

## 5. Decisions to make (my recommendation in brackets)

1. ~~Bilinear vs SetConv in the code~~ — decided: vanilla `RBFSetConv` stays. Still open: the paper's Eq. (1)/App. A.4/Table A5 wording (fix the text, or re-run with `--interpolation setconv` as an ablation).
2. Regenerate the dense maps with the 1B-M 2017 models, or disclose the v1 provenance [regenerate if `dataset_timestamp_global` + 2017 dense patches can be produced in time; otherwise disclose].
3. Document the VAE `auxon` auxiliary heads in App. A.3 [yes; check what `aux` supervises].
4. Fold the VAE encoder into this repo vs. keep `tessera-patch-encoder` as a sibling pinned by commit [fold in; one repo per paper].
5. Cut the flat `snapshot_v1` layout (`SnapshotDownscalingDataset`, `snapshot_6y_eu`, legacy `dataset_timestamp/`) [cut].
6. Cut `data/tessera.py` (geotessera download extractor) [cut; the local-mount extractor is byte-identical and documented].
7. Keep the `_tessera_1B-M_2024` folders/runs [drop configs, keep run dirs until submission].
8. Which optional notebooks survive (`uncertainty_estimation`, `stratified_performance`, `tessera_1bm_variant_shortlist`, `single_folder_analysis`) [keep the first three].
9. ~~History~~ — done: fresh repo at `/home/pmms2/tessera-downscaling`, verbatim import commit `0e6ea1c`.
10. ~~`wind_energy/`~~ — decided: stays in the monorepo for a later separate migration.

## 6. Gaps found (not blockers, but worth knowing)

- App. E (US-vs-EU orography displacement: 24 %/5 % > 200 m, ρ = 0.83/0.32) has no located producer.
- Fig 9 code was reconstructed on 2026-08-12 (`make_paper_figures.fig09`) and reproduces the paper's R²/hit/β̂/n; the original generating code was never found.
- `residual_probe_spaces.py` (Europe/t2m elev+mTPI = 0.75, Tessera = 0.30, n = 5,934) ≠ Fig 5 (0.65 / 0.19, n ≈ 898): different station set and RF params — label it as an independent check.
- `snapshot_14y_aurora_zeroshot` has a notebook and 26 tracked outputs but no runs on `/data`.
- Three duplicates to collapse: `interpret_plots.py` vs `summary_table.py` (Spearman), `generate_aurora_forecasts.py:284` vs `preprocessing/helpers.compute_grid_crop_indices`, `replot_paper_maps.FIGS` vs `make_paper_figures.PAPER_MAPS`.
- `evaluate_simple_baselines.py` and `residual_probe_spaces.py` share code via `sys.path` — becomes a normal import once baselines live in the package.
