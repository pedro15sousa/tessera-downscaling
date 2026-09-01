"""Shared helpers for the snapshot-experiment analysis notebooks.

Used by the notebooks in this directory: loading the per-run
``test_summary.json`` files of one or more experiment folders into a tidy
dataframe, printing the per-distribution summary tables, shortlisting and
baseline resolution, and the config-driven run classification behind the
centralised slice analysis of ``cross_folder_analysis.ipynb``.

Source of truth for "what experiments belong to a folder" is each folder's
``scripts/experiments/<folder>/experiments.yaml``. Source of truth for "what
runs exist on disk" is ``<data_root>/training_runs/<folder>/`` (see
``tessera_downscaling.paths``).
"""

from __future__ import annotations

import json
import re
import textwrap
from collections.abc import Iterable
from enum import Enum
from pathlib import Path
from typing import Literal

import pandas as pd
import yaml

from tessera_downscaling.paths import training_runs_dir

# -----------------------------------------------------------------------------
# Repo / path discovery
# -----------------------------------------------------------------------------


def find_repo_root(start: Path | None = None) -> Path:
    """Walk up from `start` (or cwd) until we find the repository root
    (the directory holding ``pyproject.toml`` and ``scripts/experiments``)."""
    p = (start or Path.cwd()).resolve()

    def _is_root(d: Path) -> bool:
        return (d / "pyproject.toml").exists() and (
            d / "scripts" / "experiments"
        ).is_dir()

    while not _is_root(p) and p != p.parent:
        p = p.parent
    if not _is_root(p):
        raise RuntimeError(
            f"Could not locate the repository root above {start or Path.cwd()}"
        )
    return p


def experiments_dir(repo_root: Path | None = None) -> Path:
    """Path to scripts/experiments/ (the experiment-folder definitions)."""
    repo_root = repo_root or find_repo_root()
    return repo_root / "scripts" / "experiments"


def output_dir_for_folder(folder: str, repo_root: Path | None = None) -> Path:
    """Map a folder name (e.g. 'snapshot_14y_us') to its training_runs/ dir
    under the data root. ``repo_root`` is accepted for signature compatibility
    and ignored: runs live under ``$TESSERA_DATA_ROOT``, not in the repo."""
    return training_runs_dir(folder)


def list_folders(repo_root: Path | None = None) -> list[str]:
    """List experiment folders that have an experiments.yaml file.

    Sorted alphabetically. Filters out any non-experiment dirs that
    might land under scripts/experiments/.
    """
    edir = experiments_dir(repo_root)
    return sorted(
        sub.name
        for sub in edir.iterdir()
        if sub.is_dir() and (sub / "experiments.yaml").exists()
    )


# -----------------------------------------------------------------------------
# YAML loading
# -----------------------------------------------------------------------------


def load_experiments_yaml(folder: str, repo_root: Path | None = None) -> list[dict]:
    """Load the YAML for a single folder; return the list of experiment entries.

    Two YAML formats are supported:

    * **Flat** (legacy): the YAML decodes to a list of experiment dicts,
      each carrying ``name``, ``label``, ``colour``, ``target_variables``,
      ``extra_args``. Multi-region YAMLs additionally have
      ``region_specs_train`` / ``region_specs_test``. Returned as-is.

    * **Data-efficiency nested**: the YAML decodes to a dict with top-
      level keys including ``architectures``, ``sweep_points``, and
      optionally ``simple_baselines``. This function then cross-products
      ``architectures × sweep_points`` to produce synthetic flat
      experiment entries with composite names ``{arch_name}_{sweep_label}``
      — matching the run-directory layout that the data-efficiency
      submit.sh scripts create — and appends the simple baselines.

    The notebooks consume the flat list returned by this function; the
    submit scripts continue to parse the raw YAML themselves so they
    can see the structured sweep info.
    """
    yaml_path = experiments_dir(repo_root) / folder / "experiments.yaml"
    with open(yaml_path) as f:
        raw = yaml.safe_load(f)
    if isinstance(raw, list):
        return raw
    if isinstance(raw, dict) and "architectures" in raw and "sweep_points" in raw:
        return _flatten_data_efficiency_yaml(raw)
    raise ValueError(
        f"Unrecognised experiments.yaml format at {yaml_path}: expected "
        f"either a list of experiments or a dict with "
        f"'architectures' + 'sweep_points' (data-efficiency format)."
    )


def _flatten_data_efficiency_yaml(spec: dict) -> list[dict]:
    """Cross-product architectures × sweep_points into flat experiment entries.

    Each synthetic entry preserves all the fields the notebooks rely on
    (``name``, ``label``, ``colour``, ``target_variables``, ``extra_args``)
    and adds two data-efficiency-specific ones (``family``, ``sweep_label``,
    ``sweep_k`` where applicable) that downstream curve-plotting code can
    use to bucket runs by axis position.

    Simple baselines (``spec['simple_baselines']``) are appended too —
    their run-directories live in the same output folder, but they don't
    cross-product with sweep_points.
    """
    out: list[dict] = []
    archs = spec.get("architectures", [])
    sweep_points = spec.get("sweep_points", [])
    # Stable family-coloured palette: TESSERA in blue tones, baselines
    # in grey, simple baselines in orange. Doesn't have to be precise —
    # downstream notebooks can override.
    _family_colour = {"tessera": "#1f77b4", "baseline": "#7f7f7f"}
    for arch in archs:
        family = arch.get("family", "tessera")
        base_colour = _family_colour.get(family, "#1f77b4")
        for sp in sweep_points:
            sweep_label = sp["label"]
            entry: dict = {
                "name": f"{arch['name']}_{sweep_label}",
                "label": f"{arch['label']} — {sweep_label}",
                "colour": arch.get("colour", base_colour),
                "target_variables": list(arch["target_variables"]),
                "extra_args": arch.get("extra_args", ""),
                "family": family,
                "arch_name": arch["name"],
                "sweep_label": sweep_label,
            }
            # Carry sweep-point fields onto the entry so curve plots
            # can read them off the dataframe directly. None for the
            # axis we don't have (e.g. active_from for station-count).
            if "active_from" in sp:
                entry["sweep_active_from"] = sp["active_from"]
            if "k_train" in sp:
                entry["sweep_k_train"] = sp["k_train"]
            out.append(entry)
    # Simple baselines — no sweep cross-product. The submit scripts
    # write these to run dirs named "{name}_seed42" (one fixed seed,
    # since the baseline is deterministic).
    for sb in spec.get("simple_baselines", []):
        out.append(
            {
                "name": sb["name"],
                "label": sb.get("label", sb["name"]),
                "colour": sb.get("colour", "#ff7f0e"),
                "target_variables": list(sb["target_variables"]),
                "extra_args": "",
                "family": "simple_baseline",
                "arch_name": sb["name"],
                "sweep_label": "all",
            }
        )
    return out


# -----------------------------------------------------------------------------
# Result loading
# -----------------------------------------------------------------------------

# Metric fields that get copied per-variable from test_summary.json.
#
# The evaluator emits a common core for every variable plus extras that
# depend on the variable's likelihood, so the metric list is split into:
#
#   - ``_COMMON_METRICS``: present for every variable, every distribution.
#     NLL and CRPS are proper scoring rules and are directly comparable
#     across distributions; n_predictions / n_test_stations / pit_chi2_*
#     are bookkeeping / calibration metrics that work for any predictive
#     distribution.
#
#   - ``_DISTRIBUTION_METRICS[dist]``: extras that only apply to a given
#     distribution. Gaussian keeps the mean-based point metrics;
#     truncated-normal point estimates are split by estimator (MAE on the
#     predictive median, RMSE/bias/correlation on the mean).
#
# The loader uses ``head_spec`` from ``test_results.json`` to dispatch.
# Older test_results.json files don't have ``head_spec`` — those are
# treated as all-Gaussian (the implicit legacy behaviour).

_COMMON_METRICS = [
    "n_predictions",
    "n_test_stations",
    "nll",
    "crps",
    "pit_chi2_stat",
    "pit_chi2_pvalue",
]

_DISTRIBUTION_METRICS = {
    "gaussian": [
        "mae",
        "rmse",
        "bias",
        "correlation",
        "mean_pred_std",
        "within_1sigma",
        "within_2sigma",
        "p50",
        "p90",
        "p95",
        "p99",
    ],
    "truncated_normal": [
        # Point estimates are split by estimator (MAE@median,
        # RMSE/bias/correlation @mean). Pre-rename eval files wrote the
        # Gaussian-style names (mae/rmse/bias/correlation, all mean-based);
        # _load_run_record aliases those to these names on load so old and
        # new result files share one schema.
        "mae_at_median",
        "rmse_at_mean",
        "bias_at_mean",
        "correlation_at_mean",
        "mean_pred_std",
        "within_1sigma",
        "within_2sigma",
        "p50",
        "p90",
        "p95",
        "p99",
    ],
}


def _metrics_for_variable(head_spec: dict | None, var: str) -> list[str]:
    """Return the metric field names to look up for ``var``.

    ``head_spec`` is the optional dict the evaluator writes into
    ``test_results.json``: ``{var: {"distribution": ..., "param_names": [...]}}``.
    Result files written before it was added have no such field — that case
    falls through to the Gaussian metric set, the implicit assumption those
    files were written under.
    """
    distribution = "gaussian"
    if head_spec is not None and var in head_spec:
        distribution = head_spec[var].get("distribution", "gaussian")
    extras = _DISTRIBUTION_METRICS.get(distribution, [])
    return _COMMON_METRICS + extras


_SEASONS = ["DJF", "MAM", "JJA", "SON"]


# -----------------------------------------------------------------------------
# Region area metadata (for station-density columns)
# -----------------------------------------------------------------------------
# Bbox lat/lon ranges, MUST stay in sync with REGIONS in
# scripts/preprocessing/preprocess_timestamp_global.py. Areas are
# computed lat-cosine-corrected so they're physically meaningful.
# Note: bbox area is a coarse upper bound — Australia, Southern Africa,
# and US bboxes contain large areas with no land or no observed stations,
# so density-per-bbox-area underestimates effective density. We expose
# both bbox area and a rough land-only area where it differs materially.

_REGION_BBOXES_DEG = {
    # name: (lat_min, lat_max, lon_min, lon_max)
    "europe": (35.0, 75.0, -24.0, 40.0),
    "us": (24.0, 50.0, -125.0, -66.0),
    "east_asia": (20.0, 46.0, 100.0, 146.0),
    "australia": (-44.0, -10.0, 112.0, 154.0),
    "southern_africa": (-35.0, -15.0, 15.0, 35.0),
}


def _bbox_area_km2(
    lat_min: float, lat_max: float, lon_min: float, lon_max: float
) -> float:
    """Approximate bbox area in km², lat-cosine corrected."""
    import math

    mean_lat = 0.5 * (lat_min + lat_max)
    cos_lat = math.cos(math.radians(mean_lat))
    lat_km = (lat_max - lat_min) * 111.0
    lon_km = (lon_max - lon_min) * 111.0 * cos_lat
    return abs(lat_km * lon_km)


REGION_BBOX_AREA_KM2 = {
    name: _bbox_area_km2(*bbox) for name, bbox in _REGION_BBOXES_DEG.items()
}


# Folder-name → region-name resolution. Folders for the per-region 14y
# experiments embed the region name in their dir; resolve back to the
# canonical key in REGION_BBOX_AREA_KM2.
#
# Region shorthands used in folder names — the datasets say "eu", the
# canonical region key is "europe".
_REGION_SYNONYMS = {"eu": "europe"}


def _folder_to_region(folder_name: str) -> str | None:
    """Resolve the canonical region for an experiment folder (or a bare region
    name), or None when it isn't a single-region folder.

    The region is the LEADING token of the name after its ``snapshot_<age>_``
    experiment prefix (``snapshot_14y_``, ``snapshot_6y_``,
    ``snapshot_global_14y_``, …), matched against the known region set.
    Matching the leading token — rather than testing for a region name as a
    bare substring anywhere — is what avoids false hits like ``us`` ⊂
    ``australia`` (which previously mislabelled the australia folder as
    ``us``) or ``us`` ⊂ ``museum``. Longest-first so multi-word regions
    (``east_asia``, ``southern_africa``) win over any shorter key, and
    compound folders (``…_eu_temporal_rollout_norway``) resolve on their
    leading ``eu`` → ``europe``. Non-region folders (aurora/cross_lead/global)
    match nothing and return None.
    """
    fn = re.sub(r"^snapshot_(?:global_)?\d+y_", "", folder_name.lower())
    candidates = sorted(
        list(REGION_BBOX_AREA_KM2) + list(_REGION_SYNONYMS),
        key=len,
        reverse=True,
    )
    for key in candidates:
        if fn == key or fn.startswith(key + "_"):
            return _REGION_SYNONYMS.get(key, key)
    return None


def _load_run_record(
    run_dir: Path,
    exp_name: str,
    exp_label: str,
    exp_colour: str,
    variables: list[str],
    seed: int,
    source_folder: str,
) -> dict | None:
    """Load metrics from a single run directory; return None if no results."""
    # Prefer test_summary.json (newer); fall back to test_results.json.
    # Skip empty / corrupt JSON files — usually the result of a job being
    # killed mid-write. We log a warning so the user knows to clean up,
    # but don't crash the whole load over one bad file.
    result = None
    for fname in ("test_summary.json", "test_results.json"):
        fpath = run_dir / fname
        if not fpath.exists():
            continue
        try:
            with open(fpath) as f:
                result = json.load(f)
            break
        except (json.JSONDecodeError, ValueError) as e:
            print(f"  [warn] skipping unparseable {fpath} ({type(e).__name__}: {e})")
            continue
    if result is None:
        return None

    config = {}
    # config.json normally sits alongside test_summary.json in run_dir. For a
    # globally-trained model the per-region eval summaries live in
    # {run_dir}/eval_{region}/ while config.json stays in the parent run dir,
    # so fall back to the parent when the eval dir has no config of its own.
    for config_path in (run_dir / "config.json", run_dir.parent / "config.json"):
        if config_path.exists():
            try:
                with open(config_path) as f:
                    config = json.load(f)
            except (json.JSONDecodeError, ValueError):
                # Config corruption is non-fatal — record stays usable
                # without it (just no config-derived columns).
                pass
            break

    record = {
        "experiment": exp_name,
        "label": exp_label,
        "colour": exp_colour,
        "seed": seed,
        "run_dir": str(run_dir),
        "source_folder": source_folder,
        "variables": variables,
        "include_elevation": config.get("include_elevation", True),
        "best_epoch": result.get("checkpoint_epoch", result.get("best_epoch")),
        "best_val_loss": result.get("best_val_loss"),
        # head_spec is written by the evaluator; older result files won't
        # have it. Kept on the record so downstream filters / shortlist
        # logic can dispatch on the per-variable distribution.
        "head_spec": result.get("head_spec"),
        # Full config dict, consumed by the centralised slice-analysis
        # helpers below for config-driven classification + pairing.
        "config": config,
    }
    head_spec = result.get("head_spec") or {}

    # Backwards-compat: truncated-normal eval files written before the
    # point-estimate rename used the Gaussian-style keys (<var>_mae /
    # _rmse / _bias / _correlation), all computed on the predictive *mean*.
    # The current evaluator writes <var>_mae_at_median / _rmse_at_mean /
    # _bias_at_mean / _correlation_at_mean (MAE on the median, the rest on
    # the mean), matching the Weibull convention. Alias the old keys to the
    # new names when a stale file is loaded so the rest of the pipeline
    # sees a single schema. RMSE/bias/correlation were already mean-based,
    # so those aliases are exact; the MAE alias is the mean-based value
    # standing in for the (smaller) median MAE — re-run evaluate.py on the
    # checkpoint to get the proper median MAE. Only the alias is applied
    # here; a re-run overwrites all four with true values.
    _TRUNC_NORMAL_ALIAS = {
        "mae": "mae_at_median",
        "rmse": "rmse_at_mean",
        "bias": "bias_at_mean",
        "correlation": "correlation_at_mean",
    }
    for var in variables:
        if head_spec.get(var, {}).get("distribution") != "truncated_normal":
            continue
        for old, new in _TRUNC_NORMAL_ALIAS.items():
            old_key, new_key = f"{var}_{old}", f"{var}_{new}"
            if new_key not in result and old_key in result:
                result[new_key] = result[old_key]

    for var in variables:
        dist = head_spec.get(var, {}).get("distribution", "gaussian")
        record[f"{var}_distribution"] = dist
        for metric in _metrics_for_variable(head_spec or None, var):
            record[f"{var}_{metric}"] = result.get(f"{var}_{metric}")
        # Canonical point-estimate columns, resolved per-row from the row's
        # own distribution: the MAE/RMSE-equivalent regardless of likelihood.
        # No recomputation — this only *selects* the value evaluate.py already
        # stored (mean-based for Gaussian, median-based for Weibull / trunc-N,
        # wet-median for Bernoulli-Gamma):
        #   Gaussian          -> <var>_mae              / <var>_rmse
        #   Weibull / trunc-N -> <var>_mae_at_median     / <var>_rmse_at_mean
        #   Bernoulli-Gamma   -> <var>_wet_mae_at_median / <var>_wet_rmse_at_mean
        # Plotting/aggregation code can then compare point accuracy across rows
        # of different heads (e.g. a Gaussian ERA5-interp baseline vs a
        # truncated-normal model) by reading one column name, instead of
        # hardcoding "<var>_mae" (which silently goes NaN for skewed heads).
        record[f"{var}_point_mae"] = result.get(
            f"{var}_{_point_estimate_mae_column(dist)}"
        )
        record[f"{var}_point_rmse"] = result.get(
            f"{var}_{_point_estimate_rmse_column(dist)}"
        )
        seasonal = result.get(f"{var}_seasonal_mae", {}) or {}
        for season in _SEASONS:
            record[f"{var}_mae_{season}"] = seasonal.get(season)
        record[f"{var}_learned_sigma"] = result.get(f"{var}_learned_sigma")
        # Data-efficiency per-subset breakdown (emitted by evaluate.py only
        # when the run had a probe_active_from_file configured at training
        # time). Older runs / non-data-efficiency runs leave these keys
        # absent — the .get() calls return None and the dataframe column
        # ends up NaN, which is the right semantic for "not measured".
        for subset in ("probe", "always_on", "spatial_test", "unmapped"):
            for sub_metric in (
                "mae",
                "rmse",
                "bias",
                "mae_macro",
                "rmse_macro",
                "bias_macro",
                "n_stations",
                "n_predictions",
            ):
                key = f"{var}_{subset}_{sub_metric}"
                record[key] = result.get(key)

    return record


def load_folder_results(
    folder: str,
    seeds: Iterable[int] = (42, 123, 456),
    repo_root: Path | None = None,
) -> pd.DataFrame:
    """Load all completed runs for a single folder into a dataframe.

    Reads the YAML for the canonical experiment list and labels, then
    looks for {output_dir}/{name}_seed{seed}/test_summary.json. Skips
    any (name, seed) without a result file — useful while the sweep
    is still in progress.

    The returned dataframe has a `source_folder` column (always equal
    to `folder` here), kept for compatibility with the cross-folder
    loader so downstream code can use the same shape regardless.
    """
    repo_root = repo_root or find_repo_root()
    output_dir = output_dir_for_folder(folder, repo_root)
    if not output_dir.exists():
        return pd.DataFrame()

    records = []
    for entry in load_experiments_yaml(folder, repo_root):
        # Single-variable runs only: every record carries exactly one
        # target variable, which the summary/pairing code relies on.
        if len(entry["target_variables"]) > 1:
            continue
        for seed in seeds:
            run_dir = output_dir / f"{entry['name']}_seed{seed}"
            if not run_dir.exists():
                continue
            rec = _load_run_record(
                run_dir=run_dir,
                exp_name=entry["name"],
                exp_label=entry["label"],
                exp_colour=entry["colour"],
                variables=list(entry["target_variables"]),
                seed=seed,
                source_folder=folder,
            )
            if rec is not None:
                records.append(rec)
    return pd.DataFrame(records)


def load_all_results(
    folders: Iterable[str] | None = None,
    seeds: Iterable[int] = (42, 123, 456),
    repo_root: Path | None = None,
) -> pd.DataFrame:
    """Load runs from multiple folders, concatenated into one dataframe.

    Each row carries `source_folder`. If `folders` is None, loads every
    folder found under scripts/experiments/.
    """
    repo_root = repo_root or find_repo_root()
    if folders is None:
        folders = list_folders(repo_root)
    frames = [load_folder_results(f, seeds=seeds, repo_root=repo_root) for f in folders]
    frames = [f for f in frames if not f.empty]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _agg_metric(rows: pd.DataFrame, col: str) -> tuple[float, float, int]:
    """(mean, std, n) over seed rows for a metric column; NaN-safe."""
    if rows is None or rows.empty or col not in rows.columns:
        return (float("nan"), float("nan"), 0)
    vals = pd.to_numeric(rows[col], errors="coerce").dropna()
    if vals.empty:
        return (float("nan"), float("nan"), 0)
    return (float(vals.mean()), float(vals.std(ddof=0)), int(vals.shape[0]))


# -----------------------------------------------------------------------------
# Summary tables (extracted verbatim shape from the original notebook)
# -----------------------------------------------------------------------------

# Shared formatting helpers for the distribution-specific table writers
# below. Pulled out so each ``_print_<dist>_table`` doesn't redefine them.


def _fmt_mean_std(mean, std, width, prec=3):
    if mean != mean:  # NaN
        return f"{'N/A':>{width}}"
    return f"{f'{mean:.{prec}f}±{std:.{prec}f}':>{width}}"


def _fmt_single(val, width, prec=2):
    if val != val:  # NaN
        return f"{'N/A':>{width}}"
    return f"{val:>{width}.{prec}f}"


def _fmt_count(val, width):
    """Compact thousands/millions formatter for prediction counts."""
    if val != val:
        return f"{'N/A':>{width}}"
    v = int(val)
    if v >= 1_000_000:
        s = f"{v / 1_000_000:.2f}M"
    elif v >= 1_000:
        s = f"{v / 1_000:.1f}K"
    else:
        s = str(v)
    return f"{s:>{width}}"


def _resolve_region_and_density(df: pd.DataFrame, region: str | None):
    """Resolve canonical region name + bbox area for a sub-dataframe.

    Caller-provided region wins; otherwise inferred from ``source_folder``
    when all rows agree. Returns ``(resolved_region, bbox_area_km2_or_None)``.
    """
    resolved_region = region
    if resolved_region is None and "source_folder" in df.columns:
        folders = df["source_folder"].dropna().unique()
        if len(folders) == 1:
            resolved_region = _folder_to_region(folders[0])
    bbox_area_km2 = (
        REGION_BBOX_AREA_KM2.get(resolved_region) if resolved_region else None
    )
    return resolved_region, bbox_area_km2


def _print_title(
    variable: str,
    distribution: str,
    title_suffix: str,
    resolved_region: str | None,
    bbox_area_km2: float | None,
    total_w: int,
) -> None:
    """Print the boxed title block above a per-distribution table."""
    title = f"ConvCNP RESULTS — {variable}"
    if distribution != "gaussian":
        title += f" [{distribution}]"
    if title_suffix:
        title += f"   [{title_suffix}]"
    if resolved_region:
        title += f"   ({resolved_region}"
        if bbox_area_km2:
            title += f", bbox≈{bbox_area_km2 / 1e6:.2f}M km²"
        title += ")"
    print(f"\n{'=' * total_w}")
    print(title)
    print(f"{'=' * total_w}")


def _station_count_and_density(
    edata: pd.DataFrame, n_stations_col: str, bbox_area_km2: float | None
) -> tuple[str, str]:
    """Return (stations_str, density_str) for the rightmost two columns."""
    if n_stations_col in edata.columns:
        ns_val = edata[n_stations_col].iloc[0]
    else:
        ns_val = float("nan")
    if ns_val != ns_val:  # NaN
        return "—", "—"
    ns_int = int(ns_val)
    if bbox_area_km2 and bbox_area_km2 > 0:
        density_str = f"{ns_int / (bbox_area_km2 / 1e6):.2f}"
    else:
        density_str = "—"
    return f"{ns_int}", density_str


def _print_gaussian_table(
    df: pd.DataFrame,
    variable: str,
    title_suffix: str,
    experiments: Iterable[str] | None,
    region: str | None,
    distribution: str = "gaussian",
) -> None:
    """Gaussian-likelihood table: legacy layout + CRPS.

    Columns: Experiment | MAE | RMSE | NLL | CRPS | Epoch | Pred σ | 1σ% |
             2σ% | Seeds | N | Stations | st/Mkm²

    CRPS is a _COMMON_METRIC emitted by evaluate.py for every head (computed
    unconditionally alongside NLL), so it's already present in the loaded
    dataframe for Gaussian rows — this column just surfaces it, making
    Gaussian rows directly CRPS-comparable to the trunc-normal table.
    """
    mae_col = f"{variable}_mae"
    rmse_col = f"{variable}_rmse"
    nll_col = f"{variable}_nll"
    crps_col = f"{variable}_crps"
    std_col = f"{variable}_mean_pred_std"
    w1_col = f"{variable}_within_1sigma"
    w2_col = f"{variable}_within_2sigma"
    n_col = f"{variable}_n_predictions"
    n_stations_col = f"{variable}_n_test_stations"

    if mae_col not in df.columns or not df[mae_col].notna().any():
        print(f"No results for {variable} [gaussian]")
        return
    vdf = df[df[mae_col].notna()].copy()
    resolved_region, bbox_area_km2 = _resolve_region_and_density(vdf, region)

    LABEL_W, NUM_W, EPOCH_W, STD_W, PCT_W = 60, 13, 8, 7, 7
    SEEDS_W, N_W, STA_W, DEN_W = 6, 9, 8, 9
    # 4 numeric columns now (MAE, RMSE, NLL, CRPS) + Epoch + σ + 1σ/2σ.
    total_w = (
        LABEL_W
        + 1
        + 4 * (NUM_W + 1)
        + EPOCH_W
        + 1
        + STD_W
        + 1
        + PCT_W
        + 1
        + PCT_W
        + 1
        + SEEDS_W
        + 1
        + N_W
        + 1
        + STA_W
        + 1
        + DEN_W
    )
    _print_title(
        variable, distribution, title_suffix, resolved_region, bbox_area_km2, total_w
    )
    header = (
        f"{'Experiment':<{LABEL_W}} "
        f"{'MAE':>{NUM_W}} "
        f"{'RMSE':>{NUM_W}} "
        f"{'NLL(test)':>{NUM_W}} "
        f"{'CRPS':>{NUM_W}} "
        f"{'Epoch':>{EPOCH_W}} "
        f"{'Pred σ':>{STD_W}} "
        f"{'1σ%':>{PCT_W}} "
        f"{'2σ%':>{PCT_W}} "
        f"{'Seeds':>{SEEDS_W}} "
        f"{'N':>{N_W}} "
        f"{'Stations':>{STA_W}} "
        f"{'st/Mkm²':>{DEN_W}}"
    )
    print(header)
    print("-" * total_w)

    for exp_name in vdf["experiment"].unique() if experiments is None else experiments:
        edata = vdf[vdf["experiment"] == exp_name]
        if edata.empty:
            continue
        n = len(edata)

        mae_mean = edata[mae_col].mean()
        mae_std = edata[mae_col].std() if n > 1 else 0.0
        rmse_mean = edata[rmse_col].mean()
        rmse_std = edata[rmse_col].std() if n > 1 else 0.0
        nll_mean = edata[nll_col].mean() if nll_col in edata.columns else float("nan")
        nll_std = edata[nll_col].std() if (nll_col in edata.columns and n > 1) else 0.0
        crps_mean = (
            edata[crps_col].mean() if crps_col in edata.columns else float("nan")
        )
        crps_std = (
            edata[crps_col].std() if (crps_col in edata.columns and n > 1) else 0.0
        )
        epoch_mean = edata["best_epoch"].mean()
        epoch_std = edata["best_epoch"].std() if n > 1 else 0.0
        pred_std = edata[std_col].mean() if std_col in edata.columns else float("nan")
        w1 = edata[w1_col].mean() if w1_col in edata.columns else float("nan")
        w2 = edata[w2_col].mean() if w2_col in edata.columns else float("nan")
        n_pred = edata[n_col].iloc[0] if n_col in edata.columns else float("nan")
        ns_str, density_str = _station_count_and_density(
            edata,
            n_stations_col,
            bbox_area_km2,
        )

        label = edata["label"].iloc[0]
        wrapped = textwrap.wrap(label, width=LABEL_W) or [""]
        first_label = wrapped[0]
        continuation_labels = wrapped[1:]

        line = (
            f"{first_label:<{LABEL_W}} "
            f"{_fmt_mean_std(mae_mean, mae_std, NUM_W):>{NUM_W}} "
            f"{_fmt_mean_std(rmse_mean, rmse_std, NUM_W):>{NUM_W}} "
            f"{_fmt_mean_std(nll_mean, nll_std, NUM_W):>{NUM_W}} "
            f"{_fmt_mean_std(crps_mean, crps_std, NUM_W):>{NUM_W}} "
            f"{_fmt_single(epoch_mean, EPOCH_W - 3, 0)}±"
            f"{_fmt_single(epoch_std, 2, 0).strip()} "
            f"{_fmt_single(pred_std, STD_W):>{STD_W}} "
            f"{_fmt_single(w1, PCT_W, prec=1):>{PCT_W}} "
            f"{_fmt_single(w2, PCT_W, prec=1):>{PCT_W}} "
            f"{n:>{SEEDS_W}} "
            f"{_fmt_count(n_pred, N_W)} "
            f"{ns_str:>{STA_W}} "
            f"{density_str:>{DEN_W}}"
        )
        print(line)
        for cont in continuation_labels:
            print(f"{cont:<{LABEL_W}}")


def _print_truncated_normal_table(
    df: pd.DataFrame,
    variable: str,
    title_suffix: str,
    experiments: Iterable[str] | None,
    region: str | None,
    distribution: str = "truncated_normal",
) -> None:
    """Truncated-Normal table.

    Columns: Experiment | MAE@med | RMSE@mean | NLL | CRPS | Epoch |
             Pred σ | 1σ% | 2σ% | Seeds | N | Stations | st/Mkm²

    Like the Gaussian table — TruncNormal keeps a σ parameter, so the
    σ-coverage diagnostics still apply — but the point estimates are split
    by estimator: MAE on the median, RMSE on the mean (the optimal
    estimators for a skewed distribution). NLL/CRPS are the proper scoring
    rules comparable across distributions, so a wind-TruncNormal row is
    directly comparable to a wind-Gaussian row. Old (pre-rename) result
    files are handled by the alias in ``_load_run_record``, so this reads
    one schema regardless of when the file was written.
    """
    mae_col = f"{variable}_mae_at_median"
    rmse_col = f"{variable}_rmse_at_mean"
    nll_col = f"{variable}_nll"
    crps_col = f"{variable}_crps"
    std_col = f"{variable}_mean_pred_std"
    w1_col = f"{variable}_within_1sigma"
    w2_col = f"{variable}_within_2sigma"
    n_col = f"{variable}_n_predictions"
    n_stations_col = f"{variable}_n_test_stations"

    if mae_col not in df.columns or not df[mae_col].notna().any():
        print(f"No results for {variable} [truncated_normal]")
        return
    vdf = df[df[mae_col].notna()].copy()
    resolved_region, bbox_area_km2 = _resolve_region_and_density(vdf, region)

    LABEL_W, NUM_W, EPOCH_W, STD_W, PCT_W = 60, 13, 8, 7, 7
    SEEDS_W, N_W, STA_W, DEN_W = 6, 9, 8, 9
    # 4 numeric columns (MAE@med, RMSE@mean, NLL, CRPS) + Epoch + σ + 1σ/2σ.
    total_w = (
        LABEL_W
        + 1
        + 4 * (NUM_W + 1)
        + EPOCH_W
        + 1
        + STD_W
        + 1
        + PCT_W
        + 1
        + PCT_W
        + 1
        + SEEDS_W
        + 1
        + N_W
        + 1
        + STA_W
        + 1
        + DEN_W
    )
    _print_title(
        variable, distribution, title_suffix, resolved_region, bbox_area_km2, total_w
    )
    header = (
        f"{'Experiment':<{LABEL_W}} "
        f"{'MAE@med':>{NUM_W}} "
        f"{'RMSE@mean':>{NUM_W}} "
        f"{'NLL(test)':>{NUM_W}} "
        f"{'CRPS':>{NUM_W}} "
        f"{'Epoch':>{EPOCH_W}} "
        f"{'Pred σ':>{STD_W}} "
        f"{'1σ%':>{PCT_W}} "
        f"{'2σ%':>{PCT_W}} "
        f"{'Seeds':>{SEEDS_W}} "
        f"{'N':>{N_W}} "
        f"{'Stations':>{STA_W}} "
        f"{'st/Mkm²':>{DEN_W}}"
    )
    print(header)
    print("-" * total_w)

    for exp_name in vdf["experiment"].unique() if experiments is None else experiments:
        edata = vdf[vdf["experiment"] == exp_name]
        if edata.empty:
            continue
        n = len(edata)

        def col_mean_std(col, edata=edata, n=n):
            if col not in edata.columns:
                return float("nan"), 0.0
            return edata[col].mean(), (edata[col].std() if n > 1 else 0.0)

        def col_mean(col, edata=edata):
            return edata[col].mean() if col in edata.columns else float("nan")

        mae_mean, mae_std = col_mean_std(mae_col)
        rmse_mean, rmse_std = col_mean_std(rmse_col)
        nll_mean, nll_std = col_mean_std(nll_col)
        crps_mean, crps_std = col_mean_std(crps_col)
        epoch_mean = edata["best_epoch"].mean()
        epoch_std = edata["best_epoch"].std() if n > 1 else 0.0
        pred_std = col_mean(std_col)
        w1 = col_mean(w1_col)
        w2 = col_mean(w2_col)
        n_pred = edata[n_col].iloc[0] if n_col in edata.columns else float("nan")
        ns_str, density_str = _station_count_and_density(
            edata,
            n_stations_col,
            bbox_area_km2,
        )

        label = edata["label"].iloc[0]
        wrapped = textwrap.wrap(label, width=LABEL_W) or [""]
        first_label = wrapped[0]
        continuation_labels = wrapped[1:]

        line = (
            f"{first_label:<{LABEL_W}} "
            f"{_fmt_mean_std(mae_mean, mae_std, NUM_W):>{NUM_W}} "
            f"{_fmt_mean_std(rmse_mean, rmse_std, NUM_W):>{NUM_W}} "
            f"{_fmt_mean_std(nll_mean, nll_std, NUM_W):>{NUM_W}} "
            f"{_fmt_mean_std(crps_mean, crps_std, NUM_W):>{NUM_W}} "
            f"{_fmt_single(epoch_mean, EPOCH_W - 3, 0)}±"
            f"{_fmt_single(epoch_std, 2, 0).strip()} "
            f"{_fmt_single(pred_std, STD_W):>{STD_W}} "
            f"{_fmt_single(w1, PCT_W, prec=1):>{PCT_W}} "
            f"{_fmt_single(w2, PCT_W, prec=1):>{PCT_W}} "
            f"{n:>{SEEDS_W}} "
            f"{_fmt_count(n_pred, N_W)} "
            f"{ns_str:>{STA_W}} "
            f"{density_str:>{DEN_W}}"
        )
        print(line)
        for cont in continuation_labels:
            print(f"{cont:<{LABEL_W}}")


_DISTRIBUTION_TABLE_PRINTERS = {
    "gaussian": _print_gaussian_table,
    "truncated_normal": _print_truncated_normal_table,
}


def _infer_distribution(df: pd.DataFrame, variable: str) -> str:
    """Auto-detect distribution from a dataframe slice.

    If every row in ``df`` has the same value in ``<variable>_distribution``,
    return that. Otherwise default to "gaussian" (the legacy assumption).
    Dataframes loaded from older result files have no ``<var>_distribution``
    column at all — those also fall back to "gaussian".
    """
    dist_col = f"{variable}_distribution"
    if dist_col not in df.columns:
        return "gaussian"
    vals = df[dist_col].dropna().unique()
    if len(vals) == 1:
        return str(vals[0])
    return "gaussian"


def print_summary(
    df: pd.DataFrame,
    variable: str,
    title_suffix: str = "",
    experiments: Iterable[str] | None = None,
    region: str | None = None,
    distribution: str | None = None,
) -> None:
    """Print a formatted summary table for a single target variable.

    The table layout depends on the variable's likelihood:

      * Gaussian: MAE/RMSE/NLL/CRPS + Pred σ + 1σ/2σ coverage.
      * Truncated-Normal: MAE@median/RMSE@mean + NLL/CRPS + the same
        σ-coverage diagnostics.

    If ``distribution`` is None, auto-detected from ``<variable>_distribution``
    in ``df`` (matches the evaluator's ``head_spec``). Rows from older
    result files default to ``"gaussian"`` — so legacy callers like
    ``print_summary(df, "t2m")`` are unchanged.

    If ``experiments`` is given, only those experiment names are printed,
    in the order given. ``region`` overrides folder-based density inference.
    """
    if distribution is None:
        distribution = _infer_distribution(df, variable)
    printer = _DISTRIBUTION_TABLE_PRINTERS.get(distribution)
    if printer is None:
        # Unknown distribution: fall back to Gaussian. Defensive guard
        # against future-likelihood rows showing up in legacy notebooks.
        printer = _print_gaussian_table
    printer(df, variable, title_suffix, experiments, region, distribution)


# -----------------------------------------------------------------------------
# Shortlisting
# -----------------------------------------------------------------------------

# Heuristic: a run is "TESSERA-enhanced" if its name contains any of these.
# Used to filter out baselines from shortlists. ``film`` names runs of the
# superseded FiLM-injection arm still present on disk; new runs use one of
# the other tokens.
_TESSERA_NAME_TOKENS = ("tessera", "vae", "film", "concat")


def is_tessera(experiment_name: str) -> bool:
    return any(t in experiment_name.lower() for t in _TESSERA_NAME_TOKENS)


def _point_estimate_mae_column(distribution: str) -> str:
    """Return the metric field name that's the "MAE-equivalent" for a
    given likelihood.

    For Gaussian: ``mae`` (predictive mean = median, so this is just MAE
    on the point estimate).
    For Truncated-Normal: ``mae_at_median`` (the median minimises expected
    MAE for a skewed distribution).
    """
    return {
        "gaussian": "mae",
        "truncated_normal": "mae_at_median",
    }.get(distribution, "mae")


def _point_estimate_rmse_column(distribution: str) -> str:
    """Return the metric field name that's the "RMSE-equivalent" for a
    given likelihood.

    For Gaussian: ``rmse`` (point estimate is the mean; RMSE on mean).
    For Truncated-Normal: ``rmse_at_mean`` (the mean minimises expected
    MSE for any distribution).
    """
    return {
        "gaussian": "rmse",
        "truncated_normal": "rmse_at_mean",
    }.get(distribution, "rmse")


def shortlist_experiments(
    df: pd.DataFrame,
    variable: str,
    top_n: int = 5,
    tessera_only: bool = True,
    print_table: bool = True,
) -> list[str]:
    """Rank experiments by composite (MAE, MAE-stability) score; return top names.

    The MAE column is picked per-distribution: ``<variable>_mae`` for
    Gaussian, ``<variable>_mae_at_median`` for Truncated-Normal. The
    dataframe must therefore carry a ``<variable>_distribution`` column
    (added by ``_load_run_record`` from ``test_results.json``'s
    ``head_spec`` field; rows from older files default to ``"gaussian"``).

    A run-level shortlist mixes runs of different distributions only if
    you explicitly evaluated the same variable under several likelihoods
    in one folder; in that case the MAE values from different
    distributions ARE comparable (they're all on the same target
    observations) but reflect different point estimators. NLL and CRPS
    are the metrics to use if you want a strictly proper-scoring-rule
    ranking.

    Returns a list of experiment names in rank order.
    """
    dist_col = f"{variable}_distribution"
    nll_col = f"{variable}_nll"

    # Pick the per-row MAE column based on the distribution. Default to
    # Gaussian's "mae" for legacy rows that don't carry a distribution.
    if dist_col in df.columns:
        # For each row, pull the right MAE-equivalent value.
        def _pick_mae(row):
            d = row[dist_col] if pd.notna(row[dist_col]) else "gaussian"
            col = f"{variable}_{_point_estimate_mae_column(d)}"
            return row.get(col)

        mae_series = df.apply(_pick_mae, axis=1)
    else:
        # No distribution column (older result files) — only Gaussian.
        mae_col = f"{variable}_mae"
        if mae_col not in df.columns or not df[mae_col].notna().any():
            if print_table:
                print(f"No results for {variable}")
            return []
        mae_series = df[mae_col]

    if mae_series.notna().sum() == 0:
        if print_table:
            print(f"No results for {variable}")
        return []
    df = df.copy()
    df["_pe_mae"] = mae_series

    rows = []
    for exp_name in df[df["_pe_mae"].notna()]["experiment"].unique():
        if tessera_only and not is_tessera(exp_name):
            continue
        edata = df[(df["experiment"] == exp_name) & df["_pe_mae"].notna()]
        if len(edata) < 2:
            continue
        rows.append(
            {
                "experiment": exp_name,
                "label": edata["label"].iloc[0],
                "mae_mean": edata["_pe_mae"].mean(),
                "mae_std": edata["_pe_mae"].std(),
                "nll_mean": edata[nll_col].mean()
                if nll_col in edata.columns
                else float("nan"),
                "best_epoch": edata["best_epoch"].mean()
                if "best_epoch" in edata.columns
                else float("nan"),
                "n_seeds": len(edata),
            }
        )

    if not rows:
        return []

    sdf = pd.DataFrame(rows)
    sdf["mae_rank"] = sdf["mae_mean"].rank()
    sdf["std_rank"] = sdf["mae_std"].rank()
    sdf["composite_rank"] = sdf["mae_rank"] + 0.3 * sdf["std_rank"]
    sdf = sdf.sort_values("composite_rank")

    unit = {"t2m": "°C", "wind": "m/s"}.get(variable, "")

    if print_table:
        max_label = max(max(len(r["label"]) for r in rows) + 2, 60)
        print(f"\n{'=' * (max_label + 50)}")
        print(f"{variable} — Shortlisted TESSERA Experiments (top {top_n})")
        print(f"{'=' * (max_label + 50)}")
        print(
            f"  {'Experiment':<{max_label}} {'MAE (' + unit + ')':>16}  "
            f"{'NLL':>8}  {'Epoch':>6}  {'Seeds':>6}"
        )
        print(f"  {'-' * (max_label + 46)}")

    shortlisted: list[str] = []
    for _, row in sdf.head(top_n).iterrows():
        if print_table:
            mae_str = f"{row['mae_mean']:.3f}±{row['mae_std']:.3f}"
            nll_str = (
                f"{row['nll_mean']:.3f}"
                if row["nll_mean"] == row["nll_mean"]
                else "N/A"
            )
            ep_str = (
                f"{row['best_epoch']:.0f}"
                if row["best_epoch"] == row["best_epoch"]
                else "?"
            )
            print(
                f"  {row['label']:<{max_label}} {mae_str:>16}  "
                f"{nll_str:>8}  {ep_str:>6}  {row['n_seeds']:>6}"
            )
        shortlisted.append(row["experiment"])

    return shortlisted


# -----------------------------------------------------------------------------
# Baseline resolution
# -----------------------------------------------------------------------------

# Per-variable, per-distribution baseline-name resolution, matching the
# baseline entries of the regional ``experiments.yaml`` files. Simple
# baselines (ERA5 interp / persistence) don't write ``head_spec``, so the
# loader defaults their ``<var>_distribution`` to "gaussian"; listing them
# under every family they should accompany lets the non-Gaussian tables
# pull them back in by name. Trained ConvCNP baselines are listed under
# the likelihood head they were trained with.
BASELINE_NAMES = {
    "t2m": {
        "gaussian": [
            "t2m_snap_era5_interp_baseline",  # no model — bilinear ERA5 interp
            "t2m_snap_era5_interp_lapse_baseline",  # + fixed lapse-rate correction
            "t2m_snap_era5_interp_lapse_fitted_baseline",  # + fitted lapse rate
            "t2m_snap_persistence_baseline",  # no model — last valid station obs
            "t2m_snap_bilinear_baseline_mtpi_wd",  # trained ConvCNP, no TESSERA
            "t2m_snap_bilinear_baseline_mtpi_extradesc_wd",  # + hand-crafted surface descriptors
        ],
    },
    "wind": {
        "gaussian": [
            "wind_snap_era5_interp_baseline",
            "wind_snap_persistence_baseline",
        ],
        "truncated_normal": [
            "wind_snap_era5_interp_baseline",  # simple references (loaded as Gaussian rows)
            "wind_snap_persistence_baseline",
            "wind_truncnormal_snap_bilinear_baseline_mtpi_wd",  # trained ConvCNP, no TESSERA
            "wind_truncnormal_snap_bilinear_baseline_mtpi_extradesc_wd",  # + hand-crafted surface descriptors
        ],
    },
}


def baselines_for(variable: str, distribution: str = "gaussian") -> list[str]:
    """Resolve baseline experiment names for a (variable, distribution) pair.

    Returns ``[]`` if nothing is registered — keeps cross-folder iteration
    safe when a (variable, distribution) cell legitimately has no baselines.
    """
    entry = BASELINE_NAMES.get(variable)
    if entry is None:
        return []
    return list(entry.get(distribution, []))


# ============================================================================
# Centralised slice analysis (config-driven classification + pairing tables)
# ============================================================================
# A "slice" is a single (folder, variable, distribution, task) cell. For each
# single-task slice we print six tables, for each multi-task slice three:
#
#   1. Simple baselines (uncapped)                              — both tasks
#   2. Trained baselines, no TESSERA (uncapped)                 — both tasks
#   3. Top-N vanilla TESSERA                                    — both tasks
#   4. Shuffled-control pairing for the top-N                   — single only
#   5. Stats-encoder counterpart pairing for the top-N          — single only
#   6. Kernel-swap pairing for the top-N                        — single only
#
# Classification is config.json-driven via the ``config`` field on each row
# (added by ``_load_run_record``). Legacy / corrupted runs fall through to
# ``UNKNOWN`` and surface in a per-slice warnings block at the end.
#
# Entry points:
#   ``print_slice_analysis(df, folder, variable, distribution, task)``
#   ``print_centralised_analysis(df, folders, target_pairs, top_n=5)``  # top-level


class RunCategory(str, Enum):
    """Mutually-exclusive structural identity of a run."""

    SIMPLE_BASELINE = "simple_baseline"
    TRAINED_NO_TESSERA = "trained_no_tessera"
    TESSERA_VANILLA = "tessera_vanilla"
    UNKNOWN = "unknown"


# Name suffixes for the no-model reference runs. These don't go through
# train.py, so they may not write a full config.json — name detection is
# the primary classifier for them.
_SIMPLE_BASELINE_NAME_SUFFIXES = (
    "_era5_interp_baseline",
    "_persistence_baseline",
)


def _config_of(record: dict) -> dict:
    cfg = record.get("config")
    return cfg if isinstance(cfg, dict) else {}


def is_simple_baseline(record: dict) -> bool:
    """True for ERA5-interp / persistence reference rows (no trained model)."""
    name = record.get("experiment", "") or ""
    return any(name.endswith(s) for s in _SIMPLE_BASELINE_NAME_SUFFIXES)


def is_tessera_using(record: dict) -> bool:
    """True if the run conditions on a precomputed per-station vector
    (the VAE latents; the hand-crafted-descriptor-only baselines do not
    set ``vae_latents_path`` and stay in the no-TESSERA category)."""
    return bool(_config_of(record).get("vae_latents_path"))


def is_shuffled(record: dict) -> bool:
    """True iff the VAE latents file is a shuffled-control sibling.

    Detection: ``_shuffle_seed`` substring in vae_latents_path — the
    convention used by scripts/data/shuffle_latents.py for its output
    filenames.
    """
    path = _config_of(record).get("vae_latents_path") or ""
    return "_shuffle_seed" in path


def classify_run(record: dict) -> RunCategory:
    """Mutually-exclusive category. Control status is orthogonal — a shuffled
    TESSERA-vanilla run is classified as TESSERA_VANILLA and tagged
    is_shuffled=True separately.
    """
    if is_simple_baseline(record):
        return RunCategory.SIMPLE_BASELINE
    cfg = _config_of(record)
    if not cfg:
        return RunCategory.UNKNOWN
    if not is_tessera_using(record):
        return RunCategory.TRAINED_NO_TESSERA
    return RunCategory.TESSERA_VANILLA


def kernel_of(record: dict) -> str:
    """Interpolation kernel: 'setconv' | 'bilinear' | 'unknown'."""
    return _config_of(record).get("interpolation", "unknown") or "unknown"


def injection_of(record: dict) -> str:
    """Embedding injection mode recorded in the config: 'concat' | 'none'."""
    if not is_tessera_using(record):
        return "none"
    return _config_of(record).get("tessera_injection", "concat") or "concat"


_LAT_FROM_PATH = re.compile(r"_lat(\d+)")


def _latent_dim_from_path(path: str | None) -> int | None:
    if not path:
        return None
    m = _LAT_FROM_PATH.search(path)
    return int(m.group(1)) if m else None


def latent_setting_of(record: dict) -> str:
    """Compact label: 'none' | 'vae_lat16' | 'vae_lat32' etc."""
    if not is_tessera_using(record):
        return "none"
    lat = _latent_dim_from_path(_config_of(record).get("vae_latents_path"))
    return f"vae_lat{lat}" if lat is not None else "vae_latUNK"


def latent_encoder_of(record: dict) -> str:
    """Identify which static-feature encoder produced the latents:
    'vae'   — VAE-encoder latents (path contains 'station_latents_lat')
    'stats' — patch summary statistics (path contains 'summary_stats')
    'none'  — runs without TESSERA latents.
    """
    if not is_tessera_using(record):
        return "none"
    path = _config_of(record).get("vae_latents_path") or ""
    if "summary_stats" in path:
        return "stats"
    return "vae"


# ----------------------------------------------------------------------------
# Pairing
# ----------------------------------------------------------------------------
# Two records are siblings along a dimension iff their pairing_key matches
# when that dimension is excluded. The same generic mechanism handles all
# three: shuffled (latents-path normalised by stripping the shuffle marker),
# stats (latents-path canonicalised to its dim token), and kernel (skip
# interpolation).

# Config fields that define a run's architectural / regularisation identity
# for pairing, with the default a config lacking the key implies.
_PAIRING_AXIS_FIELDS: dict[str, object] = {
    "interpolation": "bilinear",
    "tessera_injection": "concat",
    "vae_latents_path": None,
    "include_elevation": True,
    "include_static_fields": True,
    "weight_decay": 0.0,
}

_SHUFFLE_MARKER_RE = re.compile(r"_shuffle_seed\d+")

# Canonicalise either a VAE-latents path or a stats path to its latent-
# dimension token only. Used by the 'stats' pairing dimension so that
# a 'station_latents_lat16_…' path and a 'station_summary_stats_dim16…'
# path produce the same pairing key when all other axes agree. Matches
# the first '_lat<N>' or '_dim<N>' substring; falls back to the raw
# path if neither marker is present.
_DIM_TOKEN_RE = re.compile(r"_(?:lat|dim)(\d+)")


def _path_to_dim_token(path: str | None) -> str:
    if not path:
        return ""
    m = _DIM_TOKEN_RE.search(path)
    return f"dim{m.group(1)}" if m else path


def _normalise_for_dimension(value, field: str, dimension: str):
    if dimension == "shuffled" and field == "vae_latents_path":
        return _SHUFFLE_MARKER_RE.sub("", value or "")
    if dimension == "stats" and field == "vae_latents_path":
        # Encoder-agnostic dim token so VAE and stats paths match when
        # all other axis fields agree.
        return _path_to_dim_token(value or "")
    return value


def pairing_key(record: dict, dimension: str) -> tuple:
    """Tuple of axis values minus the pair-dimension. Always includes the
    sorted variable list and likelihood so a single-task wind run does not
    match a multi-task one."""
    cfg = _config_of(record)
    skip_fields = {
        "shuffled": set(),
        "kernel": {"interpolation"},
        "stats": set(),
    }[dimension]

    parts: list = []
    for f, default in _PAIRING_AXIS_FIELDS.items():
        if f in skip_fields:
            continue
        parts.append(_normalise_for_dimension(cfg.get(f, default), f, dimension))

    variables = record.get("variables") or []
    parts.append(tuple(sorted(variables)))
    parts.append(cfg.get("likelihood"))
    return tuple(parts)


# ----------------------------------------------------------------------------
# Dataframe augmentation
# ----------------------------------------------------------------------------


def _augment_df_with_classification(df: pd.DataFrame) -> pd.DataFrame:
    """Attach per-row classification columns. Idempotent — calling on an
    already-augmented df returns it unchanged.

    Adds: category, kernel, injection, latent_setting, latent_encoder,
    is_shuffled, n_target_variables.
    """
    if df.empty or "category" in df.columns:
        return df
    df = df.copy()
    records = df.to_dict("records")
    df["category"] = [classify_run(r).value for r in records]
    df["kernel"] = [kernel_of(r) for r in records]
    df["injection"] = [injection_of(r) for r in records]
    df["latent_setting"] = [latent_setting_of(r) for r in records]
    df["latent_encoder"] = [latent_encoder_of(r) for r in records]
    df["is_shuffled"] = [is_shuffled(r) for r in records]
    df["n_target_variables"] = df["variables"].apply(
        lambda vs: len(vs) if isinstance(vs, list | tuple) else 1
    )
    return df


# ----------------------------------------------------------------------------
# Top-N selection
# ----------------------------------------------------------------------------


def _top_n_experiments(
    df: pd.DataFrame,
    variable: str,
    distribution: str,
    category: RunCategory,
    task: str,
    top_n: int,
    encoder: str | None = None,
) -> list[str]:
    """Top-N experiment names by per-seed-averaged MAE within
    (variable, distribution, category, task). Excludes shuffled-control rows.

    If ``encoder`` is provided ('vae', 'stats', or 'none'), additionally
    restricts to runs with that ``latent_encoder``. Used to keep the
    vanilla TESSERA leaderboard clean: VAE-only in the top-N table so
    stats variants don't displace VAE rows when they outperform, and
    stats results surface in the stats-counterpart pairing table instead
    (where the signed Δ already exposes 'stats beats VAE' cases).
    """
    mae_col = f"{variable}_{_point_estimate_mae_column(distribution)}"
    if mae_col not in df.columns:
        return []
    sub = df[
        (df["category"] == category.value)
        & (~df["is_shuffled"])
        & (df[mae_col].notna())
    ]
    if encoder is not None:
        sub = sub[sub["latent_encoder"] == encoder]
    if task == "single":
        sub = sub[sub["n_target_variables"] == 1]
    else:
        sub = sub[sub["n_target_variables"] > 1]
    if sub.empty:
        return []
    agg = sub.groupby("experiment", as_index=False)[mae_col].mean()
    agg = agg.sort_values(mae_col)
    return agg["experiment"].head(top_n).tolist()


# ----------------------------------------------------------------------------
# Pairing resolution + printing
# ----------------------------------------------------------------------------


def _seed_aggregate(rows: pd.DataFrame, mae_col: str) -> tuple[float, float, int]:
    if mae_col not in rows.columns or rows.empty:
        return (float("nan"), float("nan"), 0)
    vals = pd.to_numeric(rows[mae_col], errors="coerce").dropna()
    if vals.empty:
        return (float("nan"), float("nan"), 0)
    return (
        float(vals.mean()),
        float(vals.std(ddof=0)) if len(vals) > 1 else 0.0,
        int(len(vals)),
    )


def _resolve_pairing_for_top(
    df: pd.DataFrame,
    top_experiments: list[str],
    dimension: str,
    variable: str,
    distribution: str,
) -> list[dict]:
    """For each top experiment name, find its sibling row(s) along ``dimension``.

    Sibling selection rules:
      - shuffled: same pairing_key (after path normalisation) AND is_shuffled
      - stats:    same pairing_key AND opposite latent encoder, not shuffled
      - kernel:   same pairing_key AND opposite kernel, not shuffled
    """
    mae_col = f"{variable}_{_point_estimate_mae_column(distribution)}"
    rows = []
    for exp in top_experiments:
        live_rows = df[df["experiment"] == exp]
        if live_rows.empty:
            continue
        live_first = live_rows.iloc[0].to_dict()
        target_key = pairing_key(live_first, dimension)
        live_kernel = live_first.get("kernel", "unknown")
        live_mae, live_std, live_n = _seed_aggregate(live_rows, mae_col)

        # Candidate sibling pool: every other experiment in the slice df
        # with a matching pairing_key.
        candidates = df[df["experiment"] != exp]
        if dimension == "shuffled":
            candidates = candidates[candidates["is_shuffled"]]
        elif dimension == "kernel":
            candidates = candidates[
                (~candidates["is_shuffled"]) & (candidates["kernel"] != live_kernel)
            ]
        elif dimension == "stats":
            # Pair vanilla VAE ↔ vanilla stats (not shuffled).
            live_encoder = live_first.get("latent_encoder", "vae")
            opposite_encoder = "stats" if live_encoder == "vae" else "vae"
            candidates = candidates[
                (~candidates["is_shuffled"])
                & (candidates["latent_encoder"] == opposite_encoder)
            ]

        if candidates.empty:
            rows.append(
                {
                    "live_label": live_first.get("label", exp),
                    "live_kernel": live_kernel,
                    "live_mae": live_mae,
                    "live_std": live_std,
                    "live_n": live_n,
                    "sibling_label": None,
                    "sibling_kernel": None,
                    "sibling_mae": float("nan"),
                    "sibling_std": float("nan"),
                    "sibling_n": 0,
                    "status": "pending",
                }
            )
            continue

        matched_exp = None
        for cand_exp in candidates["experiment"].drop_duplicates():
            cand_first = candidates[candidates["experiment"] == cand_exp].iloc[0]
            if pairing_key(cand_first.to_dict(), dimension) == target_key:
                matched_exp = cand_exp
                break

        if matched_exp is None:
            rows.append(
                {
                    "live_label": live_first.get("label", exp),
                    "live_kernel": live_kernel,
                    "live_mae": live_mae,
                    "live_std": live_std,
                    "live_n": live_n,
                    "sibling_label": None,
                    "sibling_kernel": None,
                    "sibling_mae": float("nan"),
                    "sibling_std": float("nan"),
                    "sibling_n": 0,
                    "status": "pending",
                }
            )
            continue

        sib_rows = candidates[candidates["experiment"] == matched_exp]
        sib_first = sib_rows.iloc[0].to_dict()
        sib_mae, sib_std, sib_n = _seed_aggregate(sib_rows, mae_col)
        rows.append(
            {
                "live_label": live_first.get("label", exp),
                "live_kernel": live_kernel,
                "live_mae": live_mae,
                "live_std": live_std,
                "live_n": live_n,
                "sibling_label": sib_first.get("label", matched_exp),
                "sibling_kernel": sib_first.get("kernel", "unknown"),
                "sibling_mae": sib_mae,
                "sibling_std": sib_std,
                "sibling_n": sib_n,
                "status": "available",
            }
        )
    return rows


_PAIRING_KIND_LABELS = {
    "shuffled": "shuffled-latents control",
    "kernel": "kernel-swap comparison",
    "stats": "stats-encoder counterpart",
}


def _print_pairing_table(
    rows: list[dict],
    pairing_kind: str,
    title_suffix: str,
    variable: str,
    distribution: str,
) -> None:
    """Pairing-coverage table.

    Two rendering modes:
      - Compact: all siblings are pending. Prints a single ``N pending …``
        summary plus the live experiments + their MAE. No alignment grid.
      - Full: at least one sibling is available. Prints a column-aligned
        table with Role / Experiment / MAE / Seeds / Kernel headers; for
        each pair the Live row, Sibling row, and a Δ summary line are
        emitted in that order. Long labels wrap onto continuation lines.
    """
    if not rows:
        return
    kind_label = _PAIRING_KIND_LABELS.get(pairing_kind, pairing_kind)
    header_text = (
        f"Pairing — {kind_label}   [{variable} ({distribution}) — {title_suffix}]"
    )

    # ---- Compact mode: all pending --------------------------------------
    all_pending = all(r["status"] == "pending" for r in rows)
    if all_pending:
        bar_w = max(80, len(header_text) + 2)
        print()
        print("=" * bar_w)
        print(header_text)
        print("=" * bar_w)
        print(
            f"All {len(rows)} top-N variants pending sibling runs "
            f"(no matching {pairing_kind} sibling found in dataframe):"
        )
        for i, r in enumerate(rows, 1):
            if r["live_n"] > 0:
                mae_str = f"{r['live_mae']:.3f}±{r['live_std']:.3f}"
            else:
                mae_str = "N/A"
            print(f"  [#{i}]  {r['live_label']}")
            print(
                f"        MAE = {mae_str}  "
                f"(seeds={r['live_n']}, kernel={r['live_kernel']})"
            )
        return

    # ---- Full mode: at least one sibling is available -------------------
    N_W = 5  # "[#N]"
    ROLE_W = 7  # "Live" / "Sibling"
    LABEL_W = 80  # Experiment label (wraps if longer)
    NUM_W = 13  # MAE column
    SEEDS_W = 5
    KERN_W = 8
    total_w = N_W + 1 + ROLE_W + 1 + LABEL_W + 1 + NUM_W + 1 + SEEDS_W + 1 + KERN_W
    bar_w = max(total_w, len(header_text) + 2)

    print()
    print("=" * bar_w)
    print(header_text)
    print("=" * bar_w)
    print(
        f"{'':<{N_W}} {'Role':<{ROLE_W}} {'Experiment':<{LABEL_W}} "
        f"{'MAE':>{NUM_W}} {'Seeds':>{SEEDS_W}} {'Kernel':>{KERN_W}}"
    )
    print("-" * bar_w)

    def _emit_data_row(n_label, role, label, mae_str, seeds_str, kernel_str):
        """Print one data row, wrapping the label onto continuation lines.
        First line carries the full set of columns; continuations only the
        label."""
        wrapped = textwrap.wrap(label, width=LABEL_W) or [""]
        for j, line in enumerate(wrapped):
            if j == 0:
                print(
                    f"{n_label:<{N_W}} {role:<{ROLE_W}} {line:<{LABEL_W}} "
                    f"{mae_str:>{NUM_W}} {seeds_str:>{SEEDS_W}} "
                    f"{kernel_str:>{KERN_W}}"
                )
            else:
                print(f"{'':<{N_W}} {'':<{ROLE_W}} {line:<{LABEL_W}}")

    for i, r in enumerate(rows, 1):
        # Live row
        if r["live_n"] > 0:
            live_mae = f"{r['live_mae']:.3f}±{r['live_std']:.3f}"
            live_seeds = str(r["live_n"])
        else:
            live_mae = "N/A"
            live_seeds = "—"
        _emit_data_row(
            f"[#{i}]",
            "Live",
            r["live_label"] or "",
            live_mae,
            live_seeds,
            r["live_kernel"],
        )

        # Sibling row
        if r["status"] == "available":
            sib_mae = f"{r['sibling_mae']:.3f}±{r['sibling_std']:.3f}"
            _emit_data_row(
                "",
                "Sibling",
                r["sibling_label"] or "",
                sib_mae,
                str(r["sibling_n"]),
                r["sibling_kernel"],
            )
            delta = r["sibling_mae"] - r["live_mae"]
            print(
                f"{'':<{N_W}} {'':<{ROLE_W}} "
                f"{'Δ (sibling − live)':<{LABEL_W}} "
                f"{f'{delta:+.3f}':>{NUM_W}}"
            )
        else:
            _emit_data_row(
                "",
                "Sibling",
                "(pending — no matching sibling in dataframe)",
                "—",
                "—",
                "—",
            )

        if i < len(rows):
            print()


# ----------------------------------------------------------------------------
# Per-slice entry point
# ----------------------------------------------------------------------------


def _print_unknown_warnings(
    slice_df: pd.DataFrame,
    title_suffix: str,
) -> None:
    unk = slice_df[slice_df["category"] == RunCategory.UNKNOWN.value]
    if unk.empty:
        return
    print()
    print(f"--- WARNING: {len(unk)} unclassified rows in [{title_suffix}] ---")
    for _, row in unk.drop_duplicates(subset=["experiment"]).iterrows():
        run_dir = row.get("run_dir", "N/A")
        print(f"  {row['experiment']}   (run_dir={run_dir})")


def print_slice_analysis(
    df: pd.DataFrame,
    folder: str,
    variable: str,
    distribution: str = "gaussian",
    task: Literal["single", "multi"] = "single",
    top_n: int = 5,
) -> None:
    """Print the 6 (single-task) or 3 (multi-task) tables for one slice."""
    if df.empty:
        return
    df = _augment_df_with_classification(df)

    sub = df[df["source_folder"] == folder]
    if sub.empty:
        return

    # Distribution filter. Pull baselines back in by name even if their
    # head_spec is missing (legacy / simple baselines).
    dist_col = f"{variable}_distribution"
    if dist_col in sub.columns:
        fam = sub[sub[dist_col] == distribution]
    else:
        if distribution != "gaussian":
            return
        fam = sub
    baseline_names = baselines_for(variable, distribution)
    extras = sub[sub["experiment"].isin(baseline_names)]
    slice_df = pd.concat(
        [extras, fam[~fam["experiment"].isin(baseline_names)]], ignore_index=True
    ).drop_duplicates(subset=["experiment", "seed"])
    if slice_df.empty:
        return
    slice_df = _augment_df_with_classification(slice_df)

    # Task filter applied at each table individually.
    if task == "single":
        task_mask = slice_df["n_target_variables"] == 1
    else:
        task_mask = slice_df["n_target_variables"] > 1
    slice_df_task = slice_df[task_mask]

    task_label = "single-task" if task == "single" else "multi-task"
    title_suffix_base = f"{folder} — {task_label}"

    # Table 1: simple baselines (ERA5 interp / persistence).
    simple = slice_df_task[
        slice_df_task["category"] == RunCategory.SIMPLE_BASELINE.value
    ]
    if not simple.empty:
        print_summary(
            slice_df,
            variable,
            distribution=distribution,
            title_suffix=f"{title_suffix_base} — simple baselines",
            experiments=simple["experiment"].drop_duplicates().tolist(),
        )

    # Table 2: trained baselines, no TESSERA.
    trained_nt = slice_df_task[
        slice_df_task["category"] == RunCategory.TRAINED_NO_TESSERA.value
    ]
    if not trained_nt.empty:
        print_summary(
            slice_df,
            variable,
            distribution=distribution,
            title_suffix=f"{title_suffix_base} — trained baselines (no TESSERA)",
            experiments=trained_nt["experiment"].drop_duplicates().tolist(),
        )

    # Table 3: vanilla TESSERA top-N.
    vanilla_top = _top_n_experiments(
        slice_df,
        variable,
        distribution,
        RunCategory.TESSERA_VANILLA,
        task,
        top_n,
        encoder="vae",  # stats variants surface in the stats-counterpart pairing instead
    )
    if vanilla_top:
        print_summary(
            slice_df,
            variable,
            distribution=distribution,
            title_suffix=f"{title_suffix_base} — top {top_n} vanilla TESSERA (VAE)",
            experiments=vanilla_top,
        )

    # Multi-task stops here (the pairing tables are single-task only).
    if task == "multi":
        _print_unknown_warnings(slice_df_task, title_suffix_base)
        return

    # Table 4: shuffled-control pairing for the top-N.
    shuffled_rows = _resolve_pairing_for_top(
        slice_df,
        vanilla_top,
        "shuffled",
        variable,
        distribution,
    )
    _print_pairing_table(
        shuffled_rows,
        "shuffled",
        title_suffix_base,
        variable,
        distribution,
    )

    # Table 5: stats-encoder counterpart pairing for the top-N.
    stats_rows = _resolve_pairing_for_top(
        slice_df,
        vanilla_top,
        "stats",
        variable,
        distribution,
    )
    _print_pairing_table(stats_rows, "stats", title_suffix_base, variable, distribution)

    # Table 6: kernel-swap pairing for the top-N.
    kernel_rows = _resolve_pairing_for_top(
        slice_df,
        vanilla_top,
        "kernel",
        variable,
        distribution,
    )
    _print_pairing_table(
        kernel_rows,
        "kernel",
        title_suffix_base,
        variable,
        distribution,
    )

    _print_unknown_warnings(slice_df_task, title_suffix_base)


# ----------------------------------------------------------------------------
# Top-level driver
# ----------------------------------------------------------------------------


def print_centralised_analysis(
    df: pd.DataFrame,
    folders: list[str],
    target_pairs: list[tuple[str, str]],
    top_n: int = 5,
) -> None:
    """For each folder in ``folders`` and each (variable, distribution) in
    ``target_pairs``, print single-task slice analysis, then multi-task.

    ``df`` is the dataframe returned by ``load_all_results`` / ``load_folder_results``.
    Augments once at the top for efficiency; downstream slices reuse the
    classification columns.
    """
    if df.empty:
        return
    df = _augment_df_with_classification(df)

    for folder in folders:
        sub_folder = df[df["source_folder"] == folder]
        if sub_folder.empty:
            continue
        print()
        print("#" * 80)
        print(f"#  {folder}")
        print("#" * 80)

        # Single-task per (variable, distribution).
        for variable, distribution in target_pairs:
            print_slice_analysis(
                df,
                folder,
                variable,
                distribution,
                task="single",
                top_n=top_n,
            )

        # Multi-task per (variable, distribution): only tables 1-3 each.
        any_multi = (
            sub_folder["variables"]
            .apply(lambda vs: isinstance(vs, list | tuple) and len(vs) > 1)
            .any()
        )
        if any_multi:
            print()
            print("-" * 80)
            print(f"-- Multi-task results for {folder}")
            print("-" * 80)
            for variable, distribution in target_pairs:
                print_slice_analysis(
                    df,
                    folder,
                    variable,
                    distribution,
                    task="multi",
                    top_n=top_n,
                )
