"""Regenerate the paper's numeric tables from the stored run outputs.

Produces the per-region skill tables of "Earth observation embeddings are
effective sub-grid descriptors for probabilistic weather downscaling" from the
``test_summary.json`` written by ``tessera-evaluate`` for every run under the
data root (see ``tessera_downscaling.paths``), and cross-checks them against
the numbers printed in the paper.

Provenance. These tables previously existed only as notebook cells:
``notebooks/cross_folder_analysis.ipynb`` cells 14-18 (Table 1 = cell 15
``CELLS_MTPI``; the shuffled-latent control of Table 4 = cell 16; the
train-station table = cell 18) and ``notebooks/extra_descriptors_stratified.ipynb``
(the hand-crafted-descriptor arms of Table 6). This script replicates their
aggregation exactly, without importing them:

* one value per (region, arm, variable, metric) = the mean over the three
  training seeds (42, 123, 456) of the metric stored in each run's
  ``test_summary.json``; the "All regions" block is the unweighted mean of the
  five per-region means;
* MAE/RMSE are the head-aware point estimates: for the Gaussian t2m head the
  predictive mean (``t2m_mae`` / ``t2m_rmse``), for the truncated-normal wind
  head the median for MAE and the mean for RMSE (``wind_mae_at_median`` /
  ``wind_rmse_at_mean``; older files use the Gaussian-style keys, which are read
  as a fallback). CRPS is ``<var>_crps``. The persistence and ERA5
  interpolation references are point forecasts and have no CRPS;
* persistence is scored only where a valid preceding observation exists and
  the ERA5 reference for t2m includes the fitted lapse-rate correction -- both
  already reflected in the stored summaries.

Tables (numbering of the preprint; the AMS submission renumbers B1 = Table 6
and C1 = Table 4):

* Table 1 -- main results: persistence, ERA5 interpolation, ConvCNP
  (topography-only), ConvCNP (hand-crafted surface descriptors), ConvCNP with
  Tessera; 2 d.p. Key ``1p`` is the preprint variant without the hand-crafted
  row.
* Table 4 -- what in the Tessera patch matters: no descriptor, shuffled VAE
  descriptor, patch summary statistics, VAE descriptor; 3 d.p.
* Table 6 -- Tessera vs. / plus hand-crafted surface descriptors; 3 d.p.
* ``train`` -- (optional) the same models scored on the *training* stations at
  the held-out years (``<run>/eval_train_stations/test_summary.json``); rows
  whose re-evaluation is absent from the data root are left blank.

Run from the repository root::

    uv run python scripts/paper/make_paper_tables.py [--tables 1,4,6]
        [--format md|tex|csv|all] [--out paper/tables] [--data-root DIR]
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from tessera_downscaling import paths
from tessera_downscaling.paths import training_runs_dir

if TYPE_CHECKING:
    from collections.abc import Callable

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DEFAULT = REPO_ROOT / "paper" / "tables"

SEEDS = (42, 123, 456)
VARIABLES = ("t2m", "wind")
METRICS = ("mae", "rmse", "crps")
COLUMNS = [(v, m) for v in VARIABLES for m in METRICS]
VARIABLE_TITLES_TEX = {
    "t2m": r"2 m temperature ($^\circ$C)",
    "wind": r"10 m wind speed (m\,s$^{-1}$)",
}

# (display name, experiment-folder stem), in table order.
REGIONS = (
    ("Europe", "snapshot_14y_eu"),
    ("United States", "snapshot_14y_us"),
    ("East Asia", "snapshot_14y_east_asia"),
    ("Southern Africa", "snapshot_14y_southern_africa"),
    ("Australia", "snapshot_14y_australia"),
)
ALL_REGIONS = "All regions"

# Runs with Tessera 1B-M (2017) latents live in a sibling folder family.
TESSERA_SUFFIX = "_tessera_1B-M_2017"
SHUFFLED_SUFFIX = "_tessera_1B-M_2017_shuffled"

# Head-aware point-estimate keys per predictive distribution: (MAE, RMSE).
POINT_KEYS = {
    "gaussian": ("mae", "rmse"),
    "truncated_normal": ("mae_at_median", "rmse_at_mean"),
}


@dataclass(frozen=True)
class RunSpec:
    """One model arm: run-name stem per variable and the folder family it lives in."""

    t2m: str  # ``<stem>_seed<N>`` on disk
    wind: str
    folder_suffix: str = ""  # appended to the region folder stem

    def run(self, var: str) -> str:
        return {"t2m": self.t2m, "wind": self.wind}[var]


ARMS: dict[str, RunSpec] = {
    "persistence": RunSpec(
        t2m="t2m_snap_persistence_baseline",
        wind="wind_snap_persistence_baseline",
    ),
    "era5_interp": RunSpec(
        t2m="t2m_snap_era5_interp_lapse_fitted_baseline",
        wind="wind_snap_era5_interp_baseline",
    ),
    "convcnp": RunSpec(
        t2m="t2m_snap_bilinear_baseline_mtpi_wd",
        wind="wind_truncnormal_snap_bilinear_baseline_mtpi_wd",
    ),
    "convcnp_hand": RunSpec(
        t2m="t2m_snap_bilinear_baseline_mtpi_extradesc_wd",
        wind="wind_truncnormal_snap_bilinear_baseline_mtpi_extradesc_wd",
    ),
    "tessera": RunSpec(
        t2m="t2m_snap_vae_crop64_lat16_auxon_concat_mtpi",
        wind="wind_truncnormal_snap_vae_crop64_lat16_auxon_concat_mtpi",
        folder_suffix=TESSERA_SUFFIX,
    ),
    "tessera_hand": RunSpec(
        t2m="t2m_snap_vae_crop64_lat16_auxon_extradesc_concat_mtpi",
        wind="wind_truncnormal_snap_vae_crop64_lat16_auxon_extradesc_concat_mtpi",
        folder_suffix=TESSERA_SUFFIX,
    ),
    "tessera_stats": RunSpec(
        t2m="t2m_snap_stats16_crop64_concat_mtpi",
        wind="wind_truncnormal_snap_stats16_crop64_concat_mtpi",
        folder_suffix=TESSERA_SUFFIX,
    ),
    "tessera_shuffled": RunSpec(
        t2m="t2m_snap_vae_crop64_lat16_auxon_shuffled_concat_mtpi",
        wind="wind_truncnormal_snap_vae_crop64_lat16_auxon_shuffled_concat_mtpi",
        folder_suffix=SHUFFLED_SUFFIX,
    ),
}


@dataclass(frozen=True)
class TableSpec:
    """One paper table: which arms appear, in which order, under which label."""

    stem: str  # output file stem
    title: str
    rows: tuple[tuple[str, str], ...]  # (arm key, row label)
    decimals: int
    eval_subdir: str = ""  # "" = held-out stations; else a re-evaluation subfolder


TABLES: dict[str, TableSpec] = {
    "1": TableSpec(
        "table1_main",
        "Table 1 -- held-out station skill per region",
        (
            ("persistence", "Persistence"),
            ("era5_interp", "ERA5 interp. (+lapse for t2m)"),
            ("convcnp", "ConvCNP (topography-only)"),
            ("convcnp_hand", "ConvCNP (hand-crafted surface)"),
            ("tessera", "ConvCNP with Tessera"),
        ),
        decimals=2,
    ),
    "1p": TableSpec(
        "table1_main_preprint",
        "Table 1 (preprint) -- held-out station skill per region",
        (
            ("persistence", "Persistence"),
            ("era5_interp", "ERA5 interp. (+lapse for t2m)"),
            ("convcnp", "ConvCNP (topography-only)"),
            ("tessera", "ConvCNP with Tessera"),
        ),
        decimals=2,
    ),
    "4": TableSpec(
        "table4_descriptor_controls",
        "Table 4 / C1 -- Tessera descriptor controls",
        (
            ("convcnp", "None (no Tessera)"),
            ("tessera_shuffled", "VAE descriptor, shuffled"),
            ("tessera_stats", "Patch summary statistics"),
            ("tessera", "VAE descriptor"),
        ),
        decimals=3,
    ),
    "6": TableSpec(
        "table6_extended_descriptors",
        "Table 6 / B1 -- hand-crafted surface descriptors",
        (
            ("convcnp", "ConvCNP (topography-only)"),
            ("convcnp_hand", "+ hand-crafted surface"),
            ("tessera", "ConvCNP with Tessera"),
            ("tessera_hand", "+ hand-crafted surface"),
        ),
        decimals=3,
    ),
    "train": TableSpec(
        "table_train_stations",
        "Training-station skill at the held-out years",
        (
            ("convcnp", "ConvCNP (topography-only)"),
            ("tessera", "ConvCNP with Tessera"),
        ),
        decimals=3,
        eval_subdir="eval_train_stations",
    ),
}
DEFAULT_TABLES = ("1", "4", "6")


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
@dataclass
class Cell:
    """Seed-mean metrics of one (region, arm, variable)."""

    values: dict[str, float] = field(
        default_factory=lambda: dict.fromkeys(METRICS, math.nan)
    )
    n_seeds: int = 0


def read_run_metrics(run_dir: Path, var: str) -> dict[str, float] | None:
    """MAE / RMSE / CRPS of ``var`` from ``run_dir/test_summary.json``."""
    path = run_dir / "test_summary.json"
    if not path.exists():
        return None
    res = json.loads(path.read_text())
    dist = (res.get("head_spec") or {}).get(var, {}).get("distribution", "gaussian")
    mae_key, rmse_key = POINT_KEYS.get(dist, POINT_KEYS["gaussian"])

    def get(*keys: str) -> float:  # first present key wins; NaN if none
        for k in keys:
            if res.get(f"{var}_{k}") is not None:
                return float(res[f"{var}_{k}"])
        return math.nan

    return {
        "mae": get(mae_key, "mae"),
        "rmse": get(rmse_key, "rmse"),
        "crps": get("crps"),
    }


def load_cell(region_stem: str, arm: RunSpec, var: str, eval_subdir: str) -> Cell:
    """Seed-mean of ``arm``'s metrics for ``var`` in one region."""
    folder = training_runs_dir(region_stem + arm.folder_suffix)
    per_seed = []
    for seed in SEEDS:
        run_dir = folder / f"{arm.run(var)}_seed{seed}"
        if eval_subdir:
            run_dir = run_dir / eval_subdir
        rec = read_run_metrics(run_dir, var)
        if rec is not None:
            per_seed.append(rec)
    if not per_seed:
        return Cell()
    means = {}
    for m in METRICS:
        vals = [r[m] for r in per_seed if not math.isnan(r[m])]
        means[m] = float(np.mean(vals)) if vals else math.nan
    return Cell(means, len(per_seed))


@dataclass
class Row:
    """One table row: an arm scored in one region (or averaged over all)."""

    region: str
    arm: str  # key into ARMS
    label: str
    cells: dict[str, Cell]  # variable -> seed-mean metrics

    def value(self, var: str, metric: str) -> float:
        return self.cells[var].values[metric]


def region_rows(spec: TableSpec) -> list[Row]:
    """Rows of the per-region blocks, in table order."""
    rows = []
    for region_name, stem in REGIONS:
        for arm_key, label in spec.rows:
            arm = ARMS[arm_key]
            cells = {v: load_cell(stem, arm, v, spec.eval_subdir) for v in VARIABLES}
            for v, cell in cells.items():
                if cell.n_seeds < len(SEEDS):
                    where = f"{stem}{arm.folder_suffix}/{arm.run(v)}"
                    if spec.eval_subdir:
                        where += f"/{spec.eval_subdir}"
                    print(f"  [warn] {where}: {cell.n_seeds}/{len(SEEDS)} seeds found")
            rows.append(Row(region_name, arm_key, label, cells))
    return rows


def all_regions_rows(spec: TableSpec, rows: list[Row]) -> list[Row]:
    """Build the 'All regions' block: unweighted mean of the per-region rows."""
    out = []
    for arm_key, label in spec.rows:
        per_region = [r for r in rows if r.arm == arm_key]
        cells = {}
        for v in VARIABLES:
            means = {}
            for m in METRICS:
                vals = [r.value(v, m) for r in per_region]
                if all(math.isnan(x) for x in vals):  # e.g. CRPS of a point forecast
                    means[m] = math.nan
                elif any(math.isnan(x) for x in vals):  # never average fewer regions
                    print(
                        f"  [warn] {spec.stem}: '{label}' {v} {m} missing in some "
                        "region; 'All regions' left blank"
                    )
                    means[m] = math.nan
                else:
                    means[m] = float(np.mean(vals))
            cells[v] = Cell(means, min(r.cells[v].n_seeds for r in per_region))
        out.append(Row(ALL_REGIONS, arm_key, label, cells))
    return out


def build_table(spec: TableSpec) -> list[Row]:
    rows = region_rows(spec)
    return rows + all_regions_rows(spec, rows)


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------
def fmt(x: float, decimals: int, dash: str = "—") -> str:
    return dash if math.isnan(x) else f"{x:.{decimals}f}"


def _blocks(rows: list[Row]) -> list[list[Row]]:
    """Group consecutive rows by region (table order preserved)."""
    blocks: list[list[Row]] = []
    for row in rows:
        if blocks and blocks[-1][0].region == row.region:
            blocks[-1].append(row)
        else:
            blocks.append([row])
    return blocks


def to_markdown(spec: TableSpec, rows: list[Row]) -> str:
    head = ["Region", "Model"] + [f"{v} {m.upper()}" for v, m in COLUMNS]
    lines = [
        f"**{spec.title}** (mean over seeds {', '.join(map(str, SEEDS))})",
        "",
        "| " + " | ".join(head) + " |",
        "|" + "|".join(["---", "---"] + ["---:"] * len(COLUMNS)) + "|",
    ]
    for block in _blocks(rows):
        for i, row in enumerate(block):
            cells = [row.region if i == 0 else "", row.label]
            cells += [fmt(row.value(v, m), spec.decimals) for v, m in COLUMNS]
            lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def to_latex(spec: TableSpec, rows: list[Row]) -> str:
    n = len(METRICS)
    groups = " & ".join(
        rf"\multicolumn{{{n}}}{{c}}{{{VARIABLE_TITLES_TEX[v]}}}" for v in VARIABLES
    )
    rules = " ".join(
        rf"\cmidrule(lr){{{3 + i * n}-{2 + (i + 1) * n}}}"
        for i in range(len(VARIABLES))
    )
    lines = [
        "% " + spec.title,
        "% generated by scripts/paper/make_paper_tables.py -- do not edit by hand",
        r"\begin{tabular}{ll" + "r" * len(COLUMNS) + "}",
        r"\toprule",
        " & & " + groups + r" \\",
        rules,
        "Region & Model & " + " & ".join(m.upper() for _, m in COLUMNS) + r" \\",
        r"\midrule",
    ]
    for b, block in enumerate(_blocks(rows)):
        if b:
            lines.append(r"\addlinespace")
        for i, row in enumerate(block):
            cells = [row.region if i == 0 else "", row.label]
            cells += [
                fmt(row.value(v, m), spec.decimals, dash="---") for v, m in COLUMNS
            ]
            lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def to_csv(_spec: TableSpec, rows: list[Row]) -> str:
    """Wide CSV at full precision, plus the number of seeds behind each variable."""
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(
        ["region", "model"]
        + [f"{v}_{m}" for v, m in COLUMNS]
        + [f"{v}_n_seeds" for v in VARIABLES]
    )
    for row in rows:
        writer.writerow(
            [row.region, row.label]
            + [fmt(row.value(v, m), 6, dash="") for v, m in COLUMNS]
            + [row.cells[v].n_seeds for v in VARIABLES]
        )
    return buf.getvalue()


RENDERERS: dict[str, Callable[[TableSpec, list[Row]], str]] = {
    "md": to_markdown,
    "tex": to_latex,
    "csv": to_csv,
}


# ---------------------------------------------------------------------------
# Cross-checks against the published tables
# ---------------------------------------------------------------------------
# PUBLISHED[table key][region][arm key] = (t2m MAE, RMSE, CRPS, wind MAE, RMSE,
# CRPS) as printed in the paper (2 d.p. Table 1, 3 d.p. Tables 4 and 6); None
# marks a "---". Two Table 1 cells are known not to reproduce and are reported
# as [warn]: Southern Africa / hand-crafted t2m RMSE (recomputed 2.7246, printed
# 2.73 -- the paper's own Table 6 prints 2.725, so this is a double rounding)
# and All regions / hand-crafted wind MAE (recomputed 1.2949 = the mean of the
# five printed per-region values, printed 1.30 -- a typo for 1.29).
_ = None
PUBLISHED: dict[str, dict[str, dict[str, tuple[float | None, ...]]]] = {
    "1": {
        "Europe": {
            "persistence": (3.53, 4.94, _, 1.49, 2.13, _),
            "era5_interp": (1.35, 1.95, _, 1.44, 2.15, _),
            "convcnp": (1.17, 1.67, 0.84, 1.28, 1.87, 0.92),
            "convcnp_hand": (1.15, 1.65, 0.83, 1.24, 1.81, 0.89),
            "tessera": (1.10, 1.58, 0.79, 1.19, 1.74, 0.86),
        },
        "United States": {
            "persistence": (3.80, 5.29, _, 1.68, 2.32, _),
            "era5_interp": (1.52, 2.23, _, 1.63, 2.12, _),
            "convcnp": (1.41, 2.07, 1.03, 1.45, 1.96, 1.04),
            "convcnp_hand": (1.38, 2.02, 1.01, 1.42, 1.91, 1.01),
            "tessera": (1.30, 1.95, 0.95, 1.36, 1.84, 0.97),
        },
        "East Asia": {
            "persistence": (3.07, 4.05, _, 1.46, 2.04, _),
            "era5_interp": (1.22, 1.70, _, 1.44, 1.99, _),
            "convcnp": (1.35, 1.87, 0.97, 1.23, 1.71, 0.88),
            "convcnp_hand": (1.21, 1.67, 0.87, 1.23, 1.70, 0.88),
            "tessera": (1.10, 1.53, 0.79, 1.16, 1.62, 0.84),
        },
        "Southern Africa": {
            "persistence": (5.36, 6.90, _, 1.48, 2.07, _),
            "era5_interp": (1.69, 2.56, _, 1.38, 1.83, _),
            "convcnp": (2.00, 2.75, 1.45, 1.23, 1.65, 0.88),
            "convcnp_hand": (1.98, 2.73, 1.44, 1.25, 1.67, 0.90),
            "tessera": (1.79, 2.55, 1.31, 1.16, 1.57, 0.84),
        },
        "Australia": {
            "persistence": (5.60, 6.62, _, 2.03, 2.61, _),
            "era5_interp": (1.40, 2.06, _, 1.46, 1.88, _),
            "convcnp": (1.57, 2.16, 1.15, 1.42, 1.81, 1.02),
            "convcnp_hand": (1.50, 2.16, 1.12, 1.33, 1.71, 0.96),
            "tessera": (1.32, 1.92, 0.98, 1.29, 1.69, 0.95),
        },
        ALL_REGIONS: {
            "persistence": (4.27, 5.56, _, 1.63, 2.23, _),
            "era5_interp": (1.44, 2.10, _, 1.47, 1.99, _),
            "convcnp": (1.50, 2.10, 1.09, 1.32, 1.80, 0.95),
            "convcnp_hand": (1.44, 2.04, 1.05, 1.30, 1.76, 0.93),
            "tessera": (1.32, 1.90, 0.96, 1.23, 1.69, 0.89),
        },
    },
    "4": {
        "Europe": {
            "convcnp": (1.169, 1.673, 0.843, 1.283, 1.874, 0.918),
            "tessera_shuffled": (1.128, 1.616, 0.811, 1.236, 1.796, 0.892),
            "tessera_stats": (1.102, 1.583, 0.792, 1.212, 1.767, 0.867),
            "tessera": (1.098, 1.577, 0.789, 1.191, 1.739, 0.861),
        },
        "United States": {
            "convcnp": (1.413, 2.070, 1.029, 1.446, 1.960, 1.038),
            "tessera_shuffled": (1.339, 2.000, 0.976, 1.406, 1.905, 1.011),
            "tessera_stats": (1.315, 1.975, 0.960, 1.365, 1.847, 0.977),
            "tessera": (1.299, 1.946, 0.947, 1.362, 1.841, 0.973),
        },
        "East Asia": {
            "convcnp": (1.347, 1.867, 0.974, 1.231, 1.705, 0.882),
            "tessera_shuffled": (1.124, 1.563, 0.808, 1.173, 1.628, 0.840),
            "tessera_stats": (1.097, 1.533, 0.788, 1.164, 1.620, 0.833),
            "tessera": (1.097, 1.530, 0.788, 1.164, 1.622, 0.836),
        },
        "Southern Africa": {
            "convcnp": (1.998, 2.749, 1.447, 1.231, 1.646, 0.885),
            "tessera_shuffled": (1.879, 2.637, 1.367, 1.239, 1.659, 0.889),
            "tessera_stats": (1.928, 2.664, 1.394, 1.175, 1.588, 0.845),
            "tessera": (1.791, 2.546, 1.313, 1.164, 1.566, 0.835),
        },
        "Australia": {
            "convcnp": (1.573, 2.164, 1.149, 1.422, 1.812, 1.025),
            "tessera_shuffled": (1.358, 1.980, 1.006, 1.327, 1.727, 0.964),
            "tessera_stats": (1.313, 1.933, 0.982, 1.265, 1.639, 0.906),
            "tessera": (1.323, 1.922, 0.977, 1.292, 1.690, 0.945),
        },
    },
    "6": {
        "Europe": {
            "convcnp": (1.169, 1.673, 0.843, 1.283, 1.874, 0.918),
            "convcnp_hand": (1.152, 1.647, 0.829, 1.244, 1.813, 0.893),
            "tessera": (1.098, 1.577, 0.789, 1.191, 1.739, 0.861),
            "tessera_hand": (1.066, 1.522, 0.766, 1.182, 1.709, 0.845),
        },
        "United States": {
            "convcnp": (1.413, 2.070, 1.029, 1.446, 1.960, 1.038),
            "convcnp_hand": (1.381, 2.023, 1.007, 1.416, 1.910, 1.013),
            "tessera": (1.299, 1.946, 0.947, 1.362, 1.841, 0.973),
            "tessera_hand": (1.280, 1.918, 0.935, 1.353, 1.818, 0.965),
        },
        "East Asia": {
            "convcnp": (1.347, 1.867, 0.974, 1.231, 1.705, 0.882),
            "convcnp_hand": (1.207, 1.669, 0.870, 1.233, 1.697, 0.879),
            "tessera": (1.097, 1.530, 0.788, 1.164, 1.622, 0.836),
            "tessera_hand": (1.101, 1.530, 0.790, 1.149, 1.611, 0.825),
        },
        "Southern Africa": {
            "convcnp": (1.998, 2.749, 1.447, 1.231, 1.646, 0.885),
            "convcnp_hand": (1.980, 2.725, 1.438, 1.250, 1.671, 0.896),
            "tessera": (1.791, 2.546, 1.313, 1.164, 1.566, 0.835),
            "tessera_hand": (1.776, 2.540, 1.302, 1.173, 1.570, 0.842),
        },
        "Australia": {
            "convcnp": (1.573, 2.164, 1.149, 1.422, 1.812, 1.025),
            "convcnp_hand": (1.498, 2.159, 1.123, 1.331, 1.713, 0.961),
            "tessera": (1.323, 1.922, 0.977, 1.292, 1.690, 0.945),
            "tessera_hand": (1.369, 1.959, 1.004, 1.279, 1.660, 0.922),
        },
    },
}
PUBLISHED["1p"] = PUBLISHED["1"]  # the preprint rows are a subset of the AMS ones


def cross_check(key: str, spec: TableSpec, rows: list[Row]) -> tuple[int, int]:
    """Compare every published cell with its recomputed value.

    A cell passes when the recomputed value rounds to the printed one, i.e. it
    lies within half a unit of the printed precision. Returns (n_ok, n_bad).
    """
    published = PUBLISHED.get(key)
    if not published:
        print(f"  [info] {spec.stem}: no published values to check against")
        return 0, 0
    tol = 0.5 * 10**-spec.decimals + 1e-9
    n_ok = n_bad = 0
    for row in rows:
        want = published.get(row.region, {}).get(row.arm)
        if want is None:
            continue
        for (v, m), w in zip(COLUMNS, want, strict=True):
            got = row.value(v, m)
            where = f"{spec.stem} {row.region} / {row.label} {v} {m}"
            if w is None:
                if math.isnan(got):
                    n_ok += 1
                else:
                    n_bad += 1
                    print(f"  [warn] {where}: recomputed {got:.4f}, paper prints '---'")
            elif math.isnan(got) or abs(got - w) > tol:
                n_bad += 1
                print(
                    f"  [warn] {where}: recomputed {fmt(got, spec.decimals + 2)} "
                    f"!= published {w:.{spec.decimals}f}"
                )
            else:
                n_ok += 1
    tag = "[ok]" if n_bad == 0 else "[warn]"
    print(
        f"  {tag} {spec.stem}: {n_ok} published values reproduced, {n_bad} mismatch(es)"
    )
    return n_ok, n_bad


# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help=f"data root (default: ${paths.ENV_VAR} or {paths.DEFAULT_DATA_ROOT})",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=OUT_DEFAULT,
        help=f"output directory (default: {OUT_DEFAULT.relative_to(REPO_ROOT)}/)",
    )
    ap.add_argument("--format", choices=(*RENDERERS, "all"), default="all")
    ap.add_argument(
        "--tables",
        default=",".join(DEFAULT_TABLES),
        help=f"comma-separated table keys from {{{','.join(TABLES)}}} "
        f"(default: {','.join(DEFAULT_TABLES)})",
    )
    ap.add_argument("--quiet", action="store_true", help="do not echo the tables")
    args = ap.parse_args(argv)

    if args.data_root is not None:
        os.environ[paths.ENV_VAR] = str(args.data_root)
    keys = [k.strip() for k in args.tables.split(",") if k.strip()]
    unknown = [k for k in keys if k not in TABLES]
    if unknown:
        ap.error(f"unknown table key(s) {unknown}; choose from {list(TABLES)}")
    formats = list(RENDERERS) if args.format == "all" else [args.format]

    print(f"data root: {paths.data_root()}")
    args.out.mkdir(parents=True, exist_ok=True)
    total_bad = 0
    for key in keys:
        spec = TABLES[key]
        print(f"\n== {spec.title}")
        rows = build_table(spec)
        for fmt_name in formats:
            path = args.out / f"{spec.stem}.{fmt_name}"
            path.write_text(RENDERERS[fmt_name](spec, rows))
            print(f"  wrote {path}")
        if not args.quiet:
            print()
            print(to_markdown(spec, rows))
        total_bad += cross_check(key, spec, rows)[1]

    tag = "[ok]" if total_bad == 0 else "[warn]"
    print(
        f"\n{tag} cross-checks: {total_bad} mismatch(es) against the published tables"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
