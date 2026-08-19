"""Shared region configuration for the dense-downscaling map scripts.

Every script in scripts/maps/ generates figures for ONE region, selected via the
REGION environment variable (default "iberia"):

    REGION=norway uv run python scripts/maps/generate_maps.py

A "region" is a dense 0.05deg grid crop whose TESSERA latents live in
    <data_root>/processed/dense/<region>/<region>_<grid>_<year>.npz
sitting inside a parent ERA5 region (here always `europe`) whose trained ConvCNP
models and context grid we reuse. Iberia and Norway are both crops of europe, so
they share `training_runs_snapshot_14y_eu` and the europe snapshots/static fields.

Outputs are written region-prefixed under OUTPUTS/<region>/, e.g.
    <data_root>/paper_figure_outputs/maps_outputs/norway/norway_t2m_2022-07-18-12.png
OUTPUTS defaults to ``paths.paper_figure_inputs_dir()`` (the canonical location
that ``scripts/paper/make_paper_figures.py`` reads from) and can be redirected
with the TESSERA_MAPS_OUT environment variable.

Adding a region: drop its dense npz under processed/dense/<region>/ and add an
entry to REGIONS below (only the snapshot `dates` are really region-specific;
grid dims are derived from the npz at run time).
"""
from __future__ import annotations

import os
from pathlib import Path

from tessera_downscaling.paths import (
    dataset_dir,
    paper_figure_inputs_dir,
    processed_dir,
    training_runs_dir,
)

OUTPUTS = Path(os.environ.get("TESSERA_MAPS_OUT") or paper_figure_inputs_dir())

# Parent ERA5 region: context grid + static fields + per-timestamp snapshots, and
# the trained ConvCNP runs. Shared by every crop that lives inside europe.
EUROPE = dataset_dir("dataset_timestamp_global") / "regions" / "europe"
EU_RUNS = training_runs_dir("snapshot_14y_eu")

G = 9.80665            # standard gravity, geopotential z -> metres
Z_STATIC_IDX = 7       # 'z' channel in europe static_fields.npy
SDFOR_IDX = 10         # 'sdfor' = std dev of sub-grid orography (ruggedness)
SEEDS = [42, 123, 456]

# variable -> snapshot timestamp + run stems + plot styling. Shared across regions
# by default (same Europe models / test windows); a region may override via "jobs".
DEFAULT_JOBS = {
    "t2m": dict(
        ts="2022-07-18-12",
        tessera="t2m_snap_vae_lat16_concat_with_elev_no_static_wd",
        baseline="t2m_snap_bilinear_baseline_wd",
        cmap="turbo", unit="°C", unit_plain="C", title="2 m temperature",
    ),
    "wind": dict(
        ts="2022-12-12-12",
        # Truncated-normal head (matches the main results / cross_folder_analysis;
        # NOT the Gaussian wind_snap_* runs). Still no-mTPI: mTPI is undefined at map
        # scale. Point estimate is the head median (MAE@median), see run_model.
        tessera="wind_truncnormal_snap_vae_lat16_concat_with_elev_no_static_wd",
        baseline="wind_truncnormal_snap_bilinear_baseline_wd",
        cmap="viridis", unit="m s$^{-1}$", unit_plain="m/s", title="10 m wind speed",
    ),
}

# Per-region overrides. Dense-npz / DEM / output paths are DERIVED from the name;
# only behaviour that can't be derived lives here.
REGIONS = {
    "iberia": dict(
        region_data=EUROPE, runs=EU_RUNS, grid="0.05deg", year=2024,
        # Per-variable snapshot picked by select_dates.py (max spatial std on test).
        # NB the paper's Iberia panels use the DEFAULT_JOBS dates (2022-07-18-12 /
        # 2022-12-12-12; regenerate with MAPS_DATES="t2m=2022-07-18-12,wind=2022-12-12-12");
        # both date sets are cached under OUTPUTS/iberia/.
        dates={"t2m": "2022-07-22-18", "wind": "2022-12-13-18"},
    ),
    "norway": dict(
        region_data=EUROPE, runs=EU_RUNS, grid="0.05deg", year=2024,
        dates={"t2m": "2023-01-02-00", "wind": "2022-01-30-00"},
    ),
}


class Region:
    """Resolved paths + config for one region (paths derived from `name`)."""

    def __init__(self, name: str):
        if name not in REGIONS:
            raise SystemExit(f"unknown REGION={name!r}; known: {sorted(REGIONS)}")
        c = REGIONS[name]
        self.name = name
        self.region_data = Path(c["region_data"])
        self.runs = Path(c["runs"])
        # Jobs = shared styling/run-stems from DEFAULT_JOBS, with the per-variable
        # snapshot timestamp overridden by this region's `dates` (see select_dates.py).
        if "jobs" in c:
            self.jobs = c["jobs"]
        else:
            dates = c.get("dates", {})
            self.jobs = {v: {**spec, "ts": dates.get(v, spec["ts"])}
                         for v, spec in DEFAULT_JOBS.items()}
        # Optional ad-hoc date override, e.g. MAPS_DATES="t2m=2022-07-18-12,wind=2022-12-12-12"
        # — lets one build figures for any snapshot without editing REGIONS.
        env_dates = os.environ.get("MAPS_DATES")
        if env_dates:
            for kv in env_dates.split(","):
                v, _, tsv = kv.partition("=")
                v, tsv = v.strip(), tsv.strip()
                if v in self.jobs and tsv:
                    self.jobs[v] = {**self.jobs[v], "ts": tsv}
        grid, year = c.get("grid", "0.05deg"), c.get("year", 2024)
        ddir = processed_dir("dense", name)
        self.dense_npz = ddir / f"{name}_{grid}_{year}.npz"
        self.dem_path = ddir / f"{name}_{grid}_dem.npy"
        self.out_dir = OUTPUTS / name

    def fig(self, var: str, ts: str, tail: str = "") -> Path:
        """Output path under OUTPUTS/<region>/<var>_<ts>/, region+var+ts prefixed.

        e.g. fig('t2m', '2023-01-02-00', '_dem.png') ->
        OUTPUTS/norway/t2m_2023-01-02-00/norway_t2m_2023-01-02-00_dem.png
        """
        sub = self.out_dir / f"{var}_{ts}"
        sub.mkdir(parents=True, exist_ok=True)
        return sub / f"{self.name}_{var}_{ts}{tail}"


def get_region() -> Region:
    """Region selected by the REGION env var (default 'iberia')."""
    return Region(os.environ.get("REGION", "iberia"))
