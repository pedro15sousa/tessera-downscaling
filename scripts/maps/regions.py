"""Shared region configuration for the dense-downscaling map scripts.

Every script in scripts/maps/ generates figures for ONE region, selected via the
REGION environment variable (default "iberia"):

    REGION=norway .venv/bin/python projects/tessera_downscaling/scripts/maps/generate_maps.py

A "region" is a dense 0.05deg grid crop whose TESSERA latents live in
    .tmp_output/processed/dense/<region>/<region>_<grid>_<year>.npz
sitting inside a parent ERA5 region (here always `europe`) whose trained ConvCNP
models and context grid we reuse. Iberia and Norway are both crops of europe, so
they share `training_runs_snapshot_14y_eu` and the europe snapshots/static fields.

Outputs are written region-prefixed under outputs/<region>/, e.g.
    outputs/norway/norway_t2m_2022-07-18-12.png

Adding a region: drop its dense npz under processed/dense/<region>/ and add an
entry to REGIONS below (only `zoom_box` is really region-specific; grid dims are
derived from the npz at run time).
"""
from __future__ import annotations

import os
from pathlib import Path

REPO = Path("/lus/lfs1aip2/projects/u6do/pmms2/end-to-end-forecasting")
PROJ = REPO / "projects/tessera_downscaling"
BASE = PROJ / ".tmp_output"
OUTPUTS = PROJ / "scripts/maps/outputs"

# Parent ERA5 region: context grid + static fields + per-timestamp snapshots, and
# the trained ConvCNP runs. Shared by every crop that lives inside europe.
EUROPE = BASE / "dataset_timestamp_global/regions/europe"
EU_RUNS = BASE / "training_runs_snapshot_14y_eu"

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
# only behaviour that can't be derived lives here. `zoom_box`=[lon0,lon1,lat0,lat1]
# is an inland rugged-terrain close-up used by interpret_plots.py.
REGIONS = {
    "iberia": dict(
        region_data=EUROPE, runs=EU_RUNS, grid="0.05deg", year=2024,
        zoom_box=[-3.9, -2.1, 36.7, 37.6],   # Sierra Nevada / Betic massif
        # Per-variable snapshot picked by select_dates.py (max spatial std on test).
        dates={"t2m": "2022-07-22-18", "wind": "2022-12-13-18"},
    ),
    "norway": dict(
        region_data=EUROPE, runs=EU_RUNS, grid="0.05deg", year=2024,
        zoom_box=[7.0, 9.2, 61.0, 62.3],     # Jotunheimen massif (inland)
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
        self.zoom_box = c.get("zoom_box")
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
        ddir = BASE / "processed/dense" / name
        self.dense_npz = ddir / f"{name}_{grid}_{year}.npz"
        self.dem_path = ddir / f"{name}_{grid}_dem.npy"
        self.out_dir = OUTPUTS / name

    def fig(self, var: str, ts: str, tail: str = "") -> Path:
        """Output path under outputs/<region>/<var>_<ts>/, region+var+ts prefixed.

        e.g. fig('t2m', '2023-01-02-00', '_dem.png') ->
        outputs/norway/t2m_2023-01-02-00/norway_t2m_2023-01-02-00_dem.png
        """
        sub = self.out_dir / f"{var}_{ts}"
        sub.mkdir(parents=True, exist_ok=True)
        return sub / f"{self.name}_{var}_{ts}{tail}"


def get_region() -> Region:
    """Region selected by the REGION env var (default 'iberia')."""
    return Region(os.environ.get("REGION", "iberia"))
