"""Stage 1 of the Aurora-context pipeline: generate Aurora forecasts, cropped
to the regions of interest.

This script runs the pretrained Aurora 0.25 deg model from ERA5 initial
conditions and, for each harvested forecast frame, crops Aurora's global output
to the regions of interest *before* writing, so we never persist the ~286 MB
global frame (a full trainval run would be ~15 TB). Each region's crop is
written in the SAME on-disk layout as the ERA5 WeatherBench2 staging, under a
per-region subtree:

    <output_root>/lead{L}h/<region>/processed/era5_wb2_quarter_<var>/data/<valid_ts>.nc

The crop uses ``compute_grid_crop_indices`` from
``tessera_downscaling.preprocessing.helpers`` -- the same function the dataset
preprocessor uses, including the longitude roll that makes Europe's
0-deg-crossing box contiguous -- and is asserted bit-for-bit against each
region's reference grid in ``dataset_timestamp_global``, so the Aurora context
aligns exactly with the ERA5 dataset. Region bboxes come from the global
dataset's ``metadata.json``. Because the per-region crops are already regional,
the companion Stage-2 script (``scripts/preprocessing/preprocess_aurora.py``)
consumes them directly (no second crop).

Design decisions (see session notes):
  * Model: Aurora 0.25 deg *Pretrained*. The fine-tuned 0.25 checkpoint is
    matched to IFS HRES T0 inputs; we initialise from ERA5, so the pretrained
    model is the in-distribution, defensible choice. It also keeps the
    experiment a single distribution shift (reanalysis -> forecast).
  * We keep Aurora's full native field set (4 surface + 5 atmospheric vars at
    all 13 pressure levels = 69 dynamic fields per frame), fp32, leaving any
    channel/level subsetting to Stage 2 -- only the *spatial* extent is cropped
    here.
  * Only ``rollout``'s predictions are used, so no source patch to the upstream
    ``aurora`` package is required.

Lead times are harvested from a SINGLE rollout per init:
    6 h  -> step 1,   24 h -> step 4,   72 h -> step 12
so adding the 24 h lead costs no extra rollouts. Per-init we only roll out as
many steps as the longest lead that init actually feeds.

Inputs: 13-level ERA5 staging (``scripts/data/download_era5_wb2.py --levels aurora``)
and the ERA5 static file (z / lsm / slt); see ``scripts/aurora/submit_aurora_forecasts.sh``.
Needs the ``aurora`` extra (``uv sync --extra aurora``).

Usage (dry run first -- no model load, just accounting):
    uv run python scripts/aurora/generate_aurora_forecasts.py \
        --global-metadata <data root>/dataset_timestamp_global/metadata.json \
        --era5-staging-root <data root>/ingest/aurora_inputs \
        --static-file <data root>/ingest/processed/era5_static/era5_static_0p25_all.nc \
        --output-root <data root>/ingest/aurora \
        --dry-run

Real run (GPU node):        ... --model pretrained
Plumbing smoke test (GPU):  ... --model small --limit 3
"""

from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path

import pandas as pd

from tessera_downscaling.io_utils import WB_LEVELS  # the 13 Aurora pressure levels

# --------------------------------------------------------------------------- #
# Constants / mappings (no heavy imports -- this section is unit-testable).
# --------------------------------------------------------------------------- #

TIME_DELTA_HOURS = 6  # Aurora 0.25 native rollout step.
DEFAULT_LEADS_HOURS = [6, 24, 72]  # short / medium / long.

# Regions of interest cropped at write time; bboxes are read from the global
# dataset metadata so they match the ERA5 dataset exactly. The paper's Aurora
# datasets use europe and east_asia only.
DEFAULT_REGIONS = ["europe", "us", "east_asia", "australia", "southern_africa"]


# Aurora short name -> WeatherBench2 / ERA5-staging variable name. The staged
# .nc files (and the DataArray name inside them) use the WB2 name, so Stage 2's
# preprocessing reads them unchanged.
SURF_AURORA_TO_WB2 = {
    "2t": "2m_temperature",
    "10u": "10m_u_component_of_wind",
    "10v": "10m_v_component_of_wind",
    "msl": "mean_sea_level_pressure",
}
ATMOS_AURORA_TO_WB2 = {
    "t": "temperature",
    "u": "u_component_of_wind",
    "v": "v_component_of_wind",
    "q": "specific_humidity",
    "z": "geopotential",
}

# Aurora static_vars keys we feed from the ERA5 static file.
STATIC_KEYS = ["z", "slt", "lsm"]

# The 9 ERA5 staging variables Aurora consumes as input (used by the pre-flight
# existence check): 4 surface + 5 atmospheric.
INPUT_WB2_VARS = list(SURF_AURORA_TO_WB2.values()) + list(ATMOS_AURORA_TO_WB2.values())


def lead_to_step(lead_hours: int) -> int:
    """Rollout step index (1-based) whose valid time is `lead_hours` ahead."""
    if lead_hours % TIME_DELTA_HOURS != 0:
        raise ValueError(
            f"Lead {lead_hours}h is not a multiple of {TIME_DELTA_HOURS}h."
        )
    return lead_hours // TIME_DELTA_HOURS


# --------------------------------------------------------------------------- #
# Metadata / schedule (unit-testable with numpy+pandas only).
# --------------------------------------------------------------------------- #


def parse_ts(s: str) -> pd.Timestamp:
    """Parse a 'YYYY-MM-DD-HH' valid-timestamp string."""
    return pd.Timestamp(dt.datetime.strptime(s, "%Y-%m-%d-%H"))


def load_times(
    global_metadata_path: str | Path,
    split: str = "test",
    start_date: str | None = None,
    end_date: str | None = None,
) -> list[pd.Timestamp]:
    """Return the valid-times for a split, replicating the dataset's lexicographic
    rule (episodes_for_split) so the Aurora set matches the ERA5 dataset exactly:

        train    : s <= train_end            (all hours of the train_end day -> train)
        val      : train_end < s <= val_end
        test     : s > val_end
        trainval : s <= val_end               (train + val; the cross-lead training set)
        all      : every valid timestamp

    ``start_date``/``end_date`` (``YYYY-MM-DD`` or ``YYYY-MM-DD-HH``) optionally
    restrict the result to a sub-window, e.g. for chunking a big run by year.
    Counts are checked against metadata only when no date filter is applied.
    """
    import json

    meta = json.loads(Path(global_metadata_path).read_text())
    ts = meta["temporal_split"]
    te, ve = ts["train_end"], ts["val_end"]
    vt = meta["valid_timestamps"]
    if split == "train":
        sel = [s for s in vt if s <= te]
    elif split == "val":
        sel = [s for s in vt if te < s <= ve]
    elif split == "test":
        sel = [s for s in vt if s > ve]
    elif split == "trainval":
        sel = [s for s in vt if s <= ve]
    elif split == "all":
        sel = list(vt)
    else:
        raise ValueError(f"Unknown split: {split!r}")
    times = sorted(parse_ts(s) for s in sel)
    if start_date is not None:
        times = [t for t in times if t >= parse_ts_loose(start_date)]
    if end_date is not None:
        times = [t for t in times if t <= parse_ts_loose(end_date)]
    expected = {
        "train": ts.get("n_train_timestamps"),
        "val": ts.get("n_val_timestamps"),
        "test": ts.get("n_test_timestamps"),
    }.get(split)
    if (
        start_date is None
        and end_date is None
        and expected is not None
        and len(times) != expected
    ):
        raise ValueError(
            f"Derived {len(times)} {split} times but metadata says {expected}; check the split logic."
        )
    return times


def parse_ts_loose(s: str) -> pd.Timestamp:
    """Parse 'YYYY-MM-DD' or 'YYYY-MM-DD-HH' to a Timestamp."""
    return pd.Timestamp(s.replace("-", "/", 2)) if s.count("-") < 3 else parse_ts(s)


def build_schedule(
    test_times: list[pd.Timestamp], leads_hours: list[int]
) -> tuple[list[pd.Timestamp], dict[pd.Timestamp, list[tuple[int, int, pd.Timestamp]]]]:
    """Map each init time to the (lead, step, valid_time) harvests it produces.

    For valid time V and lead L the init is T0 = V - L. One rollout from T0
    yields V at step L/6. Returns the sorted union of init times and, per init,
    the list of harvests to take from its rollout.
    """
    per_init: dict[pd.Timestamp, list[tuple[int, int, pd.Timestamp]]] = {}
    for v in test_times:
        for lead in leads_hours:
            t0 = v - pd.Timedelta(hours=lead)
            per_init.setdefault(t0, []).append((lead, lead_to_step(lead), v))
    inits = sorted(per_init)
    return inits, per_init


def required_input_frames(inits: list[pd.Timestamp]) -> list[pd.Timestamp]:
    """Global ERA5 frames needed as input: each init T0 plus its T0-6h."""
    frames: set[pd.Timestamp] = set()
    for t0 in inits:
        frames.add(t0)
        frames.add(t0 - pd.Timedelta(hours=TIME_DELTA_HOURS))
    return sorted(frames)


def missing_input_frames(
    era5_staging_root: Path, inits: list[pd.Timestamp]
) -> list[tuple[str, pd.Timestamp]]:
    """Return (var, frame) pairs whose staged ERA5 input file is absent.

    Cheap pre-flight (1516 frames x 9 vars stat calls). This is what makes
    --dry-run a real go/no-go gate: it would have caught the 2021-12-27 gap.
    """
    missing: list[tuple[str, pd.Timestamp]] = []
    for f in required_input_frames(inits):
        for v in INPUT_WB2_VARS:
            if not input_path(era5_staging_root, v, f).exists():
                missing.append((v, f))
    return missing


# --------------------------------------------------------------------------- #
# Path helpers (unit-testable).
# --------------------------------------------------------------------------- #


def _file_name(t: pd.Timestamp) -> str:
    return f"{t.year:04d}-{t.month:02d}-{t.day:02d}-{t.hour:02d}.nc"


def input_path(era5_staging_root: Path, wb2_var: str, frame: pd.Timestamp) -> Path:
    return (
        era5_staging_root / f"era5_wb2_quarter_{wb2_var}" / "data" / _file_name(frame)
    )


def lead_root(output_root: Path, lead_hours: int, region: str) -> Path:
    """Root for one (lead, region) staging tree -- what Stage 2 points `root` at."""
    return output_root / f"lead{lead_hours}h" / region / "processed"


def output_path(
    output_root: Path, lead_hours: int, region: str, wb2_var: str, valid: pd.Timestamp
) -> Path:
    return (
        lead_root(output_root, lead_hours, region)
        / f"era5_wb2_quarter_{wb2_var}"
        / "data"
        / _file_name(valid)
    )


def harvest_done(
    output_root: Path, lead_hours: int, valid: pd.Timestamp, region_names: list[str]
) -> bool:
    """A (lead, valid) is done only once EVERY region's 2m_temperature exists.

    Using 'all regions present' as the marker means a run interrupted mid-frame
    (some regions written, others not) is correctly re-done on resume.
    """
    return all(
        output_path(output_root, lead_hours, r, "2m_temperature", valid).exists()
        for r in region_names
    )


# --------------------------------------------------------------------------- #
# Region cropping (same crop as the dataset preprocessor).
# --------------------------------------------------------------------------- #


def resolve_region_bboxes(global_metadata_path, region_names):
    """Map region name -> (lat_min, lat_max, lon_min, lon_max) from the global metadata."""
    import json

    meta = json.loads(Path(global_metadata_path).read_text())
    regs = meta.get("regions", {})
    out = {}
    for name in region_names:
        if name in regs and regs[name].get("bbox_lat_lon"):
            out[name] = tuple(regs[name]["bbox_lat_lon"])
        else:
            raise SystemExit(
                f"Region '{name}' has no bbox in the global metadata ({sorted(regs)})."
            )
    return out


def region_n_cells(bbox, res_deg: float = 0.25) -> int:
    """Cell count for a bbox on the 0.25deg grid (inclusive bounds)."""
    lat_min, lat_max, lon_min, lon_max = bbox
    n_lat = round((lat_max - lat_min) / res_deg) + 1
    n_lon = round((lon_max - lon_min) / res_deg) + 1
    return n_lat * n_lon


def resolve_region_crops(
    bboxes, aurora_lats, aurora_lons, global_dataset_dir, logger=None
):
    """Compute per-region crop indices on Aurora's OWN output grid.

    For regions that have a reference grid in the global dataset
    (regions/<r>/lats.npy), assert the crop matches it bit-for-bit (atol 1e-4),
    exactly as Stage 2's build_region does -- this catches the dropped-pole
    offset or any grid drift. Returns an ordered list of dicts with
    name/lat_idx/lon_idx/roll/lats/lons.
    """
    import numpy as np

    from tessera_downscaling.preprocessing.helpers import compute_grid_crop_indices

    crops = []
    for name, bbox in bboxes.items():
        lat_min, lat_max, lon_min, lon_max = bbox
        lat_idx, lon_idx, roll, lats_c, lons_c = compute_grid_crop_indices(
            aurora_lats, aurora_lons, (lat_min, lat_max), (lon_min, lon_max)
        )
        ref_dir = Path(global_dataset_dir) / "regions" / name
        ref_lat_f, ref_lon_f = ref_dir / "lats.npy", ref_dir / "lons.npy"
        if ref_lat_f.exists() and ref_lon_f.exists():
            ref_lats, ref_lons = np.load(ref_lat_f), np.load(ref_lon_f)
            if not (
                lats_c.shape == ref_lats.shape
                and lons_c.shape == ref_lons.shape
                and np.allclose(lats_c, ref_lats, atol=1e-4)
                and np.allclose(lons_c, ref_lons, atol=1e-4)
            ):
                raise AssertionError(
                    f"Region '{name}': Aurora crop {lats_c.shape}x{lons_c.shape} does not match "
                    f"the dataset reference grid {ref_lats.shape}x{ref_lons.shape} -- refusing to mis-crop."
                )
            checked = "matches dataset grid"
        else:
            checked = "NEW region (no reference grid to check against)"
        if logger:
            logger(
                f"  region {name:10s}: {len(lat_idx)}x{len(lon_idx)} grid, roll={roll}  [{checked}]"
            )
        crops.append(
            {
                "name": name,
                "lat_idx": lat_idx,
                "lon_idx": lon_idx,
                "roll": roll,
                "lats": lats_c,
                "lons": lons_c,
            }
        )
    return crops


# --------------------------------------------------------------------------- #
# Accounting / dry run (unit-testable).
# --------------------------------------------------------------------------- #


def estimate_output_gb(
    per_init,
    output_root: Path,
    leads_hours: list[int],
    bytes_per_value: int,
    region_names: list[str],
    region_cells: dict,
) -> float:
    n_fields = len(SURF_AURORA_TO_WB2) + len(ATMOS_AURORA_TO_WB2) * len(WB_LEVELS)  # 69
    cells_per_frame = sum(
        region_cells[r] for r in region_names
    )  # summed across regions
    n_frames = sum(
        1
        for harvs in per_init.values()
        for (lead, _step, valid) in harvs
        if not harvest_done(output_root, lead, valid, region_names)
    )
    return n_frames * n_fields * cells_per_frame * bytes_per_value / 1e9


def dry_run_report(
    test_times,
    inits,
    per_init,
    leads_hours,
    era5_staging_root: Path,
    output_root: Path,
    dtype_bytes: int,
    region_bboxes: dict,
    region_cells: dict,
    static_file: str | None = None,
) -> None:
    region_names = list(region_bboxes)
    in_frames = required_input_frames(inits)
    n_rollouts = sum(
        1
        for t0 in inits
        if any(
            not harvest_done(output_root, lead, valid, region_names)
            for (lead, _s, valid) in per_init[t0]
        )
    )
    total_steps = 0
    for t0 in inits:
        todo_steps = [
            s
            for (lead, s, valid) in per_init[t0]
            if not harvest_done(output_root, lead, valid, region_names)
        ]
        total_steps += max(todo_steps) if todo_steps else 0
    print("=" * 70)
    print("Aurora forecast generation -- DRY RUN")
    print("=" * 70)
    print(
        f"Leads (hours)         : {leads_hours}  ->  steps {[lead_to_step(h) for h in leads_hours]}"
    )
    print(
        f"Valid-times (split)   : {len(test_times)}  ({test_times[0]} .. {test_times[-1]})"
    )
    print(f"Init times (union)    : {len(inits)}  ({inits[0]} .. {inits[-1]})")
    print(f"Rollouts to run       : {n_rollouts}  (after skipping completed)")
    print(f"Total rollout steps   : {total_steps}")
    for lead in leads_hours:
        n = sum(
            1 for harvs in per_init.values() for (h, _s, valid) in harvs if h == lead
        )
        n_todo = sum(
            1
            for harvs in per_init.values()
            for (h, _s, valid) in harvs
            if h == lead and not harvest_done(output_root, lead, valid, region_names)
        )
        print(
            f"  lead {lead:>3}h frames    : {n} total, {n_todo} to write (x {len(region_names)} regions)"
        )
    print(
        f"Input frames needed   : {len(in_frames)}  ({in_frames[0]} .. {in_frames[-1]})"
    )
    print("Regions (cropped)     :")
    for r in region_names:
        lat_min, lat_max, lon_min, lon_max = region_bboxes[r]
        print(
            f"  {r:10s} bbox=[{lat_min},{lat_max},{lon_min},{lon_max}]  ~{region_cells[r]:,} cells"
        )
    pct = 100.0 * sum(region_cells[r] for r in region_names) / (721 * 1440)
    print(f"  -> cropped extent is ~{pct:.1f}% of the global grid")

    # --- Pre-flight existence checks (the go/no-go gate) ---
    missing = missing_input_frames(era5_staging_root, inits)
    if missing:
        print(
            f"  *** MISSING INPUTS  : {len(missing)} (var, frame) pairs absent under {era5_staging_root}"
        )
        print(f"      e.g. {[(v, str(f)) for v, f in missing[:5]]}")
    else:
        print(
            f"  input check         : all {len(in_frames) * len(INPUT_WB2_VARS)} required ERA5 files present  [ok]"
        )
    if static_file is not None:
        present = Path(static_file).exists()
        print(
            f"Static file           : {'present [ok]' if present else '*** MISSING ***'}  ({static_file})"
        )

    out_gb = estimate_output_gb(
        per_init, output_root, leads_hours, dtype_bytes, region_names, region_cells
    )
    print(
        f"Output storage (todo) : ~{out_gb:,.0f} GB  (69 fields x cropped grid x {dtype_bytes}B, summed over regions)"
    )
    go = (not missing) and (static_file is None or Path(static_file).exists())
    print("=" * 70)
    print(
        "PRE-FLIGHT:",
        "READY TO RUN" if go else "*** RESOLVE ISSUES ABOVE BEFORE RUNNING ***",
    )
    print("=" * 70)


# --------------------------------------------------------------------------- #
# Heavy lifting (lazy imports of torch / aurora / xarray).
# --------------------------------------------------------------------------- #


def _load_model(kind: str, device):
    """Load the requested Aurora model.

    Both spellings of the `microsoft-aurora` API are supported: the newer named
    classes (`AuroraPretrained`, `AuroraSmallPretrained`) and the older
    `Aurora(use_lora=...)` + manual checkpoint load.
    """
    import aurora  # noqa: F401

    if kind == "pretrained":
        try:
            from aurora import AuroraPretrained as Cls  # newer API

            model = Cls()
            model.load_checkpoint()
        except ImportError:
            from aurora import Aurora as Cls  # older API

            model = Cls(use_lora=False)
            model.load_checkpoint("microsoft/aurora", "aurora-0.25-pretrained.ckpt")
    elif kind == "small":
        try:
            from aurora import AuroraSmallPretrained as Cls

            model = Cls()
            model.load_checkpoint()
        except ImportError:
            from aurora import AuroraSmall as Cls

            model = Cls()
            model.load_checkpoint(
                "microsoft/aurora", "aurora-0.25-small-pretrained.ckpt"
            )
    else:
        raise ValueError(f"Unknown model kind: {kind}")

    model.eval()
    model = model.to(device)
    return model


def _open_static(static_file: Path):
    import xarray as xr

    ds = xr.open_dataset(static_file)
    for k in STATIC_KEYS:
        if k not in ds:
            raise KeyError(
                f"Static file {static_file} is missing '{k}'. Found: {list(ds.data_vars)}"
            )
    return ds


def _read_frame_var(
    era5_staging_root: Path, wb2_var: str, frame: pd.Timestamp, levels=None
):
    import xarray as xr

    p = input_path(era5_staging_root, wb2_var, frame)
    if not p.exists():
        raise FileNotFoundError(f"Missing ERA5 input frame: {p}")
    da = xr.open_dataset(p)[wb2_var]
    if levels is not None:
        da = da.sel(level=levels)  # explicit Aurora level order
    return da


def _build_batch(era5_staging_root: Path, t0: pd.Timestamp, static_ds, device):
    """Assemble an Aurora `Batch` from frames at (t0-6h, t0)."""
    import numpy as np
    import torch
    from aurora import Batch, Metadata

    frames = [t0 - pd.Timedelta(hours=TIME_DELTA_HOURS), t0]

    surf_vars = {}
    for a_name, wb2 in SURF_AURORA_TO_WB2.items():
        stk = np.stack(
            [_read_frame_var(era5_staging_root, wb2, f).values for f in frames], axis=0
        )
        surf_vars[a_name] = torch.from_numpy(stk).float()[None]  # (1, T=2, H, W)

    atmos_vars = {}
    for a_name, wb2 in ATMOS_AURORA_TO_WB2.items():
        stk = np.stack(
            [
                _read_frame_var(era5_staging_root, wb2, f, levels=WB_LEVELS).values
                for f in frames
            ],
            axis=0,
        )  # (T=2, C=13, H, W)
        atmos_vars[a_name] = torch.from_numpy(stk).float()[None]  # (1, T=2, C, H, W)

    static_vars = {
        k: torch.from_numpy(static_ds[k].squeeze().values).float() for k in STATIC_KEYS
    }

    # Grid from any surface file.
    ref = _read_frame_var(era5_staging_root, "2m_temperature", t0)
    lat = torch.from_numpy(ref["latitude"].values.copy()).float()
    lon = torch.from_numpy(ref["longitude"].values.copy()).float()

    # Sanity: Aurora expects 721x1440, lat descending from +90, lon 0..359.75.
    assert lat.shape[0] == 721 and lon.shape[0] == 1440, (
        f"Grid is {lat.shape[0]}x{lon.shape[0]}, expected 721x1440"
    )
    assert abs(float(lat[0]) - 90.0) < 1e-4 and float(lat[0]) > float(lat[-1]), (
        "lat must descend from +90"
    )
    assert abs(float(lon[0]) - 0.0) < 1e-4, "lon must start at 0"

    batch = Batch(
        surf_vars=surf_vars,
        static_vars=static_vars,
        atmos_vars=atmos_vars,
        metadata=Metadata(
            lat=lat,
            lon=lon,
            time=(t0.to_pydatetime(),),
            atmos_levels=tuple(WB_LEVELS),
        ),
    )
    return batch.to(device)


def _write_frame(
    output_root: Path,
    lead: int,
    valid: pd.Timestamp,
    pred,
    dtype_str: str,
    region_crops: list,
) -> None:
    """Crop one harvested forecast frame to each region and write per-region staging.

    For each region, the global field is rolled (so a 0-deg-crossing box is
    contiguous) and index-cropped to the region grid, then written under
    <output_root>/lead{L}h/<region>/processed/era5_wb2_quarter_<var>/data/<ts>.nc
    with the region's -180/180 lat/lon coords. region_crops is the list returned
    by resolve_region_crops (computed once against Aurora's output grid).
    """
    import numpy as np
    import xarray as xr

    levels = list(pred.metadata.atmos_levels)

    def _crop2d(arr, rc):  # (H, W) -> (region H, region W)
        return np.roll(arr, rc["roll"], axis=-1)[rc["lat_idx"]][:, rc["lon_idx"]]

    def _crop3d(arr, rc):  # (C, H, W) -> (C, region H, region W)
        return np.roll(arr, rc["roll"], axis=-1)[:, rc["lat_idx"]][:, :, rc["lon_idx"]]

    def _write(rc, wb2_var, da):
        out = output_path(output_root, lead, rc["name"], wb2_var, valid)
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_suffix(".nc.tmp")
        da.to_dataset(name=wb2_var).to_netcdf(tmp, engine="h5netcdf")
        tmp.replace(out)  # atomic rename

    # Pull each global field to host once, then crop it to every region.
    surf = {
        wb2: pred.surf_vars[a][0, 0].cpu().numpy().astype(dtype_str)
        for a, wb2 in SURF_AURORA_TO_WB2.items()
    }
    atmos = {
        wb2: pred.atmos_vars[a][0, 0].cpu().numpy().astype(dtype_str)  # (C, H, W)
        for a, wb2 in ATMOS_AURORA_TO_WB2.items()
    }

    for rc in region_crops:
        coords2d = {"latitude": rc["lats"], "longitude": rc["lons"]}
        for wb2, arr in surf.items():
            _write(
                rc,
                wb2,
                xr.DataArray(
                    _crop2d(arr, rc),
                    dims=("latitude", "longitude"),
                    coords=coords2d,
                    name=wb2,
                ),
            )
        for wb2, arr in atmos.items():
            _write(
                rc,
                wb2,
                xr.DataArray(
                    _crop3d(arr, rc),
                    dims=("level", "latitude", "longitude"),
                    coords={"level": levels, **coords2d},
                    name=wb2,
                ),
            )


def run(args) -> None:
    output_root = Path(args.output_root)
    era5_staging_root = Path(args.era5_staging_root)
    global_dataset_dir = Path(args.global_metadata).parent

    test_times = load_times(
        args.global_metadata, args.split, args.start_date, args.end_date
    )
    inits, per_init = build_schedule(test_times, args.leads)
    dtype_bytes = 2 if args.dtype == "float16" else 4

    # Region bboxes (from the global metadata) + cell counts.
    region_bboxes = resolve_region_bboxes(args.global_metadata, args.regions)
    region_names = list(region_bboxes)
    region_cells = {r: region_n_cells(bb) for r, bb in region_bboxes.items()}

    if args.limit:
        inits = inits[: args.limit]

    if args.dry_run:
        dry_run_report(
            test_times,
            inits,
            per_init,
            args.leads,
            era5_staging_root,
            output_root,
            dtype_bytes,
            region_bboxes,
            region_cells,
            static_file=args.static_file,
        )
        return

    # Heavy imports only on the real path (keeps --dry-run torch/aurora-free).
    import torch
    from aurora import rollout
    from tqdm import tqdm

    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    if device.type == "cpu" and not args.allow_cpu:
        raise SystemExit(
            "Refusing to run global Aurora on CPU -- it will exhaust host RAM and be OOM-killed.\n"
            f"  torch.cuda.is_available() = {torch.cuda.is_available()}\n"
            f"  torch {torch.__version__} (built for CUDA {torch.version.cuda})\n"
            "This usually means the installed torch was compiled for a newer CUDA than the node's\n"
            "driver supports. Install a torch CUDA build matching the driver (mirror your working\n"
            ".venv), or pass --allow-cpu to override (debug only)."
        )
    static_ds = _open_static(Path(args.static_file))
    model = _load_model(args.model, device)

    # Crop indices are resolved once, from Aurora's actual output grid (the first
    # prediction), then reused. This mirrors Stage 2 and asserts europe/east_asia
    # against the dataset reference grid before any write happens.
    region_crops = None

    for t0 in tqdm(inits, desc=f"Aurora rollouts ({args.model})"):
        harvests = per_init[t0]
        todo = [
            (lead, step, valid)
            for (lead, step, valid) in harvests
            if not harvest_done(output_root, lead, valid, region_names)
        ]
        if not todo:
            continue
        max_step = max(step for (_l, step, _v) in todo)
        want = {step: (lead, valid) for (lead, step, valid) in todo}

        batch = _build_batch(era5_staging_root, t0, static_ds, device)
        with torch.inference_mode():
            for i, pred in enumerate(rollout(model, batch, steps=max_step), start=1):
                if i in want:
                    lead, valid = want[i]
                    pred = pred.to("cpu")
                    if region_crops is None:
                        print(
                            f"Resolving region crops on Aurora's "
                            f"{len(pred.metadata.lat)}x{len(pred.metadata.lon)} output grid:"
                        )
                        region_crops = resolve_region_crops(
                            region_bboxes,
                            pred.metadata.lat.cpu().numpy(),
                            pred.metadata.lon.cpu().numpy(),
                            global_dataset_dir,
                            logger=print,
                        )
                    _write_frame(
                        output_root, lead, valid, pred, args.dtype, region_crops
                    )


def parse_args():
    p = argparse.ArgumentParser(
        description="Generate global Aurora forecasts in ERA5-staging layout (Stage 1)."
    )
    p.add_argument(
        "--global-metadata",
        required=True,
        help="Path to dataset_timestamp_global/metadata.json",
    )
    p.add_argument(
        "--era5-staging-root",
        required=True,
        help="Root of 13-level ERA5 staging (era5_wb2_quarter_<var>/data/)",
    )
    p.add_argument(
        "--static-file",
        required=True,
        help="era5_static_0p25_all.nc (provides z/lsm/slt)",
    )
    p.add_argument(
        "--output-root",
        required=True,
        help="Aurora staging root; writes lead{L}h/<region>/processed/... under it",
    )
    p.add_argument(
        "--regions",
        nargs="+",
        default=DEFAULT_REGIONS,
        help="Regions to crop to; bboxes are read from --global-metadata "
        "(default: all five dataset regions).",
    )
    p.add_argument(
        "--leads",
        type=int,
        nargs="+",
        default=DEFAULT_LEADS_HOURS,
        help="Lead times in hours",
    )
    p.add_argument(
        "--split",
        choices=["train", "val", "trainval", "test", "all"],
        default="test",
        help="Which split's valid-times to generate forecasts for. Default 'test' "
        "(backward compatible). Use 'trainval' for the cross-lead training set.",
    )
    p.add_argument(
        "--start-date",
        default=None,
        help="Optional lower bound on valid-time (YYYY-MM-DD[-HH]); for chunking a big run, e.g. by year.",
    )
    p.add_argument(
        "--end-date",
        default=None,
        help="Optional upper bound on valid-time (YYYY-MM-DD[-HH]).",
    )
    p.add_argument("--model", choices=["pretrained", "small"], default="pretrained")
    p.add_argument("--dtype", choices=["float32", "float16"], default="float32")
    p.add_argument("--device", default=None, help="e.g. cuda, cpu (default: auto)")
    p.add_argument(
        "--allow-cpu",
        action="store_true",
        help="Permit CPU execution (debug only; global model will OOM)",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Process only the first N inits (smoke test)",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print accounting and exit (no model load)",
    )
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
