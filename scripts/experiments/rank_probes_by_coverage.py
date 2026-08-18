"""Rank a probe set by descriptor-space coverage, for the placement experiment.

The temporal-rollout schedule deploys probe stations in a *random* order
(``build_rollout_schedule.py`` uses ``rng.permutation``). This script produces
the alternative: a *coverage-driven* deployment order per descriptor space, so
we can ask whether choosing WHERE to place stations (not just how many) using
TESSERA geometry lets a region become generalisable with fewer stations than
choosing by geography / elevation / ERA5-static / random.

For each descriptor space we run two greedy selection rules, both seeded with
the existing European (non-Norway) training set so the budget is only spent on
genuine gaps:
  * k-center (farthest-first) — radius-free; minimises worst-case gap.
  * max-coverage@R95         — maximises the reachability metric directly.
The output is an ORDERED station_id list per strategy; every budget point in
the placement sweep is a prefix of that order.

Descriptor spaces mirror ``norway_rollout_descriptors/norway_descriptor_spaces.py``:
  * geographic       — (latitude, longitude)
  * elevation+mTPI   — (elevation, delta_elevation, mtpi)
  * ERA5 static      — the coarse ERA5 static grid, bilinearly interpolated
  * TESSERA lat16    — the 16-d land-surface embedding

Each space is z-scored on the rest-EU-train reference before any distance is
taken (same frame as the reachability figure), so the coverage geometry matches
the reachability metric the figures report.

Run (defaults reproduce the Norway placement set):
    uv run --project projects/tessera_downscaling python \\
        projects/tessera_downscaling/scripts/experiments/rank_probes_by_coverage.py

Outputs (9 JSONs, one per strategy) into --out-dir:
    probe_order_random.json
    probe_order_kcenter_{geographic,elevation_mtpi,era5_static,tessera}.json
    probe_order_maxcov_{geographic,elevation_mtpi,era5_static,tessera}.json
Each: {"strategy", "descriptor", "selection", "ordered_station_ids", "n", ...}.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

REPO = Path(__file__).resolve().parents[4]
DEFAULT_BASE = REPO / "projects/tessera_downscaling/.tmp_output"
NB_LAT, NB_LON = (58.0, 71.0), (4.0, 31.0)   # Norway bbox (matches the figures)

# Descriptor spaces that get a coverage ordering. "random" is emitted too (a
# seeded permutation) so all five orderings flow through identical downstream
# schedule machinery.
DESCRIPTORS = ["geographic", "elevation_mtpi", "era5_static", "tessera"]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--base-dir", type=Path, default=DEFAULT_BASE,
                   help=f"Root holding processed/ and dataset_timestamp_global/. "
                        f"Default: {DEFAULT_BASE}")
    p.add_argument("--dataset-name", type=str, default="dataset_timestamp_global",
                   help="Dataset subdir under --base-dir (has stations.csv, "
                        "regions/europe/static_fields.npy).")
    p.add_argument("--latents-npy", type=Path, default=None,
                   help="TESSERA latents .npy. Default: "
                        "<base>/processed/station_latents_lat16_grad0.5.npy")
    p.add_argument("--latents-csv", type=Path, default=None,
                   help="Row-aligned station_id CSV. Default: "
                        "<base>/processed/tessera_global/station_list_filtered.csv")
    p.add_argument("--probe-ids-json", type=Path, default=None,
                   help="probe_station_ids.json = the candidate pool to order. "
                        "Default: the Norway lat16+mtpi rollout probe set.")
    p.add_argument("--out-dir", type=Path, default=None,
                   help="Where to write probe_order_<strategy>.json. Default: "
                        "<repo>/.../snapshot_14y_eu_placement_norway/orders")
    p.add_argument("--seed", type=int, default=0,
                   help="Seed for the random-order baseline.")
    p.add_argument("--force", action="store_true",
                   help="Overwrite existing order files. Default: refuse.")
    return p.parse_args()


def _load_spaces(base: Path, dataset_name: str, latents_npy: Path,
                 latents_csv: Path) -> tuple[pd.DataFrame, dict, dict]:
    """Return (stations_df, spaces, groups) mirroring the reachability figure.

    ``spaces`` maps descriptor name -> (n_stations, d) float array, row-aligned
    with ``stations_df``. ``groups`` maps group name -> boolean mask.
    """
    ds = base / dataset_name
    lat = np.load(latents_npy)
    ll = pd.read_csv(latents_csv)
    ll["station_id"] = ll["station_id"].astype(str)
    lat_of = {s: i for i, s in enumerate(ll["station_id"])}

    st = pd.read_csv(ds / "stations.csv")
    st["station_id"] = st["station_id"].astype(str)
    st["lrow"] = st["station_id"].map(lat_of)
    st = st[st["lrow"].notna()].copy()
    st["lrow"] = st["lrow"].astype(int)
    st = st[~np.isnan(lat[st["lrow"].to_numpy()]).any(axis=1)].reset_index(drop=True)
    Z16 = lat[st["lrow"].to_numpy()]

    # ERA5 static interpolant (raw time-invariant fields -> stations).
    eu_static = ds / "regions/europe"
    sfield = np.load(eu_static / "static_fields.npy")   # (n_static, H, W)
    glat = np.load(eu_static / "lats.npy")
    glon = np.load(eu_static / "lons.npy")
    qpts = np.column_stack([
        np.clip(st["latitude"].to_numpy(), glat.min(), glat.max()),
        np.clip(st["longitude"].to_numpy(), glon.min(), glon.max()),
    ])
    era5 = np.column_stack([
        RegularGridInterpolator((glat, glon), sfield[c], method="linear",
                                bounds_error=False, fill_value=None)(qpts)
        for c in range(sfield.shape[0])
    ]).astype(np.float32)

    in_nb = (st.latitude.between(*NB_LAT) & st.longitude.between(*NB_LON)
             & (st.region == "europe")).to_numpy()
    eu = (st.region == "europe").to_numpy()
    tr = (st.spatial_split == "train").to_numpy()
    te = (st.spatial_split == "test").to_numpy()
    groups = {
        "norway_test":  in_nb & eu & te,
        "norway_probe": in_nb & eu & tr,
        "rest_train":   eu & tr & ~in_nb,
    }
    spaces = {
        "geographic":     st[["latitude", "longitude"]].to_numpy().astype(np.float64),
        "elevation_mtpi": st[["elevation", "delta_elevation", "mtpi"]].to_numpy().astype(np.float64),
        "era5_static":    era5.astype(np.float64),
        "tessera":        Z16.astype(np.float64),
    }
    return st, spaces, groups


def _r95(Xs: np.ndarray, ref_rows: np.ndarray) -> float:
    """95th-percentile of the reference set's leave-one-out NN distance.

    The in-distribution coverage radius used by the reachability metric
    (norway_descriptor_spaces.py): a query point is "reachable" iff its nearest
    reference neighbour lies within R95.
    """
    R = Xs[ref_rows]
    d_loo = NearestNeighbors(n_neighbors=2).fit(R).kneighbors(R)[0][:, 1]
    return float(np.percentile(d_loo, 95))


def _farthest_first_order(Xs: np.ndarray, cand_rows: np.ndarray,
                          ref_rows: np.ndarray) -> list[int]:
    """Greedy k-center order over ``cand_rows``, seeded with ``ref_rows``.

    Returns candidate row-indices in deployment order: at each step, the
    candidate whose nearest already-covered point (reference ∪ picked) is
    farthest. Minimises the worst-case distance from any candidate to a covered
    point (Gonzalez farthest-first traversal). Radius-free.
    """
    C = Xs[cand_rows]                                   # (m, d)
    R = Xs[ref_rows]                                    # (r, d)
    dmin = NearestNeighbors(n_neighbors=1).fit(R).kneighbors(C)[0][:, 0].copy()
    remaining = np.ones(len(cand_rows), dtype=bool)
    order: list[int] = []
    for _ in range(len(cand_rows)):
        pick = int(np.argmax(np.where(remaining, dmin, -np.inf)))
        order.append(pick)
        remaining[pick] = False
        diff = C - C[pick]
        d_to_pick = np.sqrt(np.einsum("ij,ij->i", diff, diff))
        dmin = np.minimum(dmin, d_to_pick)
    return [int(cand_rows[i]) for i in order]


def _max_coverage_order(Xs: np.ndarray, cand_rows: np.ndarray,
                        ref_rows: np.ndarray, r95: float) -> list[int]:
    """Greedy max-coverage@R95 order over ``cand_rows``, seeded with ``ref_rows``.

    Universe to cover = the candidate pool itself (a sample of the region).
    A candidate "covers" every candidate within R95 of it. At each step pick
    the candidate that brings the most currently-uncovered candidates within
    R95 (marginal coverage); ties (and the saturated tail, once everything is
    covered) break toward the most isolated candidate (largest distance to the
    reference set), so the order stays a full permutation. Directly maximises
    the reachability metric the figures plot. The universe is the candidate
    pool, NOT the held-out norway_test, so selection stays blind to evaluation.
    """
    C = Xs[cand_rows]                                   # (m, d)
    R = Xs[ref_rows]
    m = len(cand_rows)
    d_to_ref = NearestNeighbors(n_neighbors=1).fit(R).kneighbors(C)[0][:, 0]
    # within[i, j] = candidate j is within R95 of candidate i (what i covers).
    within = (cdist(C, C) <= r95).astype(np.float32)
    covered = d_to_ref <= r95                           # already covered by ref
    picked = np.zeros(m, dtype=bool)
    iso = d_to_ref / (d_to_ref.max() + 1e-9)            # isolation tiebreak in [0,1)
    order: list[int] = []
    for _ in range(m):
        marginal = within @ (~covered).astype(np.float32)   # uncovered count per cand
        score = marginal + iso                          # +<1 tiebreak never crosses an integer
        score[picked] = -np.inf
        pick = int(np.argmax(score))
        order.append(pick)
        picked[pick] = True
        covered |= within[pick] > 0
        covered[pick] = True
    return [int(cand_rows[i]) for i in order]


def main() -> None:
    args = _parse_args()
    base = args.base_dir
    latents_npy = args.latents_npy or base / "processed/station_latents_lat16_grad0.5.npy"
    latents_csv = args.latents_csv or base / "processed/tessera_global/station_list_filtered.csv"
    probe_json = args.probe_ids_json or (
        REPO / "projects/tessera_downscaling/scripts/experiments"
        / "snapshot_14y_eu_temporal_rollout_norway_lat16_mtpi/probe_station_ids.json")
    out_dir = args.out_dir or (
        REPO / "projects/tessera_downscaling/scripts/experiments"
        / "snapshot_14y_eu_placement_norway/orders")

    for pth in (latents_npy, latents_csv, probe_json):
        if not Path(pth).exists():
            sys.exit(f"ERROR: required input not found: {pth}")
    out_dir.mkdir(parents=True, exist_ok=True)

    st, spaces, groups = _load_spaces(base, args.dataset_name, latents_npy, latents_csv)
    sid = st["station_id"].to_numpy()
    id_to_row = {s: i for i, s in enumerate(sid)}

    probe_ids = [str(s) for s in json.loads(probe_json.read_text())["probe_station_ids"]]
    # Candidate pool = probe ids present in the joined table AND non-NaN in
    # EVERY space, so all five orderings cover an identical set (comparable
    # budgets). Report any drops.
    cand_rows0 = np.array([id_to_row[s] for s in probe_ids if s in id_to_row], dtype=int)
    finite = np.ones(len(cand_rows0), dtype=bool)
    for name, X in spaces.items():
        finite &= np.isfinite(X[cand_rows0]).all(axis=1)
    dropped = len(probe_ids) - int(finite.sum())
    cand_rows = cand_rows0[finite]
    print(f"probe pool: {len(probe_ids)} ids -> {len(cand_rows)} candidates "
          f"({dropped} dropped: missing from table or NaN descriptor)")
    print(f"reference (rest_train): {int(groups['rest_train'].sum())} stations")
    print(f"held-out norway_test:   {int(groups['norway_test'].sum())} stations")

    ref_rows = np.where(groups["rest_train"])[0]
    # records: strategy -> (selection, descriptor, ordered_station_ids)
    records: dict[str, tuple[str, str | None, list[str]]] = {}

    # Coverage orderings per descriptor: z-score on rest_train (its non-NaN
    # rows), then both k-center (radius-free) and max-coverage@R95.
    for name in DESCRIPTORS:
        X = spaces[name]
        ref_ok = ref_rows[np.isfinite(X[ref_rows]).all(axis=1)]
        Xs = StandardScaler().fit(X[ref_ok]).transform(X)
        r95 = _r95(Xs, ref_ok)
        kc = _farthest_first_order(Xs, cand_rows, ref_ok)
        mc = _max_coverage_order(Xs, cand_rows, ref_ok, r95)
        records[f"kcenter_{name}"] = ("kcenter_farthest_first", name, [sid[r] for r in kc])
        records[f"maxcov_{name}"] = ("max_coverage_r95", name, [sid[r] for r in mc])
        print(f"  ordered {len(kc)} candidates by {name} (r95={r95:.3f}): kcenter + maxcov")

    # Random baseline (seeded permutation of the same candidate pool).
    rng = np.random.default_rng(args.seed)
    rand = [sid[r] for r in cand_rows[rng.permutation(len(cand_rows))]]
    records["random"] = ("random", None, rand)

    for strategy, (selection, descriptor, ordered) in records.items():
        out = out_dir / f"probe_order_{strategy}.json"
        if out.exists() and not args.force:
            sys.exit(f"ERROR: {out} exists. Pass --force to overwrite.")
        out.write_text(json.dumps({
            "strategy":            strategy,
            "descriptor":          descriptor,
            "selection":           selection,
            "reference":           "rest_train (europe, train, non-Norway)",
            "seed":                args.seed if strategy == "random" else None,
            "n":                   len(ordered),
            "probe_ids_json":      str(probe_json),
            "ordered_station_ids": ordered,
        }, indent=2))
        print(f"wrote {out.name}")


if __name__ == "__main__":
    main()
