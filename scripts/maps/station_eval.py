"""Per-station evaluation of baseline vs TESSERA at the chosen snapshot.

For each variable+timestamp, pulls the Europe TEST episode (GHCNh stations with
valid latents + a valid observation), runs both 3-seed ensembles at those exact
stations, and saves per-station predictions/errors and an `improved` flag
(TESSERA abs-error < baseline abs-error). Both models share the same station
targets; each uses its own context grid (TESSERA no-static, baseline with-static).

Uses the same v1-generation runs and v1 station latents as generate_maps.py
(see its docstring for provenance).

Saves: OUTPUTS/<region>/<var>_<ts>/<region>_<var>_<ts>_stations.npz

  REGION=norway uv run python scripts/maps/station_eval.py
"""

import json
import os

import numpy as np
import torch
from generate_maps import (
    EU,
    RUNS,
    SEEDS,
    R,
    bilinear_grid_to_points,
    build_ctx,
    build_model,
)

from tessera_downscaling.data.dataset import MultiRegionSnapshotDownscalingDataset
from tessera_downscaling.paths import dataset_dir, processed_dir

VAE_LAT = processed_dir("station_latents_lat16_grad0.5.npy")
VAE_CSV = processed_dir("tessera_global", "station_list_filtered.csv")
glats = np.load(EU / "lats.npy").astype(np.float32)
glons = np.load(EU / "lons.npy").astype(np.float32)

# Region bounding box (for the in-region station-improvement print), derived from
# the dense-grid coords so it tracks whichever REGION is selected.
_dco = np.load(R.dense_npz, allow_pickle=True)["coords"]
RLON0, RLON1 = float(_dco["lon"].min()), float(_dco["lon"].max())
RLAT0, RLAT1 = float(_dco["lat"].min()), float(_dco["lat"].max())

JOBS = R.jobs


@torch.no_grad()
def predict(model, var, ctx, tc, te, tde, tt, obs):
    """Return (median, mean, crps) per station. MAE @ median, RMSE @ mean, and CRPS
    (proper score) -- matches cross_folder_analysis (gaussian median==mean; the
    truncated-normal wind head differs, so median/mean/crps are all reported)."""
    out = model(
        torch.tensor(ctx[None]),
        torch.tensor(glats),
        torch.tensor(glons),
        torch.tensor(tc[None].astype(np.float32)),
        torch.tensor(te[None].astype(np.float32)),
        torch.tensor(tde[None].astype(np.float32)),
        None,
        tt,
    )
    head = model.heads.heads[var]
    p = out[var]
    obs_t = torch.tensor(obs[None].astype(np.float32))
    return (
        head.median(p)[0].numpy().astype(np.float32),
        head.mean(p)[0].numpy().astype(np.float32),
        head.crps(p, obs_t)[0].numpy().astype(np.float32),
    )


def main():
    _only = os.environ.get("MAPS_VARS")  # e.g. MAPS_VARS=wind
    for var, job in JOBS.items():
        if _only and var not in _only.split(","):
            continue
        ts = job["ts"]
        ds = MultiRegionSnapshotDownscalingDataset(
            dataset_dir=dataset_dir("dataset_timestamp_global"),
            region_specs={"europe": "test"},
            split="test",
            target_variables=[var],
            vae_latents_path=VAE_LAT,
            vae_latents_station_csv=VAE_CSV,
            vae_latents_zscore=True,
            include_static_fields=False,
            normalisation_policy="per_region",
        )
        idx = ds.timestamps.index(ts)
        ep = ds[idx]
        tc = ep["target_coords"].numpy()
        te = ep["target_elev"].numpy()
        tde = ep["target_delta_elev"].numpy()
        obs = ep["target_values"].numpy().astype(np.float32)
        tt_np = ep["target_tessera"].numpy().astype(np.float32)
        sidx = ep["target_station_indices"].numpy()
        slat = np.asarray(ds.station_lats)[sidx].astype(np.float32)
        slon = np.asarray(ds.station_lons)[sidx].astype(np.float32)
        N = len(obs)

        tcfg = json.load(
            open(RUNS / f"{job['tessera']}_seed{SEEDS[0]}" / "config.json")
        )
        bcfg = json.load(
            open(RUNS / f"{job['baseline']}_seed{SEEDS[0]}" / "config.json")
        )
        ctx_t = build_ctx(tcfg, ts, glats, glons)
        ctx_b = build_ctx(bcfg, ts, glats, glons)
        tt = torch.tensor(tt_np[None])

        tmed, tmean, tcrps, bmed, bmean, bcrps = [], [], [], [], [], []
        for s in SEEDS:
            mt, _, _ = build_model(
                RUNS / f"{job['tessera']}_seed{s}",
                n_ctx=ctx_t.shape[0],
                latent_dim=tt_np.shape[1],
            )
            md, mn, cr = predict(mt, var, ctx_t, tc, te, tde, tt, obs)
            tmed.append(md)
            tmean.append(mn)
            tcrps.append(cr)
            mb, _, _ = build_model(
                RUNS / f"{job['baseline']}_seed{s}",
                n_ctx=ctx_b.shape[0],
                latent_dim=tt_np.shape[1],
            )
            md, mn, cr = predict(mb, var, ctx_b, tc, te, tde, None, obs)
            bmed.append(md)
            bmean.append(mn)
            bcrps.append(cr)
        # Point estimate for MAE / win-rate / residual-alignment = seed-mean median.
        tp, bp = np.mean(tmed, 0), np.mean(bmed, 0)
        # Seed-mean mean (RMSE@mean) and seed-mean per-station CRPS (proper score).
        tp_mean, bp_mean = np.mean(tmean, 0), np.mean(bmean, 0)
        tp_crps, bp_crps = np.mean(tcrps, 0), np.mean(bcrps, 0)
        be, tee = np.abs(bp - obs), np.abs(tp - obs)
        be_mean, tee_mean = np.abs(bp_mean - obs), np.abs(tp_mean - obs)
        improved = tee < be

        # ERA5 bilinear-interpolation forecast at the same stations (no model)
        era5 = np.load(EU / "era5_snapshot" / f"{ts}.npy")
        if var == "t2m":
            era5_pred = bilinear_grid_to_points(era5[0], glats, glons, tc) - 273.15
        else:
            uu = bilinear_grid_to_points(era5[1], glats, glons, tc)
            vv = bilinear_grid_to_points(era5[2], glats, glons, tc)
            era5_pred = np.sqrt(uu * uu + vv * vv)
        era5_pred = era5_pred.astype(np.float32)
        era5_err = np.abs(era5_pred - obs)

        np.savez(
            R.fig(var, ts, "_stations.npz"),
            lat=slat,
            lon=slon,
            obs=obs,
            base_pred=bp,
            tess_pred=tp,
            era5_pred=era5_pred,
            base_err=be,
            tess_err=tee,
            era5_err=era5_err,
            improved=improved,
            station_idx=sidx,
            # Mean-based point estimate + errors for RMSE@mean (== median for gaussian t2m).
            base_pred_mean=bp_mean,
            tess_pred_mean=tp_mean,
            base_err_mean=be_mean,
            tess_err_mean=tee_mean,
            # Per-station seed-mean CRPS (proper score; era5 is deterministic so CRPS==MAE).
            base_crps=bp_crps,
            tess_crps=tp_crps,
            era5_crps=era5_err,
        )
        nib = (slon >= RLON0) & (slon <= RLON1) & (slat >= RLAT0) & (slat <= RLAT1)
        print(
            f"{var} {ts}: N={N} europe test stns | baseline MAE={be.mean():.3f} TESSERA MAE={tee.mean():.3f} "
            f"| improved {100 * improved.mean():.0f}% | in-{R.name} {nib.sum()} "
            f"MAE {be[nib].mean():.2f}->{tee[nib].mean():.2f}  CRPS {bp_crps[nib].mean():.2f}->{tp_crps[nib].mean():.2f} "
            f"(improved {100 * improved[nib].mean():.0f}%)"
        )


if __name__ == "__main__":
    main()
